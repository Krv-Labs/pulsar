//! Parity for the longitudinal panel (`pulsar/mcp/longitudinal.py`): the
//! pivot under every missing-data policy, the panel preprocessing, the config
//! adaptation, the cost estimate, the representation guards and every
//! structured error, as `generate_panel.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze them. Tables are rebuilt from their typed inline columns.

use pulsar_analysis::longitudinal::{
    adapt_config_to_panel, estimate_costs, guard_representation, pivot_panel, prepare_panel_frame,
    Panel, PanelError,
};
use pulsar_data::{Column, Table};
use pulsar_parity::{assert_close, golden_dir, Manifest};
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = golden_dir().join("temporal/panel.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn table(spec: &Value) -> Table {
    let mut names = vec![];
    let mut columns = vec![];
    for col in spec.as_array().unwrap() {
        names.push(col["name"].as_str().unwrap().to_owned());
        let values = col["values"].as_array().unwrap();
        columns.push(match col["dtype"].as_str().unwrap() {
            "float64" => Column::Float(
                values
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    .collect(),
            ),
            "int64" => Column::Int(values.iter().map(|v| v.as_i64().unwrap()).collect()),
            "bool" => Column::Bool(values.iter().map(Value::as_bool).collect()),
            _ => Column::Text(
                values
                    .iter()
                    .map(|v| v.as_str().map(str::to_owned))
                    .collect(),
            ),
        });
    }
    Table::new(names, columns).unwrap()
}

fn features(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().map(|s| s.as_str().unwrap().to_owned()).collect())
        .unwrap_or_default()
}

fn config(yaml: &Value) -> pulsar_config::PulsarConfig {
    pulsar_config::from_yaml_str(yaml.as_str().unwrap()).unwrap()
}

fn error_json(e: &PanelError) -> Value {
    json!({
        "reason": e.reason,
        "error_code": e.error_code,
        "agent_action": e.agent_action,
        "details": e.details,
    })
}

fn check_panel(label: &str, got: &Panel, want: &Value, rtol: f64) {
    assert_eq!(
        got.snapshots.len(),
        want["snapshots"].as_array().unwrap().len(),
        "{label}"
    );
    for (k, (snap, w)) in got
        .snapshots
        .iter()
        .zip(want["snapshots"].as_array().unwrap())
        .enumerate()
    {
        assert_eq!(
            json!(snap.shape()),
            w["shape"],
            "{label}: snapshot {k} shape"
        );
        let want_data: Vec<f64> = w["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let got_data: Vec<f64> = snap.iter().copied().collect();
        assert_close(
            &format!("{label}: snapshot {k}"),
            &got_data,
            &want_data,
            rtol,
        );
    }
    assert_eq!(
        json!(got.entity_ids),
        want["entity_ids"],
        "{label}: entity_ids"
    );
    assert_eq!(json!(got.times), want["times"], "{label}: times");
    assert_eq!(
        json!(got.snapshot_entity_ids),
        want["snapshot_entity_ids"],
        "{label}: snapshot_entity_ids"
    );
    assert_eq!(
        json!(got.feature_columns),
        want["feature_columns"],
        "{label}: features"
    );
    assert_eq!(got.report, want["report"], "{label}: report");
    assert_eq!(
        json!(got.is_aligned()),
        want["is_aligned"],
        "{label}: is_aligned"
    );
}

fn check_result(label: &str, got: Result<Panel, PanelError>, want: &Value, rtol: f64) {
    match (got, want.get("ok")) {
        (Ok(panel), Some(ok)) => check_panel(label, &panel, ok, rtol),
        (Err(e), None) => assert_eq!(error_json(&e), want["error"], "{label}: error"),
        (Ok(_), None) => panic!("{label}: expected {}", want["error"]),
        (Err(e), Some(_)) => panic!("{label}: unexpected error {e:?}"),
    }
}

#[test]
fn pivots_match_python_under_every_policy() {
    let fx = fixture();
    let rtol = Manifest::load().rtol("panel");
    for case in fx["pivots"].as_array().unwrap() {
        let label = case["name"].as_str().unwrap();
        let got = pivot_panel(
            &table(&case["table"]),
            case["entity_column"].as_str().unwrap(),
            case["time_column"].as_str().unwrap(),
            &features(&case["feature_columns"]),
            case["on_missing"].as_str().unwrap(),
        );
        check_result(label, got, &case["result"], rtol);
    }
}

#[test]
fn preprocessing_then_pivot_matches_python() {
    let fx = fixture();
    let rtol = Manifest::load().rtol("panel");
    for case in fx["prepared"].as_array().unwrap() {
        let label = case["name"].as_str().unwrap();
        let cfg = config(&case["config_yaml"]);
        let feats = features(&case["feature_columns"]);
        let frame = prepare_panel_frame(
            &table(&case["table"]),
            &cfg.preprocessing,
            "patient",
            "visit",
            &feats,
        )
        .unwrap();
        let ok = &case["result"]["ok"];
        assert_eq!(json!(frame.names), ok["columns"], "{label}: columns");
        let panel = pivot_panel(
            &frame,
            "patient",
            "visit",
            &feats,
            case["on_missing"].as_str().unwrap(),
        );
        check_result(label, panel, &json!({"ok": ok["panel"]}), rtol);
    }
}

#[test]
fn config_adaptation_costs_and_guards_match_python() {
    let fx = fixture();
    let long = table(&fx["adapted_table"]);
    let panel = pivot_panel(&long, "e", "t", &[], "drop_entity").unwrap();
    for case in fx["adapted"].as_array().unwrap() {
        let label = case["name"].as_str().unwrap();
        let (adapted, notes) =
            adapt_config_to_panel(&config(&case["config_yaml"]), &panel).unwrap();
        assert_eq!(
            json!(adapted.projection.dimensions),
            case["dimensions"],
            "{label}: dims"
        );
        assert_eq!(
            json!(adapted.ball_mapper.epsilons),
            case["epsilons"],
            "{label}: eps"
        );
        assert_eq!(Value::Object(notes), case["notes"], "{label}: notes");
        assert_eq!(
            estimate_costs(&panel, &adapted),
            case["costs"],
            "{label}: costs"
        );
        for (rep, want) in case["guards"].as_object().unwrap() {
            match guard_representation(&panel, rep) {
                Ok(()) => assert!(want.get("ok").is_some(), "{label}/{rep}: expected error"),
                Err(e) => assert_eq!(error_json(&e), want["error"], "{label}/{rep}"),
            }
        }
    }
}

#[test]
fn every_panel_error_matches_python() {
    let fx = fixture();
    let plain = pulsar_config::from_yaml_str(
        "run: {name: golden_panel}\nsweep:\n  projection: {method: jl, dimensions: {values: [2, 3]}, seed: {values: [42, 7]}}\n  ball_mapper: {epsilon: {values: [0.5, 1.0]}}\n",
    )
    .unwrap();
    for case in fx["errors"].as_array().unwrap() {
        let label = case["name"].as_str().unwrap();
        let t = table(&case["table"]);
        let entity = case["entity"].as_str().unwrap();
        let time = case["time"].as_str().unwrap();
        let feats = features(&case["features"]);
        let err = match case["kind"].as_str().unwrap() {
            "pivot" => pivot_panel(
                &t,
                entity,
                time,
                &feats,
                case["policy"].as_str().unwrap_or("drop_entity"),
            )
            .unwrap_err(),
            "prepare" => prepare_panel_frame(
                &t,
                &config(&case["config_yaml"]).preprocessing,
                entity,
                time,
                &feats,
            )
            .unwrap_err(),
            "adapt" => {
                let panel = pivot_panel(&t, entity, time, &feats, "drop_entity").unwrap();
                adapt_config_to_panel(&plain, &panel).unwrap_err()
            }
            _ => {
                let panel = pivot_panel(&t, entity, time, &[], "allow_ragged").unwrap();
                guard_representation(&panel, case["representation"].as_str().unwrap()).unwrap_err()
            }
        };
        assert_eq!(error_json(&err), case["error"], "{label}");
    }
}
