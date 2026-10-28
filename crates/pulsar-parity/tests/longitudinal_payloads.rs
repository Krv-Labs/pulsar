//! Parity for the longitudinal payload builders: what
//! `generate_longitudinal_payloads.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from
//! the trajectory and temporal builds of `trajectory.json` / `temporal.json`,
//! synthetic label sequences for the levenshtein/dtw cohorts, and the entity
//! sample. Payloads are compared exactly (they are rounded as Python rounds).

use ndarray::{Array2, ArrayD, IxDyn};
use pulsar_analysis::longitudinal::{
    archetype_payload, classify_labels, classify_trajectories_payload, cross_time_payload,
    select_longitudinal_entities, temporal_diagnosis, trajectory_diagnosis,
};
use pulsar_core::TemporalCosmicGraph;
use pulsar_data::{Column, Key, Table};
use pulsar_parity::golden_dir;
use pulsar_pipeline::{CosmicTrajectory, EntityIds, Observation};
use serde_json::{json, Value};

fn load(name: &str) -> Value {
    let path = golden_dir().join("temporal").join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn key(v: &Value) -> Key {
    match v {
        Value::String(s) => Key::Text(s.clone()),
        Value::Bool(b) => Key::Bool(*b),
        v if v.is_i64() => Key::Int(v.as_i64().unwrap()),
        v => Key::Float(v.as_f64().unwrap()),
    }
}

fn keys(v: &Value) -> Vec<Key> {
    v.as_array().unwrap().iter().map(key).collect()
}

fn dense(m: &Value) -> ArrayD<f64> {
    let s: Vec<usize> = serde_json::from_value(m["shape"].clone()).unwrap();
    let d: Vec<f64> = serde_json::from_value(m["data"].clone()).unwrap();
    ArrayD::from_shape_vec(IxDyn(&s), d).unwrap()
}

fn matrix(m: &Value) -> Array2<f64> {
    dense(m).into_dimensionality().unwrap()
}

fn build(input: &Value) -> CosmicTrajectory {
    let snapshots: Vec<Array2<f64>> = input["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .map(matrix)
        .collect();
    let config = pulsar_config::from_yaml_str(input["config_yaml"].as_str().unwrap()).unwrap();
    let ids = match input["ids_kind"].as_str().unwrap() {
        "shared" => EntityIds::Shared(keys(&input["ids"])),
        "per_snapshot" => {
            EntityIds::PerSnapshot(input["ids"].as_array().unwrap().iter().map(keys).collect())
        }
        _ => EntityIds::Positional,
    };
    let stamps = (!input["timestamps"].is_null()).then(|| keys(&input["timestamps"]));
    CosmicTrajectory::from_snapshots(
        &snapshots,
        &config,
        &ids,
        stamps.as_deref(),
        input["similarity_threshold"].as_f64().unwrap(),
        input["scale"].as_bool().unwrap(),
    )
    .unwrap()
}

fn same(label: &str, got: &Value, want: &Value) {
    if got != want {
        panic!(
            "{label}\n got: {}\nwant: {}",
            serde_json::to_string(got).unwrap(),
            serde_json::to_string(want).unwrap()
        );
    }
}

/// Compare a payload whose `list` is ranked by `rank`: everything else exactly,
/// the rank sequence exactly, and the items as a multiset. numpy's unstable
/// sort left the order inside a tie unspecified (Rust keeps first appearance).
/// A truncated list may cut a tie differently, so its items need only appear
/// in Python's untruncated list `full`.
fn ranked_same(label: &str, got: &Value, want: &Value, list: &str, rank: &str, full: &Value) {
    let strip = |v: &Value| {
        let mut v = v.clone();
        v[list] = json!(null);
        v
    };
    same(
        &format!("{label} (outside {list})"),
        &strip(got),
        &strip(want),
    );
    let items = |v: &Value| v[list].as_array().unwrap().clone();
    let (g, w) = (items(got), items(want));
    let ranks = |xs: &[Value]| xs.iter().map(|x| x[rank].clone()).collect::<Vec<_>>();
    assert_eq!(ranks(&g), ranks(&w), "{label}: {rank} sequence");
    let pool = items(full);
    for item in &g {
        assert!(
            pool.contains(item),
            "{label}: {item} is not among Python's candidates"
        );
    }
    let key = |x: &Value| serde_json::to_string(x).unwrap();
    let mut gs: Vec<String> = g.iter().map(key).collect();
    let mut ws: Vec<String> = w.iter().map(key).collect();
    gs.sort();
    ws.sort();
    if g.len() == pool.len() {
        assert_eq!(gs, ws, "{label}: items");
    }
}

fn result(r: Result<Value, String>) -> Value {
    match r {
        Ok(v) => json!({"ok": v}),
        Err(e) => json!({"error": e}),
    }
}

#[test]
fn trajectory_payloads_match_python() {
    let builds = load("trajectory.json");
    let fx = load("payloads.json");
    for (case, want) in builds["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(fx["trajectory"].as_array().unwrap())
    {
        let name = want["name"].as_str().unwrap();
        let ct = build(&case["input"]);
        for detail in ["summary", "full"] {
            same(
                &format!("{name} diagnosis {detail}"),
                &trajectory_diagnosis(&ct, detail),
                &want["diagnosis"][detail],
            );
        }
        for a in want["archetypes"].as_array().unwrap() {
            let got = archetype_payload(
                &ct,
                a["threshold"].as_f64().unwrap(),
                a["max_archetypes"].as_u64().unwrap() as usize,
                a["max_entities_per_archetype"].as_u64().unwrap() as usize,
            );
            let full = want["archetypes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["threshold"] == a["threshold"] && f["max_archetypes"] == 1000)
                .unwrap();
            let label = format!(
                "{name} archetypes {} {}",
                a["threshold"], a["max_archetypes"]
            );
            // A truncated run also caps the entity previews; compare the
            // archetypes by (population, sequence) against the full run.
            let core = |v: &Value| -> Value {
                let mut v = v.clone();
                for x in v["archetypes"].as_array_mut().unwrap() {
                    *x = json!({"n_entities": x["n_entities"], "sequence": x["sequence"]});
                }
                v
            };
            if a["max_archetypes"] == 1000 {
                ranked_same(
                    &label,
                    &got,
                    &a["payload"],
                    "archetypes",
                    "n_entities",
                    &full["payload"],
                );
            } else {
                ranked_same(
                    &label,
                    &core(&got),
                    &core(&a["payload"]),
                    "archetypes",
                    "n_entities",
                    &core(&full["payload"]),
                );
            }
        }
        for c in want["cross_time"].as_array().unwrap() {
            let got = cross_time_payload(
                &ct,
                c["obs_id"].as_u64().unwrap() as usize,
                c["threshold"].as_f64().unwrap(),
                c["max_neighbors"].as_u64().unwrap() as usize,
                c["direction"].as_str().unwrap(),
            );
            let label = format!(
                "{name} cross {} {} {} {}",
                c["obs_id"], c["direction"], c["threshold"], c["max_neighbors"]
            );
            let full = want["cross_time"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| {
                    f["obs_id"] == c["obs_id"]
                        && f["direction"] == c["direction"]
                        && f["threshold"] == c["threshold"]
                        && f["max_neighbors"] == 1000
                })
                .unwrap();
            ranked_same(
                &label,
                &got,
                &c["payload"],
                "neighbors",
                "weight",
                &full["payload"],
            );
        }
        for c in want["classify"].as_array().unwrap() {
            let got = classify_trajectories_payload(
                &ct,
                c["method"].as_str().unwrap(),
                c["threshold"].as_f64().unwrap(),
            );
            same(
                &format!("{name} classify {} {}", c["method"], c["threshold"]),
                &result(got),
                &c["result"],
            );
        }
        same(
            &format!("{name} unknown method"),
            &result(classify_trajectories_payload(&ct, "markov", 0.0)),
            &want["classify_unknown"],
        );
    }
}

#[test]
fn temporal_diagnosis_matches_python() {
    let temporal = load("temporal.json");
    let fx = load("payloads.json");
    let p = &temporal["pipeline"];
    let ints = |v: &Value| -> Vec<u64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    };
    let dims: Vec<usize> = ints(&p["dimensions"])
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let eps: Vec<f64> = serde_json::from_value(p["epsilons"].clone()).unwrap();
    let snaps: Vec<Array2<f64>> = p["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .map(matrix)
        .collect();
    let mut tcgs =
        vec![
            TemporalCosmicGraph::from_snapshots(&snaps, &dims, &ints(&p["seeds"]), &eps, 0.0)
                .unwrap(),
        ];
    for case in temporal["tensors"].as_array().unwrap() {
        tcgs.push(
            TemporalCosmicGraph::new(dense(&case["tensor"]), case["threshold"].as_f64().unwrap())
                .unwrap(),
        );
    }
    for (tcg, want) in tcgs.iter().zip(fx["temporal"].as_array().unwrap()) {
        for detail in ["summary", "full"] {
            same(
                &format!("{} {detail}", want["name"]),
                &temporal_diagnosis(tcg, detail),
                &want["payload"][detail],
            );
        }
    }
}

#[test]
fn sequence_cohorts_match_scipy_complete_linkage() {
    let fx = load("payloads.json");
    for (k, case) in fx["classify_random"].as_array().unwrap().iter().enumerate() {
        let obs: Vec<Observation> = case["obs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| Observation {
                entity_id: key(&o["entity_id"]),
                t: o["t"].as_u64().unwrap() as usize,
                timestamp: None,
            })
            .collect();
        let labels: Vec<usize> = serde_json::from_value(case["labels"].clone()).unwrap();
        for (method, want) in case["results"].as_object().unwrap() {
            same(
                &format!("random {k} {method}"),
                &result(classify_labels(&obs, &labels, method, 0.0)),
                want,
            );
        }
    }
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

#[test]
fn entity_sample_matches_python() {
    let fx = load("payloads.json");
    for case in fx["sample"].as_array().unwrap() {
        let (selected, eligible, rows) = select_longitudinal_entities(
            &table(&case["table"]),
            "pid",
            case["max_entities"].as_u64().unwrap() as usize,
            case["min_observations"].as_u64().unwrap() as usize,
            case["time_column"].as_str(),
            case["min_time_points"].as_u64().map(|v| v as usize),
        );
        let label = format!(
            "sample {} {} {}",
            case["kind"], case["max_entities"], case["min_observations"]
        );
        same(
            &format!("{label} selected"),
            &json!(selected),
            &case["selected"],
        );
        assert_eq!(
            eligible as u64,
            case["eligible"].as_u64().unwrap(),
            "{label} eligible"
        );
        same(&format!("{label} rows"), &json!(rows), &case["rows"]);
    }
}
