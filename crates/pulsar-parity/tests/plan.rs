//! Parity for the preprocessing-plan default rules: `populate_default_entries`
//! and `validate_role_designations` must reproduce what
//! `generate_plan.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from the Python
//! `pulsar.mcp.preprocessing_plan`. Entries, reasons and the rule
//! bookkeeping lists are compared exactly (they are strings and ids).

use pulsar_analysis::plan::{populate_default_entries, validate_role_designations, DraftContext};
use pulsar_analysis::{measure_preprocessing_evidence, EvidenceOptions};
use pulsar_data::read_csv_path;
use pulsar_parity::golden_dir;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = golden_dir().join("plan/defaults.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    serde_json::from_value(v.clone()).unwrap()
}

fn run(csv: &str, scenario: &Value) -> Value {
    let table = read_csv_path(golden_dir().join(csv)).unwrap();
    let evidence = measure_preprocessing_evidence(&table, &EvidenceOptions::default()).unwrap();
    let args = &scenario["args"];
    let (targets, entity, time) = validate_role_designations(
        &table.names,
        &strings(&args["targets"]),
        args["entity"].as_str().unwrap(),
        args["time"].as_str().unwrap(),
    )
    .unwrap();
    let mut ctx = DraftContext::new(&table, &evidence, &targets, &entity, &time);
    populate_default_entries(&mut ctx);
    json!({
        "entries": ctx.entries,
        "date_cols": ctx.date_cols,
        "mixed_value_cols": ctx.mixed_value_cols,
        "conditional_by_partner": ctx.conditional_by_partner,
        "rule7b_cols": ctx.rule7b_cols,
        "grouped_7b": ctx.grouped_7b,
        "missing_groups": ctx.missing_groups,
    })
}

fn compare(name: &str, csv: &str) {
    let fx = fixture();
    let want = &fx[name];
    let got = run(csv, want);
    for key in [
        "date_cols",
        "mixed_value_cols",
        "conditional_by_partner",
        "rule7b_cols",
        "grouped_7b",
        "missing_groups",
    ] {
        assert_eq!(got[key], want[key], "{name}.{key}");
    }
    // Same columns in the same order, then entry by entry for a readable diff.
    let keys =
        |v: &Value| -> Vec<String> { v["entries"].as_object().unwrap().keys().cloned().collect() };
    assert_eq!(keys(&got), keys(want), "{name}: column order");
    for (col, want_entry) in want["entries"].as_object().unwrap() {
        assert_eq!(&got["entries"][col], want_entry, "{name}.entries.{col}");
    }
}

#[test]
fn plain_defaults_match_python() {
    compare("plain", "plan/input.csv");
}

#[test]
fn designated_roles_match_python() {
    compare("roles", "plan/input.csv");
}

#[test]
fn penguins_defaults_match_python() {
    compare("penguins", "penguins/input.csv");
}

#[test]
fn every_default_rule_is_exercised_by_the_fixture() {
    // Guards against a vacuous comparison if the fixture table drifts.
    let fx = fixture();
    let entries = fx["plain"]["entries"].as_object().unwrap();
    let has = |pred: &dyn Fn(&Value) -> bool| entries.values().any(pred);
    let reason = |e: &Value| e["reason"].as_str().unwrap().to_string();
    assert!(has(&|e| e["role"] == "id"));
    assert!(has(&|e| reason(e).starts_with("date/time text")));
    assert!(has(
        &|e| reason(e).starts_with("mostly numbers") && reason(e).contains("kept as")
    ));
    assert!(has(
        &|e| reason(e).contains("treated as blank with median fill")
    ));
    assert!(has(
        &|e| reason(e).starts_with("numeric text (") && e["indicator"] == "own"
    ));
    assert!(has(
        &|e| reason(e).contains("blank on the exact rows where flag is 0")
    ));
    assert!(has(&|e| reason(e).contains("exceeds sqrt(40)")));
    assert!(has(&|e| reason(e).starts_with("all 40 rows are blank")));
    assert!(has(&|e| reason(e).starts_with("one distinct value")));
    assert!(has(&|e| e["indicator"] == "group:mg2"));
    assert!(has(&|e| reason(e).starts_with("complete ")));
}

#[test]
fn role_validation_errors_match_python() {
    let fx = fixture();
    let table = read_csv_path(golden_dir().join("plan/input.csv")).unwrap();
    let names = &table.names;
    let none: Vec<String> = vec![];
    let cases = [
        (
            "entity",
            validate_role_designations(names, &none, "nope", ""),
        ),
        ("time", validate_role_designations(names, &none, "", "when")),
        (
            "targets",
            validate_role_designations(
                names,
                &["age".into(), "bad one".into(), "it's".into()],
                "",
                "",
            ),
        ),
    ];
    for (key, got) in cases {
        let err = got.unwrap_err();
        let want = &fx["errors"][key];
        assert_eq!(err.code, want["code"].as_str().unwrap(), "{key}");
        assert_eq!(err.message, want["message"].as_str().unwrap(), "{key}");
        assert_eq!(Value::Object(err.details), want["details"], "{key}");
    }
}

#[test]
fn empty_and_blank_role_arguments_are_ignored() {
    let names = vec!["a".to_string(), "b".to_string()];
    let (targets, entity, time) =
        validate_role_designations(&names, &["".into(), "a".into()], "  ", " b ").unwrap();
    assert_eq!(
        (targets, entity.as_str(), time.as_str()),
        (vec!["a".to_string()], "", "b")
    );
}
