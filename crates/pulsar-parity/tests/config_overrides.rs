//! Parity for the override engine behind `refine_config`: the Rust port must
//! reproduce what `generate_config_overrides.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze from Python.
//!
//! Exact: the applied paths (including the caller's own spelling appended
//! after the canonical path), the diff, and the message of every rejected
//! override. The resulting configs are compared through the Rust emitter:
//! Python's YAML text is loaded and re-emitted, so only the config's meaning
//! is compared, not PyYAML's formatting.

use pulsar_analysis::config_overrides::{check_overrides, OverrideError};
use pulsar_parity::golden_dir;
use serde_json::{Map, Value};

fn fixture() -> Value {
    let path = golden_dir().join("plan/config_overrides.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn mapping(yaml: &str) -> Map<String, Value> {
    match serde_yaml_ng::from_str::<Value>(yaml).unwrap() {
        Value::Object(m) => m,
        other => panic!("not a mapping: {other}"),
    }
}

#[test]
fn every_override_case_matches_python() {
    let fx = fixture();
    let base = fx["base"].as_str().unwrap();
    let cases = fx["cases"].as_object().unwrap();
    assert!(cases.len() >= 30);
    for (name, case) in cases {
        let overrides = case["overrides"].as_object().unwrap();
        let mut raw = mapping(base);
        let outcome = check_overrides(overrides).and_then(|c| c.apply(&mut raw));

        if let Some(py_error) = case["error"].as_str() {
            match outcome {
                // Rejected by the override engine: the message is exact.
                Err(OverrideError::UnknownKey(msg)) => assert_eq!(msg, py_error, "{name}"),
                Err(OverrideError::Invalid(_)) => {}
                // Accepted by the engine, rejected as a config.
                Ok(_) => assert!(
                    pulsar_config::from_value(Value::Object(raw)).is_err(),
                    "{name}: Python rejects this config ({py_error}), Rust accepted it"
                ),
            }
            continue;
        }

        let applied = outcome.unwrap_or_else(|e| panic!("{name}: {e}"));
        let want_applied: Vec<&str> = case["applied"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        assert_eq!(applied.applied, want_applied, "{name}: applied");
        assert_eq!(Value::Array(applied.diff), case["diff"], "{name}: diff");

        let got = pulsar_config::from_value(Value::Object(raw))
            .unwrap_or_else(|e| panic!("{name}: Rust rejects the refined config: {e:?}"));
        let want = pulsar_config::from_yaml_str(case["config_yaml"].as_str().unwrap())
            .unwrap_or_else(|e| panic!("{name}: Rust cannot load Python's config: {e:?}"));
        assert_eq!(
            pulsar_config::to_yaml(&got),
            pulsar_config::to_yaml(&want),
            "{name}: refined config"
        );
    }
}

#[test]
fn the_fixtures_cover_each_kind_of_outcome() {
    let fx = fixture();
    let cases = fx["cases"].as_object().unwrap().values();
    let (mut applied, mut removed, mut rejected) = (false, false, false);
    for c in cases {
        rejected |= c["error"].is_string();
        applied |= c["applied"].as_array().is_some_and(|a| !a.is_empty());
        removed |= c["diff"]
            .as_array()
            .is_some_and(|d| d.iter().any(|r| r["removed"] == true));
    }
    assert!(applied && removed && rejected);
}
