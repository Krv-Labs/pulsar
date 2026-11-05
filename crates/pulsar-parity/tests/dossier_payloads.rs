//! Parity for the evidence-index payloads: the Rust port must reproduce what
//! `generate_dossier_payloads.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from
//! Python (`cluster_profile_payload`, `feature_signal_payload`,
//! `signal_matrix_payload`).
//!
//! Each case rebuilds the feature evidence index (itself under parity in
//! `dossier_evidence`) from the frozen labels and edges, then replays every
//! recorded call. Whole payloads are compared: rows, ordering, truncation and
//! omission counts, and the Markdown report. Floats within 1e-7.

use pulsar_analysis::dossier_payloads::{
    cluster_profile_payload, feature_signal_payload, signal_matrix_payload,
};
use pulsar_analysis::feature_evidence::{build_feature_evidence_index, FeatureEvidenceIndex};
use pulsar_data::read_csv_path;
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = golden_dir().join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn index_for(case: &Value) -> FeatureEvidenceIndex {
    let table = read_csv_path(golden_dir().join(case["csv"].as_str().unwrap())).unwrap();
    let labels: Vec<i64> = case["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let edges: Vec<(usize, usize, f64)> = case["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e[0].as_u64().unwrap() as usize,
                e[1].as_u64().unwrap() as usize,
                e[2].as_f64().unwrap(),
            )
        })
        .collect();
    let exclude: Vec<String> = case["exclude"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let max = case["max_clusters"].as_u64().map(|m| m as usize);
    build_feature_evidence_index(&table, &labels, &edges, &exclude, max).unwrap()
}

fn ids(v: &Value) -> Option<Vec<i64>> {
    v.as_array()
        .map(|a| a.iter().map(|i| i.as_i64().unwrap()).collect())
}

fn check(label: &str, got: Result<Value, String>, want: &Value) {
    match (got, want["error"].as_str()) {
        (Err(msg), Some(want_msg)) => assert_eq!(msg, want_msg, "{label}"),
        (Ok(value), None) => assert_json_close(label, &value, &want["value"], 1e-7),
        (got, want_error) => panic!("{label}: got {got:?}, Python error {want_error:?}"),
    }
}

#[test]
fn every_recorded_call_matches_python() {
    let evidence = load("dossier/feature_evidence.json");
    let payloads = load("dossier/payloads.json");
    let mut calls = 0;
    for (name, recorded) in payloads.as_object().unwrap() {
        let index = index_for(&evidence[name]);

        for c in recorded["profile"].as_array().unwrap() {
            let a = &c["args"];
            let got = cluster_profile_payload(
                &index,
                a["cluster_id"].as_i64().unwrap(),
                a["detail"].as_str().unwrap(),
                a["max_features"].as_u64().unwrap() as usize,
            )
            .map_err(|e| e.to_string());
            check(&format!("{name} profile {a}"), got, &c["out"]);
            calls += 1;
        }
        for c in recorded["signal"].as_array().unwrap() {
            let a = &c["args"];
            let names: Vec<String> = a["feature_names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect();
            let got = feature_signal_payload(
                &index,
                &names,
                ids(&a["cluster_ids"]).as_deref(),
                a["detail"].as_str().unwrap(),
                a["max_clusters"].as_u64().map(|m| m as usize),
            );
            check(&format!("{name} signal {a}"), Ok(got), &c["out"]);
            calls += 1;
        }
        for c in recorded["matrix"].as_array().unwrap() {
            let a = &c["args"];
            let got = signal_matrix_payload(
                &index,
                ids(&a["cluster_ids"]).as_deref(),
                a["include_context_tier"].as_bool().unwrap(),
                a["max_clusters"].as_u64().map(|m| m as usize),
                a["return_markdown"].as_bool().unwrap(),
            );
            check(&format!("{name} matrix {a}"), Ok(got), &c["out"]);
            calls += 1;
        }
    }
    assert!(calls >= 150, "only {calls} calls replayed");
}

#[test]
fn the_calls_exercise_truncation_tiers_and_failures() {
    // Guards against a vacuous comparison if the fixtures drift.
    let payloads = load("dossier/payloads.json");
    let all = |sec: &str| -> Vec<Value> {
        payloads
            .as_object()
            .unwrap()
            .values()
            .flat_map(|r| r[sec].as_array().unwrap().clone())
            .collect()
    };
    let profiles = all("profile");
    assert!(
        profiles.iter().any(|c| c["out"]["error"].is_string()),
        "an unknown cluster"
    );
    assert!(
        profiles
            .iter()
            .any(|c| c["out"]["value"]["features_omitted"]["total"].as_u64() > Some(0)),
        "a profile truncated by max_features"
    );
    let signals = all("signal");
    assert!(
        signals
            .iter()
            .any(|c| c["out"]["value"]["clusters_omitted"].as_u64() > Some(0)),
        "omitted signal clusters"
    );
    assert!(
        signals
            .iter()
            .any(|c| c["out"]["value"]["clusters_returned"] == 0),
        "an empty selection"
    );
    let matrices = all("matrix");
    assert!(
        matrices.iter().any(|c| c["out"]["value"]["markdown_report"]
            .as_str()
            .is_some_and(|m| m.contains("#### Omitted Clusters"))),
        "omitted-cluster telemetry in the Markdown report"
    );
    assert!(
        matrices.iter().any(|c| c["out"]["value"]["numeric_rows"]
            .as_array()
            .is_some_and(|r| !r.is_empty())),
        "a JSON matrix with rows"
    );
}
