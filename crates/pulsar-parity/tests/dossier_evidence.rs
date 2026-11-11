//! Parity for the per-cluster feature evidence index: the Rust port must
//! reproduce what `generate_feature_evidence.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze from Python (`build_feature_evidence_index`).
//!
//! The whole index is compared as one document: every row of every bundle
//! (effect sizes, tests, FDR q-values, aggregate and percentile scores, tiers),
//! the global rankings, the signal matrix and the metadata. Strings, counts,
//! tiers and orderings are exact; floats are within 1e-7. Known divergences
//! (module header of `feature_evidence`): tiers use the exact optimal 1-D
//! k-means partition, Fisher's test sums its support directly.

use pulsar_analysis::feature_evidence::build_feature_evidence_index;
use pulsar_data::read_csv_path;
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::Value;

fn fixture() -> Value {
    let path = golden_dir().join("dossier/feature_evidence.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn run(case: &Value) -> Result<Value, String> {
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
    build_feature_evidence_index(&table, &labels, &edges, &exclude, max)
        .map(|index| serde_json::to_value(index).unwrap())
        .map_err(|e| e.to_string())
}

#[test]
fn every_case_matches_the_python_index() {
    let fx = fixture();
    let cases = fx.as_object().unwrap();
    assert!(cases.len() >= 7);
    for (name, case) in cases {
        match (run(case), case["error"].as_str()) {
            (Err(got), Some(want)) => assert_eq!(got, want, "{name}"),
            (Ok(got), None) => assert_json_close(name, &got, &case["index"], 1e-7),
            (got, want) => panic!("{name}: got {got:?}, Python error {want:?}"),
        }
    }
}

#[test]
fn the_fixtures_exercise_the_interesting_paths() {
    // Guards against a vacuous comparison if the fixtures drift.
    let fx = fixture();
    let rows = |kind: &str| -> Vec<Value> {
        fx.as_object()
            .unwrap()
            .values()
            .filter_map(|c| c["index"]["cluster_bundles"].as_object())
            .flat_map(|b| b.values())
            .flat_map(|b| b[kind].as_array().unwrap().clone())
            .collect()
    };
    let (numeric, categorical) = (rows("numeric"), rows("categorical"));
    for tier in ["core", "supporting", "context"] {
        assert!(
            numeric.iter().any(|r| r["signal_tier"] == tier),
            "numeric {tier}"
        );
        assert!(
            categorical.iter().any(|r| r["signal_tier"] == tier),
            "categorical {tier}"
        );
    }
    for method in ["chi2", "fisher"] {
        assert!(
            categorical.iter().any(|r| r["test_method"] == method),
            "{method}"
        );
    }
    assert!(
        numeric.iter().any(|r| r["neighbor_cluster_id"].is_null()),
        "a cluster with no neighbor"
    );
    assert!(
        numeric.iter().any(|r| r["neighbor_cluster_id"].is_number()),
        "a cluster with a neighbor"
    );
    let meta = |k: &str| fx["synth"]["index"]["metadata"][k].clone();
    assert!(
        meta("categorical_columns_gated")
            .as_array()
            .is_some_and(|g| !g.is_empty()),
        "a gated column"
    );
    assert!(fx["synth_capped"]["index"]["metadata"]["omitted_clusters_count"].as_u64() > Some(0));
    assert!(fx["synth_misaligned"]["error"].is_string());
}
