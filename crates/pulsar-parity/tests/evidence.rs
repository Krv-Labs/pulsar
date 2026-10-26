//! Parity for the preprocessing-evidence port: `measure_preprocessing_evidence`
//! and `neighborhood_overlap` must reproduce what
//! `scripts/golden/generate_evidence.py` froze from the Python
//! `pulsar.analysis.preprocessing_evidence`.
//!
//! Numbers agree to 1e-9 (plain vs pairwise summation); everything else, in
//! particular kinds, groups, dependencies and relation order, is exact.
//! `cost.elapsed_ms` is wall time and is not compared.

use ndarray::Array2;
use pulsar_analysis::{
    measure_preprocessing_evidence, neighborhood_overlap, EvidenceOptions, PreprocessingEvidence,
};
use pulsar_data::read_csv_path;
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::Value;

const RTOL: f64 = 1e-9;

fn measured(case: &str) -> PreprocessingEvidence {
    let table = read_csv_path(golden_dir().join(case).join("input.csv")).unwrap();
    measure_preprocessing_evidence(&table, &EvidenceOptions::default()).unwrap()
}

fn compare(case: &str, artifact_dir: &str) {
    let mut got = serde_json::to_value(measured(case)).unwrap();
    got["cost"].as_object_mut().unwrap().remove("elapsed_ms");
    let path = golden_dir().join(artifact_dir).join("evidence.json");
    let want: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_json_close(case, &got, &want, RTOL);
}

#[test]
fn synthetic_table_matches_python() {
    // Guard against a vacuous comparison: the fixture must exercise every
    // evidence feature, and the Rust side must find all of them.
    let ev = measured("evidence");
    let kinds: Vec<&str> = ev.relations.iter().map(|r| r.kind.as_str()).collect();
    for kind in ["duplicate", "sum_of", "strong_correlation"] {
        assert!(kinds.contains(&kind), "no {kind} relation measured");
    }
    assert_eq!(ev.missing_groups.len(), 1);
    assert_eq!(ev.mask_dependencies.len(), 1);
    let column_kinds: Vec<&str> = ev.columns.iter().map(|c| c.kind.as_str()).collect();
    for kind in [
        "numeric",
        "boolean",
        "numeric_text",
        "datetime_like",
        "text",
    ] {
        assert!(column_kinds.contains(&kind), "no {kind} column");
    }
    compare("evidence", "evidence");
}

#[test]
fn penguins_matches_python() {
    compare("penguins", "penguins");
}

#[test]
fn neighborhood_overlap_matches_python() {
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(golden_dir().join("evidence/overlap.json")).unwrap(),
    )
    .unwrap();
    let matrix = |key: &str| {
        let rows: Vec<Vec<f64>> = serde_json::from_value(fixture[key].clone()).unwrap();
        let (n, p) = (rows.len(), rows[0].len());
        Array2::from_shape_vec((n, p), rows.into_iter().flatten().collect()).unwrap()
    };
    let (x, y) = (matrix("x"), matrix("y"));
    let k = fixture["k"].as_u64().unwrap() as usize;
    let got = neighborhood_overlap(x.view(), y.view(), k).unwrap();
    let want = fixture["overlap"].as_f64().unwrap();
    assert!((got - want).abs() < 1e-12, "overlap {got} vs {want}");
    assert_eq!(
        neighborhood_overlap(x.view(), x.view(), k).unwrap(),
        fixture["self_overlap"].as_f64().unwrap()
    );
}
