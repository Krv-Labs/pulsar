//! Parity for the baseline config of `create_config`: the Rust port must
//! reproduce what `generate_create_config.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze from Python.
//!
//! Exact: the calibration outcome, the feature count, the elbow and the
//! dimension grid, and every line of the YAML except the epsilon range. The
//! epsilon bounds come from k-NN percentiles, which differ from scikit-learn's
//! dot-product distances by about 1e-8, so they are compared within one unit
//! of the fourth decimal they are printed at.
//!
//! The table test recomputes the variance curves (raw and processed) with
//! Python's probe set ([`PcaProbes::Python`]); the edge-case grid test reads
//! frozen curves from the fixture.

use pulsar_analysis::characterization::{characterize_dataset, PcaProbes};
use pulsar_analysis::config_build::{
    build_initial_config_yaml, grid_ceiling, initial_pca_grid, variance_elbow_dimension,
    CeilingSource,
};
use pulsar_analysis::plan::{calibrate_processed_space, draft_plan, DraftOptions};
use pulsar_data::read_csv_path;
use pulsar_parity::golden_dir;
use serde_json::Value;

/// Python clamped the grid at 16; production Rust does not (see the
/// divergences in `config_build`), so the frozen fixtures are reproduced with
/// the clamp.
const PYTHON_DIMENSION_CAP: Option<usize> = Some(16);

/// Python probed variance only at 2, 3, 5, 10, 15 and 20; production Rust
/// probes up to the feature count (see the divergences in `characterization`).
const PYTHON_PROBES: PcaProbes<'static> = PcaProbes::Python;

/// Python kept row-index columns (penguins' `rowid`) as features; production
/// Rust drafts them as identifiers.
const PYTHON_ROW_INDEX_IDS: bool = false;

/// Python did not bound the grid by the row count; the edge-case fixtures
/// carry no row count, so they are reproduced unbounded.
const UNBOUNDED_ROWS: usize = usize::MAX;

fn fixture() -> Value {
    let path = golden_dir().join("plan/create_config.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

/// Same document, epsilon bounds within `tol`, whatever the quoting.
fn close_docs(got: &Value, want: &Value, path: &str) {
    match (got, want) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!((a - b).abs() <= 1.5e-4, "{path}: {a} vs {b}");
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path}: keys"
            );
            for (k, v) in a {
                close_docs(v, &b[k], &format!("{path}.{k}"));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: length");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                close_docs(x, y, &format!("{path}[{i}]"));
            }
        }
        _ => assert_eq!(got, want, "{path}"),
    }
}

fn compare_yaml(csv: &str, got: &str, want: &str) {
    // PyYAML escapes non-ASCII text that serde_yaml_ng writes as is; the two
    // documents are identical once parsed.
    if want.contains("\\x") || want.contains("\\u") {
        let parse = |s: &str| serde_yaml_ng::from_str::<Value>(s).unwrap();
        close_docs(&parse(got), &parse(want), csv);
        return;
    }
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    assert_eq!(g.len(), w.len(), "{csv}: line count\n{got}\n---\n{want}");
    // The header names the plan by its random id; the fingerprint after it is
    // the deterministic part.
    let mask_id = |l: &str| {
        if l.starts_with("# preprocessing plan pp_") {
            l.split_whitespace()
                .enumerate()
                .filter(|(i, _)| *i != 3)
                .map(|(_, t)| t)
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            l.to_owned()
        }
    };
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        let (a, b) = (&mask_id(a), &mask_id(b));
        let bound = |l: &str| {
            let t = l.trim();
            t.strip_prefix("min: ")
                .or_else(|| t.strip_prefix("max: "))
                .and_then(|n| n.parse::<f64>().ok())
        };
        // Only the epsilon bounds (the sweep section) are inexact.
        match (bound(a), bound(b)) {
            (Some(x), Some(y)) if a.starts_with("        ") => {
                assert!((x - y).abs() <= 1.5e-4, "{csv}: line {i}: {a:?} vs {b:?}")
            }
            _ => assert_eq!(a, b, "{csv}: line {i}"),
        }
    }
}

#[test]
fn every_table_gets_the_python_baseline_config() {
    let fx = fixture();
    let tables = fx["tables"].as_object().unwrap();
    assert!(tables.len() >= 8);
    for (csv, want) in tables {
        let table = read_csv_path(golden_dir().join(csv)).unwrap();
        let geo = characterize_dataset(&table, 1000, 42, PYTHON_PROBES);
        if let Some(py_error) = want["error"].as_str() {
            assert_eq!(geo.unwrap_err().to_string(), py_error, "{csv}");
            continue;
        }
        let geo = geo.unwrap();
        let opts = DraftOptions {
            stakes: false,
            row_index_ids: PYTHON_ROW_INDEX_IDS,
            ..DraftOptions::default()
        };
        let plan = draft_plan(&table, "ds_fixture", &opts).unwrap();
        let processed = calibrate_processed_space(&table, &plan, PYTHON_PROBES).ok();
        assert_eq!(
            processed.is_some(),
            want["calibrated"].as_bool().unwrap(),
            "{csv}: calibrated"
        );
        if let Some(p) = &processed {
            assert_eq!(
                p.n_features as u64,
                want["processed"]["n_features"].as_u64().unwrap(),
                "{csv}: processed feature count"
            );
        }

        let yaml = build_initial_config_yaml(
            &geo,
            csv,
            "initial_sweep",
            processed.as_ref(),
            &plan,
            PYTHON_DIMENSION_CAP,
        );
        compare_yaml(csv, &yaml, want["config_yaml"].as_str().unwrap());

        let (n_features, n_rows, cumulative) = match &processed {
            Some(p) => (
                p.n_features,
                p.n_samples_profiled,
                p.pca_cumulative_variance.clone(),
            ),
            None => (
                geo.n_features_profiled,
                geo.n_samples_profiled,
                geo.pca_cumulative_variance.clone(),
            ),
        };
        let knee = variance_elbow_dimension(&cumulative, n_features);
        assert_eq!(knee as u64, want["knee"].as_u64().unwrap(), "{csv}: knee");
        assert_eq!(
            initial_pca_grid(n_features, n_rows, knee, &cumulative, PYTHON_DIMENSION_CAP),
            usizes(&want["grid"]),
            "{csv}: grid"
        );
    }
}

#[test]
fn python_probes_on_the_frozen_wide_curve_stop_at_the_probe_limit() {
    // 58 processed features, but Python's probes stop at 20 with 73.8% of
    // the variance: uncapped, the grid tops out at the probe set's end.
    let want = &fixture()["tables"]["plan/characterize_wide.csv"]["processed"];
    let n_features = want["n_features"].as_u64().unwrap() as usize;
    let n_rows = want["n_samples_profiled"].as_u64().unwrap() as usize;
    let cumulative: Vec<(usize, f64)> = want["pca_cumulative_variance"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_f64().unwrap()))
        .collect();
    assert_eq!((n_features, cumulative.last().unwrap().0), (58, 20));
    let knee = variance_elbow_dimension(&cumulative, n_features);
    assert_eq!(
        initial_pca_grid(n_features, n_rows, knee, &cumulative, None),
        vec![10, 15, 20]
    );
    let ceiling = grid_ceiling(&cumulative, n_features, n_rows);
    assert_eq!(
        (ceiling.dim, ceiling.source),
        (20, CeilingSource::ProbeLimit)
    );
}

#[test]
fn dimension_grids_match_python_on_edge_cases() {
    for case in fixture()["grids"].as_array().unwrap() {
        let n_features = case["n_features"].as_u64().unwrap() as usize;
        let cumulative: Vec<(usize, f64)> = case["cumulative"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_f64().unwrap()))
            .collect();
        let knee = variance_elbow_dimension(&cumulative, n_features);
        assert_eq!(knee as u64, case["knee"].as_u64().unwrap(), "{case}");
        assert_eq!(
            initial_pca_grid(
                n_features,
                UNBOUNDED_ROWS,
                knee,
                &cumulative,
                PYTHON_DIMENSION_CAP
            ),
            usizes(&case["grid"]),
            "{case}"
        );
    }
}

#[test]
fn the_fixtures_cover_both_calibration_outcomes() {
    let fx = fixture();
    let calibrated: Vec<bool> = fx["tables"]
        .as_object()
        .unwrap()
        .values()
        .filter(|t| t["error"].is_null())
        .map(|t| t["calibrated"].as_bool().unwrap())
        .collect();
    assert!(calibrated.contains(&true) && calibrated.contains(&false));
}
