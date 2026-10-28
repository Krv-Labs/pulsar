//! Parity for dataset characterization and column probes: the Rust port must
//! reproduce what `generate_characterization.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze from Python, on the golden tables.
//!
//! Exact: every column profile (dtype, counts, rounded statistics, previews)
//! and the probe payloads and Markdown. Floats of the geometry (k-NN means,
//! percentiles) within 1e-6, because scikit-learn's brute-force path computes
//! distances from dot products (about 1e-8 apart from the exact Euclidean
//! distances used here). Every table here is at most 1000 rows, so no row
//! sample is drawn and numpy's generator is not involved. The variance
//! curve is probed with Python's fixed probe set ([`PcaProbes::Python`]);
//! production probes up to the feature count (see the divergences in
//! `characterization`).

use pulsar_analysis::characterization::{
    characterization_payload_to_markdown, characterize_dataset, compact_characterization_payload,
    probe_columns_payload, probe_columns_payload_to_markdown, profile_column_details,
    ColumnProfile, DatasetProfile, PcaProbes,
};
use pulsar_analysis::minhash::{massive_dataset_advisory, recommend_d, STANDARD_DEPTHS};
use pulsar_data::read_csv_path;
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::{json, Value};

const GEOMETRY_RTOL: f64 = 1e-6;

fn fixture() -> Value {
    let path = golden_dir().join("plan/characterization.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn lines(s: &str) -> Vec<String> {
    s.lines().map(str::to_string).collect()
}

fn compare_table(csv: &str, want: &Value) {
    let table = read_csv_path(golden_dir().join(csv)).unwrap();
    let result = characterize_dataset(&table, 1000, 42, PcaProbes::Python);

    if let Some(py_error) = want["error"].as_str() {
        assert_eq!(
            result.unwrap_err().to_string(),
            py_error,
            "{csv}: error message"
        );
        return;
    }
    let profile = result.unwrap_or_else(|e| panic!("{csv}: {e}"));
    let got = serde_json::to_value(&profile).unwrap();

    // Counts and the per-column map view are exact.
    for key in ["n_samples", "n_features", "n_columns_total"] {
        assert_eq!(got[key], want["profile"][key], "{csv}: {key}");
    }
    assert_eq!(
        got["column_profiles"], want["profile"]["column_profiles"],
        "{csv}: column profiles"
    );
    // Geometry within tolerance; the variance curve (rounded to 4 places) exact.
    for key in [
        "missingness_pct",
        "knn_k5_mean",
        "knn_k10_mean",
        "knn_k20_mean",
    ] {
        assert_json_close(
            &format!("{csv}: {key}"),
            &got[key],
            &want["profile"][key],
            GEOMETRY_RTOL,
        );
    }
    assert_eq!(
        got["pca_cumulative_variance"], want["profile"]["pca_cumulative_variance"],
        "{csv}: PCA variance"
    );

    // The tool payload and its Markdown.
    let payload = compact_characterization_payload(&profile);
    let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(
        keys(&payload),
        keys(&want["payload"]),
        "{csv}: payload keys"
    );
    assert_json_close(
        &format!("{csv}: payload"),
        &payload,
        &want["payload"],
        GEOMETRY_RTOL,
    );
    assert_eq!(
        lines(&characterization_payload_to_markdown(&payload)),
        lines(want["markdown"].as_str().unwrap()),
        "{csv}: characterization markdown"
    );

    // Every column's detailed profile.
    for (col, w) in want["details"].as_object().unwrap() {
        let p = profile_column_details(&table, col, 10)
            .unwrap_or_else(|| panic!("{csv}: no column {col}"));
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            *w,
            "{csv}: details of {col}"
        );
    }

    // probe_columns over the first 20 columns plus one that does not exist.
    let requested: Vec<String> =
        serde_json::from_value(want["probe"]["requested"].clone()).unwrap();
    let (mut found, mut missing): (Vec<ColumnProfile>, Vec<String>) = (vec![], vec![]);
    for c in &requested {
        match profile_column_details(&table, c, 10) {
            Some(p) => found.push(p),
            None => missing.push(c.clone()),
        }
    }
    let probe = probe_columns_payload(&found, &requested, &missing);
    assert_eq!(probe, want["probe"]["payload"], "{csv}: probe payload");
    assert_eq!(
        lines(&probe_columns_payload_to_markdown(&probe)),
        lines(want["probe"]["markdown"].as_str().unwrap()),
        "{csv}: probe markdown"
    );
}

#[test]
fn every_table_characterizes_like_python() {
    let fx = fixture();
    let tables = fx["tables"].as_object().unwrap();
    assert!(tables.len() >= 10);
    for (csv, want) in tables {
        compare_table(csv, want);
    }
}

#[test]
fn a_massive_dataset_gets_the_minhash_advisory() {
    let fx = fixture();
    let m = &fx["massive"];
    let profiles: Vec<ColumnProfile> = m["geo"]["column_profiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| ColumnProfile {
            name: c["name"].as_str().unwrap().into(),
            dtype: c["dtype"].as_str().unwrap().into(),
            is_numeric: c["is_numeric"].as_bool().unwrap(),
            n_unique: c["n_unique"].as_u64().unwrap() as usize,
            n_missing: c["n_missing"].as_u64().unwrap() as usize,
            missing_pct: c["missing_pct"].as_f64().unwrap(),
            sample_values: vec![],
            mean: None,
            std: None,
            min_val: None,
            max_val: None,
            top_values: None,
        })
        .collect();
    let g = &m["geo"];
    let geo = DatasetProfile {
        n_samples: g["n_samples"].as_u64().unwrap() as usize,
        n_features: 1,
        n_columns_total: 2,
        missingness_pct: 0.0,
        knn_k5_mean: 0.1,
        knn_k10_mean: 0.2,
        knn_k20_mean: 0.3,
        pca_cumulative_variance: vec![(1, 1.0)],
        n_samples_profiled: 1000,
        n_features_profiled: 1,
        column_profiles: profiles,
    };
    let payload = compact_characterization_payload(&geo);
    assert_eq!(payload, m["payload"], "massive payload");
    assert!(payload["minhash_advisory"].is_object());
    assert_eq!(
        lines(&characterization_payload_to_markdown(&payload)),
        lines(m["markdown"].as_str().unwrap()),
        "massive markdown"
    );

    for (n, want) in m["advisories"].as_object().unwrap() {
        let got = massive_dataset_advisory(n.parse().unwrap(), 256).unwrap_or(Value::Null);
        assert_eq!(got, *want, "advisory at n={n}");
    }
    assert_eq!(
        m["advisories"]["499999"],
        Value::Null,
        "below the threshold there is no advisory"
    );

    let rec = |n, budget, eps| {
        serde_json::to_value(recommend_d(n, budget, eps, &STANDARD_DEPTHS, 0.05)).unwrap()
    };
    let r = &m["recommend"];
    assert_eq!(rec(1000, None, None), r["default_small"]);
    assert_eq!(rec(2_000_000, None, None), r["default_massive"]);
    assert_eq!(rec(1000, None, Some(0.1)), r["eps_loose"]);
    assert_eq!(rec(1000, None, Some(0.02)), r["eps_tight"]);
    assert_eq!(rec(1_000_000, Some(2_000_000_000), None), r["memory_fits"]);
    assert_eq!(rec(10_000_000, Some(1_000_000), None), r["memory_none_fit"]);
}

#[test]
fn the_fixtures_exercise_the_interesting_paths() {
    // Guards against a vacuous comparison if the fixtures drift.
    let fx = fixture();
    let tables = fx["tables"].as_object().unwrap();
    let all_columns = || {
        tables
            .values()
            .filter_map(|t| t["profile"]["column_profiles"].as_array())
            .flatten()
    };
    let dtypes: std::collections::BTreeSet<&str> = all_columns()
        .map(|c| c["dtype"].as_str().unwrap())
        .collect();
    for d in ["float64", "int64", "bool", "object", "str"] {
        assert!(
            dtypes.contains(d),
            "no {d} column in the fixtures: {dtypes:?}"
        );
    }
    assert!(
        tables.values().any(|t| t["error"].is_string()),
        "a rejected table"
    );
    assert!(
        tables
            .values()
            .any(|t| t["payload"]["omitted_column_profiles"]
                .as_u64()
                .unwrap_or(0)
                > 0),
        "a capped column preview"
    );
    assert!(
        tables.values().any(
            |t| t["payload"]["column_name_preview"]["numeric"]["omitted"]
                .as_u64()
                .unwrap_or(0)
                > 0
                || t["payload"]["column_name_preview"]["categorical"]["omitted"]
                    .as_u64()
                    .unwrap_or(0)
                    > 0
        ),
        "a capped name preview"
    );
    assert!(
        tables
            .values()
            .any(|t| t["probe"]["payload"]["column_profiles"]
                .as_array()
                .is_some_and(|ps| {
                    ps.iter().any(|p| {
                        p["truncation"]["sample_values_truncated"]
                            .as_u64()
                            .unwrap_or(0)
                            > 0
                    })
                })),
        "truncated long values"
    );
    assert!(
        tables.values().any(|t| t["profile"]["column_profiles"]
            .as_array()
            .is_some_and(|ps| ps.iter().any(|p| p["missing_pct"] == json!(100.0)))),
        "an all-missing column"
    );
}
