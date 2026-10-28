//! Parity for `compare_clusters` and the two tests under it: the Rust port must
//! reproduce what `generate_compare.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from
//! Python and SciPy.
//!
//! The Welch t-test and the exact two-sample KS p-value are checked on 59 input
//! pairs (equal and unequal sizes, ties, shifts, degenerate samples); the full
//! comparison table and its Markdown on three evidence tables. P-values are
//! compared within an absolute 1e-9 (SciPy's exact KS value floors near 1e-16);
//! everything else within 1e-7 relative.

use pulsar_analysis::compare::{compare_clusters, comparison_to_markdown};
use pulsar_analysis::stats::{ks_two_sample_pvalue, welch_t_test};
use pulsar_data::read_csv_path;
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::Value;

fn fixture() -> Value {
    let path = golden_dir().join("dossier/compare.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn p_close(label: &str, got: f64, want: &Value) {
    match want.as_f64() {
        None => assert!(got.is_nan(), "{label}: got {got}, want NaN"),
        Some(w) => assert!(
            (got - w).abs() <= 1e-9 + 1e-7 * w.abs(),
            "{label}: got {got}, want {w}"
        ),
    }
}

#[test]
fn welch_and_ks_p_values_match_scipy() {
    let fx = fixture();
    let pairs = fx["pairs"].as_array().unwrap();
    assert!(pairs.len() >= 50);
    let (mut exact_unequal, mut tiny) = (0, 0);
    for (i, case) in pairs.iter().enumerate() {
        let (a, b) = (nums(&case["a"]), nums(&case["b"]));
        let label = format!("pair {i} ({}x{})", a.len(), b.len());
        let (_, p_t) = welch_t_test(&a, &b);
        p_close(&format!("{label} welch"), p_t, &case["p_t"]);
        let p_ks = ks_two_sample_pvalue(&a, &b).unwrap();
        p_close(&format!("{label} ks"), p_ks, &case["p_ks"]);
        exact_unequal += usize::from(a.len() != b.len());
        tiny += usize::from(case["p_ks"].as_f64().is_some_and(|p| p < 1e-6));
    }
    assert!(
        exact_unequal > 10 && tiny > 3,
        "the pairs should cover unequal sizes and strong effects"
    );
}

#[test]
fn compare_clusters_matches_python() {
    let fx = fixture();
    for (name, case) in fx["compare"].as_object().unwrap() {
        let table = read_csv_path(golden_dir().join(case["csv"].as_str().unwrap())).unwrap();
        let labels: Vec<i64> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        for call in case["calls"].as_array().unwrap() {
            let (a, b) = (call["a"].as_i64().unwrap(), call["b"].as_i64().unwrap());
            let rows = compare_clusters(&table, &labels, a, b);
            let got = serde_json::to_value(&rows).unwrap();
            let want = &call["rows"];
            // Column order is by effect size; p-values get the absolute tolerance.
            assert_eq!(
                got.as_array().unwrap().len(),
                want.as_array().unwrap().len(),
                "{name} {a}v{b}"
            );
            for (g, w) in got.as_array().unwrap().iter().zip(want.as_array().unwrap()) {
                let label = format!("{name} {a}v{b} {}", w["column"]);
                assert_eq!(g["column"], w["column"], "{label}: order");
                for key in ["mean_a", "mean_b", "cohens_d"] {
                    assert_json_close(&format!("{label}.{key}"), &g[key], &w[key], 1e-7);
                }
                for key in ["p_val_t", "p_val_ks"] {
                    p_close(
                        &format!("{label}.{key}"),
                        g[key].as_f64().unwrap_or(f64::NAN),
                        &w[key],
                    );
                }
            }
            if !rows.is_empty() {
                // The Markdown shows rounded values, so it matches wherever rounding agrees.
                let md = comparison_to_markdown(a, b, &rows);
                let want_md = call["markdown"].as_str().unwrap();
                assert_eq!(
                    md.lines().take(4).collect::<Vec<_>>(),
                    want_md.lines().take(4).collect::<Vec<_>>(),
                    "{name} {a}v{b} header"
                );
                assert_eq!(
                    md.lines().count(),
                    want_md.lines().count(),
                    "{name} {a}v{b}"
                );
                assert_eq!(md, want_md, "{name} {a}v{b}: markdown");
            }
        }
    }
}

#[test]
fn the_fixtures_cover_small_clusters_and_every_table() {
    let fx = fixture();
    let compare = fx["compare"].as_object().unwrap();
    assert_eq!(compare.len(), 3);
    assert!(compare
        .values()
        .flat_map(|c| c["calls"].as_array().unwrap())
        .any(|c| c["rows"].as_array().unwrap().is_empty()));
    assert!(compare
        .values()
        .flat_map(|c| c["calls"].as_array().unwrap())
        .any(|c| !c["rows"].as_array().unwrap().is_empty()));
}
