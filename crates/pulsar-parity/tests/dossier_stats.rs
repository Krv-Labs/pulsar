//! Parity for the statistical kernel of the cluster interpreter: the Rust port
//! must reproduce what `generate_stats.py` (Python generator, see `tests/fixtures/golden/README.md`) froze
//! from Python (`pulsar/mcp/interpreter.py`).
//!
//! Exact: ranks, counts, tie corrections, percentiles and every failure
//! message. Floats within 1e-9 (relative, with a 1e-12 absolute floor): numpy
//! sums pairwise and scipy evaluates its special functions with Cephes, while
//! this port sums plainly and uses `statrs`. NaN is null in the fixture.

use pulsar_analysis::stats::{
    bh_fdr, build_numeric_precompute, direction_from_effects, empirical_percentiles,
    kruskal_wallis_pvalue, ks_two_sample_stat, mwu_asymptotic_two_sided_pvalue,
    mwu_one_vs_rest_pvalue, mwu_pair_pvalue, pooled_std, safe_mad, safe_mean, safe_median,
    safe_std, safe_var, wasserstein_distance_1d, StatsError,
};
use pulsar_parity::golden_dir;
use serde_json::Value;

fn fixture() -> Value {
    let path = golden_dir().join("dossier/stats.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// null is NaN; "inf" and "-inf" are the infinities.
fn num(v: &Value) -> f64 {
    match v {
        Value::Null => f64::NAN,
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        other => other
            .as_f64()
            .unwrap_or_else(|| panic!("not a number: {other}")),
    }
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|b| b.as_bool().unwrap())
        .collect()
}

fn close(label: &str, got: f64, want: f64) {
    if got.is_nan() || want.is_nan() {
        assert!(got.is_nan() && want.is_nan(), "{label}: {got} vs {want}");
    } else if got.is_infinite() || want.is_infinite() {
        assert_eq!(got, want, "{label}");
    } else {
        assert!(
            (got - want).abs() <= 1e-12 + 1e-9 * want.abs(),
            "{label}: {got} vs {want}"
        );
    }
}

fn close_all(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        close(&format!("{label}[{i}]"), *g, *w);
    }
}

/// A frozen outcome: a value (null = NaN) or the Python error message.
fn check(label: &str, got: Result<f64, StatsError>, want: &Value) {
    match (got, want["error"].as_str()) {
        (Err(e), Some(msg)) => assert_eq!(e.to_string(), msg, "{label}"),
        (Ok(v), None) => close(label, v, num(&want["value"])),
        (got, want_error) => panic!("{label}: got {got:?}, Python error {want_error:?}"),
    }
}

#[test]
fn fdr_percentiles_and_moments_match() {
    let fx = fixture();
    for (i, case) in fx["bh"].as_array().unwrap().iter().enumerate() {
        let nan_at = bools(&case["p_raw_nan"].to_owned());
        let p: Vec<Option<f64>> = case["p"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&nan_at)
            .map(|(v, &is_nan)| {
                if is_nan {
                    Some(f64::NAN)
                } else {
                    (!v.is_null()).then(|| num(v))
                }
            })
            .collect();
        let got = bh_fdr(&p);
        let want = case["out"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "bh {i}");
        for (g, w) in got.iter().zip(want) {
            match (g, w.is_null()) {
                (None, true) => {}
                (Some(g), false) => close(&format!("bh {i}"), *g, num(w)),
                _ => panic!("bh {i}: {got:?} vs {want:?}"),
            }
        }
    }
    for case in fx["percentiles"].as_array().unwrap() {
        assert_eq!(
            empirical_percentiles(&nums(&case["values"])),
            nums(&case["out"]),
            "{}",
            case["values"]
        );
    }
    for case in fx["moments"].as_array().unwrap() {
        let v = nums(&case["values"]);
        let label = case["values"].to_string();
        close(&format!("mean {label}"), safe_mean(&v), num(&case["mean"]));
        close(
            &format!("median {label}"),
            safe_median(&v),
            num(&case["median"]),
        );
        close(&format!("std {label}"), safe_std(&v), num(&case["std"]));
        close(&format!("var {label}"), safe_var(&v), num(&case["var"]));
        close(&format!("mad {label}"), safe_mad(&v), num(&case["mad"]));
    }
    for case in fx["pooled"].as_array().unwrap() {
        close(
            &case.to_string(),
            pooled_std(&nums(&case["a"]), &nums(&case["b"])),
            num(&case["pooled"]),
        );
    }
    for case in fx["direction"].as_array().unwrap() {
        assert_eq!(
            direction_from_effects(&nums(&case["effects"])),
            case["out"].as_str().unwrap()
        );
    }
}

#[test]
fn the_batched_column_statistics_match() {
    let fx = fixture();
    let p = &fx["precompute"];
    let columns: Vec<Vec<f64>> = p["columns"].as_array().unwrap().iter().map(nums).collect();
    let pre = build_numeric_precompute(&columns);
    let out = &p["out"];
    let names = p["names"].as_array().unwrap();
    assert_eq!(pre.x.len(), names.len());
    assert!(p["empty_columns"].as_u64() == Some(0));
    assert!(build_numeric_precompute(&[]).x.is_empty());

    let per_column: [(&str, &Vec<f64>); 7] = [
        ("mean", &pre.col_mean),
        ("std_pop", &pre.col_std_pop),
        ("std_sample", &pre.col_std_sample),
        ("var_sample", &pre.col_var_sample),
        ("median", &pre.col_median),
        ("mad", &pre.col_mad),
        ("iqr", &pre.col_iqr),
    ];
    for (key, got) in per_column {
        close_all(key, got, &nums(&out[key]));
    }
    let n_total: Vec<usize> = out["n_total"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(pre.n_total, n_total);
    // The fixture stores the ranks per column.
    let want_ranks = out["ranks"].as_array().unwrap();
    for (j, name) in names.iter().enumerate() {
        let want = nums(&want_ranks[j]);
        let got = &pre.ranks[j];
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g.is_nan() && w.is_nan()) || g == w,
                "ranks of {name}[{i}]: {g} vs {w}"
            );
        }
    }
    close_all(
        "tie_correction",
        &pre.tie_correction,
        &nums(&out["tie_correction"]),
    );
}

fn precompute() -> pulsar_analysis::stats::NumericPrecompute {
    let fx = fixture();
    let columns: Vec<Vec<f64>> = fx["precompute"]["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(nums)
        .collect();
    build_numeric_precompute(&columns)
}

#[test]
fn mann_whitney_matches_exact_and_asymptotic() {
    let fx = fixture();
    let pre = precompute();
    let mwu = &fx["mwu"];
    let masks = mwu["masks"].as_object().unwrap();

    let mut exact_branch = 0;
    for case in mwu["one_vs_rest"].as_array().unwrap() {
        let mask = bools(&masks[case["mask"].as_str().unwrap()]);
        let j = case["column"].as_u64().unwrap() as usize;
        let label = format!("one-vs-rest {} col {j}", case["mask"]);
        check(&label, mwu_one_vs_rest_pvalue(&mask, j, &pre), &case["out"]);
        exact_branch += usize::from(
            case["out"]["value"].is_number() && {
                let n1 = mask
                    .iter()
                    .zip(&pre.valid[j])
                    .filter(|(m, v)| **m && **v)
                    .count();
                n1 > 0 && n1 <= 8
            },
        );
    }
    assert!(
        exact_branch > 0,
        "no one-vs-rest case reached the exact branch"
    );

    for case in mwu["pairs"].as_array().unwrap() {
        let (x, y) = (nums(&case["x"]), nums(&case["y"]));
        check(
            &format!("pair {}x{}", x.len(), y.len()),
            mwu_pair_pvalue(&x, &y),
            &case["out"],
        );
    }
    for case in mwu["asymptotic"].as_array().unwrap() {
        let got = mwu_asymptotic_two_sided_pvalue(
            num(&case["u1"]),
            case["n1"].as_u64().unwrap() as usize,
            case["n2"].as_u64().unwrap() as usize,
            num(&case["tie_sum"]),
        );
        close(&case.to_string(), got, num(&case["out"]));
    }
}

#[test]
fn kruskal_wallis_matches() {
    let fx = fixture();
    let pre = precompute();
    let mut nan_results = 0;
    for case in fx["kruskal"].as_array().unwrap() {
        let j = case["column"].as_u64().unwrap() as usize;
        let masks: Vec<Vec<bool>> = case["masks"]
            .as_array()
            .unwrap()
            .iter()
            .map(bools)
            .collect();
        let got = kruskal_wallis_pvalue(&pre.ranks[j], &masks, pre.tie_correction[j]);
        nan_results += usize::from(matches!(&got, Ok(v) if v.is_nan()));
        check(
            &format!("kruskal {} col {j}", case["grouping"]),
            got,
            &case["out"],
        );
    }
    assert!(nan_results > 0, "no fully tied column in the fixtures");
}

#[test]
fn ks_and_wasserstein_match() {
    let fx = fixture();
    for case in fx["distance"].as_array().unwrap() {
        let (a, b) = (nums(&case["a"]), nums(&case["b"]));
        check("ks", ks_two_sample_stat(&a, &b), &case["ks"]);
        check(
            "wasserstein",
            wasserstein_distance_1d(&a, &b),
            &case["wasserstein"],
        );
    }
}
