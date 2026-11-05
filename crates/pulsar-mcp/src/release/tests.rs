use std::collections::{BTreeMap, BTreeSet};

use pulsar_data::{Column, Table};
use serde_json::{json, Value};

use super::ReleaseBuilder;

fn table() -> Table {
    let species = (0..30)
        .map(|row| Some(if row < 12 { "Adelie" } else { "Gentoo" }.to_owned()))
        .collect();
    Table::new(vec!["species".to_owned()], vec![Column::Text(species)]).unwrap()
}

fn cluster() -> Value {
    let adelie = json!({
        "column": "species",
        "value": "Adelie",
        "cluster_id": 0,
        "count": 12,
        "count_cluster": 12,
        "count_rest": 0,
        "cluster_size": 20,
        "global_count": 12,
        "prevalence_cluster": 60.0,
        "prevalence_rest": 0.0,
        "prevalence_global": 40.0,
        "lift": 1.5,
        "log_lift": 0.405_465_108_118_159_83,
        "specificity": 60.0,
        "global_recall": 100.0,
        "neighbor_specificity": 0.0,
        "mi_contrib": 0.1,
        "fisher_p": 0.03,
        "fisher_q": 0.04,
        "global_p": 0.05,
        "global_q": 0.06,
        "aggregate_score": 0.8,
        "percentile_score": 90.0,
        "signal_tier": "core",
        "in_cluster_prevalence": 60.0,
        "concentration": 60.0,
        "test_method": "chi2",
        "failure_reasons": [],
        "evidence_vector": {},
    });
    let gentoo = json!({
        "column": "species",
        "value": "Gentoo",
        "cluster_id": 0,
        "count": 8,
        "count_cluster": 8,
        "count_rest": 10,
        "cluster_size": 20,
        "global_count": 18,
        "prevalence_cluster": 40.0,
        "prevalence_rest": 100.0,
        "prevalence_global": 60.0,
        "lift": 0.666_666_666_666_666_6,
        "log_lift": -0.405_465_108_118_159_9,
        "specificity": -60.0,
        "global_recall": 44.44,
        "neighbor_specificity": -100.0,
        "mi_contrib": 0.2,
        "fisher_p": 0.01,
        "fisher_q": 0.02,
        "global_p": 0.05,
        "global_q": 0.06,
        "aggregate_score": 0.9,
        "percentile_score": 95.0,
        "signal_tier": "core",
        "in_cluster_prevalence": 40.0,
        "concentration": 40.0,
        "test_method": "fisher",
        "failure_reasons": [],
        "evidence_vector": {},
    });
    let numeric = json!({
        "column": "age",
        "cluster_id": 0,
        "direction": "High",
        "mean": 5.0,
        "mean_rest": 3.0,
        "median": 5.0,
        "median_rest": 3.0,
        "std_cluster": 1.0,
        "std_rest": 1.0,
        "mad_cluster": 1.0,
        "mad_rest": 1.0,
        "global_mean": 4.0,
        "global_mean_rest": 3.0,
        "z_score": 1.0,
        "homogeneity": 0.5,
        "effect_mean_std": 2.0,
        "effect_median_mad": 2.0,
        "ks_stat": 0.5,
        "wasserstein_norm": 0.4,
        "variance_ratio_log": 0.1,
        "concentration_gain": 0.2,
        "one_vs_rest_p": 0.01,
        "one_vs_rest_q": 0.02,
        "global_p": 0.03,
        "global_q": 0.04,
        "global_cluster_signal": 0.5,
        "neighbor_cluster_id": 1,
        "neighbor_effect": 0.3,
        "neighbor_p": 0.05,
        "specificity_score": 0.7,
        "aggregate_score": 0.8,
        "percentile_score": 90.0,
        "signal_tier": "core",
        "sparkline": "",
        "failure_reasons": [],
        "evidence_vector": {},
    });
    let preview = json!([
        {
            "kind": "numeric",
            "column": "age",
            "signal_tier": "core",
            "aggregate_score": 0.8,
            "direction": "High",
            "mean": 5.0,
            "mean_rest": 3.0,
        },
        {
            "kind": "categorical",
            "column": "species",
            "signal_tier": "core",
            "aggregate_score": 0.8,
            "value": "Adelie",
            "lift": 1.5,
            "prevalence_cluster": 60.0,
        },
    ]);
    let mut cluster = json!({
        "cluster_id": 0,
        "size": 20,
        "cluster_size": 20,
        "size_pct": 66.67,
        "central_rows": [],
    });
    cluster["feature_preview"] = preview;
    cluster["numeric_features"] = json!([numeric]);
    cluster["categorical_features"] = json!([adelie, gentoo]);
    cluster
}

fn decode_b64(encoded: &str) -> Vec<u8> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let val = |b: u8| TABLE.iter().position(|&c| c == b).unwrap() as u32;
    let mut out = Vec::new();
    for chunk in encoded.as_bytes().chunks(4) {
        let b0 = val(chunk[0]);
        let b1 = val(chunk[1]);
        let b2 = if chunk[2] == b'=' { 0 } else { val(chunk[2]) };
        let b3 = if chunk[3] == b'=' { 0 } else { val(chunk[3]) };
        let n = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
        out.push((n >> 16) as u8);
        if chunk[2] != b'=' {
            out.push((n >> 8) as u8);
        }
        if chunk[3] != b'=' {
            out.push(n as u8);
        }
    }
    out
}

fn group_rows(release: &Value) -> BTreeMap<String, BTreeSet<usize>> {
    let mut groups = BTreeMap::new();
    for group in release["groups"].as_array().unwrap() {
        let id = group["id"].as_str().unwrap().to_owned();
        let mut rows = BTreeSet::new();
        match group["rows"]["enc"].as_str().unwrap() {
            "ranges" => {
                for range in group["rows"]["v"].as_array().unwrap() {
                    let start = range[0].as_u64().unwrap() as usize;
                    let end = range[1].as_u64().unwrap() as usize;
                    rows.extend(start..end);
                }
            }
            "bitmap_b64" => {
                for (byte_at, byte) in decode_b64(group["rows"]["v"].as_str().unwrap())
                    .into_iter()
                    .enumerate()
                {
                    for bit in 0..8usize {
                        if byte & (1 << bit) != 0 {
                            rows.insert(byte_at * 8 + bit);
                        }
                    }
                }
            }
            other => panic!("unexpected row encoding {other}"),
        }
        assert!(groups.insert(id, rows).is_none(), "duplicate group");
    }
    groups
}

#[test]
fn labels_match_their_twin_rows() {
    let labels: Vec<i64> = (0..30).map(|row| if row < 20 { 0 } else { 1 }).collect();
    let mut builder = ReleaseBuilder::for_test("/tmp/twin.csv", &labels, table()).unwrap();
    let cluster = cluster();
    let matrix = json!({
        "numeric_rows": [{"cluster_id": 0, "values": {"age": 1.0}}],
        "categorical_rows": [
            {"cluster_id": 0, "values": {"species=Adelie": 60.0}},
            {"cluster_id": 0, "values": {"species=Gentoo": {
                "value": 40.0,
                "prevalence": 40.0,
                "aggregate_score": 0.9,
                "signal_tier": "core",
            }}},
        ],
    });
    builder.cluster_entry("/clusters/0", &cluster);
    builder.signal_matrix("/signal_matrix", &matrix);
    let release = builder.finish();
    let payload = json!({"clusters": [cluster], "signal_matrix": matrix});

    let groups = group_rows(&release);
    let all = labels.len();
    let mut paths = BTreeSet::new();
    for label in release["labels"].as_array().unwrap() {
        let path = label["path"].as_str().unwrap();
        paths.insert(path.to_owned());
        let target = payload
            .pointer(path)
            .unwrap_or_else(|| panic!("{path} does not resolve"));
        match label["kind"].as_str().unwrap() {
            "count" => {
                let group = groups.get(label["group"].as_str().unwrap()).unwrap();
                assert_eq!(
                    target.as_f64().unwrap(),
                    group.len() as f64,
                    "{path} does not count its group"
                );
            }
            "share" => {
                let group = groups.get(label["group"].as_str().unwrap()).unwrap();
                let of = label["of"].as_str().unwrap();
                let of_len = if of == "all" {
                    all
                } else {
                    groups.get(of).unwrap().len()
                };
                let scale = if target.as_f64().unwrap() > 1.0 {
                    100.0
                } else {
                    1.0
                };
                let expected = (group.len() as f64 / of_len as f64 * scale * 100.0).round() / 100.0;
                assert_eq!(
                    target.as_f64().unwrap(),
                    expected,
                    "{path} does not describe its twin-row share"
                );
            }
            "label" => assert!(target.is_string(), "{path} is not a string"),
            _ => {}
        }
    }

    // Group ids name levels by index, never by raw value.
    for id in groups.keys() {
        assert!(!id.contains("Adelie"), "{id} carries a raw value");
        assert!(!id.contains("Gentoo"), "{id} carries a raw value");
    }
    for id in ["c0:species=0", "c0:species=1", "v:species=0", "v:species=1"] {
        assert!(groups.contains_key(id), "missing group {id}");
    }

    // Numeric rows label only c0-vs-rest statistics.
    for key in [
        "neighbor_cluster_id",
        "global_mean",
        "mean_rest",
        "median_rest",
        "neighbor_effect",
    ] {
        let path = format!("/clusters/0/numeric_features/0/{key}");
        assert!(!paths.contains(&path), "{path} must stay unlabelled");
    }
    for key in [
        "z_score",
        "effect_mean_std",
        "effect_median_mad",
        "ks_stat",
        "wasserstein_norm",
        "variance_ratio_log",
        "one_vs_rest_p",
        "one_vs_rest_q",
    ] {
        let path = format!("/clusters/0/numeric_features/0/{key}");
        assert!(paths.contains(&path), "{path} should be labelled");
    }

    // Rest/global preview values are never attributed to c0.
    for path in [
        "/clusters/0/feature_preview/0/mean_rest",
        "/clusters/0/feature_preview/1/lift",
        "/clusters/0/categorical_features/0/prevalence_rest",
        "/clusters/0/categorical_features/0/lift",
    ] {
        assert!(!paths.contains(path), "{path} must stay unlabelled");
    }
    assert!(paths.contains("/clusters/0/feature_preview/1/prevalence_cluster"));

    // Matrix cells are self-scoped shares of their categorical sub-group.
    assert!(paths.contains("/signal_matrix/categorical_rows/0/values/species=Adelie"));
    assert!(paths.contains("/signal_matrix/categorical_rows/1/values/species=Gentoo/value"));
}

#[test]
fn neighbor_cluster_id_and_global_mean_carry_no_label() {
    let labels: Vec<i64> = (0..30).map(|row| if row < 20 { 0 } else { 1 }).collect();
    let mut builder = ReleaseBuilder::for_test("/tmp/twin.csv", &labels, table()).unwrap();
    let cluster = cluster();
    builder.cluster_entry("/clusters/0", &cluster);
    let release = builder.finish();
    let paths: BTreeSet<String> = release["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["path"].as_str().unwrap().to_owned())
        .collect();
    assert!(
        !paths.contains("/clusters/0/numeric_features/0/neighbor_cluster_id"),
        "neighbor_cluster_id must carry no label"
    );
    assert!(
        !paths.contains("/clusters/0/numeric_features/0/global_mean"),
        "global_mean must carry no label"
    );
}

#[test]
fn group_ids_contain_no_raw_values() {
    let labels: Vec<i64> = (0..30).map(|row| if row < 20 { 0 } else { 1 }).collect();
    let mut builder = ReleaseBuilder::for_test("/tmp/twin.csv", &labels, table()).unwrap();
    let cluster = cluster();
    builder.cluster_entry("/clusters/0", &cluster);
    let release = builder.finish();
    for g in release["groups"].as_array().unwrap() {
        let id = g["id"].as_str().unwrap();
        assert!(
            !id.contains("Adelie"),
            "group id {id} contains raw value Adelie"
        );
        assert!(
            !id.contains("Gentoo"),
            "group id {id} contains raw value Gentoo"
        );
    }
}

#[test]
fn fragmented_groups_use_bitmap_b64_and_stay_compact() {
    let labels: Vec<i64> = (0..10_000).map(|i| i % 2).collect();
    let groups = super::groups_of(&labels);
    assert_eq!(groups.len(), 2);
    for g in &groups {
        assert_eq!(g["rows"]["enc"], "bitmap_b64");
    }
    let release = json!({"groups": groups});
    let chars = crate::budget::json_chars(&release);
    assert!(chars < 4_000, "{chars}");
    let decoded = group_rows(&release);
    assert_eq!(decoded["c0"].len(), 5_000);
    assert_eq!(decoded["c1"].len(), 5_000);
    assert!(decoded["c0"].iter().all(|r| r % 2 == 0));
    assert!(decoded["c1"].iter().all(|r| r % 2 == 1));
}
