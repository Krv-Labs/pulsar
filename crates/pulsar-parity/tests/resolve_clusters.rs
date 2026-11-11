//! Parity for cluster resolution: the Rust port must reproduce what
//! `generate_resolve_clusters.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from
//! Python (`resolve_clusters` in `pulsar/mcp/interpreter.py`).
//!
//! Components and threshold-stability cuts are compared label for label, with
//! plateaus and thresholds within `resolve_clusters` rtol. Spectral cuts use a
//! different label solver (see the `clustering::resolve` module header), so
//! they are compared as partitions (`resolve_clusters_spectral` min ARI), with
//! the chosen `k`, cluster count, residual numbering text and, when the
//! partition is identical, the silhouette. `NO_STABLE_SPECTRAL_CUT` cases
//! compare the structural diagnostics; per-`k` scores only when Python had no
//! candidate to score.

use pulsar_analysis::clustering::resolve::{resolve_clusters, ResolveError, ResolveMethod};
use pulsar_parity::{adjusted_rand_index, assert_json_close, golden_dir, Manifest, Tolerance};
use pulsar_pipeline::Graph;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = golden_dir().join("dossier/resolve_clusters.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn resolve_clusters_matches_python() {
    let manifest = Manifest::load();
    let rtol = manifest.rtol("resolve_clusters");
    let min_ari = match manifest.tolerances.get("resolve_clusters_spectral") {
        Some(Tolerance::Partition { min_ari }) => *min_ari,
        other => panic!("resolve_clusters_spectral: expected a partition tolerance, got {other:?}"),
    };
    let fx = fixture();
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 10, "fixture has {} cases", cases.len());

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let graph = Graph {
            n: case["n"].as_u64().unwrap() as usize,
            edges: case["edges"]
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
                .collect(),
        };
        let method = ResolveMethod::parse(case["method"].as_str().unwrap()).unwrap();
        let max_k = case["max_k"].as_u64().unwrap() as usize;
        let threshold = case["interpretation_edge_weight_threshold"]
            .as_f64()
            .unwrap();
        let got = resolve_clusters(&graph, None, method, max_k, threshold);

        if let Some(want) = case.get("error") {
            let Err(ResolveError::NoStableSpectralCut(diag)) = got else {
                panic!("{name}: expected NO_STABLE_SPECTRAL_CUT, got {got:?}");
            };
            assert_eq!(
                want["error_code"],
                json!("NO_STABLE_SPECTRAL_CUT"),
                "{name}"
            );
            let mut got_details = diag.details(true);
            let mut want_details = want["details"].clone();
            let python_scored = !want_details["candidate_scores"]
                .as_array()
                .unwrap()
                .is_empty();
            if python_scored {
                for key in ["best_k", "best_silhouette_score", "candidate_scores"] {
                    got_details.as_object_mut().unwrap().remove(key);
                    want_details.as_object_mut().unwrap().remove(key);
                }
            }
            assert_json_close(
                &format!("{name}.details"),
                &got_details,
                &want_details,
                rtol,
            );
            continue;
        }

        let want = &case["result"];
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            got.method_used,
            want["method_used"].as_str().unwrap(),
            "{name}: method_used"
        );
        assert_eq!(
            got.n_clusters as u64,
            want["n_clusters"].as_u64().unwrap(),
            "{name}: n_clusters"
        );
        assert_eq!(
            got.failure_reason.as_deref(),
            want["failure_reason"].as_str(),
            "{name}: failure_reason"
        );
        assert_json_close(
            &format!("{name}.threshold"),
            &json!(got.interpretation_edge_weight_threshold_applied),
            &want["interpretation_edge_weight_threshold_applied"],
            rtol,
        );
        assert_json_close(
            &format!("{name}.stability_plateaus"),
            &serde_json::to_value(&got.stability_plateaus).unwrap(),
            &want["stability_plateaus"],
            rtol,
        );
        let want_labels = usizes(&want["labels"]);
        if got.method_used == "spectral" {
            let ari = adjusted_rand_index(&got.labels, &want_labels);
            assert!(ari >= min_ari, "{name}: spectral ARI {ari} < {min_ari}");
            if ari == 1.0 {
                assert_json_close(
                    &format!("{name}.silhouette"),
                    &json!(got.silhouette_score),
                    &want["silhouette_score"],
                    rtol,
                );
            }
        } else {
            assert_eq!(got.labels, want_labels, "{name}: labels");
            assert!(
                got.silhouette_score.is_none() && want["silhouette_score"].is_null(),
                "{name}"
            );
        }
    }
}
