//! Parity for the graph-analysis port (#64): threshold options, structural
//! breakpoints and clustering must reproduce what `generate.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze from the Python `thresholds.py` and `cosmic_clusters`.
//!
//! The generator computed these on the `construction: exact` fit: the full
//! weighted affinity (`weighted_edges(threshold=0.0)`) feeds the threshold
//! graph, the H0 curve comes from `find_stable_thresholds_sparse`, and the
//! clusterings run on the constructed (thresholded) cosmic graph.

use pulsar_analysis::{
    agent_threshold_options, cluster_agglomerative, structural_breakpoints, Policy, ThresholdGraph,
};
use pulsar_config::{Construction, PulsarConfig};
use pulsar_data::{read_csv_path, Table};
use pulsar_parity::{
    adjusted_rand_index, assert_json_close, golden_dir, load, Manifest, Tolerance,
};
use pulsar_pipeline::{fit, FitOptions, Fitted};
use serde::Deserialize;

#[derive(Deserialize)]
struct Clusters {
    n_clusters: usize,
    labels: Vec<usize>,
}

fn inputs(case: &str) -> (PulsarConfig, Table) {
    let dir = golden_dir().join(case);
    let cfg = pulsar_config::from_path(dir.join("config.yaml"))
        .unwrap_or_else(|e| panic!("{case}: config.yaml failed to load:\n{e}"));
    let table = read_csv_path(dir.join("input.csv"))
        .unwrap_or_else(|e| panic!("{case}: input.csv failed to read: {e}"));
    (cfg, table)
}

fn exact_fit(case: &str) -> Fitted {
    let (mut cfg, table) = inputs(case);
    cfg.cosmic_graph.construction = Construction::Exact;
    fit(&cfg, &table, FitOptions::default(), None)
        .unwrap_or_else(|e| panic!("{case}: fit failed: {e}"))
}

fn min_ari(manifest: &Manifest, artifact: &str) -> f64 {
    match manifest.tolerances.get(artifact) {
        Some(Tolerance::Partition { min_ari }) => *min_ari,
        other => panic!("{artifact}: expected a partition tolerance, got {other:?}"),
    }
}

#[test]
fn threshold_options_and_breakpoints_match_golden() {
    let manifest = Manifest::load();
    for case in manifest.case_names() {
        let fitted = exact_fit(&case);
        let stability = fitted
            .stability
            .as_ref()
            .expect("golden cases use auto threshold");
        let graph = ThresholdGraph::from_edges(fitted.n, &fitted.weighted_edges(Some(0.0)));
        let plateaus: Vec<_> = stability.plateaus.iter().take(10).cloned().collect();

        let options = agent_threshold_options(
            &graph,
            &plateaus,
            &stability.thresholds,
            &stability.component_counts,
            Policy::Balanced,
            7,
        );
        let got = serde_json::to_value(&options).unwrap();
        let want: serde_json::Value = load(&case, "threshold_options");
        assert_json_close(
            &format!("{case} threshold_options"),
            &got,
            &want,
            manifest.rtol("threshold_options"),
        );

        let breakpoints = structural_breakpoints(
            &graph,
            &stability.thresholds,
            &stability.component_counts,
            5,
            16,
        );
        let got = serde_json::to_value(&breakpoints).unwrap();
        let want: serde_json::Value = load(&case, "breakpoints");
        assert_json_close(
            &format!("{case} breakpoints"),
            &got,
            &want,
            manifest.rtol("breakpoints"),
        );
    }
}

#[test]
fn agglomerative_clusters_match_golden_partitions() {
    let manifest = Manifest::load();
    let floor = min_ari(&manifest, "clusters_agglomerative");
    for case in manifest.case_names() {
        let fitted = exact_fit(&case);
        let want: Clusters = load(&case, "clusters_agglomerative");
        let got = cluster_agglomerative(&fitted.graph, want.n_clusters).unwrap();
        let ari = adjusted_rand_index(&got, &want.labels);
        assert!(ari >= floor, "{case}: agglomerative ARI {ari} < {floor}");
    }
}

/// Spectral clustering is only determined when the Laplacian nullspace
/// (one dimension per connected component) fits inside the embedding: with
/// more components than clusters, *any* orthonormal basis of the nullspace
/// is a valid embedding and scikit-learn's unseeded ARPACK start picked one
/// at random when the fixture was frozen (`penguins` has 20 components at
/// its construction threshold and asks for 3 clusters). The ARI floor is
/// enforced where the problem is well posed; elsewhere the invariant every
/// correct embedding guarantees is checked instead: a connected component
/// is never split across clusters.
#[test]
fn spectral_clusters_match_golden_partitions() {
    let manifest = Manifest::load();
    let floor = min_ari(&manifest, "clusters_spectral");
    let mut well_posed = 0;
    for case in manifest.case_names() {
        let fitted = exact_fit(&case);
        let want: Clusters = load(&case, "clusters_spectral");
        let components = fitted.graph.component_labels();
        let n_components = fitted.graph.n_components();
        for (name, force_iterative) in [("dense", false), ("LOBPCG", true)] {
            let got = pulsar_analysis::clustering::cluster_spectral_with(
                &fitted.graph,
                want.n_clusters,
                42,
                force_iterative,
            )
            .unwrap();
            assert_eq!(got.len(), want.labels.len());
            let ari = adjusted_rand_index(&got, &want.labels);
            if n_components <= want.n_clusters {
                assert!(
                    ari >= floor,
                    "{case} ({name}): spectral ARI {ari} < {floor}"
                );
            } else {
                for i in 0..got.len() {
                    for j in i + 1..got.len() {
                        if components[i] == components[j] {
                            assert_eq!(
                                got[i], got[j],
                                "{case} ({name}): component split across clusters at nodes {i}, {j}"
                            );
                        }
                    }
                }
                eprintln!(
                    "{case} ({name}): {n_components} components > {} clusters; ARI vs the \
                     unseeded sklearn draw is {ari:.3} (informational)",
                    want.n_clusters
                );
            }
        }
        if n_components <= want.n_clusters {
            well_posed += 1;
        }
    }
    assert!(
        well_posed >= 1,
        "at least one case must exercise the ARI floor"
    );
}
