//! Parity for the stages that were already Rust in v0.5, chained exactly as
//! `ThemaRS.fit` called them. Each stage starts from the *fixture* output of the
//! previous stage, so a failure points at one kernel rather than cascading.

use ndarray::Array2;
use pulsar_core::{
    accumulate_pseudo_laplacians_sparse, ball_mapper_grid, find_stable_thresholds_from_edges,
    jl_grid, pca_grid, CosmicGraphInner, MinHashSignatures, StandardScalerInner,
};
use pulsar_parity::{assert_close, assert_edges_close, load, Manifest, Matrix};
use serde::Deserialize;

#[derive(Deserialize)]
struct Embeddings {
    method: String,
    center: bool,
    grid: Vec<GridEmbedding>,
}

#[derive(Deserialize)]
struct PcaEmbeddings {
    grid: Vec<GridEmbedding>,
}

#[derive(Deserialize)]
struct GridEmbedding {
    dimension: usize,
    seed: u64,
    #[serde(flatten)]
    matrix: Matrix,
}

#[derive(Deserialize)]
struct BallMap {
    embedding_index: usize,
    epsilon: f64,
    nodes: Vec<Vec<usize>>,
    edges: Vec<[usize; 2]>,
}

#[derive(Deserialize)]
struct Cosmic {
    n: usize,
    edges: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct MinHashCosmic {
    n: usize,
    minhash_d: usize,
    minhash_seed: u64,
    edges: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct Graph {
    weight_scale: f64,
}

#[derive(Deserialize)]
struct Stability {
    num_bins: usize,
    optimal_threshold: f64,
    thresholds: Vec<f64>,
    component_counts: Vec<usize>,
    plateaus: Vec<Plateau>,
}

#[derive(Deserialize)]
struct Plateau {
    start: f64,
    end: f64,
    component_count: usize,
}

fn for_each_case(check: impl Fn(&Manifest, &str)) {
    let manifest = Manifest::load();
    let cases = manifest.case_names();
    assert!(!cases.is_empty(), "manifest lists no cases");
    for case in &cases {
        check(&manifest, case);
    }
}

fn epsilons(maps: &[BallMap]) -> Vec<f64> {
    let first = maps[0].embedding_index;
    maps.iter()
        .take_while(|m| m.embedding_index == first)
        .map(|m| m.epsilon)
        .collect()
}

#[test]
fn standard_scaler_matches_golden() {
    for_each_case(|manifest, case| {
        let pre: Matrix = load(case, "preprocessed");
        let want: Matrix = load(case, "scaled");
        let (got, _) = StandardScalerInner::fit_transform(&pre.to_array());
        assert_eq!(got.dim(), (want.shape[0], want.shape[1]), "{case}: shape");
        assert_close(
            &format!("{case}/scaled"),
            got.as_slice().unwrap(),
            &want.data,
            manifest.rtol("scaled"),
        );
    });
}

#[test]
fn jl_grid_matches_golden() {
    for_each_case(|manifest, case| {
        let scaled = load::<Matrix>(case, "scaled").to_array();
        let want: Embeddings = load(case, "embeddings");
        assert_eq!(want.method, "jl", "{case}: fixture projection method");
        let (dims, seeds) = grid_axes(&want.grid);
        let got = jl_grid(&scaled, &dims, &seeds, want.center).expect("jl_grid");
        assert_grid(
            &format!("{case}/embeddings"),
            &got,
            &want.grid,
            manifest.rtol("embeddings"),
        );
    });
}

#[test]
fn pca_grid_matches_golden() {
    for_each_case(|manifest, case| {
        let scaled = load::<Matrix>(case, "scaled").to_array();
        let want: PcaEmbeddings = load(case, "pca_embeddings");
        let (dims, seeds) = grid_axes(&want.grid);
        // n_oversamples=10, n_power_iter=2: the Python binding's defaults.
        let got = pca_grid(&scaled, &dims, &seeds, 10, 2).expect("pca_grid");
        assert_grid(
            &format!("{case}/pca_embeddings"),
            &got,
            &want.grid,
            manifest.rtol("pca_embeddings"),
        );
    });
}

/// Dimensions and seeds of a grid, in first-seen order.
fn grid_axes(grid: &[GridEmbedding]) -> (Vec<usize>, Vec<u64>) {
    (
        dedup(grid.iter().map(|g| g.dimension)),
        dedup(grid.iter().map(|g| g.seed)),
    )
}

/// Grid kernels return seed-outer, dimension-inner order; the fixture labels
/// each cell, so check the labels as well as the values.
fn assert_grid(label: &str, got: &[Array2<f64>], want: &[GridEmbedding], rtol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: grid size");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        let cell = format!("{label}[{k}] d={} seed={}", w.dimension, w.seed);
        assert_eq!(g.ncols(), w.dimension, "{cell}: column count");
        assert_eq!(w.matrix.shape[1], w.dimension, "{cell}: fixture label");
        assert_close(
            &cell,
            g.as_standard_layout().as_slice().unwrap(),
            &w.matrix.data,
            rtol,
        );
    }
    let seeds: Vec<u64> = want.iter().map(|w| w.seed).collect();
    let mut seed_outer = seeds.clone();
    seed_outer.dedup();
    assert_eq!(
        seed_outer.len(),
        dedup(seeds.into_iter()).len(),
        "{label}: fixture grid is not seed-outer"
    );
}

#[test]
fn ball_mapper_grid_matches_golden() {
    for_each_case(|_, case| {
        let embeddings: Embeddings = load(case, "embeddings");
        let want: Vec<BallMap> = load(case, "ball_maps");
        let arrays: Vec<Array2<f64>> = embeddings
            .grid
            .iter()
            .map(|g| g.matrix.to_array())
            .collect();
        let got = ball_mapper_grid(&arrays, &epsilons(&want));
        assert_eq!(got.len(), want.len(), "{case}: ball map count");
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.eps, w.epsilon, "{case}: ball map {k} epsilon");
            let nodes: Vec<Vec<usize>> = g
                .nodes
                .iter()
                .map(|n| {
                    let mut n = n.clone();
                    n.sort_unstable();
                    n
                })
                .collect();
            assert_eq!(nodes, w.nodes, "{case}: ball map {k} nodes");
            let mut edges: Vec<[usize; 2]> = g.edges.iter().map(|&(a, b)| [a, b]).collect();
            edges.sort_unstable();
            assert_eq!(edges, w.edges, "{case}: ball map {k} edges");
        }
    });
}

#[test]
fn exact_cosmic_graph_matches_golden() {
    for_each_case(|manifest, case| {
        let maps: Vec<BallMap> = load(case, "ball_maps");
        let want: Cosmic = load(case, "cosmic_exact");
        let per_map: Vec<&[Vec<usize>]> = maps.iter().map(|m| m.nodes.as_slice()).collect();
        let lap = accumulate_pseudo_laplacians_sparse(&per_map, want.n)
            .expect("golden ball maps index points below n");
        let cosmic =
            CosmicGraphInner::from_pseudo_laplacian_sparse(lap.n, &lap.diag, &lap.offdiag, 0.0);
        assert_edges_close(
            &format!("{case}/cosmic_exact"),
            &cosmic.weighted_edges(),
            &want.edges,
            manifest.rtol("cosmic_exact"),
        );
    });
}

#[test]
fn minhash_cosmic_graph_matches_golden() {
    for_each_case(|manifest, case| {
        let maps: Vec<BallMap> = load(case, "ball_maps");
        let want: MinHashCosmic = load(case, "cosmic_minhash");
        let balls: Vec<&[usize]> = maps
            .iter()
            .flat_map(|m| m.nodes.iter().map(Vec::as_slice))
            .collect();
        let mut sig = MinHashSignatures::new(want.n, want.minhash_d, want.minhash_seed);
        sig.accumulate(&balls);
        assert_edges_close(
            &format!("{case}/cosmic_minhash"),
            &sig.edges(),
            &want.edges,
            manifest.rtol("cosmic_minhash"),
        );
    });
}

#[test]
fn stability_matches_golden() {
    for_each_case(|manifest, case| {
        let cosmic: Cosmic = load(case, "cosmic_exact");
        let graph: Graph = load(case, "graph");
        let want: Stability = load(case, "stability");
        let rtol = manifest.rtol("stability");
        // ThemaRS normalizes weights by max(1, max_weight) before the H0 scan.
        let edges: Vec<(usize, usize, f64)> = cosmic
            .edges
            .iter()
            .map(|e| (e[0] as usize, e[1] as usize, e[2] / graph.weight_scale))
            .filter(|&(_, _, w)| w > 0.0)
            .collect();
        let got =
            find_stable_thresholds_from_edges(cosmic.n, &edges, want.num_bins).expect("stability");
        let label = format!("{case}/stability");
        assert_close(
            &label,
            &[got.optimal_threshold],
            &[want.optimal_threshold],
            rtol,
        );
        assert_close(&label, &got.thresholds, &want.thresholds, rtol);
        assert_eq!(got.component_counts, want.component_counts, "{label}");
        assert_eq!(got.plateaus.len(), want.plateaus.len(), "{label}: plateaus");
        for (g, w) in got.plateaus.iter().zip(&want.plateaus) {
            assert_close(
                &label,
                &[g.start_threshold, g.end_threshold],
                &[w.start, w.end],
                rtol,
            );
            assert_eq!(g.component_count, w.component_count, "{label}");
        }
    });
}

fn dedup<T: PartialEq + Copy>(items: impl Iterator<Item = T>) -> Vec<T> {
    let mut out = Vec::new();
    for x in items {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}
