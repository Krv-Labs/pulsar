//! End-to-end parity for the ported pipeline (#63): reading each case's
//! `input.csv` and `config.yaml` through `pulsar-data` + `pulsar-pipeline`
//! must reproduce what `scripts/golden/generate.py` froze from `ThemaRS.fit`.
//!
//! The generator fitted the config twice: once with `construction: exact`
//! (source of `preprocessed`, `ball_maps`, `cosmic_exact`, `stability` and
//! `graph`) and once as written (`cosmic_minhash`).

use pulsar_config::{Construction, PulsarConfig};
use pulsar_data::{preprocess, read_csv_path, Table};
use pulsar_parity::{assert_close, assert_edges_close, golden_dir, load, Manifest, Matrix};
use pulsar_pipeline::{fit, fit_multi, FitOptions, Fitted};
use serde::Deserialize;

#[derive(Deserialize)]
struct PreprocessedFixture {
    columns: Vec<String>,
    #[serde(flatten)]
    matrix: Matrix,
}

#[derive(Deserialize)]
struct Cosmic {
    n: usize,
    edges: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct GraphFixture {
    construction_threshold: f64,
    weight_scale: f64,
    edges: Vec<[usize; 2]>,
}

#[derive(Deserialize)]
struct Stability {
    optimal_threshold: f64,
    thresholds: Vec<f64>,
    component_counts: Vec<usize>,
}

#[derive(Deserialize)]
struct BallMap {
    embedding_index: usize,
    epsilon: f64,
}

fn inputs(case: &str) -> (PulsarConfig, Table) {
    let dir = golden_dir().join(case);
    let cfg = pulsar_config::from_path(dir.join("config.yaml"))
        .unwrap_or_else(|e| panic!("{case}: config.yaml failed to load:\n{e}"));
    let table = read_csv_path(dir.join("input.csv"))
        .unwrap_or_else(|e| panic!("{case}: input.csv failed to read: {e}"));
    (cfg, table)
}

fn for_each_case(check: impl Fn(&Manifest, &str)) {
    let manifest = Manifest::load();
    let cases = manifest.case_names();
    assert!(!cases.is_empty(), "manifest lists no cases");
    for case in &cases {
        check(&manifest, case);
    }
}

fn exact(mut cfg: PulsarConfig) -> PulsarConfig {
    cfg.cosmic_graph.construction = Construction::Exact;
    cfg
}

fn fit_case(cfg: &PulsarConfig, table: &Table) -> Fitted {
    fit(cfg, table, FitOptions::default(), None).expect("fit")
}

#[test]
fn preprocess_matches_golden() {
    for_each_case(|manifest, case| {
        let (cfg, table) = inputs(case);
        let want: PreprocessedFixture = load(case, "preprocessed");
        let got = preprocess(&table, &cfg.preprocessing, None, None)
            .unwrap_or_else(|e| panic!("{case}: preprocess failed: {e}"));
        assert_eq!(got.feature_names(), want.columns, "{case}: feature names");
        assert_eq!(
            got.matrix.dim(),
            (want.matrix.shape[0], want.matrix.shape[1]),
            "{case}: shape"
        );
        assert_close(
            &format!("{case}/preprocessed"),
            got.matrix.as_slice().unwrap(),
            &want.matrix.data,
            manifest.rtol("preprocessed"),
        );
    });
}

#[test]
fn exact_fit_matches_golden() {
    for_each_case(|manifest, case| {
        let (cfg, table) = inputs(case);
        let model = fit_case(&exact(cfg), &table);
        let label = |artifact: &str| format!("{case}/{artifact}");

        let maps: Vec<BallMap> = load(case, "ball_maps");
        assert_eq!(model.ball_maps.len(), maps.len(), "{case}: ball map count");
        let n_eps = model.config.ball_mapper.epsilons.len();
        for (k, (g, w)) in model.ball_maps.iter().zip(&maps).enumerate() {
            assert_eq!(g.eps, w.epsilon, "{case}: ball map {k} epsilon");
            assert_eq!(
                k / n_eps,
                w.embedding_index,
                "{case}: ball map {k} embedding"
            );
        }
        assert_eq!(
            model.embeddings.len(),
            maps.len() / n_eps,
            "{case}: embeddings"
        );

        let want: Cosmic = load(case, "cosmic_exact");
        assert_eq!(model.n, want.n, "{case}: n");
        assert_edges_close(
            &label("cosmic_exact"),
            &model.cosmic.weighted_edges(),
            &want.edges,
            manifest.rtol("cosmic_exact"),
        );

        let want: Stability = load(case, "stability");
        let stability = model
            .stability
            .as_ref()
            .expect("auto threshold yields stability");
        let rtol = manifest.rtol("stability");
        assert_close(
            &label("stability"),
            &[stability.optimal_threshold],
            &[want.optimal_threshold],
            rtol,
        );
        assert_close(
            &label("stability"),
            &stability.thresholds,
            &want.thresholds,
            rtol,
        );
        assert_eq!(
            stability.component_counts, want.component_counts,
            "{case}: counts"
        );

        let want: GraphFixture = load(case, "graph");
        assert_close(
            &label("graph"),
            &[model.construction_threshold, model.weight_scale],
            &[want.construction_threshold, want.weight_scale],
            rtol,
        );
        let got: Vec<[usize; 2]> = model.graph.edges.iter().map(|&(i, j, _)| [i, j]).collect();
        assert_eq!(got, want.edges, "{case}: constructed graph edges");
        assert_eq!(model.graph.n, want_n(&model), "{case}: graph node count");
        assert_eq!(
            model.weighted_edges(None).len(),
            want.edges.len(),
            "{case}: weighted_edges at the resolved threshold"
        );
    });
}

fn want_n(model: &Fitted) -> usize {
    model.n
}

#[test]
fn minhash_fit_matches_golden() {
    for_each_case(|manifest, case| {
        let (cfg, table) = inputs(case);
        assert_eq!(cfg.cosmic_graph.construction, Construction::Minhash);
        let model = fit_case(&cfg, &table);
        let want: Cosmic = load(case, "cosmic_minhash");
        assert_edges_close(
            &format!("{case}/cosmic_minhash"),
            &model.cosmic.weighted_edges(),
            &want.edges,
            manifest.rtol("cosmic_minhash"),
        );
    });
}

/// Folding the same table twice doubles every co-occurrence count and every
/// diagonal, so the exact weights `count / (d_i + d_j - count)` are unchanged:
/// `fit_multi` must agree with `fit` on this known answer.
#[test]
fn fit_multi_of_identical_tables_matches_single_fit() {
    for_each_case(|manifest, case| {
        let (cfg, table) = inputs(case);
        let cfg = exact(cfg);
        let single = fit_case(&cfg, &table);
        let multi = fit_multi(&cfg, &[&table, &table], FitOptions::default(), None)
            .unwrap_or_else(|e| panic!("{case}: fit_multi failed: {e}"));
        let want: Vec<[f64; 3]> = single
            .cosmic
            .weighted_edges()
            .into_iter()
            .map(|(i, j, w)| [i as f64, j as f64, w])
            .collect();
        assert_edges_close(
            &format!("{case}/fit_multi"),
            &multi.cosmic.weighted_edges(),
            &want,
            manifest.rtol("cosmic_exact"),
        );
        assert_eq!(multi.graph, single.graph, "{case}: constructed graph");
        // fit_multi pins the shared vocabulary on the layout; the feature
        // layout itself is identical.
        assert_eq!(
            multi.layout.feature_names, single.layout.feature_names,
            "{case}: layout"
        );
        assert_eq!(multi.layout.n_rows, single.layout.n_rows);
        assert_eq!(multi.ball_maps.len(), 2 * single.ball_maps.len());
    });
}

#[test]
fn cached_embeddings_reproduce_the_fit() {
    for_each_case(|_, case| {
        let (cfg, table) = inputs(case);
        let cfg = exact(cfg);
        let first = fit_case(&cfg, &table);
        let mut seen: Vec<String> = Vec::new();
        let mut cb = |stage: &str, _: f64| seen.push(stage.to_string());
        let again = fit(
            &cfg,
            &table,
            FitOptions {
                precomputed_embeddings: Some(first.embeddings.clone()),
                ballmap_batch_size: None,
                ..FitOptions::default()
            },
            Some(&mut cb),
        )
        .expect("fit with cached embeddings");
        assert_eq!(again.graph, first.graph, "{case}");
        assert!(seen.iter().any(|s| s == "projection (cached)"), "{seen:?}");
    });
}

#[test]
fn progress_is_monotonic_and_ends_at_one() {
    let (cfg, table) = inputs("blobs");
    let mut fractions: Vec<f64> = Vec::new();
    let mut cb = |_: &str, f: f64| fractions.push(f);
    fit(&cfg, &table, FitOptions::default(), Some(&mut cb)).expect("fit");
    assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "{fractions:?}");
    assert_eq!(fractions.last().copied(), Some(1.0));
}

#[test]
fn representatives_are_a_subset_of_the_sweep() {
    let (cfg, table) = inputs("blobs");
    let model = fit_case(&cfg, &table);
    let reps = model.select_representatives(Some(3));
    assert!(!reps.is_empty() && reps.len() <= 3, "{reps:?}");
    assert!(reps.iter().all(|&i| i < model.ball_maps.len()));
    assert_eq!(
        model.select_representatives(Some(100)).len(),
        model.ball_maps.len()
    );
}
