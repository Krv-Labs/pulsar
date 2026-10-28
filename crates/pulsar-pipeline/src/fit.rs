//! `fit` and `fit_multi`: the sweep, streamed into one cosmic graph.

use ndarray::Array2;
use pulsar_config::{
    Construction, CosmicGraph as CosmicSpec, Projection, ProjectionMethod, PulsarConfig, Threshold,
};
use pulsar_core::{
    accumulate_pseudo_laplacians_sparse, ball_mapper_grid, find_stable_thresholds_from_edges,
    jl_grid, pca_grid, BallMapper, CosmicGraphInner, MinHashSignatures, SparsePseudoLaplacian,
    StabilityResult, StandardScalerInner, DEFAULT_NUM_BINS,
};
use pulsar_data::{preprocess, read_table, shared_vocab, Layout, Preprocessed, Table};

use crate::error::PipelineError;
use crate::graph::Graph;
use crate::progress::{ProgressFn, ProgressTracker};
use crate::representatives::select_representatives;

/// Randomized-PCA defaults the Python binding used (`n_oversamples`,
/// `n_power_iter`).
const PCA_OVERSAMPLES: usize = 10;
const PCA_POWER_ITER: usize = 2;

/// Knobs that do not belong in `params.yaml`.
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Embeddings per Ball Mapper batch. `Some(1)` (the default) bounds the
    /// sparse accumulator's peak memory; `None` runs the whole grid at once.
    pub ballmap_batch_size: Option<usize>,
    /// Keep every fitted [`BallMapper`] on the model (needed by
    /// `select_representatives`). Off, batches are freed as they stream.
    pub store_ball_maps: bool,
    /// Projection embeddings from an earlier fit on the same data (see
    /// [`crate::projection_fingerprint`]); skips the projection stage.
    pub precomputed_embeddings: Option<Vec<Array2<f64>>>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            ballmap_batch_size: Some(1),
            store_ball_maps: true,
            precomputed_embeddings: None,
        }
    }
}

/// A fitted model.
///
/// Surfaces, labelled as the MCP contract requires:
/// - `base_cosmic`: the full weighted affinity straight from accumulation,
///   raw weights, never thresholded;
/// - `cosmic`: `base_cosmic` after optional spectral sparsification, the
///   object `weight_scale` and the threshold are resolved on;
/// - `graph`: the constructed graph, weights divided by `weight_scale` and
///   kept only above `construction_threshold`.
pub struct Fitted {
    pub config: PulsarConfig,
    pub n: usize,
    /// Preprocessed table of a single-table fit; `None` after `fit_multi`.
    pub preprocessed: Option<Preprocessed>,
    pub layout: Layout,
    /// Projection embeddings of a single-table fit (reusable across
    /// retries); empty after `fit_multi`.
    pub embeddings: Vec<Array2<f64>>,
    pub ball_maps: Vec<BallMapper>,
    pub base_cosmic: CosmicGraphInner,
    pub cosmic: CosmicGraphInner,
    /// `max(1, max edge weight of cosmic)`: maps weights onto `[0, 1]`.
    pub weight_scale: f64,
    /// Present when `construction_threshold` was `auto`.
    pub stability: Option<StabilityResult>,
    pub construction_threshold: f64,
    pub graph: Graph,
}

impl Fitted {
    /// Edges of `cosmic` on the `[0, 1]` scale, strictly above `threshold`
    /// (default: the resolved construction threshold).
    pub fn weighted_edges(&self, threshold: Option<f64>) -> Vec<(usize, usize, f64)> {
        let cutoff = threshold.unwrap_or(self.construction_threshold);
        self.cosmic
            .weighted_edges()
            .into_iter()
            .map(|(i, j, w)| (i, j, w / self.weight_scale))
            .filter(|&(_, _, w)| w > cutoff)
            .collect()
    }

    /// Spectrally sparsify `base_cosmic` (opt-in, post-construction). With
    /// `update`, the result replaces `cosmic` and the threshold, stability
    /// and graph are recomputed on it.
    #[allow(clippy::too_many_arguments)]
    pub fn spectral_sparsify(
        &mut self,
        epsilon: f64,
        seed: u64,
        sketch_dim: Option<usize>,
        sample_count: Option<usize>,
        pcg_tol: f64,
        max_iter: usize,
        update: bool,
    ) -> Result<CosmicGraphInner, PipelineError> {
        let sparse = self.base_cosmic.spectral_sparsify(
            epsilon,
            seed,
            sketch_dim,
            sample_count,
            pcg_tol,
            max_iter,
        )?;
        if update {
            let mut tracker = ProgressTracker::new(&[], None);
            let fin = finalize(
                clone_cosmic(&sparse),
                self.config.cosmic_graph.construction_threshold,
                &mut tracker,
            )?;
            self.apply(fin);
        }
        Ok(sparse)
    }

    /// Re-resolve the construction surface at a fixed threshold.
    pub fn with_threshold(&mut self, threshold: f64) {
        let mut tracker = ProgressTracker::new(&[], None);
        let fin = finalize(
            clone_cosmic(&self.cosmic),
            Threshold::Value(threshold),
            &mut tracker,
        )
        .expect("a fixed threshold cannot fail");
        self.apply(fin);
    }

    fn apply(&mut self, fin: Finalized) {
        self.cosmic = fin.cosmic;
        self.weight_scale = fin.weight_scale;
        self.stability = fin.stability;
        self.construction_threshold = fin.threshold;
        self.graph = fin.graph;
    }

    /// Indices into `ball_maps` of `n_reps` structurally diverse maps
    /// (default `n_reps`: the config's `output.n_reps`).
    pub fn select_representatives(&self, n_reps: Option<usize>) -> Vec<usize> {
        let k = n_reps.unwrap_or(self.config.n_reps as usize);
        select_representatives(&self.ball_maps, k, 42)
    }
}

/// Load `config.data` and fit it.
pub fn fit_path(
    cfg: &PulsarConfig,
    opts: FitOptions,
    progress: Option<&mut ProgressFn<'_>>,
) -> Result<Fitted, PipelineError> {
    if cfg.data.is_empty() {
        return Err(PipelineError::NoData);
    }
    let table = read_table(&cfg.data)?;
    fit(cfg, &table, opts, progress)
}

/// Run the full pipeline on one table.
pub fn fit(
    cfg: &PulsarConfig,
    table: &Table,
    opts: FitOptions,
    progress: Option<&mut ProgressFn<'_>>,
) -> Result<Fitted, PipelineError> {
    let batch_size = check_batch_size(opts.ballmap_batch_size)?;
    check_epsilons(&cfg.ball_mapper.epsilons)?;
    let use_cached = opts.precomputed_embeddings.is_some();
    let variants = cfg.preprocessing_variants();
    let n_variants = variants.len().max(1);
    let projection_count = opts
        .precomputed_embeddings
        .as_ref()
        .map_or_else(|| grid_size(&cfg.projection) * n_variants, Vec::len);
    let batch_count = batch_count(projection_count, batch_size);
    let n_eps = cfg.ball_mapper.epsilons.len().max(1) as f64;
    let stages = vec![
        ("load".to_string(), 1.0),
        ("preprocess".to_string(), 1.0),
        ("scale".to_string(), 1.0),
        (
            "projection".to_string(),
            if use_cached {
                0.1
            } else {
                projection_count as f64
            },
        ),
        ("sweep_batches".to_string(), batch_count as f64 * n_eps),
        ("cosmic_graph".to_string(), cosmic_weight(&cfg.cosmic_graph)),
        ("stability".to_string(), stability_weight(&cfg.cosmic_graph)),
        ("graph".to_string(), 1.0),
    ];
    let mut tracker = ProgressTracker::new(&stages, progress);
    tracker.complete("load", None);

    let pre = preprocess(table, &variants[0].preprocessing, None, None)?;
    tracker.complete("preprocess", None);
    let n = pre.matrix.nrows();

    let embeddings = match opts.precomputed_embeddings {
        Some(e) => {
            tracker.complete("scale", None);
            if let Some(bad) = e.iter().find(|m| m.nrows() != n) {
                return Err(PipelineError::EmbeddingRows {
                    got: bad.nrows(),
                    expected: n,
                });
            }
            e
        }
        None => {
            let (x_scaled, _) = StandardScalerInner::fit_transform(&pre.matrix);
            tracker.complete("scale", None);
            let mut all_embeddings = projection_grid(&x_scaled, &variants[0].projection)?;
            drop(x_scaled);
            tracker.update(
                "projection",
                1.0 / n_variants as f64,
                Some(&format!("projection variant 1/{n_variants}")),
            );
            for (idx, variant_cfg) in variants[1..].iter().enumerate() {
                let pre_k = preprocess(table, &variant_cfg.preprocessing, None, None)?;
                let (x_k_scaled, _) = StandardScalerInner::fit_transform(&pre_k.matrix);
                all_embeddings.extend(projection_grid(&x_k_scaled, &variant_cfg.projection)?);
                tracker.update(
                    "projection",
                    (idx + 2) as f64 / n_variants as f64,
                    Some(&format!("projection variant {}/{n_variants}", idx + 2)),
                );
            }
            all_embeddings
        }
    };
    tracker.complete(
        "projection",
        if use_cached {
            Some("projection (cached)")
        } else {
            None
        },
    );

    let mut builder = CosmicBuilder::new(n, &cfg.cosmic_graph);
    let mut ball_maps = Vec::new();
    sweep(
        &embeddings,
        &cfg.ball_mapper.epsilons,
        batch_size,
        &mut builder,
        opts.store_ball_maps.then_some(&mut ball_maps),
        &mut tracker,
        "",
    )?;

    let base = builder.build();
    let cosmic = sparsify(&base, &cfg.cosmic_graph)?;
    tracker.complete("cosmic_graph", None);
    let fin = finalize(
        cosmic,
        cfg.cosmic_graph.construction_threshold,
        &mut tracker,
    )?;

    Ok(Fitted {
        config: cfg.clone(),
        n,
        layout: pre.layout.clone(),
        preprocessed: Some(pre),
        embeddings,
        ball_maps,
        base_cosmic: base,
        cosmic: fin.cosmic,
        weight_scale: fin.weight_scale,
        stability: fin.stability,
        construction_threshold: fin.threshold,
        graph: fin.graph,
    })
}

/// Run the sweep on several representations of the same points and fold
/// every ball map into one cosmic graph: a high weight means two points are
/// neighbours across *all* representations.
///
/// Every table must preprocess to the same layout and row count; one-hot
/// vocabularies are shared across tables so blocks line up.
pub fn fit_multi(
    cfg: &PulsarConfig,
    tables: &[&Table],
    opts: FitOptions,
    progress: Option<&mut ProgressFn<'_>>,
) -> Result<Fitted, PipelineError> {
    if tables.is_empty() {
        return Err(PipelineError::EmptyDatasets);
    }
    let batch_size = check_batch_size(opts.ballmap_batch_size)?;
    check_epsilons(&cfg.ball_mapper.epsilons)?;
    let count = tables.len();
    let projection_count = grid_size(&cfg.projection);
    let batch_count = batch_count(projection_count, batch_size);
    let n_eps = cfg.ball_mapper.epsilons.len().max(1) as f64;
    let mut stages = Vec::new();
    for i in 0..count {
        let p = format!("Dataset {}/{}: ", i + 1, count);
        stages.push((format!("{p}load"), 1.0));
        stages.push((format!("{p}preprocess"), 1.0));
        stages.push((format!("{p}scale"), 1.0));
        stages.push((format!("{p}projection"), projection_count as f64));
        stages.push((format!("{p}sweep_batches"), batch_count as f64 * n_eps));
    }
    stages.push(("cosmic_graph".to_string(), cosmic_weight(&cfg.cosmic_graph)));
    stages.push(("stability".to_string(), stability_weight(&cfg.cosmic_graph)));
    stages.push(("graph".to_string(), 1.0));
    let mut tracker = ProgressTracker::new(&stages, progress);

    let vocab = if cfg.preprocessing.encode.is_empty() {
        None
    } else {
        Some(shared_vocab(tables, &cfg.preprocessing))
    };

    let mut builder: Option<CosmicBuilder> = None;
    let mut ball_maps = Vec::new();
    let mut n: Option<usize> = None;
    let mut ref_layout: Option<Layout> = None;
    for (i, table) in tables.iter().enumerate() {
        let p = format!("Dataset {}/{}: ", i + 1, count);
        tracker.complete(&format!("{p}load"), None);
        let pre = preprocess(
            table,
            &cfg.preprocessing,
            vocab.as_ref(),
            ref_layout.as_ref(),
        )?;
        tracker.complete(&format!("{p}preprocess"), None);
        let rows = pre.matrix.nrows();
        match n {
            None => n = Some(rows),
            Some(expected) if expected != rows => {
                return Err(PipelineError::RowCountMismatch {
                    index: i,
                    got: rows,
                    expected,
                })
            }
            Some(_) => {}
        }
        let (x_scaled, _) = StandardScalerInner::fit_transform(&pre.matrix);
        tracker.complete(&format!("{p}scale"), None);
        let embeddings = projection_grid(&x_scaled, &cfg.projection)?;
        drop(x_scaled);
        tracker.complete(&format!("{p}projection"), None);

        let b = builder.get_or_insert_with(|| CosmicBuilder::new(rows, &cfg.cosmic_graph));
        sweep(
            &embeddings,
            &cfg.ball_mapper.epsilons,
            batch_size,
            b,
            opts.store_ball_maps.then_some(&mut ball_maps),
            &mut tracker,
            &p,
        )?;
        if ref_layout.is_none() {
            ref_layout = Some(pre.layout);
        }
    }
    let (Some(builder), Some(n), Some(layout)) = (builder, n, ref_layout) else {
        return Err(PipelineError::EmptyDatasets);
    };

    let base = builder.build();
    let cosmic = sparsify(&base, &cfg.cosmic_graph)?;
    tracker.complete("cosmic_graph", None);
    let fin = finalize(
        cosmic,
        cfg.cosmic_graph.construction_threshold,
        &mut tracker,
    )?;

    Ok(Fitted {
        config: cfg.clone(),
        n,
        preprocessed: None,
        layout,
        embeddings: Vec::new(),
        ball_maps,
        base_cosmic: base,
        cosmic: fin.cosmic,
        weight_scale: fin.weight_scale,
        stability: fin.stability,
        construction_threshold: fin.threshold,
        graph: fin.graph,
    })
}

/// The projection grid: JL by default, randomized PCA as the legacy mode.
/// Returns seed-outer, dimension-inner order.
pub fn projection_grid(
    x_scaled: &Array2<f64>,
    proj: &Projection,
) -> Result<Vec<Array2<f64>>, PipelineError> {
    let out = match proj.method {
        ProjectionMethod::Jl => jl_grid(x_scaled, &proj.dimensions, &proj.seeds, proj.center)?,
        ProjectionMethod::Pca => pca_grid(
            x_scaled,
            &proj.dimensions,
            &proj.seeds,
            PCA_OVERSAMPLES,
            PCA_POWER_ITER,
        )?,
    };
    Ok(out)
}

fn grid_size(proj: &Projection) -> usize {
    (proj.dimensions.len() * proj.seeds.len()).max(1)
}

fn check_batch_size(size: Option<usize>) -> Result<Option<usize>, PipelineError> {
    match size {
        Some(0) => Err(PipelineError::InvalidOption {
            what: "ballmap_batch_size",
        }),
        other => Ok(other),
    }
}

fn batch_count(projection_count: usize, batch_size: Option<usize>) -> usize {
    match batch_size {
        None => 1,
        Some(b) => projection_count.div_ceil(b).max(1),
    }
}

fn cosmic_weight(spec: &CosmicSpec) -> f64 {
    1.0 + if spec.sparsify { 2.0 } else { 0.0 }
}

fn stability_weight(spec: &CosmicSpec) -> f64 {
    match spec.construction_threshold {
        Threshold::Auto => 2.0,
        Threshold::Value(_) => 0.0,
    }
}

/// Ball Mapper needs a finite radius > 0. The config loader enforces this,
/// but `PulsarConfig` fields are public, so programmatic configs are
/// re-checked before any ball map is built.
pub(crate) fn check_epsilons(epsilons: &[f64]) -> Result<(), PipelineError> {
    if epsilons.iter().all(|&eps| eps.is_finite() && eps > 0.0) {
        Ok(())
    } else {
        Err(PipelineError::InvalidOption {
            what: "ball_mapper.epsilons",
        })
    }
}

/// Ball Mapper over `embeddings` in batches, folded into `builder`.
fn sweep(
    embeddings: &[Array2<f64>],
    epsilons: &[f64],
    batch_size: Option<usize>,
    builder: &mut CosmicBuilder,
    mut store: Option<&mut Vec<BallMapper>>,
    tracker: &mut ProgressTracker<'_, '_>,
    prefix: &str,
) -> Result<(), PipelineError> {
    let batches: Vec<&[Array2<f64>]> = match batch_size {
        None => vec![embeddings],
        Some(b) => embeddings.chunks(b).collect(),
    };
    let total = batches.len();
    let stage = format!("{prefix}sweep_batches");
    for (k, batch) in batches.into_iter().enumerate() {
        let maps = ball_mapper_grid(batch, epsilons);
        builder.accumulate(&maps)?;
        if let Some(store) = store.as_deref_mut() {
            store.extend(maps);
        }
        tracker.update(
            &stage,
            (k + 1) as f64 / total as f64,
            Some(&format!(
                "{prefix}sweep batch {}/{total}: ball_mapper + construction",
                k + 1
            )),
        );
    }
    Ok(())
}

/// Streams ball maps into a cosmic graph by the configured construction.
/// Both paths are constant-memory in the number of ball maps and never
/// allocate an n×n matrix.
pub(crate) enum CosmicBuilder {
    /// Unbiased MinHash Jaccard sketch (default).
    MinHash(MinHashSignatures),
    /// Exact sparse pseudo-Laplacian merge: the bit-identical backbone.
    Exact {
        n: usize,
        acc: Option<SparsePseudoLaplacian>,
    },
}

impl CosmicBuilder {
    pub(crate) fn new(n: usize, spec: &CosmicSpec) -> Self {
        match spec.construction {
            Construction::Minhash => Self::MinHash(MinHashSignatures::new(
                n,
                spec.minhash_d as usize,
                spec.minhash_seed,
            )),
            Construction::Exact => Self::Exact { n, acc: None },
        }
    }

    pub(crate) fn accumulate(&mut self, maps: &[BallMapper]) -> Result<(), PipelineError> {
        match self {
            Self::MinHash(sig) => {
                let balls: Vec<&[usize]> = maps
                    .iter()
                    .flat_map(|m| m.nodes.iter().map(Vec::as_slice))
                    .collect();
                sig.accumulate(&balls);
            }
            Self::Exact { n, acc } => {
                let per_map: Vec<&[Vec<usize>]> = maps.iter().map(|m| m.nodes.as_slice()).collect();
                let block = accumulate_pseudo_laplacians_sparse(&per_map, *n)?;
                match acc {
                    None => *acc = Some(block),
                    Some(existing) => existing.merge_in_place(&block)?,
                }
            }
        }
        Ok(())
    }

    pub(crate) fn build(self) -> CosmicGraphInner {
        match self {
            Self::MinHash(sig) => CosmicGraphInner::Sparse {
                n: sig.n(),
                edges: sig.edges(),
            },
            Self::Exact { n, acc } => match acc {
                Some(l) => {
                    CosmicGraphInner::from_pseudo_laplacian_sparse(l.n, &l.diag, &l.offdiag, 0.0)
                }
                None => CosmicGraphInner::Sparse {
                    n,
                    edges: Vec::new(),
                },
            },
        }
    }
}

fn clone_cosmic(c: &CosmicGraphInner) -> CosmicGraphInner {
    CosmicGraphInner::Sparse {
        n: c.n(),
        edges: c.weighted_edges(),
    }
}

fn sparsify(base: &CosmicGraphInner, spec: &CosmicSpec) -> Result<CosmicGraphInner, PipelineError> {
    if !spec.sparsify {
        return Ok(clone_cosmic(base));
    }
    Ok(base.spectral_sparsify(
        spec.sparsify_epsilon,
        spec.sparsify_seed,
        spec.sparsify_sketch_dim.map(|v| v as usize),
        spec.sparsify_sample_count.map(|v| v as usize),
        spec.sparsify_pcg_tol,
        spec.sparsify_max_iter as usize,
    )?)
}

struct Finalized {
    cosmic: CosmicGraphInner,
    weight_scale: f64,
    stability: Option<StabilityResult>,
    threshold: f64,
    graph: Graph,
}

/// Normalize weights onto `[0, 1]` by `max(1, max_weight)` (a no-op unless
/// sparsification pushed weights above 1), resolve `auto` through the H0
/// stability scan, and build the thresholded graph.
fn finalize(
    cosmic: CosmicGraphInner,
    threshold: Threshold,
    tracker: &mut ProgressTracker<'_, '_>,
) -> Result<Finalized, PipelineError> {
    let n = cosmic.n();
    let edges = cosmic.weighted_edges();
    let max_w = edges.iter().map(|e| e.2).fold(0.0_f64, f64::max);
    let weight_scale = max_w.max(1.0);
    let normalized: Vec<(usize, usize, f64)> = edges
        .iter()
        .map(|&(i, j, w)| (i, j, w / weight_scale))
        .collect();
    let (stability, threshold) = match threshold {
        Threshold::Auto => {
            let s = find_stable_thresholds_from_edges(n, &normalized, DEFAULT_NUM_BINS)?;
            let t = s.optimal_threshold;
            tracker.complete("stability", None);
            (Some(s), t)
        }
        Threshold::Value(v) => (None, v),
    };
    let mut kept: Vec<(usize, usize, f64)> = normalized
        .into_iter()
        .filter(|&(_, _, w)| w > threshold)
        .collect();
    kept.sort_by_key(|e| (e.0, e.1));
    let graph = Graph { n, edges: kept };
    tracker.complete("graph", None);
    Ok(Finalized {
        cosmic,
        weight_scale,
        stability,
        threshold,
        graph,
    })
}
