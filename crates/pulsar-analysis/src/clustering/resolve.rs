//! Cluster resolution for the dossier (`resolve_clusters` in
//! `pulsar/mcp/interpreter.py`): connected components at an interpretation
//! threshold, an H0 threshold-stability cut, or a spectral cut chosen by
//! silhouette over `k = 2..=max_k`.
//!
//! Only the sparse (edge-list) path is ported; the Python dense
//! `weighted_adjacency` path was a fallback for models without
//! `weighted_edges`, and every Rust model has an edge list.
//!
//! Known divergences from Python:
//! - The spectral cut labels the embedding with k-means++ ([`cluster_spectral`],
//!   seed 42), where Python used scikit-learn's `assign_labels="discretize"`.
//!   Neither solver is bit-reproducible across implementations, so the golden
//!   fixture compares spectral partitions by ARI and the chosen `k`, not label
//!   by label, and per-`k` candidate scores on degenerate affinities (where
//!   every partition ties) are not compared.
//! - Silhouette distances (`1 − clip(w, 0, 1)`, 1 between non-adjacent nodes)
//!   are looked up from sorted adjacency rows instead of a dense `n × n`
//!   matrix; the values are the same.

use pulsar_core::{find_stable_thresholds_from_edges, StabilityResult, DEFAULT_NUM_BINS};
use pulsar_pipeline::Graph;
use serde::Serialize;
use serde_json::{json, Value};

use super::spectral::cluster_spectral;
use crate::components::ThresholdGraph;
use crate::error::AnalysisError;
use crate::pyround::round_to;
use crate::silhouette::silhouette_score;

/// Stability plateaus with this many components or more are not a usable cut.
const MAX_COMPONENTS: usize = 30;
/// A stability cut is rejected when more than this fraction of nodes are singletons.
const MAX_SINGLETON_RATIO: f64 = 0.5;
/// Smallest `k` the spectral search tries.
pub const SPECTRAL_K_MIN: usize = 2;
/// A spectral cut is accepted only above this mean silhouette.
pub const ACCEPTED_SILHOUETTE_MIN: f64 = 0.05;
/// Seed for the spectral k-means.
const SPECTRAL_SEED: u64 = 42;

/// Clustering strategy requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveMethod {
    /// Components when the interpretation threshold is positive; otherwise a
    /// threshold-stability cut, then components (disconnected graph) or spectral.
    Auto,
    Components,
    Spectral,
}

impl ResolveMethod {
    pub fn parse(name: &str) -> Result<Self, AnalysisError> {
        match name {
            "auto" => Ok(Self::Auto),
            "components" => Ok(Self::Components),
            "spectral" => Ok(Self::Spectral),
            other => Err(AnalysisError::InvalidArgument(format!(
                "unknown cluster resolution method {other:?}; expected auto, components or spectral"
            ))),
        }
    }
}

/// One of the longest stability plateaus, as reported with a stability cut.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlateauSummary {
    pub start: f64,
    pub end: f64,
    pub component_count: usize,
    pub length: f64,
}

/// A clustering with provenance (`ClusterResult`).
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterResult {
    /// Cluster label per node.
    pub labels: Vec<usize>,
    /// `"components"`, `"threshold_stability"` or `"spectral"`.
    pub method_used: &'static str,
    pub n_clusters: usize,
    pub silhouette_score: Option<f64>,
    pub failure_reason: Option<String>,
    pub interpretation_edge_weight_threshold_applied: f64,
    /// Up to five longest plateaus, for threshold-stability cuts only.
    pub stability_plateaus: Option<Vec<PlateauSummary>>,
}

/// How one `k` of the spectral search went.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateScore {
    Scored {
        k: usize,
        silhouette_score: f64,
        n_labels: usize,
    },
    SingleLabel {
        k: usize,
        n_labels: usize,
    },
    SolverError {
        k: usize,
        error: String,
    },
}

impl CandidateScore {
    fn to_json(&self) -> Value {
        match self {
            Self::Scored {
                k,
                silhouette_score,
                n_labels,
            } => json!({
                "k": k, "status": "scored", "silhouette_score": silhouette_score, "n_labels": n_labels,
            }),
            Self::SingleLabel { k, n_labels } => {
                json!({"k": k, "status": "single_label", "n_labels": n_labels})
            }
            Self::SolverError { k, error } => {
                json!({"k": k, "status": "solver_error", "error": error})
            }
        }
    }
}

/// Why no spectral cut was accepted (`SpectralClusterCutError.diagnostics`).
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralCutDiagnostics {
    pub interpretation_edge_weight_threshold: f64,
    pub affinity_component_count: usize,
    pub giant_component_size: usize,
    pub n: usize,
    pub max_k: usize,
    /// `None` when no `k` produced a score of at least 0.
    pub best_k: Option<usize>,
    pub best_silhouette_score: Option<f64>,
    pub candidate_scores: Vec<CandidateScore>,
}

impl SpectralCutDiagnostics {
    pub const ERROR_CODE: &'static str = "NO_STABLE_SPECTRAL_CUT";
    pub const REASON: &'static str = "No stable spectral cut found.";

    /// The diagnostics object of the error envelope; `candidate_scores` only
    /// at `detail="full"`.
    pub fn details(&self, full: bool) -> Value {
        let mut d = json!({
            "method": "spectral",
            "interpretation_edge_weight_threshold": round_to(self.interpretation_edge_weight_threshold, 6),
            "affinity_component_count": self.affinity_component_count,
            "giant_component_size": self.giant_component_size,
            "giant_component_fraction": round_to(self.giant_component_size as f64 / self.n.max(1) as f64, 6),
            "residual_node_count": self.n - self.giant_component_size,
            "k_min": SPECTRAL_K_MIN,
            "max_k": self.max_k,
            "best_k": self.best_k,
            "best_silhouette_score": self.best_silhouette_score.map(|s| round_to(s, 6)),
            "accepted_silhouette_min": ACCEPTED_SILHOUETTE_MIN,
        });
        if full {
            d["candidate_scores"] = Value::Array(
                self.candidate_scores
                    .iter()
                    .map(CandidateScore::to_json)
                    .collect(),
            );
        }
        d
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("No stable spectral cut found.")]
    NoStableSpectralCut(Box<SpectralCutDiagnostics>),
    #[error(transparent)]
    Analysis(#[from] AnalysisError),
}

impl ResolveError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoStableSpectralCut(_) => SpectralCutDiagnostics::ERROR_CODE,
            Self::Analysis(e) => e.code(),
        }
    }
}

/// Resolve clusters on the constructed graph. `stability` is the model's
/// cached H0 stability result; `None` recomputes it from the edges.
///
/// `interpretation_threshold` must be finite; negative values act as 0.
pub fn resolve_clusters(
    graph: &Graph,
    stability: Option<&StabilityResult>,
    method: ResolveMethod,
    max_k: usize,
    interpretation_threshold: f64,
) -> Result<ClusterResult, ResolveError> {
    if graph.n == 0 {
        return Err(AnalysisError::EmptyGraph.into());
    }
    if !interpretation_threshold.is_finite() {
        return Err(AnalysisError::InvalidArgument(format!(
            "interpretation_edge_weight_threshold must be finite, got {interpretation_threshold}"
        ))
        .into());
    }
    if let Some(&(i, j, _)) = graph
        .edges
        .iter()
        .find(|&&(i, j, _)| i >= graph.n || j >= graph.n)
    {
        return Err(AnalysisError::InvalidArgument(format!(
            "edge ({i}, {j}) is out of range for a graph of {} nodes",
            graph.n
        ))
        .into());
    }
    // The model's `weighted_edges(threshold=0.0)`: positive edges only.
    let edges: Vec<(usize, usize, f64)> = graph
        .edges
        .iter()
        .copied()
        .filter(|&(_, _, w)| w > 0.0)
        .collect();
    let tg = ThresholdGraph::from_edges(graph.n, &edges);
    let thresh = interpretation_threshold.max(0.0);

    if method == ResolveMethod::Components
        || (method == ResolveMethod::Auto && interpretation_threshold > 0.0)
    {
        return Ok(components_result(&tg, thresh));
    }
    if method == ResolveMethod::Auto {
        let computed;
        let stability = match stability {
            Some(s) => s,
            None => {
                computed = find_stable_thresholds_from_edges(graph.n, &edges, DEFAULT_NUM_BINS)
                    .map_err(AnalysisError::from)?;
                &computed
            }
        };
        if let Some(result) = threshold_stability_cut(&tg, stability) {
            return Ok(result);
        }
        if tg.component_count_at(0.0) > 1 {
            return Ok(components_result(&tg, thresh));
        }
    }
    spectral_result(graph.n, &edges, &tg, max_k, thresh)
}

fn components_result(tg: &ThresholdGraph, thresh: f64) -> ClusterResult {
    let state = tg.state_at(thresh);
    ClusterResult {
        n_clusters: state.component_count(),
        labels: state.labels,
        method_used: "components",
        silhouette_score: None,
        failure_reason: None,
        interpretation_edge_weight_threshold_applied: thresh,
        stability_plateaus: None,
    }
}

/// Component labels at `thresh`, renumbered largest component first (ties
/// keep the lower original label), and the component count.
fn labels_by_size(tg: &ThresholdGraph, thresh: f64) -> (Vec<usize>, usize) {
    let state = tg.state_at(thresh);
    let sizes = state.size_by_label();
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));
    let mut remap = vec![0; sizes.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new;
    }
    (
        state.labels.iter().map(|&l| remap[l]).collect(),
        sizes.len(),
    )
}

fn threshold_stability_cut(
    tg: &ThresholdGraph,
    stability: &StabilityResult,
) -> Option<ClusterResult> {
    let n = tg.n_nodes();
    let plateaus: Vec<PlateauSummary> = stability
        .plateaus
        .iter()
        .take(5)
        .map(|p| PlateauSummary {
            start: p.start_threshold,
            end: p.end_threshold,
            component_count: p.component_count,
            length: p.length(),
        })
        .collect();
    for plateau in &stability.plateaus {
        if plateau.component_count <= 1 || plateau.component_count >= MAX_COMPONENTS {
            continue;
        }
        let thresh = plateau.midpoint();
        let (labels, n_clusters) = labels_by_size(tg, thresh);
        let mut sizes = vec![0usize; n_clusters];
        for &l in &labels {
            sizes[l] += 1;
        }
        let singletons = sizes.iter().filter(|&&s| s == 1).count();
        if singletons as f64 / n as f64 > MAX_SINGLETON_RATIO {
            continue;
        }
        return Some(ClusterResult {
            labels,
            method_used: "threshold_stability",
            n_clusters,
            silhouette_score: None,
            failure_reason: None,
            interpretation_edge_weight_threshold_applied: thresh,
            stability_plateaus: Some(plateaus),
        });
    }
    None
}

struct BestCut {
    labels: Option<Vec<usize>>,
    k: usize,
    score: f64,
    candidates: Vec<CandidateScore>,
}

/// Sweep `k` over a connected affinity on local nodes `0..n`.
fn spectral_best_cut(sub: &Graph, max_k: usize) -> Result<BestCut, AnalysisError> {
    let n = sub.n;
    // Sorted adjacency rows for silhouette distance lookups.
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(i, j, w) in &sub.edges {
        rows[i].push((j, w));
        rows[j].push((i, w));
    }
    for row in &mut rows {
        row.sort_by_key(|&(j, _)| j);
    }
    let distance = |i: usize, j: usize| {
        if i == j {
            return 0.0;
        }
        match rows[i].binary_search_by_key(&j, |&(v, _)| v) {
            Ok(pos) => 1.0 - rows[i][pos].1.clamp(0.0, 1.0),
            Err(_) => 1.0,
        }
    };

    let mut best = BestCut {
        labels: None,
        k: SPECTRAL_K_MIN,
        score: -1.0,
        candidates: Vec::new(),
    };
    for k in SPECTRAL_K_MIN..(max_k + 1).min(n) {
        let labels = match cluster_spectral(sub, k, SPECTRAL_SEED) {
            Ok(labels) => labels,
            Err(e) => {
                best.candidates.push(CandidateScore::SolverError {
                    k,
                    error: e.to_string(),
                });
                continue;
            }
        };
        let mut seen = vec![false; k];
        for &l in &labels {
            seen[l] = true;
        }
        let n_labels = seen.iter().filter(|&&s| s).count();
        if n_labels <= 1 {
            best.candidates
                .push(CandidateScore::SingleLabel { k, n_labels });
            continue;
        }
        let score = silhouette_score(&labels, distance)?;
        best.candidates.push(CandidateScore::Scored {
            k,
            silhouette_score: round_to(score, 6),
            n_labels,
        });
        if score > best.score {
            best.score = score;
            best.k = k;
            best.labels = Some(labels);
        }
    }
    Ok(best)
}

fn spectral_result(
    n: usize,
    edges: &[(usize, usize, f64)],
    tg: &ThresholdGraph,
    max_k: usize,
    thresh: f64,
) -> Result<ClusterResult, ResolveError> {
    let (component_labels, n_components) = labels_by_size(tg, thresh);
    // Giant component (label 0 after renumbering), in ascending node order.
    let mut local = vec![usize::MAX; n];
    let mut giant = Vec::new();
    for (node, &l) in component_labels.iter().enumerate() {
        if l == 0 {
            local[node] = giant.len();
            giant.push(node);
        }
    }
    let sub = Graph {
        n: giant.len(),
        edges: edges
            .iter()
            .filter(|&&(i, j, w)| w > thresh && local[i] != usize::MAX && local[j] != usize::MAX)
            .map(|&(i, j, w)| (local[i], local[j], w))
            .collect(),
    };
    let cut = spectral_best_cut(&sub, max_k)?;
    let best_labels = match cut.labels {
        Some(labels) if cut.score > ACCEPTED_SILHOUETTE_MIN => labels,
        _ => {
            let scored = cut.score >= 0.0;
            return Err(ResolveError::NoStableSpectralCut(Box::new(
                SpectralCutDiagnostics {
                    interpretation_edge_weight_threshold: thresh,
                    affinity_component_count: n_components,
                    giant_component_size: giant.len(),
                    n,
                    max_k,
                    best_k: scored.then_some(cut.k),
                    best_silhouette_score: scored.then_some(cut.score),
                    candidate_scores: cut.candidates,
                },
            )));
        }
    };

    if n_components <= 1 {
        return Ok(ClusterResult {
            labels: best_labels,
            method_used: "spectral",
            n_clusters: cut.k,
            silhouette_score: Some(cut.score),
            failure_reason: None,
            interpretation_edge_weight_threshold_applied: thresh,
            stability_plateaus: None,
        });
    }

    // Every off-giant node becomes its own cluster, numbered after the cut,
    // component by component (largest first), ascending node order within.
    let mut labels = vec![0usize; n];
    for (local_id, &node) in giant.iter().enumerate() {
        labels[node] = best_labels[local_id];
    }
    let mut by_component: Vec<Vec<usize>> = vec![Vec::new(); n_components];
    for (node, &l) in component_labels.iter().enumerate() {
        if l != 0 {
            by_component[l].push(node);
        }
    }
    let mut next_id = cut.k;
    for node in by_component.into_iter().flatten() {
        labels[node] = next_id;
        next_id += 1;
    }
    let residual = n - giant.len();
    Ok(ClusterResult {
        labels,
        method_used: "spectral",
        n_clusters: next_id,
        silhouette_score: Some(cut.score),
        failure_reason: Some(format!(
            "Graph disconnected: clustered giant component ({} of {n} nodes) spectrally; \
             {residual} residual node(s) in {} smaller component(s) isolated as singleton clusters.",
            giant.len(),
            n_components - 1
        )),
        interpretation_edge_weight_threshold_applied: thresh,
        stability_plateaus: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(n: usize, edges: &[(usize, usize, f64)]) -> Graph {
        Graph {
            n,
            edges: edges.to_vec(),
        }
    }

    #[test]
    fn components_known_answer() {
        // 0-1 strong, 1-2 weak, 3 isolated: at 0.5 only 0-1 survives.
        let g = graph(4, &[(0, 1, 0.9), (1, 2, 0.3)]);
        let r = resolve_clusters(&g, None, ResolveMethod::Components, 15, 0.5).unwrap();
        assert_eq!(r.labels, vec![0, 0, 1, 2]);
        assert_eq!(r.n_clusters, 3);
        assert_eq!(r.method_used, "components");
        assert_eq!(r.interpretation_edge_weight_threshold_applied, 0.5);
    }

    #[test]
    fn auto_with_positive_threshold_uses_components() {
        let g = graph(3, &[(0, 1, 0.9), (1, 2, 0.3)]);
        let r = resolve_clusters(&g, None, ResolveMethod::Auto, 15, 0.5).unwrap();
        assert_eq!(r.method_used, "components");
        assert_eq!(r.labels, vec![0, 0, 1]);
    }

    #[test]
    fn negative_threshold_acts_as_zero() {
        let g = graph(3, &[(0, 1, 0.9)]);
        let r = resolve_clusters(&g, None, ResolveMethod::Components, 15, -0.2).unwrap();
        assert_eq!(r.interpretation_edge_weight_threshold_applied, 0.0);
        assert_eq!(r.labels, vec![0, 0, 1]);
    }

    #[test]
    fn spectral_too_small_reports_no_candidates() {
        let g = graph(2, &[(0, 1, 0.7)]);
        let err = resolve_clusters(&g, None, ResolveMethod::Spectral, 15, 0.0).unwrap_err();
        assert_eq!(err.code(), "NO_STABLE_SPECTRAL_CUT");
        assert_eq!(err.to_string(), "No stable spectral cut found.");
        let ResolveError::NoStableSpectralCut(d) = err else {
            unreachable!()
        };
        assert_eq!(d.best_k, None);
        assert!(d.candidate_scores.is_empty());
        let details = d.details(false);
        assert!(details.get("candidate_scores").is_none());
        assert_eq!(details["giant_component_fraction"], json!(1.0));
    }

    #[test]
    fn empty_graph_is_rejected() {
        let err = resolve_clusters(&graph(0, &[]), None, ResolveMethod::Auto, 15, 0.0).unwrap_err();
        assert_eq!(err.to_string(), "empty graph: the model has no nodes");
    }

    #[test]
    fn out_of_range_edge_is_rejected() {
        let err = resolve_clusters(
            &graph(2, &[(0, 5, 0.5)]),
            None,
            ResolveMethod::Auto,
            15,
            0.0,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid argument: edge (0, 5) is out of range for a graph of 2 nodes"
        );
    }

    #[test]
    fn non_finite_threshold_is_rejected() {
        let err =
            resolve_clusters(&graph(2, &[]), None, ResolveMethod::Auto, 15, f64::NAN).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid argument: interpretation_edge_weight_threshold must be finite, got NaN"
        );
    }

    #[test]
    fn unknown_method_is_rejected() {
        let err = ResolveMethod::parse("threshold").unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid argument: unknown cluster resolution method \"threshold\"; expected auto, components or spectral"
        );
    }
}
