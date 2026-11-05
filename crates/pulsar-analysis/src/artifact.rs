//! The spectral-sparsifier graph artifact (`_build_sparsifier_artifact_worker`
//! in `pulsar/mcp/tools/diagnostics.py`): sparsify the base cosmic graph,
//! put its weights on `[0, 1]` and measure it like a constructed graph.

use pulsar_core::CosmicGraphInner;
use pulsar_pipeline::Graph;

use crate::diagnostics::{GraphAnalysis, GraphMetrics};
use crate::error::AnalysisError;

/// The sparsified graph (weights divided by `max(1, max weight)`, zero
/// weights dropped) and its metrics at threshold 0 with no ball maps.
#[allow(clippy::too_many_arguments)]
pub fn spectral_sparsifier_graph(
    base: &CosmicGraphInner,
    epsilon: f64,
    seed: u64,
    sketch_dim: Option<usize>,
    sample_count: Option<usize>,
    pcg_tol: f64,
    max_iter: usize,
) -> Result<(Graph, GraphMetrics), AnalysisError> {
    let sparse = base
        .spectral_sparsify(epsilon, seed, sketch_dim, sample_count, pcg_tol, max_iter)
        .map_err(|e| AnalysisError::InvalidArgument(e.to_string()))?;
    let raw = sparse.weighted_edges();
    let scale = raw.iter().map(|e| e.2).fold(0.0, f64::max).max(1.0);
    let edges = raw
        .iter()
        .map(|&(i, j, w)| (i, j, w / scale))
        .filter(|&(_, _, w)| w > 0.0)
        .collect();
    let graph = Graph {
        n: sparse.n(),
        edges,
    };
    let metrics = GraphAnalysis::new(&graph)?.metrics(0.0, 0, Some(&graph.edges), None);
    Ok((graph, metrics))
}
