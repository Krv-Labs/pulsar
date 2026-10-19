//! Spectral clustering of the cosmic graph (`cosmic_clusters(method=
//! "spectral")`): the scikit-learn recipe with `affinity="precomputed"`.
//!
//! 1. Symmetric normalized Laplacian `L = I − D^{-1/2} A D^{-1/2}` with the
//!    diagonal set to 1 everywhere (isolated nodes included), as
//!    `spectral_embedding` builds it.
//! 2. The `k` eigenvectors of the smallest eigenvalues, each entry divided
//!    by `sqrt(degree)` (1 for isolated nodes).
//! 3. k-means++ (`n_init = 10`) on the `n × k` embedding.
//!
//! Small graphs use a dense symmetric eigensolver; larger ones LOBPCG on the
//! sparse operator. scikit-learn's ARPACK path and its unseeded k-means make
//! the fixture a partition comparison (ARI ≥ 0.9), not an exact one.

use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::Array2;
use pulsar_core::kmeans;
use pulsar_pipeline::Graph;

use super::lobpcg::lobpcg_smallest;
use crate::error::AnalysisError;

/// Below this size the Laplacian is diagonalized densely (`O(n³)`, exact).
pub const DENSE_LIMIT: usize = 600;

struct NormalizedLaplacian<'a> {
    graph: &'a Graph,
    /// `sqrt(degree)`, or 1 for isolated nodes.
    dd: Vec<f64>,
}

impl<'a> NormalizedLaplacian<'a> {
    fn new(graph: &'a Graph) -> Self {
        let mut degree = vec![0.0_f64; graph.n];
        for &(i, j, w) in &graph.edges {
            degree[i] += w;
            degree[j] += w;
        }
        let dd = degree
            .iter()
            .map(|&d| if d == 0.0 { 1.0 } else { d.sqrt() })
            .collect();
        Self { graph, dd }
    }

    /// `Y = X − N X` with `N = D^{-1/2} A D^{-1/2}`.
    fn apply(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut y = x.clone();
        for &(i, j, w) in &self.graph.edges {
            let nw = w / (self.dd[i] * self.dd[j]);
            for c in 0..x.ncols() {
                y[(i, c)] -= nw * x[(j, c)];
                y[(j, c)] -= nw * x[(i, c)];
            }
        }
        y
    }

    fn dense(&self) -> DMatrix<f64> {
        let n = self.graph.n;
        let mut m = DMatrix::<f64>::identity(n, n);
        for &(i, j, w) in &self.graph.edges {
            let nw = w / (self.dd[i] * self.dd[j]);
            m[(i, j)] -= nw;
            m[(j, i)] -= nw;
        }
        m
    }
}

/// The `n × k` spectral embedding (rows are nodes). `force_iterative`
/// bypasses the dense path regardless of size.
pub fn spectral_embedding(
    graph: &Graph,
    k: usize,
    seed: u64,
    force_iterative: bool,
) -> Result<Array2<f64>, AnalysisError> {
    let n = graph.n;
    if n == 0 {
        return Err(AnalysisError::EmptyGraph);
    }
    if k == 0 || k > n {
        return Err(AnalysisError::InvalidArgument(format!(
            "n_components must be in 1..={n} for a graph with {n} nodes, got {k}"
        )));
    }
    if graph
        .edges
        .iter()
        .any(|&(_, _, w)| !w.is_finite() || w < 0.0)
    {
        return Err(AnalysisError::InvalidArgument(
            "spectral clustering needs finite, non-negative edge weights".into(),
        ));
    }
    let lap = NormalizedLaplacian::new(graph);
    let use_dense = !force_iterative && (n <= DENSE_LIMIT || 3 * k > n);
    let vectors: DMatrix<f64> = if use_dense {
        let eig = SymmetricEigen::new(lap.dense());
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
        let mut v = DMatrix::<f64>::zeros(n, k);
        for (c, &i) in order.iter().take(k).enumerate() {
            v.set_column(c, &eig.eigenvectors.column(i));
        }
        v
    } else {
        lobpcg_smallest(&|x| lap.apply(x), n, k, seed, 1e-6, 500)?.vectors
    };
    let mut embedding = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for c in 0..k {
            embedding[[i, c]] = vectors[(i, c)] / lap.dd[i];
        }
    }
    Ok(embedding)
}

/// Cluster labels from k-means on the spectral embedding.
pub fn cluster_spectral(
    graph: &Graph,
    n_clusters: usize,
    seed: u64,
) -> Result<Vec<usize>, AnalysisError> {
    cluster_spectral_with(graph, n_clusters, seed, false)
}

pub fn cluster_spectral_with(
    graph: &Graph,
    n_clusters: usize,
    seed: u64,
    force_iterative: bool,
) -> Result<Vec<usize>, AnalysisError> {
    let embedding = spectral_embedding(graph, n_clusters, seed, force_iterative)?;
    Ok(kmeans(&embedding, n_clusters, seed, 10)?.labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cliques(sizes: &[usize], bridge: f64) -> Graph {
        let mut edges = Vec::new();
        let mut base = 0;
        let mut ends = Vec::new();
        for &s in sizes {
            for a in 0..s {
                for b in a + 1..s {
                    edges.push((base + a, base + b, 1.0));
                }
            }
            ends.push((base, base + s - 1));
            base += s;
        }
        for w in ends.windows(2) {
            edges.push((w[0].1, w[1].0, bridge));
        }
        Graph { n: base, edges }
    }

    fn same_partition(a: &[usize], b: &[usize]) -> bool {
        a.len() == b.len()
            && (0..a.len()).all(|i| (0..a.len()).all(|j| (a[i] == a[j]) == (b[i] == b[j])))
    }

    #[test]
    fn dense_and_iterative_paths_agree_on_weakly_bridged_cliques() {
        let g = cliques(&[30, 30, 30], 0.05);
        let dense = cluster_spectral_with(&g, 3, 42, false).unwrap();
        let iterative = cluster_spectral_with(&g, 3, 42, true).unwrap();
        let truth: Vec<usize> = (0..90).map(|i| i / 30).collect();
        assert!(same_partition(&dense, &truth), "{dense:?}");
        assert!(same_partition(&iterative, &truth), "{iterative:?}");
    }

    #[test]
    fn isolated_nodes_and_bad_input_are_handled() {
        let mut g = cliques(&[10, 10], 0.0);
        g.n += 2; // two isolated nodes
        let labels = cluster_spectral(&g, 2, 1).unwrap();
        assert_eq!(labels.len(), 22);
        assert_eq!(
            cluster_spectral(&g, 0, 1).unwrap_err().code(),
            "INVALID_ARGUMENT"
        );
        assert_eq!(
            cluster_spectral(&g, 23, 1).unwrap_err().code(),
            "INVALID_ARGUMENT"
        );
        let bad = Graph {
            n: 3,
            edges: vec![(0, 1, -1.0)],
        };
        assert_eq!(
            cluster_spectral(&bad, 2, 1).unwrap_err().code(),
            "INVALID_ARGUMENT"
        );
        assert_eq!(
            cluster_spectral(
                &Graph {
                    n: 0,
                    edges: vec![]
                },
                1,
                1
            )
            .unwrap_err()
            .code(),
            "EMPTY_GRAPH"
        );
    }
}
