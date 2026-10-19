//! Average-linkage agglomerative clustering of the cosmic graph
//! (`cosmic_clusters(method="agglomerative")`).
//!
//! scikit-learn ran `AgglomerativeClustering(metric="precomputed",
//! linkage="average")` on `1 - adjacency`, which SciPy solves with the
//! fastcluster nearest-neighbour-chain algorithm. `kodama` is a port of the
//! same algorithm with the same cluster labelling, so the partition matches
//! the golden fixture exactly (ARI 1.0).

use kodama::{linkage, Method};
use pulsar_core::UnionFind;
use pulsar_pipeline::Graph;

use crate::error::AnalysisError;

/// Condensed upper-triangle index of `(i, j)`, `i < j`, in the SciPy
/// `squareform` order.
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < n);
    n * i - i * (i + 1) / 2 + (j - i - 1)
}

/// Distance `1 - w` for connected pairs and `1` otherwise, in condensed
/// form. `O(n²)` memory: the method is defined on the full matrix.
fn condensed_distances(graph: &Graph) -> Vec<f64> {
    let n = graph.n;
    let mut dist = vec![1.0_f64; n * (n - 1) / 2];
    for &(i, j, w) in &graph.edges {
        if i == j {
            continue;
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        dist[condensed_index(n, a, b)] = 1.0 - w;
    }
    dist
}

/// Cluster labels (`0..n_clusters`, numbered by first node) from average
/// linkage on `1 - weight`.
pub fn cluster_agglomerative(
    graph: &Graph,
    n_clusters: usize,
) -> Result<Vec<usize>, AnalysisError> {
    let n = graph.n;
    if n == 0 {
        return Err(AnalysisError::EmptyGraph);
    }
    if n_clusters == 0 || n_clusters > n {
        return Err(AnalysisError::InvalidArgument(format!(
            "n_clusters must be in 1..={n} for a graph with {n} nodes, got {n_clusters}"
        )));
    }
    if n_clusters == n {
        return Ok((0..n).collect());
    }
    let mut dist = condensed_distances(graph);
    let dendrogram = linkage(&mut dist, n, Method::Average);

    // Cut: apply the first n - n_clusters merges. Cluster labels >= n name
    // the cluster created at step `label - n`; `rep` maps each to a leaf.
    let mut uf = UnionFind::new(n);
    let mut rep: Vec<usize> = (0..n).collect();
    for step in dendrogram.steps().iter().take(n - n_clusters) {
        let a = rep[step.cluster1];
        let b = rep[step.cluster2];
        uf.union(a, b);
        rep.push(uf.find(a));
    }
    let mut label_of_root = vec![usize::MAX; n];
    let mut next = 0;
    let labels = (0..n)
        .map(|node| {
            let root = uf.find(node);
            if label_of_root[root] == usize::MAX {
                label_of_root[root] = next;
                next += 1;
            }
            label_of_root[root]
        })
        .collect();
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condensed_index_matches_squareform() {
        let n = 5;
        let mut expect = 0;
        for i in 0..n {
            for j in i + 1..n {
                assert_eq!(condensed_index(n, i, j), expect);
                expect += 1;
            }
        }
        assert_eq!(expect, n * (n - 1) / 2);
    }

    #[test]
    fn known_answer_two_cliques_with_a_weak_bridge() {
        let mut edges = vec![
            (0, 1, 0.9),
            (0, 2, 0.9),
            (1, 2, 0.9),
            (3, 4, 0.8),
            (3, 5, 0.8),
            (4, 5, 0.8),
        ];
        edges.push((2, 3, 0.1));
        let g = Graph { n: 6, edges };
        assert_eq!(
            cluster_agglomerative(&g, 2).unwrap(),
            vec![0, 0, 0, 1, 1, 1]
        );
        assert_eq!(cluster_agglomerative(&g, 1).unwrap(), vec![0; 6]);
        assert_eq!(
            cluster_agglomerative(&g, 6).unwrap(),
            vec![0, 1, 2, 3, 4, 5]
        );
        let three = cluster_agglomerative(&g, 3).unwrap();
        assert_eq!(three[3], three[4]);
        assert_ne!(
            three[4], three[5],
            "the third merge is the first of clique B"
        );
        assert_eq!(three[0], three[2]);
        assert_eq!(
            cluster_agglomerative(&g, 7).unwrap_err().code(),
            "INVALID_ARGUMENT"
        );
        assert_eq!(
            cluster_agglomerative(
                &Graph {
                    n: 0,
                    edges: vec![]
                },
                1
            )
            .unwrap_err()
            .code(),
            "EMPTY_GRAPH"
        );
    }
}
