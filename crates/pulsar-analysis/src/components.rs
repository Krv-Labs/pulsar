//! Connected components of a weighted graph at a threshold.
//!
//! [`ThresholdGraph`] is the Rust replacement for `prepare_threshold_graph`:
//! the full weighted affinity, sorted by weight so a query at threshold `t`
//! touches only the edges above `t`. Components come from union-find in one
//! pass; labels are numbered by the lowest node index in each component,
//! as `scipy.sparse.csgraph.connected_components` numbers them.

use pulsar_core::UnionFind;
use pulsar_pipeline::Graph;

/// Component structure at one threshold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentState {
    /// Component sizes, descending.
    pub sizes: Vec<usize>,
    /// Component label per node, numbered by first (lowest) node index.
    pub labels: Vec<usize>,
}

impl ComponentState {
    /// Size of the component each label names (indexed by label).
    pub fn size_by_label(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.labels.iter().copied().max().map_or(0, |m| m + 1)];
        for &l in &self.labels {
            counts[l] += 1;
        }
        counts
    }

    pub fn component_count(&self) -> usize {
        self.sizes.len()
    }
}

/// A weighted undirected graph prepared for repeated threshold queries.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdGraph {
    n: usize,
    /// `(i, j, w)` sorted by `w` descending.
    edges: Vec<(usize, usize, f64)>,
}

impl ThresholdGraph {
    /// From an edge list on `n` nodes. Edges with non-finite weight are
    /// dropped; the rest are kept whatever their sign.
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut edges: Vec<(usize, usize, f64)> = edges
            .iter()
            .copied()
            .filter(|&(i, j, w)| i < n && j < n && w.is_finite())
            .collect();
        edges.sort_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        });
        Self { n, edges }
    }

    pub fn from_graph(graph: &Graph) -> Self {
        Self::from_edges(graph.n, &graph.edges)
    }

    pub fn n_nodes(&self) -> usize {
        self.n
    }

    /// Edges above `threshold` (strict), heaviest first.
    pub fn edges_above(&self, threshold: f64) -> &[(usize, usize, f64)] {
        let end = self.edges.partition_point(|&(_, _, w)| w > threshold);
        &self.edges[..end]
    }

    /// Number of stored (undirected) edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Stored nonzeros as a symmetric matrix would count them (`2 × edges`).
    pub fn nnz(&self) -> usize {
        2 * self.edges.len()
    }

    /// `nnz / n²`, rounded to 6 decimals as the Python helper reported it.
    pub fn density(&self) -> f64 {
        let possible = (self.n * self.n).max(1);
        crate::pyround::round_to(self.nnz() as f64 / possible as f64, 6)
    }

    /// Components when only edges with `w > threshold` are kept.
    pub fn state_at(&self, threshold: f64) -> ComponentState {
        let mut uf = UnionFind::new(self.n);
        for &(i, j, _) in self.edges_above(threshold) {
            uf.union(i, j);
        }
        let mut label_of_root = vec![usize::MAX; self.n];
        let mut labels = Vec::with_capacity(self.n);
        let mut sizes = Vec::new();
        for node in 0..self.n {
            let root = uf.find(node);
            if label_of_root[root] == usize::MAX {
                label_of_root[root] = sizes.len();
                sizes.push(0);
            }
            let label = label_of_root[root];
            sizes[label] += 1;
            labels.push(label);
        }
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        ComponentState { sizes, labels }
    }

    /// Component count only (no labels), for curve sampling.
    pub fn component_count_at(&self, threshold: f64) -> usize {
        let mut uf = UnionFind::new(self.n);
        for &(i, j, _) in self.edges_above(threshold) {
            uf.union(i, j);
        }
        uf.components()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_follow_the_strict_threshold_and_scipy_label_order() {
        // 0-1 (0.9), 1-2 (0.5), 3-4 (0.5), 5 isolated.
        let g = ThresholdGraph::from_edges(6, &[(3, 4, 0.5), (0, 1, 0.9), (1, 2, 0.5)]);
        assert_eq!(g.n_edges(), 3);
        assert_eq!(g.edges_above(0.5).len(), 1, "strictly above");
        let s = g.state_at(0.0);
        assert_eq!(s.sizes, vec![3, 2, 1]);
        assert_eq!(s.labels, vec![0, 0, 0, 1, 1, 2]);
        assert_eq!(s.size_by_label(), vec![3, 2, 1]);
        let s = g.state_at(0.5);
        assert_eq!(s.sizes, vec![2, 1, 1, 1, 1]);
        assert_eq!(s.labels, vec![0, 0, 1, 2, 3, 4]);
        assert_eq!(g.component_count_at(0.95), 6);
        assert_eq!(g.density(), 0.166667);
    }
}
