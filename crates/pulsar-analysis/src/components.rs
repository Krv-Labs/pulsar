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
        let cmp = |a: &(usize, usize, f64), b: &(usize, usize, f64)| {
            b.2.total_cmp(&a.2)
                .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        };
        if edges
            .iter()
            .all(|&(i, j, w)| i < n && j < n && w.is_finite())
            && edges.is_sorted_by(|a, b| cmp(a, b).is_le())
        {
            return Self {
                n,
                edges: edges.to_vec(),
            };
        }
        let mut edges: Vec<(usize, usize, f64)> = edges
            .iter()
            .copied()
            .filter(|&(i, j, w)| i < n && j < n && w.is_finite())
            .collect();
        edges.sort_by(cmp);
        Self { n, edges }
    }

    pub fn from_graph(graph: &Graph) -> Self {
        Self::from_edges(graph.n, &graph.edges)
    }

    pub fn n_nodes(&self) -> usize {
        self.n
    }

    /// All stored edges, sorted by weight descending.
    pub fn edges(&self) -> &[(usize, usize, f64)] {
        &self.edges
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

    /// Component structure for each threshold in `thresholds`, computed in a
    /// single descending pass over the sorted edges.
    pub fn states_at(&self, thresholds: &[f64]) -> Vec<ComponentState> {
        if thresholds.is_empty() {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..thresholds.len()).collect();
        order.sort_by(|&a, &b| thresholds[b].total_cmp(&thresholds[a]));
        let mut uf = UnionFind::new(self.n);
        let mut edge_idx = 0usize;
        let mut last: Option<(usize, ComponentState)> = None;
        let mut out: Vec<Option<ComponentState>> = vec![None; thresholds.len()];
        for q_idx in order {
            let t = thresholds[q_idx];
            while edge_idx < self.edges.len() && self.edges[edge_idx].2 > t {
                let (i, j, _) = self.edges[edge_idx];
                uf.union(i, j);
                edge_idx += 1;
            }
            let state = match &last {
                Some((prev_idx, prev)) if *prev_idx == edge_idx => prev.clone(),
                _ => {
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
                    let s = ComponentState { sizes, labels };
                    last = Some((edge_idx, s.clone()));
                    s
                }
            };
            out[q_idx] = Some(state);
        }
        out.into_iter()
            .map(|s| s.expect("all slots filled"))
            .collect()
    }

    /// Descending component sizes for each threshold in `thresholds`, computed
    /// in a single descending pass over the sorted edges without building
    /// per-node label vectors.
    pub fn sizes_at(&self, thresholds: &[f64]) -> Vec<Vec<usize>> {
        if thresholds.is_empty() {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..thresholds.len()).collect();
        order.sort_by(|&a, &b| thresholds[b].total_cmp(&thresholds[a]));
        let mut parent: Vec<usize> = (0..self.n).collect();
        let mut rank = vec![0u8; self.n];
        let mut comp_size = vec![1usize; self.n];
        let find = |parent: &mut [usize], mut x: usize| -> usize {
            while parent[x] != x {
                let p = parent[x];
                parent[x] = parent[p];
                x = p;
            }
            x
        };
        let mut edge_idx = 0usize;
        let mut last: Option<(usize, Vec<usize>)> = None;
        let mut out: Vec<Option<Vec<usize>>> = vec![None; thresholds.len()];
        for q_idx in order {
            let t = thresholds[q_idx];
            while edge_idx < self.edges.len() && self.edges[edge_idx].2 > t {
                let (i, j, _) = self.edges[edge_idx];
                let rx = find(&mut parent, i);
                let ry = find(&mut parent, j);
                if rx != ry {
                    if rank[rx] < rank[ry] {
                        parent[rx] = ry;
                        comp_size[ry] += comp_size[rx];
                        comp_size[rx] = 0;
                    } else if rank[rx] > rank[ry] {
                        parent[ry] = rx;
                        comp_size[rx] += comp_size[ry];
                        comp_size[ry] = 0;
                    } else {
                        parent[ry] = rx;
                        rank[rx] += 1;
                        comp_size[rx] += comp_size[ry];
                        comp_size[ry] = 0;
                    }
                }
                edge_idx += 1;
            }
            let sizes = match &last {
                Some((prev_idx, prev)) if *prev_idx == edge_idx => prev.clone(),
                _ => {
                    let mut s: Vec<usize> = comp_size.iter().copied().filter(|&c| c > 0).collect();
                    s.sort_unstable_by(|a, b| b.cmp(a));
                    last = Some((edge_idx, s.clone()));
                    s
                }
            };
            out[q_idx] = Some(sizes);
        }
        out.into_iter()
            .map(|s| s.expect("all slots filled"))
            .collect()
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

    #[test]
    fn incremental_sweep_matches_per_threshold_state_at_on_random_graphs() {
        let mut rng_state = 0x1234_5678_9abc_def0_u64;
        let mut next_u64 = || {
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            rng_state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        for &(n, n_edges) in &[(25usize, 60usize), (80, 400), (150, 1200)] {
            let mut edges = Vec::with_capacity(n_edges);
            let mut exact_weights = Vec::new();
            for _ in 0..n_edges {
                let u = (next_u64() as usize) % n;
                let v = (next_u64() as usize) % n;
                // Quantize to 32 levels so many edges share exact boundary weights.
                let w = ((next_u64() % 32) + 1) as f64 / 32.0;
                edges.push((u, v, w));
                exact_weights.push(w);
            }
            let g = ThresholdGraph::from_edges(n, &edges);
            // Include ascending, descending, duplicate, and exact-edge-weight thresholds.
            let mut thresholds: Vec<f64> = (0..=64).map(|k| k as f64 / 64.0).collect();
            thresholds.extend(exact_weights.into_iter().take(20));
            let batch_states = g.states_at(&thresholds);
            let batch_sizes = g.sizes_at(&thresholds);
            for (idx, &t) in thresholds.iter().enumerate() {
                let expected = g.state_at(t);
                assert_eq!(batch_states[idx], expected, "state mismatch at t={t}");
                assert_eq!(batch_sizes[idx], expected.sizes, "sizes mismatch at t={t}");
            }
        }
    }
}
