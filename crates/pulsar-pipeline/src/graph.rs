//! The constructed cosmic graph as an edge list, with CSR and component
//! helpers. No networkx equivalent exists in the public API (ADR 0001).

use pulsar_core::UnionFind;
use serde::Serialize;

/// An undirected weighted graph on `n` nodes. Edges are `(i, j, w)` with
/// `i < j`, sorted, and `w` on the `[0, 1]` fraction-of-max scale.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<(usize, usize, f64)>,
}

/// Compressed sparse row view of a [`Graph`] (both directions stored).
#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    pub offsets: Vec<usize>,
    pub neighbors: Vec<usize>,
    pub weights: Vec<f64>,
}

impl Csr {
    pub fn neighbors(&self, u: usize) -> &[usize] {
        &self.neighbors[self.offsets[u]..self.offsets[u + 1]]
    }

    pub fn weights(&self, u: usize) -> &[f64] {
        &self.weights[self.offsets[u]..self.offsets[u + 1]]
    }
}

impl Graph {
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn csr(&self) -> Csr {
        let mut degree = vec![0usize; self.n];
        for &(i, j, _) in &self.edges {
            degree[i] += 1;
            degree[j] += 1;
        }
        let mut offsets = vec![0usize; self.n + 1];
        for u in 0..self.n {
            offsets[u + 1] = offsets[u] + degree[u];
        }
        let mut fill = offsets.clone();
        let mut neighbors = vec![0usize; offsets[self.n]];
        let mut weights = vec![0.0f64; offsets[self.n]];
        for &(i, j, w) in &self.edges {
            neighbors[fill[i]] = j;
            weights[fill[i]] = w;
            fill[i] += 1;
            neighbors[fill[j]] = i;
            weights[fill[j]] = w;
            fill[j] += 1;
        }
        Csr {
            offsets,
            neighbors,
            weights,
        }
    }

    pub fn degree(&self) -> Vec<usize> {
        let mut d = vec![0usize; self.n];
        for &(i, j, _) in &self.edges {
            d[i] += 1;
            d[j] += 1;
        }
        d
    }

    pub fn weighted_degree(&self) -> Vec<f64> {
        let mut d = vec![0.0f64; self.n];
        for &(i, j, w) in &self.edges {
            d[i] += w;
            d[j] += w;
        }
        d
    }

    /// Connected-component label per node, numbered in order of first node.
    pub fn component_labels(&self) -> Vec<usize> {
        let mut uf = UnionFind::new(self.n);
        for &(i, j, _) in &self.edges {
            uf.union(i, j);
        }
        let mut label_of_root = vec![usize::MAX; self.n];
        let mut next = 0;
        (0..self.n)
            .map(|u| {
                let r = uf.find(u);
                if label_of_root[r] == usize::MAX {
                    label_of_root[r] = next;
                    next += 1;
                }
                label_of_root[r]
            })
            .collect()
    }

    pub fn n_components(&self) -> usize {
        self.component_labels().iter().max().map_or(0, |m| m + 1)
    }

    /// Component sizes, largest first.
    pub fn component_sizes(&self) -> Vec<usize> {
        let labels = self.component_labels();
        let mut sizes = vec![0usize; self.n_components()];
        for l in labels {
            sizes[l] += 1;
        }
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_and_components_known_answer() {
        let g = Graph {
            n: 5,
            edges: vec![(0, 1, 0.5), (1, 2, 0.25), (3, 4, 1.0)],
        };
        let csr = g.csr();
        assert_eq!(csr.neighbors(1), [0, 2]);
        assert_eq!(csr.weights(1), [0.5, 0.25]);
        assert_eq!(csr.neighbors(4), [3]);
        assert_eq!(g.degree(), [1, 2, 1, 1, 1]);
        assert_eq!(g.weighted_degree(), [0.5, 0.75, 0.25, 1.0, 1.0]);
        assert_eq!(g.component_labels(), [0, 0, 0, 1, 1]);
        assert_eq!(g.n_components(), 2);
        assert_eq!(g.component_sizes(), [3, 2]);
        let empty = Graph {
            n: 3,
            edges: vec![],
        };
        assert_eq!(empty.n_components(), 3);
    }
}
