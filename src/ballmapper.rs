use kiddo::{KdTree, SquaredEuclidean};
use ndarray::{Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

/// Squared Euclidean distance between two equal-length slices.
/// Using squared distance avoids sqrt, which is sufficient for comparisons.
#[inline(always)]
fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Core Ball Mapper algorithm optimized for large datasets.
///
/// Returns `(nodes, edges)` where:
/// - `nodes[k]` is the list of point indices in ball `k`
/// - `edges` contains pairs `(a, b)` with `a < b` for balls sharing points
///
/// Complexity: O(n * k) for centre selection + O(n * k) for membership
/// where k = number of balls (typically k << n for reasonable epsilon).
fn fit_inner(points: ArrayView2<f64>, eps: f64) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let n = points.nrows();
    let d = points.ncols();
    let eps_sq = eps * eps;
    assert!(d > 0, "BallMapper input must have at least one column");

    // Step 1: greedy centre selection - O(n * k) where k = num centres
    let mut center_indices: Vec<usize> = Vec::new();
    'outer: for i in 0..n {
        let pi = points.row(i).to_slice().unwrap();
        for &c in &center_indices {
            let pc = points.row(c).to_slice().unwrap();
            if l2_sq(pi, pc) <= eps_sq {
                continue 'outer;
            }
        }
        center_indices.push(i);
    }

    // Step 2: build membership. Low-dimensional projected data uses a KD-tree
    // radius query; wider embeddings keep the allocation-free linear path.
    let nodes = match d {
        1 => fit_inner_kd::<1>(points, &center_indices, eps_sq),
        2 => fit_inner_kd::<2>(points, &center_indices, eps_sq),
        3 => fit_inner_kd::<3>(points, &center_indices, eps_sq),
        4 => fit_inner_kd::<4>(points, &center_indices, eps_sq),
        5 => fit_inner_kd::<5>(points, &center_indices, eps_sq),
        6 => fit_inner_kd::<6>(points, &center_indices, eps_sq),
        7 => fit_inner_kd::<7>(points, &center_indices, eps_sq),
        8 => fit_inner_kd::<8>(points, &center_indices, eps_sq),
        9 => fit_inner_kd::<9>(points, &center_indices, eps_sq),
        10 => fit_inner_kd::<10>(points, &center_indices, eps_sq),
        11 => fit_inner_kd::<11>(points, &center_indices, eps_sq),
        12 => fit_inner_kd::<12>(points, &center_indices, eps_sq),
        13 => fit_inner_kd::<13>(points, &center_indices, eps_sq),
        14 => fit_inner_kd::<14>(points, &center_indices, eps_sq),
        15 => fit_inner_kd::<15>(points, &center_indices, eps_sq),
        16 => fit_inner_kd::<16>(points, &center_indices, eps_sq),
        _ => fit_inner_linear(points, &center_indices, eps_sq),
    };

    // Step 3: build edges - O(k²) but k is small
    let n_balls = nodes.len();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for a in 0..n_balls {
        let set_a: HashSet<usize> = nodes[a].iter().copied().collect();
        for b in (a + 1)..n_balls {
            if nodes[b].iter().any(|x| set_a.contains(x)) {
                edges.push((a, b));
            }
        }
    }

    (nodes, edges)
}

fn fit_inner_linear(
    points: ArrayView2<f64>,
    center_indices: &[usize],
    eps_sq: f64,
) -> Vec<Vec<usize>> {
    let n = points.nrows();
    center_indices
        .iter()
        .map(|&c| {
            let pc = points.row(c).to_slice().unwrap();
            (0..n)
                .filter(|&i| {
                    let pi = points.row(i).to_slice().unwrap();
                    l2_sq(pi, pc) <= eps_sq
                })
                .collect()
        })
        .collect()
}

fn fit_inner_kd<const K: usize>(
    points: ArrayView2<f64>,
    center_indices: &[usize],
    eps_sq: f64,
) -> Vec<Vec<usize>> {
    let mut tree: KdTree<f64, K> = KdTree::new();
    for i in 0..points.nrows() {
        tree.add(&point_array::<K>(points, i), i as u64);
    }

    center_indices
        .iter()
        .map(|&c| {
            let center = point_array::<K>(points, c);
            let mut members: Vec<usize> = tree
                .within_unsorted::<SquaredEuclidean>(&center, eps_sq)
                .into_iter()
                .map(|entry| entry.item as usize)
                .collect();
            members.sort_unstable();
            members
        })
        .collect()
}

fn point_array<const K: usize>(points: ArrayView2<f64>, row: usize) -> [f64; K] {
    let mut out = [0.0; K];
    for j in 0..K {
        out[j] = points[[row, j]];
    }
    out
}

/// A fitted Ball Mapper complex.
///
/// Ball Mapper decomposes a point cloud into overlapping balls and
/// represents connectivity as a graph. Designed for large-scale EHR data.
#[pyclass]
pub struct BallMapper {
    pub eps: f64,
    pub nodes: Vec<Vec<usize>>,
    pub edges: Vec<(usize, usize)>,
}

#[pymethods]
impl BallMapper {
    /// Create a new Ball Mapper with given radius.
    #[new]
    pub fn new(eps: f64) -> Self {
        BallMapper {
            eps,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Fit the Ball Mapper to a point cloud.
    pub fn fit(&mut self, points: PyReadonlyArray2<f64>) -> PyResult<()> {
        let arr = points.as_array();
        let (nodes, edges) = fit_inner(arr, self.eps);
        self.nodes = nodes;
        self.edges = edges;
        Ok(())
    }

    #[getter]
    pub fn nodes(&self) -> Vec<Vec<usize>> {
        self.nodes.clone()
    }

    #[getter]
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.edges.clone()
    }

    #[getter]
    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Run Ball Mapper for every (embedding, epsilon) pair in parallel.
///
/// This is the main entry point for grid search. Parallelised across all
/// combinations using rayon for maximum throughput on large datasets.
///
/// Complexity per fit: O(n * k) where k = number of balls.
/// No O(n²) memory allocation - scales to large EHR datasets.
#[pyfunction]
pub fn ball_mapper_grid(
    embeddings: Vec<PyReadonlyArray2<f64>>,
    epsilons: Vec<f64>,
) -> PyResult<Vec<BallMapper>> {
    let owned: Vec<Array2<f64>> = embeddings.iter().map(|e| e.as_array().to_owned()).collect();

    let results: Vec<BallMapper> = owned
        .par_iter()
        .flat_map(|emb| {
            let eps_clone = epsilons.clone();
            eps_clone
                .into_par_iter()
                .map(move |eps| {
                    let (nodes, edges) = fit_inner(emb.view(), eps);
                    BallMapper { eps, nodes, edges }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(results)
}
