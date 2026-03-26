use std::collections::HashSet;
use ndarray::{Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Squared Euclidean distance between two equal-length slices.
///
/// We compare **squared** distances throughout Ball Mapper to avoid computing
/// a square root on every pair — distance comparisons only need the order
/// relationship, not the actual distance value.  The caller must compare
/// against `eps * eps` (not `eps`).
#[inline(always)]
fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Core Ball Mapper algorithm.  Returns `(nodes, edges)` where:
/// - `nodes[k]` is the list of point indices whose distance to ball `k`'s
///   centre is ≤ `eps`.
/// - `edges` contains pairs `(a, b)` (with `a < b`) for every pair of balls
///   that share at least one point.
///
/// # Algorithm (three steps)
///
/// **Step 1 — centre selection (greedy)**
/// Walk through points in order `0..n`.  A point becomes a new ball centre if
/// no existing centre is within distance `eps`.  This produces the minimum
/// number of balls needed to cover the data with balls of radius `eps`.
///
/// **Step 2 — membership**
/// For each centre, collect every point within distance `eps`.  A point can
/// belong to multiple balls (overlapping coverage is the whole point of Ball
/// Mapper — overlaps reveal topological connections between regions).
///
/// **Step 3 — edges**
/// Two balls `a` and `b` are connected by an edge if they share at least one
/// point.  Only pairs with `a < b` are recorded to avoid duplicates.
///
/// # Parameters
/// - `points` — array view of shape `(n_points, n_dims)`.
/// - `eps` — radius of each ball.
fn fit_inner(points: ArrayView2<f64>, eps: f64) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let n = points.nrows();
    let eps_sq = eps * eps;

    // Step 1: select ball centres
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

    // Step 2: build membership sets (node id = index into center_indices)
    let nodes: Vec<Vec<usize>> = center_indices
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
        .collect();

    // Step 3: build edges (pairs of balls sharing at least one point)
    let n_balls = nodes.len();
    let mut edges: Vec<(usize, usize)> = Vec::new();
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

/// A fitted Ball Mapper complex.
///
/// A Ball Mapper decomposes a point cloud into overlapping balls and
/// represents their connectivity as a graph.  It is the topological analogue
/// of a Mapper complex using distance balls as the cover.
///
/// # Fields
/// - `eps` — the ball radius used when fitting.
/// - `nodes` — `nodes[k]` contains the indices of all points in ball `k`.
/// - `edges` — pairs `(a, b)` of ball indices that share at least one point.
#[pyclass]
pub struct BallMapper {
    pub eps: f64,
    pub nodes: Vec<Vec<usize>>,
    pub edges: Vec<(usize, usize)>,
}

#[pymethods]
impl BallMapper {
    /// Create a new (unfitted) Ball Mapper with the given radius.
    ///
    /// # Parameters
    /// - `eps` — ball radius; larger values produce fewer, larger balls.
    #[new]
    pub fn new(eps: f64) -> Self {
        BallMapper { eps, nodes: Vec::new(), edges: Vec::new() }
    }

    /// Fit the Ball Mapper to a point cloud.
    ///
    /// Runs the three-step algorithm (centre selection → membership → edges)
    /// and stores the result in `self.nodes` and `self.edges`.
    ///
    /// # Parameters
    /// - `points` (`np.ndarray[float64, 2D]`, shape `(n_points, n_dims)`)
    pub fn fit(&mut self, points: PyReadonlyArray2<f64>) -> PyResult<()> {
        let arr = points.as_array();
        let (nodes, edges) = fit_inner(arr, self.eps);
        self.nodes = nodes;
        self.edges = edges;
        Ok(())
    }

    /// Ball membership lists.  `nodes[k]` is a list of point indices in ball `k`.
    #[getter]
    pub fn nodes(&self) -> Vec<Vec<usize>> {
        self.nodes.clone()
    }

    /// Edge list.  Each entry `(a, b)` (with `a < b`) means balls `a` and `b`
    /// share at least one point.
    #[getter]
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.edges.clone()
    }

    /// The ball radius this mapper was constructed with.
    #[getter]
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Number of balls (nodes) in the complex.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the complex.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Run Ball Mapper for every `(embedding, epsilon)` pair in parallel using rayon.
///
/// This is the hot path in the pipeline: the full parameter sweep can involve
/// dozens of (PCA embedding, epsilon) combinations.  Rayon distributes the
/// independent `fit_inner` calls across all available CPU cores.
///
/// # Parameters
/// - `embeddings` — list of 2-D arrays (one per PCA configuration).
/// - `epsilons` — list of ball radii to try.
///
/// # Returns
/// A flat list of `BallMapper` objects in **row-major order**: for each
/// embedding (outer loop), all epsilons (inner loop).  So if you have 3
/// embeddings and 2 epsilons you get 6 results:
/// `[(emb0, eps0), (emb0, eps1), (emb1, eps0), (emb1, eps1), ...]`
///
/// # Note on result ordering
/// Rayon's `flat_map` with `into_par_iter` preserves the logical order of the
/// outer iterator but parallelises the inner loop, so the index mapping above
/// holds even though execution is concurrent.
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
            eps_clone.into_par_iter().map(move |eps| {
                let (nodes, edges) = fit_inner(emb.view(), eps);
                BallMapper { eps, nodes, edges }
            })
        })
        .collect();

    Ok(results)
}
