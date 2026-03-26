use std::collections::HashSet;
use ndarray::{Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline(always)]
fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn fit_inner(points: ArrayView2<f64>, eps: f64) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let n = points.nrows();
    let eps_sq = eps * eps;

    // Step 1: select ball centers
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

#[pyclass]
pub struct BallMapper {
    pub eps: f64,
    pub nodes: Vec<Vec<usize>>,
    pub edges: Vec<(usize, usize)>,
}

#[pymethods]
impl BallMapper {
    #[new]
    pub fn new(eps: f64) -> Self {
        BallMapper { eps, nodes: Vec::new(), edges: Vec::new() }
    }

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

/// Run BallMapper for every (embedding, epsilon) combination in parallel using rayon.
/// Results are in row-major order: for each embedding, all epsilons.
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
