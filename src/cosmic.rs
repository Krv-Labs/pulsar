use std::collections::{HashMap, VecDeque};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use sprs::{CsMat, TriMat};

use crate::error::PulsarError;

type WeightedEdge = (usize, usize, f64);

#[derive(Copy, Clone)]
struct PcgOptions {
    tol: f64,
    max_iter: usize,
}

/// Internal Cosmic Graph data.
///
/// Dense storage preserves the historical construction path. Sparse storage is
/// used by spectral sparsification and materializes dense arrays only when a
/// compatibility getter asks for them.
pub enum CosmicGraphInner {
    Dense {
        weighted_adj: Array2<f64>,
        adj: Array2<u8>,
        n: usize,
    },
    Sparse {
        n: usize,
        edges: Vec<WeightedEdge>,
    },
}

impl CosmicGraphInner {
    pub fn from_pseudo_laplacian(l: &Array2<i64>, threshold: f64) -> CosmicGraphInner {
        let n = l.shape()[0];
        let mut wadj = Array2::<f64>::zeros((n, n));
        let mut adj = Array2::<u8>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let denom = l[[i, i]] + l[[j, j]] + l[[i, j]];
                if denom > 0 {
                    wadj[[i, j]] = -(l[[i, j]] as f64) / (denom as f64);
                }
                if wadj[[i, j]] > threshold {
                    adj[[i, j]] = 1;
                }
            }
        }

        CosmicGraphInner::Dense {
            weighted_adj: wadj,
            adj,
            n,
        }
    }

    fn n(&self) -> usize {
        match self {
            CosmicGraphInner::Dense { n, .. } | CosmicGraphInner::Sparse { n, .. } => *n,
        }
    }

    fn weighted_edges(&self) -> Vec<WeightedEdge> {
        match self {
            CosmicGraphInner::Sparse { edges, .. } => edges.clone(),
            CosmicGraphInner::Dense {
                weighted_adj, n, ..
            } => {
                let mut edges = Vec::new();
                for i in 0..*n {
                    for j in (i + 1)..*n {
                        let w = weighted_adj[[i, j]];
                        if w > 0.0 {
                            edges.push((i, j, w));
                        }
                    }
                }
                edges
            }
        }
    }

    fn weighted_adj(&self) -> Array2<f64> {
        match self {
            CosmicGraphInner::Dense { weighted_adj, .. } => weighted_adj.clone(),
            CosmicGraphInner::Sparse { n, edges } => {
                let mut out = Array2::<f64>::zeros((*n, *n));
                for &(i, j, w) in edges {
                    out[[i, j]] = w;
                    out[[j, i]] = w;
                }
                out
            }
        }
    }

    fn adj(&self) -> Array2<u8> {
        match self {
            CosmicGraphInner::Dense { adj, .. } => adj.clone(),
            CosmicGraphInner::Sparse { n, edges } => {
                let mut out = Array2::<u8>::zeros((*n, *n));
                for &(i, j, w) in edges {
                    if w > 0.0 {
                        out[[i, j]] = 1;
                        out[[j, i]] = 1;
                    }
                }
                out
            }
        }
    }
}

#[pyclass]
pub struct CosmicGraph {
    inner: CosmicGraphInner,
}

#[pymethods]
impl CosmicGraph {
    #[staticmethod]
    pub fn from_pseudo_laplacian<'py>(
        _py: Python<'py>,
        l: PyReadonlyArray2<'py, i64>,
        threshold: f64,
    ) -> PyResult<Self> {
        let arr = l.as_array().to_owned();
        let inner = CosmicGraphInner::from_pseudo_laplacian(&arr, threshold);
        Ok(CosmicGraph { inner })
    }

    /// Spielman-Srivastava style spectral sparsifier using JL resistance sketches.
    #[pyo3(signature = (epsilon, seed=42, sketch_dim=None, sample_count=None, pcg_tol=1e-6, max_iter=1000))]
    pub fn spectral_sparsify(
        &self,
        epsilon: f64,
        seed: u64,
        sketch_dim: Option<usize>,
        sample_count: Option<usize>,
        pcg_tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PulsarError::InvalidParameter {
                msg: "epsilon must be finite and positive".to_string(),
            }
            .into());
        }
        if !pcg_tol.is_finite() || pcg_tol <= 0.0 {
            return Err(PulsarError::InvalidParameter {
                msg: "pcg_tol must be finite and positive".to_string(),
            }
            .into());
        }

        let n = self.inner.n();
        let edges: Vec<WeightedEdge> = self
            .inner
            .weighted_edges()
            .into_iter()
            .filter(|&(i, j, w)| i != j && w.is_finite() && w > 0.0)
            .collect();
        if n <= 1 || edges.is_empty() {
            return Ok(CosmicGraph {
                inner: CosmicGraphInner::Sparse {
                    n,
                    edges: Vec::new(),
                },
            });
        }

        let components = connected_components(n, &edges);
        let dim = sketch_dim
            .unwrap_or_else(|| {
                ((24.0 * (n as f64).ln().max(1.0)) / (epsilon * epsilon)).ceil() as usize
            })
            .max(1);
        let samples = sample_count
            .unwrap_or_else(|| {
                ((9.0 * n as f64 * (n as f64).ln().max(1.0)) / (epsilon * epsilon)).ceil() as usize
            })
            .max(1);

        let _laplacian = sparse_laplacian(n, &edges);
        let mut resistances = vec![0.0; edges.len()];
        let diag = laplacian_diag(n, &edges);

        for row in 0..dim {
            let row_seed = seed ^ (row as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = StdRng::seed_from_u64(row_seed);
            let mut rhs = vec![0.0; n];
            for &(u, v, w) in &edges {
                let g: f64 = rng.sample(StandardNormal);
                let noise = g * w.sqrt() / (dim as f64).sqrt();
                rhs[u] += noise;
                rhs[v] -= noise;
            }

            let potentials = solve_by_component(
                n,
                &edges,
                &components,
                &diag,
                &rhs,
                PcgOptions {
                    tol: pcg_tol,
                    max_iter,
                },
            );
            for (idx, &(u, v, _)) in edges.iter().enumerate() {
                let diff = potentials[u] - potentials[v];
                resistances[idx] += diff * diff;
            }
        }

        let mut leverage: Vec<f64> = edges
            .iter()
            .zip(resistances.iter())
            .map(|(&(_, _, w), &r)| (w * r).max(0.0))
            .collect();
        let leverage_sum: f64 = leverage.iter().sum();
        if leverage_sum <= f64::EPSILON {
            let weight_sum: f64 = edges.iter().map(|&(_, _, w)| w).sum();
            leverage = edges.iter().map(|&(_, _, w)| w / weight_sum).collect();
        } else {
            for tau in &mut leverage {
                *tau /= leverage_sum;
            }
        }

        let mut cdf = Vec::with_capacity(leverage.len());
        let mut running = 0.0;
        for p in leverage {
            running += p;
            cdf.push(running);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        let mut aggregated: HashMap<(usize, usize), f64> = HashMap::new();
        for sample_idx in 0..samples {
            let sample_seed = seed ^ (sample_idx as u64 + 11).wrapping_mul(0xD1B5_4A32_D192_ED03);
            let mut rng = StdRng::seed_from_u64(sample_seed);
            let draw: f64 = rng.gen();
            let edge_idx = cdf.partition_point(|&p| p < draw).min(edges.len() - 1);
            let (u, v, w) = edges[edge_idx];
            let p = (cdf[edge_idx]
                - if edge_idx == 0 {
                    0.0
                } else {
                    cdf[edge_idx - 1]
                })
            .max(f64::EPSILON);
            let sampled_weight = w / (samples as f64 * p);
            *aggregated.entry((u.min(v), u.max(v))).or_insert(0.0) += sampled_weight;
        }

        let mut sparse_edges: Vec<WeightedEdge> = aggregated
            .into_iter()
            .map(|((u, v), w)| (u, v, w))
            .filter(|&(_, _, w)| w.is_finite() && w > 0.0)
            .collect();
        sparse_edges.sort_by_key(|&(u, v, _)| (u, v));

        Ok(CosmicGraph {
            inner: CosmicGraphInner::Sparse {
                n,
                edges: sparse_edges,
            },
        })
    }

    #[getter]
    pub fn weighted_adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.weighted_adj().into_pyarray_bound(py))
    }

    #[getter]
    pub fn adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        Ok(self.inner.adj().into_pyarray_bound(py))
    }

    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n()
    }

    pub fn weighted_edges(&self) -> Vec<WeightedEdge> {
        self.inner.weighted_edges()
    }

    #[getter]
    pub fn n_edges(&self) -> usize {
        self.inner.weighted_edges().len()
    }
}

fn sparse_laplacian(n: usize, edges: &[WeightedEdge]) -> CsMat<f64> {
    let mut tri = TriMat::<f64>::with_capacity((n, n), edges.len() * 4);
    for &(u, v, w) in edges {
        tri.add_triplet(u, u, w);
        tri.add_triplet(v, v, w);
        tri.add_triplet(u, v, -w);
        tri.add_triplet(v, u, -w);
    }
    tri.to_csr()
}

fn laplacian_diag(n: usize, edges: &[WeightedEdge]) -> Vec<f64> {
    let mut diag = vec![0.0; n];
    for &(u, v, w) in edges {
        diag[u] += w;
        diag[v] += w;
    }
    diag
}

fn connected_components(n: usize, edges: &[WeightedEdge]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v, _) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }

    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some(u) = queue.pop_front() {
            component.push(u);
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    queue.push_back(v);
                }
            }
        }
        components.push(component);
    }
    components
}

fn solve_by_component(
    n: usize,
    edges: &[WeightedEdge],
    components: &[Vec<usize>],
    diag: &[f64],
    rhs: &[f64],
    options: PcgOptions,
) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for component in components {
        if component.len() <= 1 {
            continue;
        }
        let anchor = component[0];
        let x = pcg_component(n, edges, component, anchor, diag, rhs, options);
        for &idx in component {
            out[idx] = x[idx];
        }
    }
    out
}

fn pcg_component(
    n: usize,
    edges: &[WeightedEdge],
    component: &[usize],
    anchor: usize,
    diag: &[f64],
    rhs: &[f64],
    options: PcgOptions,
) -> Vec<f64> {
    let PcgOptions { tol, max_iter } = options;
    let in_component = {
        let mut flags = vec![false; n];
        for &idx in component {
            flags[idx] = true;
        }
        flags
    };

    let mut x = vec![0.0; n];
    let mut r = rhs.to_vec();
    zero_outside_component(&mut r, &in_component);
    r[anchor] = 0.0;

    let mut z = apply_jacobi(&r, diag, &in_component, anchor);
    let mut p = z.clone();
    let mut rz_old = dot_component(&r, &z, component);
    let tol_sq = tol * tol * dot_component(rhs, rhs, component).max(1.0);
    if rz_old <= tol_sq {
        return x;
    }

    for _ in 0..max_iter {
        let ap = laplacian_matvec(n, edges, &p, &in_component, anchor);
        let denom = dot_component(&p, &ap, component);
        if denom.abs() <= f64::EPSILON {
            break;
        }
        let alpha = rz_old / denom;
        for &idx in component {
            if idx == anchor {
                continue;
            }
            x[idx] += alpha * p[idx];
            r[idx] -= alpha * ap[idx];
        }
        if dot_component(&r, &r, component) <= tol_sq {
            break;
        }
        z = apply_jacobi(&r, diag, &in_component, anchor);
        let rz_new = dot_component(&r, &z, component);
        if rz_old.abs() <= f64::EPSILON {
            break;
        }
        let beta = rz_new / rz_old;
        for &idx in component {
            if idx == anchor {
                p[idx] = 0.0;
            } else {
                p[idx] = z[idx] + beta * p[idx];
            }
        }
        rz_old = rz_new;
    }
    x
}

fn laplacian_matvec(
    n: usize,
    edges: &[WeightedEdge],
    x: &[f64],
    in_component: &[bool],
    anchor: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for &(u, v, w) in edges {
        if !in_component[u] || !in_component[v] {
            continue;
        }
        let diff = x[u] - x[v];
        out[u] += w * diff;
        out[v] -= w * diff;
    }
    out[anchor] = 0.0;
    out
}

fn apply_jacobi(r: &[f64], diag: &[f64], in_component: &[bool], anchor: usize) -> Vec<f64> {
    let mut z = vec![0.0; r.len()];
    for i in 0..r.len() {
        if in_component[i] && i != anchor && diag[i] > f64::EPSILON {
            z[i] = r[i] / diag[i];
        }
    }
    z
}

fn dot_component(a: &[f64], b: &[f64], component: &[usize]) -> f64 {
    component.iter().map(|&idx| a[idx] * b[idx]).sum()
}

fn zero_outside_component(values: &mut [f64], in_component: &[bool]) {
    for (idx, value) in values.iter_mut().enumerate() {
        if !in_component[idx] {
            *value = 0.0;
        }
    }
}
