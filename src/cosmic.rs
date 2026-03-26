use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Internal Cosmic Graph data: weighted and binary adjacency matrices.
///
/// The Cosmic Graph summarises topological proximity between data points by
/// normalising the pseudo-Laplacian into a weighted adjacency matrix.
pub struct CosmicGraphInner {
    /// Weighted adjacency matrix, shape `(n, n)`, values in `[0, 1]`.
    /// Entry `(i, j)` is the normalised co-membership weight between points
    /// `i` and `j` (see [`from_pseudo_laplacian`] for the formula).
    pub weighted_adj: Array2<f64>,
    /// Binary adjacency matrix (0/1), shape `(n, n)`.
    /// Entry `(i, j) = 1` iff `weighted_adj[i,j] > threshold`.
    pub adj: Array2<u8>,
    /// Number of data points (side length of both matrices).
    pub n: usize,
}

impl CosmicGraphInner {
    /// Construct a Cosmic Graph from an accumulated pseudo-Laplacian.
    ///
    /// This is a direct port of Thema's `normalize_cosmicGraph` in
    /// `starHelpers.py`.
    ///
    /// # Weight formula
    ///
    /// For each off-diagonal pair `(i, j)`:
    ///
    /// ```text
    /// denom = L[i,i] + L[j,j] + L[i,j]
    ///
    ///                 -L[i,j]
    /// W[i,j] =  ─────────────────   if denom > 0
    ///               denom
    ///
    ///         = 0                    otherwise
    /// ```
    ///
    /// **Intuition:**
    /// - `L[i,j]` is negative (number of shared balls, negated), so `-L[i,j]`
    ///   is positive: the raw shared co-membership count.
    /// - `L[i,i]` and `L[j,j]` are the individual membership counts (degrees).
    /// - Dividing by their sum normalises the weight to `[0, 1]`.
    /// - The maximum weight of 1 is reached when `L[i,j] = −L[i,i] = −L[j,j]`,
    ///   i.e. both points appear in exactly the same balls.
    ///
    /// Pairs where `denom ≤ 0` (disconnected points) keep weight 0.
    ///
    /// # Parameters
    /// - `l` — accumulated pseudo-Laplacian, shape `(n, n)`, dtype `i64`.
    /// - `threshold` — minimum weight for an edge to appear in the binary
    ///   adjacency matrix.  Use `0.0` to include all positive-weight edges.
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

        CosmicGraphInner { weighted_adj: wadj, adj, n }
    }
}

/// Python-facing Cosmic Graph class.
///
/// ```python
/// from pulsar._pulsar import CosmicGraph, pseudo_laplacian
/// import numpy as np
///
/// # Accumulate pseudo-Laplacians across all ball maps in the sweep
/// galactic_L = np.zeros((n, n), dtype=np.int64)
/// for bm in ball_maps:
///     galactic_L += pseudo_laplacian(bm.nodes, n)
///
/// # Build the Cosmic Graph
/// cg = CosmicGraph.from_pseudo_laplacian(galactic_L, threshold=0.0)
/// print(cg.weighted_adj)   # float weights in [0, 1]
/// print(cg.adj)            # binary adjacency (uint8)
/// ```
#[pyclass]
pub struct CosmicGraph {
    inner: CosmicGraphInner,
}

#[pymethods]
impl CosmicGraph {
    /// Build a Cosmic Graph from an accumulated pseudo-Laplacian matrix.
    ///
    /// # Parameters
    /// - `l` (`np.ndarray[int64, 2D]`, shape `(n, n)`) — summed pseudo-Laplacian
    ///   from all Ball Maps in the parameter sweep.
    /// - `threshold` (`float`) — edges with weight ≤ `threshold` are excluded
    ///   from the binary adjacency matrix.  Typical value: `0.0`.
    ///
    /// # Returns
    /// A `CosmicGraph` instance.
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

    /// Weighted adjacency matrix, shape `(n, n)`, values in `[0, 1]`.
    ///
    /// Entry `(i, j)` represents normalised co-membership between points `i`
    /// and `j` across all Ball Maps in the sweep.
    #[getter]
    pub fn weighted_adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.weighted_adj.clone().into_pyarray_bound(py))
    }

    /// Binary adjacency matrix, shape `(n, n)`, dtype `uint8`.
    ///
    /// Entry `(i, j) = 1` iff `weighted_adj[i, j] > threshold`.
    #[getter]
    pub fn adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        Ok(self.inner.adj.clone().into_pyarray_bound(py))
    }

    /// Number of data points (side length of both adjacency matrices).
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n
    }
}
