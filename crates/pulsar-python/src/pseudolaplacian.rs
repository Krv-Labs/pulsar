use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use pulsar_core::pseudolaplacian as core;

use crate::ballmapper::BallMapper;

/// Accumulate pseudo-Laplacians from all ball maps in parallel.
///
/// This is the optimized entry point that replaces sequential Python loops.
/// Uses rayon parallel map-reduce for maximum throughput.
///
/// ```python
/// # Single call replaces 4000+ Python/Rust crossings
/// galactic_L = accumulate_pseudo_laplacians(ball_maps, n)
/// ```
#[pyfunction]
pub fn accumulate_pseudo_laplacians<'py>(
    py: Python<'py>,
    ball_maps: Vec<PyRef<'py, BallMapper>>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let all_nodes: Vec<&[Vec<usize>]> = ball_maps
        .iter()
        .map(|bm| bm.core.nodes.as_slice())
        .collect();

    let galactic_l = core::accumulate_pseudo_laplacians(&all_nodes, n);

    Ok(galactic_l.into_pyarray_bound(py))
}

/// Sparse pseudo-Laplacian: diagonal counts + deduped, `(i,j)`-sorted upper-triangle
/// off-diagonal co-occurrence counts. Feeds `CosmicGraph.from_pseudo_laplacian_sparse`
/// directly without densifying.
#[pyclass]
pub struct SparsePseudoLaplacian {
    pub core: core::SparsePseudoLaplacian,
}

#[pymethods]
impl SparsePseudoLaplacian {
    /// Number of points (matrix dimension n).
    #[getter]
    pub fn n(&self) -> usize {
        self.core.n
    }

    /// Diagonal counts `diag[i]` = number of (ball-map, ball) pairs containing i.
    #[getter]
    pub fn diag<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.core.diag.clone().into_pyarray_bound(py)
    }

    /// Upper-triangle off-diagonal co-occurrence counts `(i, j, count)` with `i < j`,
    /// sorted by `(i, j)`.
    #[getter]
    pub fn offdiag(&self) -> Vec<(usize, usize, i64)> {
        self.core.offdiag.clone()
    }

    /// Number of stored off-diagonal entries (nonzeros in the upper triangle).
    #[getter]
    pub fn nnz(&self) -> usize {
        self.core.nnz()
    }

    /// Fold another sparse Laplacian (same n) into this one: sum diagonals and
    /// merge off-diagonal counts. Used to accumulate across datasets in `fit_multi`
    /// without ever building an n×n matrix.
    pub fn merge_in_place(&mut self, other: PyRef<SparsePseudoLaplacian>) -> PyResult<()> {
        self.core.merge_in_place(&other.core)?;
        Ok(())
    }
}

/// Sparse counterpart of [`accumulate_pseudo_laplacians`]. Accumulates the
/// co-membership pseudo-Laplacian across all ball maps as a COO edge list plus an
/// O(n) diagonal, never allocating an n×n matrix.
///
/// Reduce strategy: each ball map maps to a thread-local `(diag, offdiag)`
/// contribution; every contribution is sorted/merged locally, and the rayon reduce
/// merges sorted COO buffers while adding diagonals. This avoids carrying raw
/// duplicate co-membership pairs across the whole sweep.
#[pyfunction]
pub fn accumulate_pseudo_laplacians_sparse<'py>(
    _py: Python<'py>,
    ball_maps: Vec<PyRef<'py, BallMapper>>,
    n: usize,
) -> PyResult<SparsePseudoLaplacian> {
    let all_nodes: Vec<&[Vec<usize>]> = ball_maps
        .iter()
        .map(|bm| bm.core.nodes.as_slice())
        .collect();

    Ok(SparsePseudoLaplacian {
        core: core::accumulate_pseudo_laplacians_sparse(&all_nodes, n),
    })
}
