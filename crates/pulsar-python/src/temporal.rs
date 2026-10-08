use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;

use pulsar_core::temporal as core;

use crate::ballmapper::BallMapper;

/// Accumulate pseudo-Laplacians across time steps into a 3D tensor.
///
/// This function processes ball maps from multiple time steps in parallel,
/// producing a 3D tensor of shape `(n, n, T)` where each slice `[:, :, t]`
/// is the accumulated pseudo-Laplacian for time step `t`.
///
/// # Parameters
/// - `ball_maps_per_time` (`list[list[BallMapper]]`) — For each time step,
///   a list of BallMapper objects from the parameter sweep at that time.
/// - `n` (`int`) — Number of nodes (must be consistent across all time steps).
///
/// # Returns
/// A numpy array of shape `(n, n, T)` with dtype `int64`.
///
/// # Example
/// ```python
/// from pulsar._pulsar import accumulate_temporal_pseudo_laplacians
///
/// # ball_maps_per_time[t] contains all BallMappers for time step t
/// L_tensor = accumulate_temporal_pseudo_laplacians(ball_maps_per_time, n)
/// print(L_tensor.shape)  # (n, n, T)
/// ```
#[pyfunction]
pub fn accumulate_temporal_pseudo_laplacians<'py>(
    py: Python<'py>,
    ball_maps_per_time: Vec<Vec<PyRef<'py, BallMapper>>>,
    n: usize,
) -> PyResult<Bound<'py, PyArray3<i64>>> {
    // Extract node references for each time step
    let nodes_per_time: Vec<Vec<&Vec<Vec<usize>>>> = ball_maps_per_time
        .iter()
        .map(|bms| bms.iter().map(|bm| &bm.core.nodes).collect())
        .collect();

    let tensor = core::accumulate_temporal_pseudo_laplacians_inner(&nodes_per_time, n);

    Ok(tensor.into_pyarray_bound(py))
}

/// Normalize a 3D pseudo-Laplacian tensor into weighted adjacency matrices.
///
/// Applies the cosmic graph normalization formula independently at each time step,
/// producing a 3D tensor of edge weights in `[0, 1]`.
///
/// # Parameters
/// - `l` (`np.ndarray[int64, 3D]`, shape `(n, n, T)`) — The accumulated
///   pseudo-Laplacian tensor from `accumulate_temporal_pseudo_laplacians`.
///
/// # Returns
/// A numpy array of shape `(n, n, T)` with dtype `float64`, where each
/// slice `[:, :, t]` contains edge weights in `[0, 1]`.
///
/// # Example
/// ```python
/// from pulsar._pulsar import (
///     accumulate_temporal_pseudo_laplacians,
///     normalize_temporal_laplacian,
/// )
///
/// L_tensor = accumulate_temporal_pseudo_laplacians(ball_maps_per_time, n)
/// W_tensor = normalize_temporal_laplacian(L_tensor)
/// print(W_tensor.shape)  # (n, n, T)
/// print(W_tensor.min(), W_tensor.max())  # 0.0, ~1.0
/// ```
#[pyfunction]
pub fn py_normalize_temporal_laplacian<'py>(
    py: Python<'py>,
    l: numpy::PyReadonlyArray3<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let arr = l.as_array().to_owned();
    let w = core::normalize_temporal_laplacian(&arr);
    Ok(w.into_pyarray_bound(py))
}
