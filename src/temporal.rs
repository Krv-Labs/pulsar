//! Temporal Cosmic Graph support for longitudinal time-series data.
//!
//! This module extends Pulsar to handle data where the same set of nodes (e.g., patients)
//! are observed across multiple time steps. Instead of a single 2D pseudo-Laplacian,
//! we accumulate a 3D tensor `L[i, j, t]` representing connectivity at each time step.
//!
//! ## Core Data Structure
//!
//! The temporal pseudo-Laplacian tensor has shape `(n, n, T)` where:
//! - `n` is the number of nodes (fixed across time)
//! - `T` is the number of time steps
//! - `L[i, j, t]` follows the same semantics as the standard pseudo-Laplacian at time `t`
//!
//! ## Parallelization Strategy
//!
//! Time steps are processed in parallel using rayon. Each time step's ball maps
//! are accumulated independently, then results are combined into the 3D tensor.

use ndarray::{Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ballmapper::BallMapper;
use crate::pseudolaplacian::pseudo_laplacian_inner;

/// Accumulate pseudo-Laplacians for each time step, returning a 3D tensor.
///
/// This is the temporal extension of `accumulate_pseudo_laplacians`. Each time step
/// has its own set of BallMapper results from running the standard Pulsar pipeline
/// on that time step's data.
///
/// # Arguments
/// * `ball_maps_per_time` - Vector of length T, where each element is a vector of
///   BallMapper references for that time step
/// * `n` - Number of nodes (must be consistent across all time steps)
///
/// # Returns
/// A 3D array of shape `(n, n, T)` containing the accumulated pseudo-Laplacian
/// for each time step.
///
/// # Example
/// ```python
/// # Run Pulsar pipeline at each time step
/// ball_maps_per_time = []
/// for t in range(T):
///     embeddings_t = pca_grid(X_scaled[t], dims, seeds)
///     ball_maps_t = ball_mapper_grid(embeddings_t, epsilons)
///     ball_maps_per_time.append(ball_maps_t)
///
/// # Accumulate into 3D tensor
/// L_tensor = accumulate_temporal_pseudo_laplacians(ball_maps_per_time, n)
/// # L_tensor has shape (n, n, T)
/// ```
pub fn accumulate_temporal_pseudo_laplacians_inner(
    ball_maps_per_time: &[Vec<&Vec<Vec<usize>>>],
    n: usize,
) -> Array3<i64> {
    let t_steps = ball_maps_per_time.len();

    // Process each time step in parallel
    let laplacians: Vec<Array2<i64>> = ball_maps_per_time
        .par_iter()
        .map(|ball_maps_at_t| {
            // Accumulate all ball maps at this time step
            ball_maps_at_t
                .iter()
                .map(|nodes| pseudo_laplacian_inner(nodes, n))
                .fold(Array2::<i64>::zeros((n, n)), |mut acc, l| {
                    acc += &l;
                    acc
                })
        })
        .collect();

    // Stack into 3D tensor (n, n, T)
    let mut tensor = Array3::<i64>::zeros((n, n, t_steps));
    for (t, laplacian) in laplacians.into_iter().enumerate() {
        tensor.index_axis_mut(Axis(2), t).assign(&laplacian);
    }

    tensor
}

/// Normalize a 3D pseudo-Laplacian tensor into weighted adjacency matrices.
///
/// Applies the same normalization formula as `CosmicGraph::from_pseudo_laplacian`
/// independently at each time step.
///
/// # Weight formula (per time step)
///
/// For each off-diagonal pair `(i, j)` at time `t`:
///
/// ```text
/// denom = L[i,i,t] + L[j,j,t] + L[i,j,t]
///
///                 -L[i,j,t]
/// W[i,j,t] =  ─────────────────   if denom > 0
///                  denom
///
///           = 0                    otherwise
/// ```
pub fn normalize_temporal_laplacian(l: &Array3<i64>) -> Array3<f64> {
    let (n, _, t_steps) = l.dim();
    let mut w = Array3::<f64>::zeros((n, n, t_steps));

    for t in 0..t_steps {
        for i in 0..n {
            // Exploit symmetry: compute only for j > i and mirror to (j, i)
            for j in (i + 1)..n {
                let denom = l[[i, i, t]] + l[[j, j, t]] + l[[i, j, t]];
                if denom > 0 {
                    let weight = -(l[[i, j, t]] as f64) / (denom as f64);
                    w[[i, j, t]] = weight;
                    w[[j, i, t]] = weight;
                }
            }
        }
    }

    w
}

// ============================================================================
// Python bindings
// ============================================================================

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
        .map(|bms| bms.iter().map(|bm| &bm.nodes).collect())
        .collect();

    let tensor = accumulate_temporal_pseudo_laplacians_inner(&nodes_per_time, n);

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
    let w = normalize_temporal_laplacian(&arr);
    Ok(w.into_pyarray_bound(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_accumulation_single_timestep() {
        // Single time step should match non-temporal behavior
        let nodes1 = vec![vec![0, 1], vec![1, 2]];
        let ball_maps_per_time = vec![vec![&nodes1]];

        let tensor = accumulate_temporal_pseudo_laplacians_inner(&ball_maps_per_time, 3);

        assert_eq!(tensor.dim(), (3, 3, 1));
        // Check diagonal (membership counts)
        assert_eq!(tensor[[0, 0, 0]], 1); // node 0 in 1 ball
        assert_eq!(tensor[[1, 1, 0]], 2); // node 1 in 2 balls
        assert_eq!(tensor[[2, 2, 0]], 1); // node 2 in 1 ball
    }

    #[test]
    fn test_temporal_accumulation_multiple_timesteps() {
        let nodes1 = vec![vec![0, 1]];
        let nodes2 = vec![vec![1, 2]];
        let ball_maps_per_time = vec![vec![&nodes1], vec![&nodes2]];

        let tensor = accumulate_temporal_pseudo_laplacians_inner(&ball_maps_per_time, 3);

        assert_eq!(tensor.dim(), (3, 3, 2));

        // Time step 0: nodes 0,1 connected
        assert_eq!(tensor[[0, 1, 0]], -1);
        assert_eq!(tensor[[1, 2, 0]], 0);

        // Time step 1: nodes 1,2 connected
        assert_eq!(tensor[[0, 1, 1]], 0);
        assert_eq!(tensor[[1, 2, 1]], -1);
    }

    #[test]
    fn test_normalize_temporal() {
        // Create a simple 3D Laplacian tensor
        let mut l = Array3::<i64>::zeros((2, 2, 1));
        l[[0, 0, 0]] = 1;
        l[[1, 1, 0]] = 1;
        l[[0, 1, 0]] = -1;
        l[[1, 0, 0]] = -1;

        let w = normalize_temporal_laplacian(&l);

        // Both nodes in same ball: weight should be 1.0
        // denom = 1 + 1 + (-1) = 1
        // weight = -(-1) / 1 = 1.0
        assert!((w[[0, 1, 0]] - 1.0).abs() < 1e-10);
        assert!((w[[1, 0, 0]] - 1.0).abs() < 1e-10);
    }
}
