use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// Compute the pseudo-Laplacian matrix from a Ball Mapper's node membership.
///
/// This is a direct port of `mapper_pseudo_laplacian(neighborhood="node")` from
/// Thema's `starHelpers.py`.
///
/// # What it computes
///
/// Given `n` data points and a list of balls (nodes), each containing a subset
/// of point indices, the pseudo-Laplacian `L` is an `n × n` integer matrix
/// where:
/// - `L[i, i]` = number of balls that contain point `i` (diagonal: degree-like)
/// - `L[i, j]` for `i ≠ j` = **negative** number of balls that contain *both*
///   points `i` and `j` (off-diagonal: negative shared membership count)
///
/// Intuitively, `L` is a discrete Laplacian over the point cloud: if two
/// points frequently co-occur in the same balls they have a strong negative
/// off-diagonal entry, reflecting local proximity in the topological structure.
///
/// # Algorithm
/// For each ball `b` with members `{i, j, k, ...}`:
/// - Add 1 to `L[m, m]` for every member `m` (diagonal contribution)
/// - Subtract 1 from `L[i, j]` for every pair `(i, j)` with `i ≠ j` (off-diagonal)
///
/// The double loop `for i in members { for j in members { ... } }` handles
/// both directions at once, so the resulting matrix is always symmetric.
///
/// # Parameters
/// - `nodes` — slice of membership lists, one per ball.
/// - `n` — total number of data points (size of the output matrix).
///
/// # Returns
/// `Array2<i64>` of shape `(n, n)`.
pub fn pseudo_laplacian_inner(nodes: &[Vec<usize>], n: usize) -> Array2<i64> {
    let mut l = Array2::<i64>::zeros((n, n));
    for members in nodes {
        for &i in members {
            for &j in members {
                if i == j {
                    l[[i, j]] += 1;
                } else {
                    l[[i, j]] -= 1;
                }
            }
        }
    }
    l
}

/// Python-facing wrapper around [`pseudo_laplacian_inner`].
///
/// Accumulating pseudo-Laplacians across a grid of Ball Maps is done in Python:
/// ```python
/// galactic_L = np.zeros((n, n), dtype=np.int64)
/// for bm in ball_maps:
///     galactic_L += pseudo_laplacian(bm.nodes, n)
/// ```
///
/// # Parameters
/// - `nodes` (`list[list[int]]`) — ball membership lists from a fitted `BallMapper`.
/// - `n` (`int`) — number of data points (row/column size of the output).
///
/// # Returns
/// `np.ndarray[int64, 2D]` of shape `(n, n)`.
#[pyfunction]
pub fn pseudo_laplacian<'py>(
    py: Python<'py>,
    nodes: Vec<Vec<usize>>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let l = pseudo_laplacian_inner(&nodes, n);
    Ok(l.into_pyarray_bound(py))
}
