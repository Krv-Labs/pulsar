use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::ballmapper::BallMapper;

/// Compute pseudo-Laplacian matrix from Ball Mapper node membership.
///
/// Given n data points and a list of balls, the pseudo-Laplacian L is n×n where:
/// - L[i,i] = number of balls containing point i
/// - L[i,j] for i≠j = negative count of balls containing both i and j
///
/// This reflects topological proximity: frequently co-occurring points have
/// strong negative off-diagonal entries.
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
    let all_nodes: Vec<&Vec<Vec<usize>>> = ball_maps.iter().map(|bm| &bm.nodes).collect();

    let galactic_l: Array2<i64> = all_nodes
        .par_iter()
        .map(|nodes| pseudo_laplacian_inner(nodes, n))
        .reduce(
            || Array2::<i64>::zeros((n, n)),
            |mut acc, l| {
                acc += &l;
                acc
            },
        );

    Ok(galactic_l.into_pyarray_bound(py))
}
