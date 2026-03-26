use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

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

#[pyfunction]
pub fn pseudo_laplacian<'py>(
    py: Python<'py>,
    nodes: Vec<Vec<usize>>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let l = pseudo_laplacian_inner(&nodes, n);
    Ok(l.into_pyarray_bound(py))
}
