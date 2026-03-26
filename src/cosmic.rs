use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub struct CosmicGraphInner {
    pub weighted_adj: Array2<f64>,
    pub adj: Array2<u8>,
    pub n: usize,
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

        CosmicGraphInner { weighted_adj: wadj, adj, n }
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

    #[getter]
    pub fn weighted_adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.weighted_adj.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        Ok(self.inner.adj.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n
    }
}
