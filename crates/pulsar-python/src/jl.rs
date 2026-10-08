use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use pulsar_core::JLProjectionInner;

#[pyclass]
pub struct JLProjection {
    n_components: usize,
    seed: u64,
    center: bool,
    inner: Option<JLProjectionInner>,
}

#[pymethods]
impl JLProjection {
    #[new]
    #[pyo3(signature = (n_components, seed, center=true))]
    pub fn new(n_components: usize, seed: u64, center: bool) -> Self {
        Self {
            n_components,
            seed,
            center,
            inner: None,
        }
    }

    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = data.as_array().to_owned();
        let inner = JLProjectionInner::fit(&arr, self.n_components, self.seed, self.center)?;
        let projection = inner.transform(&arr)?;
        self.inner = Some(inner);
        Ok(projection.into_pyarray_bound(py))
    }

    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("call fit_transform before transform")
        })?;
        let arr = data.as_array().to_owned();
        Ok(inner.transform(&arr)?.into_pyarray_bound(py))
    }
}

/// Compute JL embeddings for multiple dimensions and seeds in parallel.
///
/// Returns arrays in row-major grid order: seed outer, dimensions inner.
#[pyfunction]
#[pyo3(signature = (data, dimensions, seeds, center=true))]
pub fn jl_grid<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    dimensions: Vec<usize>,
    seeds: Vec<u64>,
    center: bool,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let arr = data.as_array();
    let embeddings = pulsar_core::jl_grid(&arr, &dimensions, &seeds, center)?;
    Ok(embeddings
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect())
}
