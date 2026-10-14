use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use pulsar_core::{JLProjectionInner, PulsarError};

/// Core centering uses `mean_axis(...).expect(..)`, which panics on 0 rows.
fn validate_rows(nrows: usize) -> Result<(), PulsarError> {
    if nrows == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "input data has 0 rows (samples)".to_string(),
        });
    }
    Ok(())
}

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
        validate_rows(data.as_array().nrows())?;
        let arr = data.as_array().to_owned();
        let (n_components, seed, center) = (self.n_components, self.seed, self.center);
        let (inner, projection) = py.allow_threads(|| {
            JLProjectionInner::fit(&arr, n_components, seed, center)
                .and_then(|inner| inner.transform(&arr).map(|proj| (inner, proj)))
        })?;
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
        let out = py.allow_threads(|| inner.transform(&arr))?;
        Ok(out.into_pyarray_bound(py))
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
    validate_rows(data.as_array().nrows())?;
    // Owned snapshot by choice, not necessity: rust-numpy borrows stay valid
    // across allow_threads, but copying matches the other bindings in this
    // crate and shields the computation from unchecked Python-side writes to
    // the buffer while the GIL is released. Cost: one O(n*d) copy.
    let arr = data.as_array().to_owned();
    let embeddings =
        py.allow_threads(|| pulsar_core::jl_grid(&arr.view(), &dimensions, &seeds, center))?;
    Ok(embeddings
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect())
}
