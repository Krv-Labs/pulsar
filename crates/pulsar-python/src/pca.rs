use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use pulsar_core::PCAInner;

/// Randomized PCA optimized for large datasets.
///
/// Uses randomized SVD (Halko et al. 2011) which is O(n*d*k) instead of
/// O(n*d² + d³) for exact SVD. Different seeds produce different (but
/// equally valid) principal components, enabling ensemble diversity.
///
/// ```python
/// from pulsar._pulsar import PCA
///
/// pca = PCA(n_components=10, seed=42)
/// X_reduced = pca.fit_transform(X)
/// ```
#[pyclass]
pub struct PCA {
    n_components: usize,
    seed: u64,
    n_oversamples: usize,
    n_power_iter: usize,
    inner: Option<PCAInner>,
}

#[pymethods]
impl PCA {
    /// Create a new PCA with randomized SVD.
    ///
    /// # Parameters
    /// - `n_components` — number of principal components to keep
    /// - `seed` — random seed for stochastic projection (different seeds = different embeddings)
    /// - `n_oversamples` — extra dimensions for approximation quality (default: 10)
    /// - `n_power_iter` — power iterations for slowly decaying spectra (default: 2)
    #[new]
    #[pyo3(signature = (n_components, seed, n_oversamples=10, n_power_iter=2))]
    pub fn new(n_components: usize, seed: u64, n_oversamples: usize, n_power_iter: usize) -> Self {
        PCA { n_components, seed, n_oversamples, n_power_iter, inner: None }
    }

    /// Fit PCA and return the low-dimensional projection.
    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = data.as_array().to_owned();
        let inner = PCAInner::fit(
            &arr,
            self.n_components,
            self.seed,
            self.n_oversamples,
            self.n_power_iter,
        )?;
        let projection = inner.transform(&arr)?;
        self.inner = Some(inner);
        Ok(projection.into_pyarray_bound(py))
    }

    /// Project new data using fitted components.
    pub fn transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("call fit_transform before transform")
        })?;
        let arr = data.as_array().to_owned();
        let out = inner.transform(&arr)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// Explained variance per component.
    #[getter]
    pub fn explained_variance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("call fit_transform first")
        })?;
        let arr = numpy::ndarray::Array1::from_vec(inner.explained_variance.clone());
        Ok(arr.into_pyarray_bound(py))
    }
}

/// Compute PCA embeddings for multiple dimensions and seeds in parallel.
///
/// Optimized for grid search: computes one SVD per seed at max dimension,
/// then slices for each requested dimension. Parallelised across seeds.
///
/// # Returns
/// List of 2D arrays in row-major order: for each seed (outer), all dimensions (inner).
/// So `pca_grid(X, [2,3], [42,7])` returns `[X_s42_d2, X_s42_d3, X_s7_d2, X_s7_d3]`.
#[pyfunction]
#[pyo3(signature = (data, dimensions, seeds, n_oversamples=10, n_power_iter=2))]
pub fn pca_grid<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    dimensions: Vec<usize>,
    seeds: Vec<u64>,
    n_oversamples: usize,
    n_power_iter: usize,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let arr = data.as_array().to_owned();
    let embeddings =
        pulsar_core::pca_grid(&arr, &dimensions, &seeds, n_oversamples, n_power_iter)?;
    Ok(embeddings
        .into_iter()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect())
}
