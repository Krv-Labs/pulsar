use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use pulsar_core::StandardScalerInner;

/// Python-facing standard scaler.
///
/// Call `fit_transform` first to fit the scaler and scale the training data.
/// Then call `transform` on new data using the stored statistics, or
/// `inverse_transform` to recover the original scale.
///
/// ```python
/// from pulsar._pulsar import StandardScaler
///
/// scaler = StandardScaler()
/// X_scaled = scaler.fit_transform(X_train)
/// X_test_scaled = scaler.transform(X_test)
/// X_recovered = scaler.inverse_transform(X_scaled)
/// ```
#[pyclass]
pub struct StandardScaler {
    /// `None` until `fit_transform` is called.
    inner: Option<StandardScalerInner>,
}

#[pymethods]
impl StandardScaler {
    /// Create a new, unfitted scaler.
    #[new]
    pub fn new() -> Self {
        StandardScaler { inner: None }
    }

    /// Fit the scaler to `data` and return the scaled matrix.
    ///
    /// Stores column means and standard deviations internally so that
    /// `transform` / `inverse_transform` can be called later.
    ///
    /// # Parameters
    /// - `data` (`np.ndarray[float64, 2D]`, shape `(n_samples, n_features)`)
    ///
    /// # Returns
    /// `np.ndarray[float64, 2D]` — scaled matrix with mean ≈ 0, std ≈ 1 per column.
    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = data.as_array().to_owned();
        let (scaled, scaler) = StandardScalerInner::fit_transform(&arr);
        self.inner = Some(scaler);
        Ok(scaled.into_pyarray_bound(py))
    }

    /// Scale `data` using statistics from `fit_transform`.
    ///
    /// # Raises
    /// `ValueError` — if `fit_transform` has not been called yet, or if
    /// `data` has a different number of columns than the fitted data.
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

    /// Undo scaling: `x_orig = x_scaled * σ + μ`.
    ///
    /// # Raises
    /// `ValueError` — if `fit_transform` has not been called yet, or if
    /// `data` has a different number of columns than the fitted data.
    pub fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "call fit_transform before inverse_transform",
            )
        })?;
        let arr = data.as_array().to_owned();
        let out = inner.inverse_transform(&arr)?;
        Ok(out.into_pyarray_bound(py))
    }
}
