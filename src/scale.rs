use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::PulsarError;

/// Internal (non-Python) standard scaler that stores fitted parameters.
///
/// Standard scaling transforms each feature column `x` to `(x - μ) / σ`,
/// where `μ` is the column mean and `σ` is the **population** standard
/// deviation (ddof=0).  This matches scikit-learn's `StandardScaler` default.
///
/// A std of zero would cause a division-by-zero.  Constant columns (σ < 1e-10)
/// have their std clamped to `1.0` so the scaled value is `0.0` everywhere.
pub struct StandardScalerInner {
    /// Per-column means, in column order.
    pub means: Vec<f64>,
    /// Per-column standard deviations (population, clamped ≥ 1.0 if near-zero).
    pub stds: Vec<f64>,
}

impl StandardScalerInner {
    /// Compute column statistics and return the scaled matrix together with the
    /// fitted scaler.
    ///
    /// Uses **population std** (ddof=0):
    /// `σ = sqrt( Σ(xᵢ − μ)² / n )`
    pub fn fit_transform(data: &Array2<f64>) -> (Array2<f64>, StandardScalerInner) {
        let (nrows, ncols) = (data.nrows(), data.ncols());
        let mut means = vec![0.0f64; ncols];
        let mut stds = vec![1.0f64; ncols];

        for j in 0..ncols {
            let col = data.column(j);
            let mean = col.sum() / nrows as f64;
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nrows as f64;
            means[j] = mean;
            stds[j] = if variance.sqrt() < 1e-10 { 1.0 } else { variance.sqrt() };
        }

        let scaler = StandardScalerInner { means: means.clone(), stds: stds.clone() };
        let mut out = data.clone();
        for j in 0..ncols {
            out.column_mut(j).mapv_inplace(|x| (x - means[j]) / stds[j]);
        }
        (out, scaler)
    }

    /// Apply previously fitted statistics to a new matrix.
    ///
    /// # Errors
    /// [`PulsarError::ShapeMismatch`] if `data` has a different number of
    /// columns than the matrix used during `fit_transform`.
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PulsarError> {
        let ncols = data.ncols();
        if ncols != self.means.len() {
            return Err(PulsarError::ShapeMismatch {
                expected: format!("{} columns", self.means.len()),
                got: format!("{} columns", ncols),
            });
        }
        let mut out = data.clone();
        for j in 0..ncols {
            out.column_mut(j).mapv_inplace(|x| (x - self.means[j]) / self.stds[j]);
        }
        Ok(out)
    }

    /// Reverse a previous `transform`: `x_orig = x_scaled * σ + μ`.
    ///
    /// # Errors
    /// [`PulsarError::ShapeMismatch`] if `data` has a different number of
    /// columns than the matrix used during `fit_transform`.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PulsarError> {
        let ncols = data.ncols();
        if ncols != self.means.len() {
            return Err(PulsarError::ShapeMismatch {
                expected: format!("{} columns", self.means.len()),
                got: format!("{} columns", ncols),
            });
        }
        let mut out = data.clone();
        for j in 0..ncols {
            out.column_mut(j).mapv_inplace(|x| x * self.stds[j] + self.means[j]);
        }
        Ok(out)
    }
}

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
