use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::PulsarError;

pub struct StandardScalerInner {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl StandardScalerInner {
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

#[pyclass]
pub struct StandardScaler {
    inner: Option<StandardScalerInner>,
}

#[pymethods]
impl StandardScaler {
    #[new]
    pub fn new() -> Self {
        StandardScaler { inner: None }
    }

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
