use ndarray::{Array1, Array2, Axis};
use nalgebra::DMatrix;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::PulsarError;

fn ndarray_to_nalgebra(a: &Array2<f64>) -> DMatrix<f64> {
    let (nrows, ncols) = (a.nrows(), a.ncols());
    DMatrix::from_iterator(nrows, ncols, a.iter().cloned())
}

fn nalgebra_to_ndarray(m: &DMatrix<f64>) -> Array2<f64> {
    let (nrows, ncols) = (m.nrows(), m.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| m[(i, j)])
}

pub struct PCAInner {
    pub n_components: usize,
    pub components: Array2<f64>,       // (n_components, n_features)
    pub explained_variance: Vec<f64>,
    pub means: Array1<f64>,
}

impl PCAInner {
    pub fn fit(data: &Array2<f64>, n_components: usize) -> Result<PCAInner, PulsarError> {
        let (nrows, ncols) = (data.nrows(), data.ncols());
        if n_components > ncols {
            return Err(PulsarError::InvalidParameter {
                msg: format!("n_components ({}) > n_features ({})", n_components, ncols),
            });
        }

        // Center data
        let means = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &means;
        }

        // Covariance matrix: C = X^T X / (n - 1)
        let denom = (nrows as f64 - 1.0).max(1.0);
        let cov_nd = centered.t().dot(&centered) / denom;

        // SVD via nalgebra (SVD of covariance matrix)
        let cov_na = ndarray_to_nalgebra(&cov_nd);
        let svd = cov_na.svd(true, true);
        let v_t = svd.v_t.ok_or(PulsarError::SvdFailed)?;
        let singular_values = svd.singular_values;

        // V^T rows = principal components, already sorted by descending singular value
        let v_t_nd = nalgebra_to_ndarray(&v_t);
        let mut components = v_t_nd.slice(ndarray::s![..n_components, ..]).to_owned();

        // Sign convention: largest absolute value in each PC should be positive (matches sklearn)
        for i in 0..n_components {
            let row = components.row(i);
            let max_abs_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if row[max_abs_idx] < 0.0 {
                components.row_mut(i).mapv_inplace(|x| -x);
            }
        }

        let explained_variance = (0..n_components).map(|i| singular_values[i]).collect();

        Ok(PCAInner { n_components, components, explained_variance, means })
    }

    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PulsarError> {
        let ncols = data.ncols();
        if ncols != self.means.len() {
            return Err(PulsarError::ShapeMismatch {
                expected: format!("{} features", self.means.len()),
                got: format!("{} features", ncols),
            });
        }
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &self.means;
        }
        Ok(centered.dot(&self.components.t()))
    }
}

#[pyclass]
pub struct PCA {
    n_components: usize,
    #[allow(dead_code)]
    seed: u64,
    inner: Option<PCAInner>,
}

#[pymethods]
impl PCA {
    #[new]
    pub fn new(n_components: usize, seed: u64) -> Self {
        PCA { n_components, seed, inner: None }
    }

    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = data.as_array().to_owned();
        let inner = PCAInner::fit(&arr, self.n_components)?;
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
        let out = inner.transform(&arr)?;
        Ok(out.into_pyarray_bound(py))
    }

    #[getter]
    pub fn explained_variance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("call fit_transform first")
        })?;
        let arr = ndarray::Array1::from_vec(inner.explained_variance.clone());
        Ok(arr.into_pyarray_bound(py))
    }
}
