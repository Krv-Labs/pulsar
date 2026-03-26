use ndarray::{Array1, Array2, Axis};
use nalgebra::DMatrix;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::PulsarError;

/// Convert an ndarray `Array2<f64>` to a nalgebra `DMatrix<f64>`.
///
/// This conversion is needed because ndarray (used throughout the codebase for
/// array manipulation) and nalgebra (used for its robust SVD implementation)
/// are separate libraries with incompatible matrix types.  The conversion is
/// `O(n*m)` but covariance matrices are small (n_features × n_features), so
/// it is not a bottleneck.
fn ndarray_to_nalgebra(a: &Array2<f64>) -> DMatrix<f64> {
    let (nrows, ncols) = (a.nrows(), a.ncols());
    DMatrix::from_iterator(nrows, ncols, a.iter().cloned())
}

/// Convert a nalgebra `DMatrix<f64>` back to an ndarray `Array2<f64>`.
fn nalgebra_to_ndarray(m: &DMatrix<f64>) -> Array2<f64> {
    let (nrows, ncols) = (m.nrows(), m.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| m[(i, j)])
}

/// Internal PCA state stored after fitting.
///
/// PCA is computed via exact SVD of the sample covariance matrix, which is
/// equivalent to sklearn's `PCA` implementation when data is full-rank.
pub struct PCAInner {
    /// Number of principal components retained.
    pub n_components: usize,
    /// Principal component directions, shape `(n_components, n_features)`.
    /// Each row is a unit vector in feature space.  Signs are normalised so
    /// that the largest-absolute-value element of each component is positive
    /// (scikit-learn's deterministic sign convention).
    pub components: Array2<f64>,
    /// Singular values of the covariance matrix, one per component.
    /// These are the eigenvalues of `C = XᵀX / (n-1)` in descending order.
    pub explained_variance: Vec<f64>,
    /// Per-feature training means, subtracted during `transform`.
    pub means: Array1<f64>,
}

impl PCAInner {
    /// Fit PCA on `data` and return the fitted model.
    ///
    /// # Algorithm
    /// 1. **Centre** — subtract column means: `X_c = X − μ`
    /// 2. **Covariance** — `C = X_cᵀ X_c / (n − 1)` (sample covariance, ddof=1)
    /// 3. **SVD** — `C = U Σ Vᵀ` via nalgebra.  The rows of `Vᵀ` are the
    ///    principal component directions, already sorted by descending singular value.
    /// 4. **Sign flip** — for each component, if the element with the largest
    ///    absolute value is negative, negate the whole component.  This makes
    ///    the output sign-deterministic and comparable to sklearn's convention.
    /// 5. **Variance** — singular values of `C` are stored as `explained_variance`.
    ///
    /// # Errors
    /// - [`PulsarError::InvalidParameter`] — `n_components > n_features`
    /// - [`PulsarError::SvdFailed`] — SVD did not converge (rare)
    pub fn fit(data: &Array2<f64>, n_components: usize) -> Result<PCAInner, PulsarError> {
        let (nrows, ncols) = (data.nrows(), data.ncols());
        if n_components > ncols {
            return Err(PulsarError::InvalidParameter {
                msg: format!("n_components ({}) > n_features ({})", n_components, ncols),
            });
        }

        // Step 1: centre data
        let means = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &means;
        }

        // Step 2: sample covariance matrix (n×n, where n = n_features)
        // Dividing by (n_samples − 1) matches sklearn and numpy's default.
        let denom = (nrows as f64 - 1.0).max(1.0);
        let cov_nd = centered.t().dot(&centered) / denom;

        // Step 3: SVD via nalgebra
        // nalgebra's SVD gives U, Σ, Vᵀ.  For a symmetric PSD matrix (covariance),
        // U == V, so the rows of Vᵀ are the eigenvectors sorted by descending
        // singular value — exactly the principal components we need.
        let cov_na = ndarray_to_nalgebra(&cov_nd);
        let svd = cov_na.svd(true, true);
        let v_t = svd.v_t.ok_or(PulsarError::SvdFailed)?;
        let singular_values = svd.singular_values;

        let v_t_nd = nalgebra_to_ndarray(&v_t);
        let mut components = v_t_nd.slice(ndarray::s![..n_components, ..]).to_owned();

        // Step 4: sign convention — largest-abs-value element must be positive.
        // Without this, the sign of each component is arbitrary (both +v and −v
        // are valid eigenvectors).  sklearn uses the same convention, which
        // makes it possible to compare outputs directly.
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

    /// Project `data` onto the fitted principal components.
    ///
    /// Subtracts the training mean before projecting:
    /// `projection = (X − μ) · Vᵀ[:n_components].T`
    ///
    /// # Errors
    /// [`PulsarError::ShapeMismatch`] if `data` has a different number of
    /// features than the training data.
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

/// Python-facing PCA class.
///
/// ```python
/// from pulsar._pulsar import PCA
///
/// pca = PCA(n_components=3, seed=42)
/// X_reduced = pca.fit_transform(X)       # fit and project in one call
/// X_new_reduced = pca.transform(X_new)   # project new data
/// variances = pca.explained_variance     # per-component singular values
/// ```
///
/// The `seed` parameter is accepted for API symmetry with other Pulsar classes
/// but is not used (PCA is deterministic).
#[pyclass]
pub struct PCA {
    n_components: usize,
    #[allow(dead_code)]
    seed: u64,
    /// `None` until `fit_transform` is called.
    inner: Option<PCAInner>,
}

#[pymethods]
impl PCA {
    /// Create a new unfitted PCA.
    ///
    /// # Parameters
    /// - `n_components` — number of principal components to keep.
    /// - `seed` — unused; present for API symmetry.
    #[new]
    pub fn new(n_components: usize, seed: u64) -> Self {
        PCA { n_components, seed, inner: None }
    }

    /// Fit PCA on `data` and return the low-dimensional projection.
    ///
    /// # Parameters
    /// - `data` (`np.ndarray[float64, 2D]`, shape `(n_samples, n_features)`)
    ///
    /// # Returns
    /// `np.ndarray[float64, 2D]`, shape `(n_samples, n_components)`
    ///
    /// # Raises
    /// `ValueError` — if `n_components > n_features` or SVD fails.
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

    /// Project new data using the components fitted by `fit_transform`.
    ///
    /// # Raises
    /// `ValueError` — if `fit_transform` has not been called, or if `data`
    /// has a different number of features than the training data.
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

    /// Singular values of the covariance matrix, one per component, in
    /// descending order.
    ///
    /// # Raises
    /// `ValueError` — if `fit_transform` has not been called yet.
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
