use ndarray::{s, Array1, Array2, Axis};
use nalgebra::DMatrix;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

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

/// Compute the Q factor of a thin QR decomposition.
///
/// Given matrix A of shape (m, n) with m >= n, returns Q of shape (m, n)
/// such that A = Q @ R where Q has orthonormal columns.
fn qr_q(a: &Array2<f64>) -> Array2<f64> {
    let a_na = ndarray_to_nalgebra(a);
    let qr = a_na.qr();
    let q = qr.q();
    // Take only the first `ncols` columns (thin Q)
    let thin_q = q.columns(0, a.ncols());
    let (nrows, ncols) = (thin_q.nrows(), thin_q.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| thin_q[(i, j)])
}

/// Internal PCA state stored after fitting.
///
/// Uses randomized SVD (Halko et al. 2011) for efficiency on large datasets.
pub struct PCAInner {
    /// Number of principal components retained (stored for potential future use).
    #[allow(dead_code)]
    n_components: usize,
    /// Principal component directions, shape `(n_components, n_features)`.
    /// Each row is a unit vector in feature space. Signs are normalised so
    /// that the largest-absolute-value element of each component is positive
    /// (scikit-learn's deterministic sign convention).
    pub components: Array2<f64>,
    /// Singular values, one per component, in descending order.
    pub explained_variance: Vec<f64>,
    /// Per-feature training means, subtracted during `transform`.
    pub means: Array1<f64>,
}

impl PCAInner {
    /// Fit PCA using exact SVD of the covariance matrix.
    ///
    /// This is the deterministic reference implementation. For production use,
    /// prefer `fit_randomized` which is faster for large datasets.
    ///
    /// # Algorithm
    /// 1. **Centre** — subtract column means: `X_c = X − μ`
    /// 2. **Covariance** — `C = X_cᵀ X_c / (n − 1)` (sample covariance, ddof=1)
    /// 3. **SVD** — `C = U Σ Vᵀ` via nalgebra
    /// 4. **Sign flip** — make max-abs element positive per component
    /// 5. **Variance** — singular values stored as `explained_variance`
    #[allow(dead_code)]
    fn fit_exact(data: &Array2<f64>, n_components: usize) -> Result<PCAInner, PulsarError> {
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

    /// Fit PCA using randomized SVD (Halko et al. 2011).
    ///
    /// Randomized SVD is much faster for large datasets where n_samples >> n_features
    /// or when only a few components are needed. The seed controls the random
    /// projection, so different seeds produce slightly different (but equally valid)
    /// principal components.
    ///
    /// # Algorithm
    /// 1. **Centre** — subtract column means
    /// 2. **Random projection** — `Y = X @ Omega` where Omega is d x (k+p) Gaussian
    /// 3. **QR decomposition** — orthonormalise the projected space
    /// 4. **Power iterations** — improve approximation for slowly decaying spectra
    /// 5. **Small SVD** — compute SVD of the small projected matrix
    /// 6. **Recover components** — map back to original feature space
    ///
    /// # Parameters
    /// - `data` — input matrix of shape `(n_samples, n_features)`
    /// - `n_components` — number of principal components to compute
    /// - `seed` — random seed for reproducibility
    /// - `n_oversamples` — extra dimensions for better approximation (default: 10)
    /// - `n_power_iter` — power iterations for accuracy (default: 2)
    pub fn fit_randomized(
        data: &Array2<f64>,
        n_components: usize,
        seed: u64,
        n_oversamples: usize,
        n_power_iter: usize,
    ) -> Result<PCAInner, PulsarError> {
        let (n_samples, n_features) = (data.nrows(), data.ncols());
        if n_components > n_features {
            return Err(PulsarError::InvalidParameter {
                msg: format!("n_components ({}) > n_features ({})", n_components, n_features),
            });
        }

        // Step 1: centre data
        let means = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &means;
        }

        // Sketch size: k + oversampling, capped at n_features
        let sketch_size = (n_components + n_oversamples).min(n_features);

        // Step 2: generate random Gaussian matrix Omega of shape (n_features, sketch_size)
        let mut rng = StdRng::seed_from_u64(seed);
        let omega = Array2::from_shape_fn((n_features, sketch_size), |_| {
            rng.sample::<f64, _>(StandardNormal)
        });

        // Step 3: form Y = X @ Omega, shape (n_samples, sketch_size)
        let mut y = centered.dot(&omega);

        // Step 4: power iterations to improve approximation
        // Each iteration does: Y = X @ (X.T @ Q) where Q = orth(Y)
        for _ in 0..n_power_iter {
            // QR factorization to orthonormalise Y
            let q = qr_q(&y);
            // Y = X @ (X.T @ Q)
            let z = centered.t().dot(&q);
            y = centered.dot(&z);
        }

        // Final orthonormalisation
        let q = qr_q(&y); // Q is (n_samples, sketch_size)

        // Step 5: form small matrix B = Q.T @ X, shape (sketch_size, n_features)
        let b = q.t().dot(&centered);

        // Step 6: SVD of the small matrix B
        let b_na = ndarray_to_nalgebra(&b);
        let svd = b_na.svd(false, true); // We only need V_t, not U
        let v_t = svd.v_t.ok_or(PulsarError::SvdFailed)?;
        let singular_values = svd.singular_values;

        // The right singular vectors of B are the principal components
        // V_t has shape (sketch_size, n_features), we take top n_components rows
        let v_t_nd = nalgebra_to_ndarray(&v_t);
        let mut components = v_t_nd.slice(s![..n_components, ..]).to_owned();

        // Step 7: sign convention (same as exact SVD)
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

        // Explained variance: squared singular values scaled by 1/(n-1)
        let explained_variance: Vec<f64> = (0..n_components)
            .map(|i| {
                let sv = singular_values[i];
                (sv * sv) / ((n_samples as f64) - 1.0)
            })
            .collect();

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

/// Python-facing PCA class using randomized SVD.
///
/// ```python
/// from pulsar._pulsar import PCA
///
/// pca = PCA(n_components=3, seed=42)
/// X_reduced = pca.fit_transform(X)       # fit and project in one call
/// X_new_reduced = pca.transform(X_new)   # project new data
/// variances = pca.explained_variance     # per-component variances
/// ```
///
/// Uses randomized SVD (Halko et al. 2011) which is faster for large datasets
/// and produces stochastic variation controlled by `seed`. Different seeds
/// yield slightly different (but equally valid) principal components.
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
    /// Create a new unfitted PCA with randomized SVD.
    ///
    /// # Parameters
    /// - `n_components` — number of principal components to keep.
    /// - `seed` — random seed for the stochastic projection.
    /// - `n_oversamples` — extra dimensions for approximation quality (default: 10).
    /// - `n_power_iter` — power iterations for slowly decaying spectra (default: 2).
    #[new]
    #[pyo3(signature = (n_components, seed, n_oversamples=10, n_power_iter=2))]
    pub fn new(n_components: usize, seed: u64, n_oversamples: usize, n_power_iter: usize) -> Self {
        PCA { n_components, seed, n_oversamples, n_power_iter, inner: None }
    }

    /// Fit PCA on `data` and return the low-dimensional projection.
    ///
    /// Uses randomized SVD for efficiency on large datasets.
    ///
    /// # Parameters
    /// - `data` (`np.ndarray[float64, 2D]`, shape `(n_samples, n_features)`)
    ///
    /// # Returns
    /// `np.ndarray[float64, 2D]`, shape `(n_samples, n_components)`
    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = data.as_array().to_owned();
        let inner = PCAInner::fit_randomized(
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

    /// Project new data using the components fitted by `fit_transform`.
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
        let arr = ndarray::Array1::from_vec(inner.explained_variance.clone());
        Ok(arr.into_pyarray_bound(py))
    }
}

/// Compute PCA embeddings for multiple dimensions and seeds in parallel.
///
/// This is the optimized entry point for grid search over PCA configurations.
/// For each unique seed, it computes one randomized SVD at the maximum dimension,
/// then slices to produce embeddings for all requested dimensions.
///
/// # Parameters
/// - `data` — input matrix of shape `(n_samples, n_features)`
/// - `dimensions` — list of target dimensionalities (e.g., `[2, 3, 5, 10]`)
/// - `seeds` — list of random seeds for stochastic variation
///
/// # Returns
/// List of 2D arrays in row-major order: for each seed (outer), all dimensions (inner).
/// So `pca_grid(X, [2,3], [42,7])` returns `[X_s42_d2, X_s42_d3, X_s7_d2, X_s7_d3]`.
///
/// # Performance
/// - Computes `len(seeds)` SVDs instead of `len(seeds) * len(dimensions)` SVDs
/// - Parallelised across seeds using rayon
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
    let max_dim = *dimensions.iter().max().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("dimensions list cannot be empty")
    })?;

    // Compute one PCA per seed at max dimension, then slice
    // We need to collect results because we can't hold py across thread boundaries
    let embeddings: Result<Vec<Vec<Array2<f64>>>, PulsarError> = seeds
        .par_iter()
        .map(|&seed| {
            let inner = PCAInner::fit_randomized(&arr, max_dim, seed, n_oversamples, n_power_iter)?;
            let full_proj = inner.transform(&arr)?;

            // Slice for each requested dimension
            let sliced: Vec<Array2<f64>> = dimensions
                .iter()
                .map(|&dim| full_proj.slice(s![.., ..dim]).to_owned())
                .collect();

            Ok(sliced)
        })
        .collect();

    let embeddings = embeddings?;

    // Flatten and convert to Python arrays
    let result: Vec<Bound<'py, PyArray2<f64>>> = embeddings
        .into_iter()
        .flatten()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect();

    Ok(result)
}
