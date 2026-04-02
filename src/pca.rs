use ndarray::{s, Array1, Array2, Axis};
use nalgebra::DMatrix;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::error::PulsarError;

/// Convert an ndarray `Array2<f64>` to a nalgebra `DMatrix<f64>`.
fn ndarray_to_nalgebra(a: &Array2<f64>) -> DMatrix<f64> {
    let (nrows, ncols) = (a.nrows(), a.ncols());
    let data: Vec<f64> = a.as_standard_layout().iter().cloned().collect();
    DMatrix::from_row_slice(nrows, ncols, &data)
}

/// Convert a nalgebra `DMatrix<f64>` back to an ndarray `Array2<f64>`.
fn nalgebra_to_ndarray(m: &DMatrix<f64>) -> Array2<f64> {
    let (nrows, ncols) = (m.nrows(), m.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| m[(i, j)])
}

/// Compute the Q factor of a thin QR decomposition.
fn qr_q(a: &Array2<f64>) -> Array2<f64> {
    let a_na = ndarray_to_nalgebra(a);
    let qr = a_na.qr();
    let q = qr.q();
    let k = a.nrows().min(a.ncols());
    let thin_q = q.columns(0, k);
    let (nrows, ncols) = (thin_q.nrows(), thin_q.ncols());
    Array2::from_shape_fn((nrows, ncols), |(i, j)| thin_q[(i, j)])
}

/// Internal PCA state stored after fitting.
///
/// Uses randomized SVD (Halko et al. 2011) optimized for large datasets.
pub struct PCAInner {
    /// Principal component directions, shape `(n_components, n_features)`.
    /// Each row is a unit vector in feature space. Signs are normalised so
    /// that the largest-absolute-value element of each component is positive.
    pub components: Array2<f64>,
    /// Explained variance per component, in descending order.
    pub explained_variance: Vec<f64>,
    /// Per-feature training means, subtracted during `transform`.
    pub means: Array1<f64>,
}

impl PCAInner {
    /// Fit PCA using randomized SVD (Halko et al. 2011).
    ///
    /// Optimized for large datasets where n_samples >> n_features. Complexity
    /// is O(n * d * k) instead of O(n * d² + d³) for exact SVD.
    ///
    /// The seed controls the random projection, so different seeds produce
    /// slightly different (but equally valid) principal components - this is
    /// intentional for ensemble diversity in the Pulsar pipeline.
    ///
    /// # Algorithm
    /// 1. Centre data by subtracting column means
    /// 2. Random projection: Y = X @ Omega (Gaussian random matrix)
    /// 3. QR decomposition to orthonormalise
    /// 4. Power iterations for accuracy on slowly decaying spectra
    /// 5. Small SVD of projected matrix
    /// 6. Recover components in original feature space
    pub fn fit(
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

        if n_samples < 2 {
            return Err(PulsarError::InvalidParameter {
                msg: format!(
                    "PCA requires at least 2 samples, got {}",
                    n_samples
                ),
            });
        }

        if n_features == 0 {
            return Err(PulsarError::InvalidParameter {
                msg: "input data has 0 features (columns)".to_string(),
            });
        }

        // Step 1: centre data
        let means = data.mean_axis(Axis(0)).expect("non-empty axis guaranteed above");
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &means;
        }

        // Sketch size: k + oversampling, capped at n_features
        let sketch_size = (n_components + n_oversamples).min(n_features);

        // Step 2: random Gaussian projection matrix
        let mut rng = StdRng::seed_from_u64(seed);
        let omega = Array2::from_shape_fn((n_features, sketch_size), |_| {
            rng.sample::<f64, _>(StandardNormal)
        });

        // Step 3: Y = X @ Omega
        let mut y = centered.dot(&omega);

        // Step 4: power iterations
        for _ in 0..n_power_iter {
            let q = qr_q(&y);
            let z = centered.t().dot(&q);
            y = centered.dot(&z);
        }

        // Final orthonormalisation
        let q = qr_q(&y);

        // Step 5: small matrix B = Q.T @ X
        let b = q.t().dot(&centered);

        // Step 6: SVD of small matrix
        let b_na = ndarray_to_nalgebra(&b);
        let svd = b_na.svd(false, true);
        let v_t = svd.v_t.ok_or(PulsarError::SvdFailed)?;
        let singular_values = svd.singular_values;

        let v_t_nd = nalgebra_to_ndarray(&v_t);
        let n_sv = singular_values.len();
        let k = n_components.min(n_sv);
        let mut components = v_t_nd.slice(s![..k, ..]).to_owned();

        // Detect NaN in SVD output — indicates the decomposition failed to converge
        for i in 0..k {
            if components.row(i).iter().any(|v| v.is_nan()) {
                return Err(PulsarError::SvdFailed);
            }
        }

        // Sign convention: largest-abs-value element must be positive
        for i in 0..k {
            let row = components.row(i);
            // NaN-free guaranteed by the check above, so unwrap is safe
            let max_abs_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).expect("NaN excluded above"))
                .map(|(idx, _)| idx)
                .unwrap();
            if row[max_abs_idx] < 0.0 {
                components.row_mut(i).mapv_inplace(|x| -x);
            }
        }

        let explained_variance: Vec<f64> = (0..k)
            .map(|i| {
                let sv = singular_values[i];
                (sv * sv) / ((n_samples as f64) - 1.0)
            })
            .collect();

        Ok(PCAInner { components, explained_variance, means })
    }

    /// Project data onto the fitted principal components.
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
        let arr = ndarray::Array1::from_vec(inner.explained_variance.clone());
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
    let n_features = arr.ncols();
    let max_dim = *dimensions.iter().max().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("dimensions list cannot be empty")
    })?;

    if max_dim > n_features {
        return Err(PulsarError::InvalidParameter {
            msg: format!(
                "requested dimension {} exceeds number of features ({})",
                max_dim, n_features
            ),
        }
        .into());
    }

    let embeddings: Result<Vec<Vec<Array2<f64>>>, PulsarError> = seeds
        .par_iter()
        .map(|&seed| {
            let inner = PCAInner::fit(&arr, max_dim, seed, n_oversamples, n_power_iter)?;
            let full_proj = inner.transform(&arr)?;

            let sliced: Vec<Array2<f64>> = dimensions
                .iter()
                .map(|&dim| full_proj.slice(s![.., ..dim]).to_owned())
                .collect();

            Ok(sliced)
        })
        .collect();

    let embeddings = embeddings?;

    let result: Vec<Bound<'py, PyArray2<f64>>> = embeddings
        .into_iter()
        .flatten()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect();

    Ok(result)
}
