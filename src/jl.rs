use ndarray::{s, Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::error::PulsarError;

/// Johnson-Lindenstrauss Gaussian random projection.
pub struct JLProjectionInner {
    components: Array2<f64>,
    means: Array1<f64>,
    center: bool,
}

impl JLProjectionInner {
    pub fn fit(
        data: &Array2<f64>,
        n_components: usize,
        seed: u64,
        center: bool,
    ) -> Result<Self, PulsarError> {
        let n_features = data.ncols();
        if n_components == 0 {
            return Err(PulsarError::InvalidParameter {
                msg: "n_components must be positive".to_string(),
            });
        }
        if n_features == 0 {
            return Err(PulsarError::InvalidParameter {
                msg: "input data has 0 features (columns)".to_string(),
            });
        }
        let means = if center {
            data.mean_axis(Axis(0))
                .expect("non-empty axis guaranteed above")
        } else {
            Array1::zeros(n_features)
        };
        let components = gaussian_components(n_features, n_components, seed, n_components);
        Ok(Self {
            components,
            means,
            center,
        })
    }

    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PulsarError> {
        if data.ncols() != self.means.len() {
            return Err(PulsarError::ShapeMismatch {
                expected: format!("{} features", self.means.len()),
                got: format!("{} features", data.ncols()),
            });
        }
        if self.center {
            let mut centered = data.clone();
            for mut row in centered.rows_mut() {
                row -= &self.means;
            }
            Ok(centered.dot(&self.components))
        } else {
            Ok(data.dot(&self.components))
        }
    }
}

fn gaussian_components(
    n_features: usize,
    width: usize,
    seed: u64,
    scale_dim: usize,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = 1.0 / (scale_dim as f64).sqrt();
    Array2::from_shape_fn((n_features, width), |_| {
        rng.sample::<f64, _>(StandardNormal) * scale
    })
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
    let arr = data.as_array().to_owned();
    let n_features = arr.ncols();
    let max_dim = *dimensions.iter().max().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("dimensions list cannot be empty")
    })?;
    if max_dim == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "dimensions must be positive".to_string(),
        }
        .into());
    }
    if n_features == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "input data has 0 features (columns)".to_string(),
        }
        .into());
    }

    let means = if center {
        arr.mean_axis(Axis(0))
            .expect("non-empty axis guaranteed above")
    } else {
        Array1::zeros(n_features)
    };

    let embeddings: Vec<Vec<Array2<f64>>> = seeds
        .par_iter()
        .map(|&seed| {
            let full_components = gaussian_components(n_features, max_dim, seed, max_dim);
            dimensions
                .iter()
                .map(|&dim| {
                    let mut components = full_components.slice(s![.., ..dim]).to_owned();
                    let rescale = (max_dim as f64 / dim as f64).sqrt();
                    components.mapv_inplace(|x| x * rescale);
                    if center {
                        let mut centered = arr.clone();
                        for mut row in centered.rows_mut() {
                            row -= &means;
                        }
                        centered.dot(&components)
                    } else {
                        arr.dot(&components)
                    }
                })
                .collect()
        })
        .collect();

    Ok(embeddings
        .into_iter()
        .flatten()
        .map(|arr| arr.into_pyarray_bound(py))
        .collect())
}
