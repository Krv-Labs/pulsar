use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix2};
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

/// Compute JL embeddings for multiple dimensions and seeds in parallel.
///
/// Returns arrays in row-major grid order: seed outer, dimensions inner.
/// Borrows the input, including strided views; centering allocates one copy.
pub fn jl_grid<S: Data<Elem = f64>>(
    data: &ArrayBase<S, Ix2>,
    dimensions: &[usize],
    seeds: &[u64],
    center: bool,
) -> Result<Vec<Array2<f64>>, PulsarError> {
    let n_features = data.ncols();
    let max_dim = *dimensions
        .iter()
        .max()
        .ok_or_else(|| PulsarError::InvalidParameter {
            msg: "dimensions list cannot be empty".to_string(),
        })?;
    if max_dim == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "dimensions must be positive".to_string(),
        });
    }
    if n_features == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "input data has 0 features (columns)".to_string(),
        });
    }

    // Centering is invariant to seed and dimension, so do it once up front
    // rather than cloning + re-centering inside every grid cell.
    let centered = if center {
        let means = data
            .mean_axis(Axis(0))
            .expect("non-empty axis guaranteed above");
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &means;
        }
        Some(centered)
    } else {
        None
    };
    let source = match &centered {
        Some(centered) => centered.view(),
        None => data.view(),
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
                    source.dot(&components)
                })
                .collect()
        })
        .collect();

    Ok(embeddings.into_iter().flatten().collect())
}
