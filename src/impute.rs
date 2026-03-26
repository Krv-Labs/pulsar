use std::collections::HashMap;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::prelude::*;
use rand_distr::{Normal, WeightedIndex};

use crate::error::PulsarError;

/// Fill missing values (`NaN`) in a single column of `f64` values **in place**.
///
/// # Parameters
/// - `values` — mutable slice representing one column; NaN entries are filled,
///   non-NaN entries are left untouched.
/// - `method` — imputation strategy (see below).
/// - `seed` — integer seed used only by the two sampling methods; ignored for
///   deterministic methods (`fill_mean`, `fill_median`, `fill_mode`).
///
/// # Supported methods
///
/// | Method | Description |
/// |---|---|
/// | `"sample_normal"` | Draw replacements from N(μ, σ) fitted to observed values. Uses **population std** (ddof=0). |
/// | `"sample_categorical"` | Sample from the empirical distribution of observed values (weighted by frequency). Categories are sorted by value before sampling to guarantee the same result for the same seed regardless of HashMap iteration order. |
/// | `"fill_mean"` | Replace all NaN with the column mean. |
/// | `"fill_median"` | Replace all NaN with the column median (average of two middle values for even-length arrays). |
/// | `"fill_mode"` | Replace all NaN with the most frequent observed value. |
///
/// # Errors
/// - [`PulsarError::EmptyInput`] — every value is NaN; statistics cannot be computed.
/// - [`PulsarError::InvalidParameter`] — unknown method name.
pub fn impute_column_inplace(
    values: &mut [f64],
    method: &str,
    seed: u64,
) -> Result<(), PulsarError> {
    let observed: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();

    if observed.is_empty() {
        return Err(PulsarError::EmptyInput {
            msg: "all values are NaN; cannot impute",
        });
    }

    let nan_indices: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if v.is_nan() { Some(i) } else { None })
        .collect();

    if nan_indices.is_empty() {
        return Ok(());
    }

    match method {
        "sample_normal" => {
            // Fit a Gaussian to the observed values and draw replacements.
            // Population std (ddof=0) is used for consistency with the rest of
            // the pipeline (StandardScaler also uses ddof=0).
            let n = observed.len() as f64;
            let mean = observed.iter().sum::<f64>() / n;
            let variance = observed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
            // Guard against zero-variance columns: clamp std to a small value
            // so the Normal distribution can still be constructed.
            let std = if std < 1e-10 { 1e-10 } else { std };

            let dist = Normal::new(mean, std).map_err(|_| PulsarError::InvalidParameter {
                msg: "could not create Normal distribution".into(),
            })?;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            for &i in &nan_indices {
                values[i] = dist.sample(&mut rng);
            }
        }
        "sample_categorical" => {
            // Build an empirical distribution over observed values and sample from it.
            //
            // IMPORTANT: HashMap iteration order is non-deterministic across
            // program runs.  To make sampling reproducible for a given seed,
            // we sort the unique category values numerically before building
            // the WeightedIndex.  Without this sort, two calls with the same
            // seed could draw from a different ordering of the categories.
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for &v in &observed {
                *counts.entry(v.to_bits()).or_insert(0) += 1;
            }
            let mut categories: Vec<f64> = counts.keys().map(|&b| f64::from_bits(b)).collect();
            categories.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let weights: Vec<usize> = categories.iter().map(|v| counts[&v.to_bits()]).collect();

            let dist = WeightedIndex::new(&weights).map_err(|_| PulsarError::InvalidParameter {
                msg: "could not create WeightedIndex for categorical sampling".into(),
            })?;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            for &i in &nan_indices {
                values[i] = categories[dist.sample(&mut rng)];
            }
        }
        "fill_mean" => {
            let mean = observed.iter().sum::<f64>() / observed.len() as f64;
            for &i in &nan_indices {
                values[i] = mean;
            }
        }
        "fill_median" => {
            let mut sorted = observed.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            let median = if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            };
            for &i in &nan_indices {
                values[i] = median;
            }
        }
        "fill_mode" => {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for &v in &observed {
                *counts.entry(v.to_bits()).or_insert(0) += 1;
            }
            let mode_bits = counts
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(&b, _)| b)
                .unwrap();
            let mode = f64::from_bits(mode_bits);
            for &i in &nan_indices {
                values[i] = mode;
            }
        }
        other => {
            return Err(PulsarError::InvalidParameter {
                msg: format!(
                    "unknown imputation method '{}'; expected one of: \
                     sample_normal, sample_categorical, fill_mean, fill_median, fill_mode",
                    other
                ),
            });
        }
    }

    Ok(())
}

/// Python-facing wrapper around [`impute_column_inplace`].
///
/// Clones the input array, fills NaN values using the chosen method, and
/// returns a new array.  The original array is **not** modified.
///
/// # Parameters (Python)
/// - `values` (`np.ndarray[float64, 1D]`) — column to impute.
/// - `method` (`str`) — one of `"sample_normal"`, `"sample_categorical"`,
///   `"fill_mean"`, `"fill_median"`, `"fill_mode"`.
/// - `seed` (`int`, default `0`) — RNG seed; only used by `"sample_normal"`
///   and `"sample_categorical"`.
///
/// # Returns
/// A new `np.ndarray[float64, 1D]` with NaN values replaced.
///
/// # Raises
/// `ValueError` — if all values are NaN or the method name is unrecognised.
#[pyfunction]
#[pyo3(signature = (values, method, seed=0))]
pub fn impute_column<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    method: &str,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut arr = values.as_array().to_owned();
    impute_column_inplace(arr.as_slice_mut().unwrap(), method, seed)?;
    Ok(arr.into_pyarray_bound(py))
}
