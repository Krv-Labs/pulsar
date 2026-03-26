use std::collections::HashMap;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::prelude::*;
use rand_distr::{Normal, WeightedIndex};

use crate::error::PulsarError;

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
            let n = observed.len() as f64;
            let mean = observed.iter().sum::<f64>() / n;
            let variance = observed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
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
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for &v in &observed {
                *counts.entry(v.to_bits()).or_insert(0) += 1;
            }
            // Sort categories by value for deterministic ordering
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
