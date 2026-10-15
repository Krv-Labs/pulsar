use std::collections::HashMap;
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
            let median = if n.is_multiple_of(2) {
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
                     sample_normal, sample_categorical, fill_mean, fill_median, fill_mode, fill_value",
                    other
                ),
            });
        }
    }

    Ok(())
}

/// Fill missing values (`NaN`) in a single column of `f64` values **in place**
/// with a finite constant `value`.
///
/// Works even if every entry in `values` is `NaN` (no observed values required).
///
/// # Errors
/// - [`PulsarError::InvalidParameter`] — `value` is not finite (`NaN` or `±∞`).
pub fn fill_constant_inplace(values: &mut [f64], value: f64) -> Result<(), PulsarError> {
    if !value.is_finite() {
        return Err(PulsarError::InvalidParameter {
            msg: format!("fill_value requires a finite value, got {value}"),
        });
    }
    for v in values.iter_mut() {
        if v.is_nan() {
            *v = value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_constant_inplace_replaces_nans() {
        let mut vals = [1.0, f64::NAN, 3.5, f64::NAN, -2.0];
        fill_constant_inplace(&mut vals, 0.0).unwrap();
        assert_eq!(vals, [1.0, 0.0, 3.5, 0.0, -2.0]);
    }

    #[test]
    fn test_fill_constant_inplace_all_nan() {
        let mut vals = [f64::NAN, f64::NAN, f64::NAN];
        fill_constant_inplace(&mut vals, -1.0).unwrap();
        assert_eq!(vals, [-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_fill_constant_inplace_no_nans_and_empty() {
        let mut vals = [1.0, 2.0, 3.0];
        fill_constant_inplace(&mut vals, 99.0).unwrap();
        assert_eq!(vals, [1.0, 2.0, 3.0]);

        let mut empty: [f64; 0] = [];
        fill_constant_inplace(&mut empty, 0.0).unwrap();
    }

    #[test]
    fn test_fill_constant_inplace_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut vals = [1.0, f64::NAN];
            let err = fill_constant_inplace(&mut vals, bad).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("fill_value requires a finite value"),
                "unexpected error message: {msg}"
            );
        }
    }

    #[test]
    fn test_unknown_method_error_lists_fill_value() {
        let mut vals = [1.0, f64::NAN, 3.0];
        let err = impute_column_inplace(&mut vals, "bogus", 0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown imputation method 'bogus'"));
        assert!(msg.contains("fill_value"));
    }
}

