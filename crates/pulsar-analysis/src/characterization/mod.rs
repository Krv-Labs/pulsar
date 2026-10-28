//! Dataset characterization: raw geometry (k-NN distances, PCA variance) and
//! per-column profiles (`pulsar/analysis/characterization.py`).
//!
//! Pure measurement: no thresholds, no recommendations. Geometry is measured
//! on the numeric columns after mean imputation and z-scoring.
//!
//! Differences from the Python reference, none of which change results on
//! tables of up to `subsample` rows (the default is 1000):
//!
//! * Above `subsample` rows the row sample is drawn with a different
//!   generator (numpy's PCG64 is not reproduced).
//! * k-NN distances are exact Euclidean; scikit-learn's brute-force path
//!   (more than 15 features) computes them from dot products, so the two
//!   agree to about 1e-8, not bit for bit.
//! * A boolean column with blanks is an `object` column in pandas (neither
//!   numeric nor `bool`); it is treated that way here too.

mod payload;

use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};
use pulsar_data::preprocess::py_float_repr;
use pulsar_data::{Column, Table};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use payload::{
    characterization_payload_to_markdown, compact_characterization_payload, probe_columns_payload,
    probe_columns_payload_to_markdown,
};

use crate::pyround::round_to;

/// k-NN distances stay finite and PCA stays defined above this many rows.
const MIN_ROWS: usize = 2;

#[derive(Debug, Error)]
pub enum CharacterizationError {
    #[error("Need at least 2 numeric columns for geometry analysis, found {found}.")]
    TooFewNumericColumns { found: usize },
    #[error("Need at least {MIN_ROWS} rows to measure neighbour distances, found {found}.")]
    TooFewRows { found: usize },
    #[error("No numeric column has an observed value, so there is no geometry to measure.")]
    NoObservedValues,
    #[error(transparent)]
    Core(#[from] pulsar_core::PulsarError),
}

impl CharacterizationError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::TooFewNumericColumns { .. } => "TOO_FEW_NUMERIC_COLUMNS",
            Self::TooFewRows { .. } => "TOO_FEW_ROWS",
            Self::NoObservedValues => "NO_OBSERVED_VALUES",
            Self::Core(_) => "CHARACTERIZATION_FAILED",
        }
    }
}

/// Per-column metadata for deciding how to treat a column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: String,
    pub dtype: String,
    pub is_numeric: bool,
    pub n_unique: usize,
    pub n_missing: usize,
    pub missing_pct: f64,
    pub sample_values: Vec<String>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min_val: Option<f64>,
    pub max_val: Option<f64>,
    pub top_values: Option<Vec<(String, usize)>>,
}

/// k-NN and PCA geometry of a numeric matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericProfile {
    pub knn_k5_mean: f64,
    pub knn_k10_mean: f64,
    pub knn_k20_mean: f64,
    pub knn_p5: f64,
    pub knn_p25: f64,
    pub knn_p50: f64,
    pub knn_p75: f64,
    pub knn_p95: f64,
    pub pca_cumulative_variance: Vec<(usize, f64)>,
    pub n_features: usize,
    pub n_samples_profiled: usize,
}

/// Raw measurements only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetProfile {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_columns_total: usize,
    pub missingness_pct: f64,
    pub knn_k5_mean: f64,
    pub knn_k10_mean: f64,
    pub knn_k20_mean: f64,
    pub pca_cumulative_variance: Vec<(usize, f64)>,
    pub column_profiles: Vec<ColumnProfile>,
}

// ---- numpy-compatible helpers ---------------------------------------------------

/// `np.percentile(sorted, q)` with numpy's default linear interpolation and
/// its `lerp` rounding (so the result matches to the last digit).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = (n - 1) as f64 * (q / 100.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let t = pos - lo as f64;
    let (a, b) = (sorted[lo], sorted[hi]);
    let diff = b - a;
    let mut out = a + diff * t;
    if t >= 0.5 {
        out = b - diff * (1.0 - t);
    }
    out
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population variance of a column (`np.var`, ddof = 0).
fn variance_of(col: ndarray::ArrayView1<f64>) -> f64 {
    let n = col.len() as f64;
    let m = col.sum() / n;
    col.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n
}

// ---- geometry --------------------------------------------------------------------

/// The `k_max` smallest distances from each row to every row (itself
/// included, at distance 0), ascending. Exact Euclidean; ties need no
/// tie-breaking since only distances are kept.
fn knn_distances(x: ArrayView2<f64>, k_max: usize) -> Vec<Vec<f64>> {
    let n = x.nrows();
    (0..n)
        .into_par_iter()
        .map_init(
            || Vec::<f64>::with_capacity(n),
            |buf, i| {
                buf.clear();
                let ri = x.row(i);
                buf.extend((0..n).map(|j| {
                    ri.iter()
                        .zip(x.row(j))
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt()
                }));
                buf.select_nth_unstable_by(k_max - 1, f64::total_cmp);
                let mut near = buf[..k_max].to_vec();
                near.sort_by(f64::total_cmp);
                near
            },
        )
        .collect()
}

/// Mean k-NN distance using whatever neighbours are available.
fn bounded_knn_mean(distances: &[Vec<f64>], k: usize) -> f64 {
    let available = distances[0].len() - 1;
    if available == 0 {
        return 0.0;
    }
    let upper = k.min(available);
    let total: f64 = distances
        .iter()
        .map(|d| d[1..=upper].iter().sum::<f64>())
        .sum();
    total / (distances.len() * upper) as f64
}

/// k-NN distances and PCA variance of an already imputed and scaled matrix.
pub fn profile_numeric_matrix(
    x: ArrayView2<f64>,
    subsample: usize,
    seed: u64,
    dims_to_probe: Option<&[usize]>,
) -> Result<NumericProfile, CharacterizationError> {
    let n = x.nrows();
    let n_sub = subsample.min(n);
    let x_sub: Array2<f64> = if n_sub < n {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut idx = rand::seq::index::sample(&mut rng, n, n_sub).into_vec();
        idx.sort_unstable();
        x.select(ndarray::Axis(0), &idx)
    } else {
        x.to_owned()
    };
    if n_sub < MIN_ROWS {
        return Err(CharacterizationError::TooFewRows { found: n_sub });
    }

    let k_max = 21.min(n_sub - 1);
    let distances = knn_distances(x_sub.view(), k_max);
    let (k5, k10, k20) = (
        bounded_knn_mean(&distances, 5),
        bounded_knn_mean(&distances, 10),
        bounded_knn_mean(&distances, 20),
    );

    // k-NN distance percentiles (k = 5 across all points): the valid
    // epsilon domain for Ball Mapper.
    let k5_upper = 5.min(k_max - 1);
    let (p5, p25, p50, p75, p95) = if k5_upper > 0 {
        let mut col: Vec<f64> = distances.iter().map(|d| d[k5_upper]).collect();
        col.sort_by(f64::total_cmp);
        (
            percentile(&col, 5.0),
            percentile(&col, 25.0),
            percentile(&col, 50.0),
            percentile(&col, 75.0),
            percentile(&col, 95.0),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };

    let n_features = x_sub.ncols();
    // PCA cannot keep more components than there are rows or columns.
    let max_dim = n_features.min(n_sub);
    let dims: Vec<usize> = match dims_to_probe {
        Some(d) => d.to_vec(),
        None => {
            let d: Vec<usize> = [2, 3, 5, 10, 15, 20]
                .into_iter()
                .filter(|&d| d <= max_dim)
                .collect();
            if d.is_empty() {
                vec![2.min(max_dim)]
            } else {
                d
            }
        }
    };
    let embeddings = pulsar_core::pca_grid(&x_sub, &dims, &[seed], 10, 2)?;
    let total_variance: f64 = (0..n_features).map(|j| variance_of(x_sub.column(j))).sum();
    let pca_cumulative_variance = dims
        .iter()
        .zip(&embeddings)
        .map(|(&dim, emb)| {
            let ev: f64 = (0..emb.ncols()).map(|j| variance_of(emb.column(j))).sum();
            (dim, round_to((ev / total_variance.max(1e-12)).min(1.0), 4))
        })
        .collect();

    Ok(NumericProfile {
        knn_k5_mean: k5,
        knn_k10_mean: k10,
        knn_k20_mean: k20,
        knn_p5: p5,
        knn_p25: p25,
        knn_p50: p50,
        knn_p75: p75,
        knn_p95: p95,
        pca_cumulative_variance,
        n_features,
        n_samples_profiled: n_sub,
    })
}

// ---- columns ---------------------------------------------------------------------

/// A column's non-missing values as display strings, with their type.
struct Observed {
    /// `str(v)` of each non-missing value, in row order.
    text: Vec<String>,
    /// The same values as numbers, for numeric columns.
    numbers: Option<Vec<f64>>,
    dtype: &'static str,
    is_numeric: bool,
    n_rows: usize,
}

fn observed(column: &Column) -> Observed {
    match column {
        Column::Float(v) => {
            let numbers: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
            Observed {
                text: numbers.iter().map(|&x| py_float_repr(x)).collect(),
                numbers: Some(numbers),
                dtype: "float64",
                is_numeric: true,
                n_rows: v.len(),
            }
        }
        Column::Int(v) => Observed {
            text: v.iter().map(i64::to_string).collect(),
            numbers: Some(v.iter().map(|&x| x as f64).collect()),
            dtype: "int64",
            is_numeric: true,
            n_rows: v.len(),
        },
        Column::Bool(v) => {
            let has_blank = v.iter().any(Option::is_none);
            let present: Vec<bool> = v.iter().flatten().copied().collect();
            Observed {
                text: present
                    .iter()
                    .map(|&b| if b { "True" } else { "False" }.to_string())
                    .collect(),
                // With blanks pandas stores the column as `object`, which is not numeric.
                numbers: (!has_blank)
                    .then(|| present.iter().map(|&b| f64::from(u8::from(b))).collect()),
                dtype: if has_blank { "object" } else { "bool" },
                is_numeric: !has_blank,
                n_rows: v.len(),
            }
        }
        Column::Text(v) => Observed {
            text: v.iter().flatten().cloned().collect(),
            numbers: None,
            dtype: "str",
            is_numeric: false,
            n_rows: v.len(),
        },
    }
}

/// Distinct values in order of first appearance, with their counts.
fn counts_in_order(text: &[String]) -> Vec<(String, usize)> {
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut out: Vec<(String, usize)> = Vec::new();
    for t in text {
        match index.get(t.as_str()) {
            Some(&i) => out[i].1 += 1,
            None => {
                index.insert(t, out.len());
                out.push((t.clone(), 1));
            }
        }
    }
    out
}

/// A cheap profile of every column (the "map" view): counts only, none of
/// the per-value fields.
fn profile_columns(table: &Table) -> Vec<ColumnProfile> {
    let n_rows = table.n_rows();
    table
        .names
        .iter()
        .zip(&table.columns)
        .map(|(name, col)| {
            let obs = observed(col);
            let n_missing = n_rows - obs.text.len();
            ColumnProfile {
                name: name.clone(),
                dtype: obs.dtype.into(),
                is_numeric: obs.is_numeric,
                n_unique: counts_in_order(&obs.text).len(),
                n_missing,
                missing_pct: if n_rows > 0 {
                    round_to(n_missing as f64 / n_rows as f64 * 100.0, 2)
                } else {
                    0.0
                },
                sample_values: vec![],
                mean: None,
                std: None,
                min_val: None,
                max_val: None,
                top_values: None,
            }
        })
        .collect()
}

/// A rich profile of one column (the "magnifying glass" view).
pub fn profile_column_details(
    table: &Table,
    name: &str,
    max_sample: usize,
) -> Option<ColumnProfile> {
    let col = table.column(name)?;
    let obs = observed(col);
    let counts = counts_in_order(&obs.text);
    let n_missing = obs.n_rows - obs.text.len();
    let n_unique = counts.len();
    let sample_values: Vec<String> = counts
        .iter()
        .take(max_sample)
        .map(|(v, _)| v.clone())
        .collect();
    // value_counts: most frequent first, ties in order of first appearance.
    let top = |limit: usize| -> Vec<(String, usize)> {
        let mut c = counts.clone();
        c.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        c.truncate(limit);
        c
    };

    let (mut m, mut std, mut lo, mut hi, mut top_values) = (None, None, None, None, None);
    match (&obs.numbers, obs.is_numeric) {
        (Some(nums), true) if !nums.is_empty() => {
            let avg = mean(nums);
            m = Some(round_to(avg, 4));
            std = (nums.len() > 1).then(|| {
                let var = nums.iter().map(|v| (v - avg) * (v - avg)).sum::<f64>()
                    / (nums.len() - 1) as f64;
                round_to(var.sqrt(), 4)
            });
            lo = Some(round_to(
                nums.iter().copied().fold(f64::INFINITY, f64::min),
                4,
            ));
            hi = Some(round_to(
                nums.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                4,
            ));
            if n_unique <= 20 {
                top_values = Some(top(10));
            }
        }
        (_, false) => top_values = Some(top(10)),
        _ => {}
    }

    Some(ColumnProfile {
        name: name.to_string(),
        dtype: obs.dtype.into(),
        is_numeric: obs.is_numeric,
        n_unique,
        n_missing,
        missing_pct: if obs.n_rows > 0 {
            round_to(n_missing as f64 / obs.n_rows as f64 * 100.0, 2)
        } else {
            0.0
        },
        sample_values,
        mean: m,
        std,
        min_val: lo,
        max_val: hi,
        top_values,
    })
}

// ---- dataset ---------------------------------------------------------------------

/// Probe a table's geometry and schema before fitting.
pub fn characterize_dataset(
    table: &Table,
    subsample: usize,
    seed: u64,
) -> Result<DatasetProfile, CharacterizationError> {
    let column_profiles = profile_columns(table);
    // `select_dtypes(np.number)`: floats and integers; booleans are not numeric here.
    let numeric: Vec<&Column> = table
        .columns
        .iter()
        .filter(|c| matches!(c, Column::Float(_) | Column::Int(_)))
        .collect();
    if numeric.len() < 2 {
        return Err(CharacterizationError::TooFewNumericColumns {
            found: numeric.len(),
        });
    }

    let n = table.n_rows();
    let values: Vec<Vec<f64>> = numeric
        .iter()
        .map(|c| match c {
            Column::Float(v) => v.clone(),
            Column::Int(v) => v.iter().map(|&x| x as f64).collect(),
            _ => unreachable!("filtered to numeric columns"),
        })
        .collect();
    let missingness_pct = mean(
        &values
            .iter()
            .map(|v| v.iter().filter(|x| x.is_nan()).count() as f64 / n as f64)
            .collect::<Vec<_>>(),
    ) * 100.0;

    // Mean imputation (columns with no observed value are dropped, as
    // SimpleImputer does) then z-scoring (zero-variance columns become 0).
    let kept: Vec<Vec<f64>> = values
        .iter()
        .filter_map(|col| {
            let observed: Vec<f64> = col.iter().copied().filter(|x| !x.is_nan()).collect();
            if observed.is_empty() {
                return None;
            }
            let fill = mean(&observed);
            let imputed: Vec<f64> = col
                .iter()
                .map(|&x| if x.is_nan() { fill } else { x })
                .collect();
            let m = mean(&imputed);
            let sd = (imputed.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64).sqrt();
            let scale = if sd > 0.0 { sd } else { 1.0 };
            Some(imputed.iter().map(|v| (v - m) / scale).collect())
        })
        .collect();
    if kept.is_empty() {
        return Err(CharacterizationError::NoObservedValues);
    }
    let x = Array2::from_shape_fn((n, kept.len()), |(i, j)| kept[j][i]);

    let geo = profile_numeric_matrix(x.view(), subsample, seed, None)?;
    Ok(DatasetProfile {
        n_samples: n,
        n_features: numeric.len(),
        n_columns_total: table.n_cols(),
        missingness_pct,
        knn_k5_mean: geo.knn_k5_mean,
        knn_k10_mean: geo.knn_k10_mean,
        knn_k20_mean: geo.knn_k20_mean,
        pca_cumulative_variance: geo.pca_cumulative_variance,
        column_profiles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn percentile_matches_numpy_linear_interpolation() {
        // np.percentile([1, 2, 3, 4, 10], [5, 25, 50, 75, 95])
        let v = [1.0, 2.0, 3.0, 4.0, 10.0];
        let got: Vec<f64> = [5.0, 25.0, 50.0, 75.0, 95.0]
            .iter()
            .map(|&q| percentile(&v, q))
            .collect();
        for (g, w) in got.iter().zip([1.2, 2.0, 3.0, 4.0, 8.8]) {
            assert!((g - w).abs() < 1e-12, "{g} vs {w}");
        }
        assert_eq!(percentile(&[7.0], 33.0), 7.0);
    }

    #[test]
    fn knn_means_use_whatever_neighbours_exist() {
        // Points on a line: 0, 1, 3, 7.
        let x = array![[0.0], [1.0], [3.0], [7.0]];
        let p = profile_numeric_matrix(x.view(), 1000, 42, Some(&[1])).unwrap();
        // Python asks for min(21, n - 1) = 3 neighbours *including the point
        // itself*, so only 2 true neighbours exist and every k in {5, 10, 20}
        // averages those two. The nearest two distances per row are
        // (1, 3), (1, 2), (2, 3), (4, 6): 22 over 8 values.
        let expected = (4.0 + 3.0 + 5.0 + 10.0) / 8.0;
        for mean in [p.knn_k5_mean, p.knn_k10_mean, p.knn_k20_mean] {
            assert!((mean - expected).abs() < 1e-12, "{mean} vs {expected}");
        }
        assert_eq!((p.n_features, p.n_samples_profiled), (1, 4));
        assert_eq!(
            p.pca_cumulative_variance,
            vec![(1, 1.0)],
            "one axis holds all the variance"
        );
    }

    #[test]
    fn pca_variance_recovers_the_dominant_axis() {
        // Variance 100 on axis 0 and 1 on axis 1: one component keeps ~99%.
        let x = Array2::from_shape_fn((60, 2), |(i, j)| {
            let t = (i as f64 - 29.5) / 17.0;
            if j == 0 {
                t * 10.0
            } else {
                ((i * 7919) % 13) as f64 / 13.0 - 0.5
            }
        });
        let p = profile_numeric_matrix(x.view(), 1000, 42, Some(&[1, 2])).unwrap();
        assert!(
            p.pca_cumulative_variance[0].1 > 0.98,
            "{:?}",
            p.pca_cumulative_variance
        );
        assert_eq!(p.pca_cumulative_variance[1].1, 1.0);
    }

    #[test]
    fn edge_cases_raise_specific_errors() {
        let one_row = array![[1.0, 2.0]];
        assert_eq!(
            profile_numeric_matrix(one_row.view(), 1000, 42, None)
                .unwrap_err()
                .to_string(),
            "Need at least 2 rows to measure neighbour distances, found 1."
        );
        let table = Table::new(
            vec!["a".into(), "s".into()],
            vec![
                Column::Float(vec![1.0, 2.0]),
                Column::Text(vec![Some("x".into()), None]),
            ],
        )
        .unwrap();
        let err = characterize_dataset(&table, 1000, 42).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Need at least 2 numeric columns for geometry analysis, found 1."
        );
        assert_eq!(err.code(), "TOO_FEW_NUMERIC_COLUMNS");
    }

    #[test]
    fn fewer_rows_than_probed_dims_does_not_panic() {
        // 3 rows x 10 columns: dims 5 and 10 would exceed the 3 rows.
        let x = Array2::from_shape_fn((3, 10), |(i, j)| ((i * 10 + j) * 7 % 11) as f64);
        let p = profile_numeric_matrix(x.view(), 1000, 42, None).unwrap();
        let dims: Vec<usize> = p.pca_cumulative_variance.iter().map(|&(d, _)| d).collect();
        assert_eq!(dims, vec![2, 3]);
    }

    #[test]
    fn booleans_are_numeric_in_profiles_but_not_in_geometry() {
        let table = Table::new(
            vec!["b".into(), "bb".into(), "x".into()],
            vec![
                Column::Bool(vec![Some(true), Some(false), Some(true)]),
                Column::Bool(vec![Some(true), None, Some(false)]),
                Column::Float(vec![1.0, 2.0, 3.0]),
            ],
        )
        .unwrap();
        let p = profile_columns(&table);
        assert_eq!((p[0].dtype.as_str(), p[0].is_numeric), ("bool", true));
        assert_eq!(
            (p[1].dtype.as_str(), p[1].is_numeric),
            ("object", false),
            "blanks make a bool column object"
        );
        // One float column only: geometry needs two numeric columns.
        assert!(matches!(
            characterize_dataset(&table, 1000, 42),
            Err(CharacterizationError::TooFewNumericColumns { found: 1 })
        ));
    }
}
