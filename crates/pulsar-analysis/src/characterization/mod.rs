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
//!
//! Known divergences from Python:
//! - Python probed cumulative PCA variance only at 2, 3, 5, 10, 15 and 20
//!   dimensions (those within the feature and row counts), so the variance
//!   elbow, and the dimension grid built from it, could never exceed 20 and
//!   a narrow table was never measured at its own width. Production Rust
//!   probes a data-relative set ([`PcaProbes::DataRelative`]) that always
//!   reaches the largest measurable dimension. `characterize_dataset` and
//!   `profile_numeric_matrix` take the probe set as a parameter; the golden
//!   parity tests pass [`PcaProbes::Python`] to reproduce the frozen output.
//!   Because the one randomized SVD is fitted at the largest probe, the
//!   variance at small probes also shifts slightly in production (it is read
//!   from a wider, often exact, decomposition).

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
    #[error("Column {column:?} holds a non-finite value ({value}) at row {row}. Infinite values are not missing and are not imputed; they would turn the column's mean and z-scores into NaN.")]
    NonFiniteValue {
        column: String,
        row: usize,
        value: f64,
    },
    #[error(transparent)]
    Core(#[from] pulsar_core::PulsarError),
}

impl CharacterizationError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::TooFewNumericColumns { .. } => "TOO_FEW_NUMERIC_COLUMNS",
            Self::TooFewRows { .. } => "TOO_FEW_ROWS",
            Self::NoObservedValues => "NO_OBSERVED_VALUES",
            Self::NonFiniteValue { .. } => "NON_FINITE_VALUE",
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

/// The dimensions at which cumulative PCA variance is measured.
///
/// `max_dim` is the largest measurable dimension, `min(features, profiled
/// rows)`: PCA cannot keep more components than either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaProbes<'a> {
    /// Python's fixed set: 2, 3, 5, 10, 15 and 20, those within `max_dim`.
    /// Kept for golden parity; it cannot see past 20 dimensions.
    Python,
    /// Every dimension up to 12 when `max_dim` is that small; otherwise
    /// Python's set below 20, up to five geometrically spaced dimensions
    /// above 20, and `max_dim` itself. At most 12 probes.
    DataRelative,
    /// Exactly these dimensions (each must be within `max_dim`).
    Exact(&'a [usize]),
}

const PYTHON_PROBES: [usize; 6] = [2, 3, 5, 10, 15, 20];
/// Below this width every dimension is probed.
const DENSE_PROBE_LIMIT: usize = 12;
/// Spacing of the probes above 20 when few are needed.
const PROBE_RATIO: f64 = 1.5;
/// Probes above 20, `max_dim` included.
const MAX_WIDE_PROBES: usize = 6;

impl PcaProbes<'_> {
    /// The probed dimensions, ascending, for a matrix whose largest
    /// measurable dimension is `max_dim`.
    pub fn dims(self, max_dim: usize) -> Vec<usize> {
        match self {
            Self::Exact(d) => d.to_vec(),
            Self::Python => {
                let d: Vec<usize> = PYTHON_PROBES
                    .into_iter()
                    .filter(|&d| d <= max_dim)
                    .collect();
                if d.is_empty() {
                    vec![2.min(max_dim)]
                } else {
                    d
                }
            }
            Self::DataRelative => data_relative_probes(max_dim),
        }
    }
}

fn data_relative_probes(max_dim: usize) -> Vec<usize> {
    if max_dim <= DENSE_PROBE_LIMIT {
        return (2.min(max_dim)..=max_dim).collect();
    }
    let base = PYTHON_PROBES[PYTHON_PROBES.len() - 1];
    let mut dims: Vec<usize> = PYTHON_PROBES.into_iter().filter(|&d| d < max_dim).collect();
    if max_dim > base {
        // Geometric steps from 20 to `max_dim`: ratio 1.5 while that needs
        // few steps, a wider ratio beyond, so the count stays bounded.
        let span = max_dim as f64 / base as f64;
        let steps = ((span.ln() / PROBE_RATIO.ln()).ceil() as usize).clamp(1, MAX_WIDE_PROBES);
        let ratio = span.powf(1.0 / steps as f64);
        dims.extend((1..steps).map(|i| (base as f64 * ratio.powi(i as i32)).round() as usize));
    }
    dims.push(max_dim);
    dims.retain(|&d| d <= max_dim);
    dims.sort_unstable();
    dims.dedup();
    dims
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
    /// Rows the geometry was measured on (at most `subsample`); with the
    /// feature count it bounds the dimensions the variance curve can reach.
    pub n_samples_profiled: usize,
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
    probes: PcaProbes<'_>,
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
    // `pca_grid` fits one randomized SVD at the largest probe and slices it,
    // so more probes cost bookkeeping, not extra SVDs. The one SVD does grow
    // with the largest probe: reaching `max_dim` on a wide table means a
    // sketch as wide as the feature count, comparable to the k-NN pass above.
    let dims = probes.dims(max_dim);
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
    probes: PcaProbes<'_>,
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
    // NaN is missing and mean-imputed below; an infinity is not, and would
    // poison the column's mean and standard deviation.
    let names = table
        .names
        .iter()
        .zip(&table.columns)
        .filter(|(_, c)| matches!(c, Column::Float(_) | Column::Int(_)))
        .map(|(name, _)| name);
    for (name, col) in names.zip(&values) {
        if let Some((row, &value)) = col.iter().enumerate().find(|(_, x)| x.is_infinite()) {
            return Err(CharacterizationError::NonFiniteValue {
                column: name.clone(),
                row,
                value,
            });
        }
    }
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

    let geo = profile_numeric_matrix(x.view(), subsample, seed, probes)?;
    Ok(DatasetProfile {
        n_samples: n,
        n_features: numeric.len(),
        n_columns_total: table.n_cols(),
        missingness_pct,
        knn_k5_mean: geo.knn_k5_mean,
        knn_k10_mean: geo.knn_k10_mean,
        knn_k20_mean: geo.knn_k20_mean,
        pca_cumulative_variance: geo.pca_cumulative_variance,
        n_samples_profiled: geo.n_samples_profiled,
        column_profiles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn an_infinite_cell_is_refused_before_imputation() {
        let table = Table::new(
            vec!["a".into(), "b".into()],
            vec![
                Column::Float((0..10).map(f64::from).collect()),
                Column::Float(
                    (0..10)
                        .map(|i| {
                            if i == 6 {
                                f64::NEG_INFINITY
                            } else {
                                f64::from(i % 3)
                            }
                        })
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let err = characterize_dataset(&table, 1000, 42, PcaProbes::DataRelative).unwrap_err();
        assert_eq!(err.code(), "NON_FINITE_VALUE");
        assert!(
            matches!(&err, CharacterizationError::NonFiniteValue { column, row: 6, value }
                if column == "b" && *value == f64::NEG_INFINITY),
            "{err:?}"
        );
        assert!(err.to_string().contains("(-inf) at row 6"), "{err}");
    }

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
        let p = profile_numeric_matrix(x.view(), 1000, 42, PcaProbes::Exact(&[1])).unwrap();
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
        let p = profile_numeric_matrix(x.view(), 1000, 42, PcaProbes::Exact(&[1, 2])).unwrap();
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
            profile_numeric_matrix(one_row.view(), 1000, 42, PcaProbes::DataRelative)
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
        let err = characterize_dataset(&table, 1000, 42, PcaProbes::DataRelative).unwrap_err();
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
        for probes in [PcaProbes::Python, PcaProbes::DataRelative] {
            let p = profile_numeric_matrix(x.view(), 1000, 42, probes).unwrap();
            let dims: Vec<usize> = p.pca_cumulative_variance.iter().map(|&(d, _)| d).collect();
            assert_eq!(dims, vec![2, 3], "{probes:?}");
        }
    }

    #[test]
    fn data_relative_probes_reach_the_largest_measurable_dimension() {
        let probes = |max_dim| PcaProbes::DataRelative.dims(max_dim);
        assert_eq!(probes(1), vec![1]);
        assert_eq!(probes(2), vec![2]);
        assert_eq!(probes(8), vec![2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(probes(12), (2..=12).collect::<Vec<_>>());
        assert_eq!(probes(13), vec![2, 3, 5, 10, 13]);
        assert_eq!(probes(20), vec![2, 3, 5, 10, 15, 20]);
        assert_eq!(probes(30), vec![2, 3, 5, 10, 15, 20, 30]);
        assert_eq!(probes(58), vec![2, 3, 5, 10, 15, 20, 29, 41, 58]);
        assert_eq!(
            probes(200),
            vec![2, 3, 5, 10, 15, 20, 29, 43, 63, 93, 136, 200]
        );
        for max_dim in 1..=5000 {
            let d = probes(max_dim);
            assert_eq!(d.last(), Some(&max_dim), "{max_dim}");
            assert!(d.len() <= 12, "{max_dim}: {d:?}");
            assert!(d.windows(2).all(|w| w[0] < w[1]), "{max_dim}: {d:?}");
        }
        // Python's set stops at 20 whatever the width.
        assert_eq!(PcaProbes::Python.dims(58), vec![2, 3, 5, 10, 15, 20]);
        assert_eq!(PcaProbes::Python.dims(8), vec![2, 3, 5]);
    }

    #[test]
    fn a_wide_table_is_measured_up_to_its_feature_count() {
        // 30 columns of independent noise: variance spreads evenly, so 20
        // components hold about two thirds of it and 90% needs nearly all 30.
        let names: Vec<String> = (0..30).map(|j| format!("f{j}")).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let columns = (0..30)
            .map(|_| {
                Column::Float(
                    (0..200)
                        .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                        .collect(),
                )
            })
            .collect();
        let table = Table::new(names, columns).unwrap();
        let geo = characterize_dataset(&table, 1000, 42, PcaProbes::DataRelative).unwrap();
        let curve = &geo.pca_cumulative_variance;
        assert_eq!(curve.last().map(|&(d, _)| d), Some(30), "{curve:?}");
        let at_20 = curve.iter().find(|&&(d, _)| d == 20).unwrap().1;
        assert!(at_20 < 0.9, "{curve:?}");
        let ceiling =
            crate::config_build::grid_ceiling(curve, geo.n_features, geo.n_samples_profiled);
        assert!(ceiling.dim > 20, "{ceiling:?}");
        assert!(
            matches!(
                ceiling.source,
                crate::config_build::CeilingSource::NFeatures
                    | crate::config_build::CeilingSource::VarianceElbow
            ),
            "{ceiling:?}"
        );
        // Python's probes stop at 20 on the same table.
        let py = characterize_dataset(&table, 1000, 42, PcaProbes::Python).unwrap();
        assert_eq!(py.pca_cumulative_variance.last().map(|&(d, _)| d), Some(20));
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
            characterize_dataset(&table, 1000, 42, PcaProbes::DataRelative),
            Err(CharacterizationError::TooFewNumericColumns { found: 1 })
        ));
    }
}
