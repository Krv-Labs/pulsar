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
//!   probes a data-relative set ([`PcaProbes::DataRelative`]): twelve
//!   dimensions spread from 2 to the largest measurable one. `characterize_dataset` and
//!   `profile_numeric_matrix` take the probe set as a parameter; the golden
//!   parity tests pass [`PcaProbes::Python`] to reproduce the frozen output.
//! - Python's curve was a randomized-SVD estimate (Halko et al., 10
//!   oversamples, 2 power iterations), which can only undershoot, most on
//!   flat spectra. Production Rust reads the exact curve from the
//!   eigenvalues of the centred subsample's Gram matrix, one decomposition
//!   for every probe; only [`PcaProbes::Python`] keeps the randomized
//!   estimate, for parity.

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
    #[error("Probe dimension {dim} is outside 1..={max_dim}; PCA keeps at most min(features, profiled rows) = {max_dim} components here.")]
    ProbeOutOfRange { dim: usize, max_dim: usize },
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
            Self::ProbeOutOfRange { .. } => "PROBE_OUT_OF_RANGE",
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
    /// Twelve probes from 2 to `max_dim`: every dimension when `max_dim` is
    /// 12 or less; otherwise twelve geometrically spaced integers (ratio
    /// `(max_dim / 2)^(1/11)`, rounded), with any duplicates that rounding
    /// leaves at the low end replaced by the midpoint of the widest
    /// remaining gap (the highest gap on a tie). Geometric spacing leaves
    /// its widest gaps at the top, which is where a flat spectrum crosses
    /// 90%. For example 30 gives 2, 3, 4, 5, 7, 9, 11, 14, 18, 23, 26, 30.
    DataRelative,
    /// Exactly these dimensions, each between 1 and `max_dim`;
    /// [`profile_numeric_matrix`] refuses any other.
    Exact(&'a [usize]),
}

const PYTHON_PROBES: [usize; 6] = [2, 3, 5, 10, 15, 20];
/// The number of data-relative probes. The exact curve costs one
/// eigendecomposition however many dimensions are read from it, so the
/// budget only bounds the payload.
const DATA_RELATIVE_PROBES: usize = 12;

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
    if max_dim <= DATA_RELATIVE_PROBES {
        return (2.min(max_dim)..=max_dim).collect();
    }
    let steps = (DATA_RELATIVE_PROBES - 1) as i32;
    let ratio = (max_dim as f64 / 2.0).powf(1.0 / f64::from(steps));
    let mut dims: Vec<usize> = (1..steps)
        .map(|i| ((2.0 * ratio.powi(i)).round() as usize).clamp(2, max_dim))
        .collect();
    // The ends are set exactly rather than trusted to the rounding.
    dims.extend([2, max_dim]);
    dims.sort_unstable();
    dims.dedup();
    // `max_dim` > 12 leaves at least 12 integers in 2..=max_dim, so while
    // fewer are probed some gap spans two or more.
    while dims.len() < DATA_RELATIVE_PROBES {
        let Some((at, gap)) = dims
            .windows(2)
            .map(|w| w[1] - w[0])
            .enumerate()
            .max_by_key(|&(at, gap)| (gap, at))
            .filter(|&(_, gap)| gap > 1)
        else {
            break;
        };
        dims.insert(at + 1, dims[at] + gap / 2);
    }
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
    /// Rows the geometry was measured on (at most `subsample`); with
    /// `n_features_profiled` it bounds the dimensions the variance curve can
    /// reach.
    pub n_samples_profiled: usize,
    /// Numeric columns the geometry was measured on: `n_features` less the
    /// columns with no observed value, which imputation drops.
    pub n_features_profiled: usize,
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
    let n_features = x_sub.ncols();
    // Every probe set refuses an empty matrix the way Python's PCA did.
    if n_features == 0 {
        return Err(pulsar_core::PulsarError::InvalidParameter {
            msg: "input data has 0 features (columns)".to_string(),
        }
        .into());
    }
    // PCA cannot keep more components than there are rows or columns.
    let max_dim = n_features.min(n_sub);
    if let PcaProbes::Exact(dims) = probes {
        if let Some(&dim) = dims.iter().find(|&&d| d == 0 || d > max_dim) {
            return Err(CharacterizationError::ProbeOutOfRange { dim, max_dim });
        }
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

    let dims = probes.dims(max_dim);
    let pca_cumulative_variance = match probes {
        PcaProbes::Python => randomized_cumulative_variance(&x_sub, &dims, seed)?,
        PcaProbes::DataRelative | PcaProbes::Exact(_) => exact_cumulative_variance(&x_sub, &dims)?,
    };

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

/// Python's estimate: one randomized SVD (`pca_grid`) fitted at the largest
/// probe, with each prefix's projected variance over the total variance.
fn randomized_cumulative_variance(
    x: &Array2<f64>,
    dims: &[usize],
    seed: u64,
) -> Result<Vec<(usize, f64)>, CharacterizationError> {
    let embeddings = pulsar_core::pca_grid(x, dims, &[seed], 10, 2)?;
    let total_variance: f64 = (0..x.ncols()).map(|j| variance_of(x.column(j))).sum();
    Ok(dims
        .iter()
        .zip(&embeddings)
        .map(|(&dim, emb)| {
            let ev: f64 = (0..emb.ncols()).map(|j| variance_of(emb.column(j))).sum();
            (dim, round_to((ev / total_variance.max(1e-12)).min(1.0), 4))
        })
        .collect())
}

/// The exact curve: the eigenvalues of the centred matrix's Gram matrix are
/// its squared singular values, so one symmetric eigendecomposition gives the
/// variance held by every prefix of components at once, whatever `dims` asks.
///
/// The Gram is formed on the smaller side (`XᵀX` when there are fewer
/// features than rows, `XXᵀ` otherwise; both share the nonzero spectrum), so
/// memory is `min(rows, features)²` and a wide table never costs `features²`.
fn exact_cumulative_variance(
    x: &Array2<f64>,
    dims: &[usize],
) -> Result<Vec<(usize, f64)>, CharacterizationError> {
    pulsar_core::error::ensure_finite(x)?;
    if x.is_empty() {
        return Ok(dims.iter().map(|&dim| (dim, 0.0)).collect());
    }
    let n = x.nrows() as f64;
    let mut centred = x.to_owned();
    if let Some(means) = x.mean_axis(ndarray::Axis(0)) {
        centred -= &means;
    }
    let gram = if x.ncols() < x.nrows() {
        gram_of_rows(centred.t().as_standard_layout().view())
    } else {
        gram_of_rows(centred.view())
    };
    let mut eigenvalues: Vec<f64> = gram
        .symmetric_eigenvalues()
        .iter()
        // Rounding leaves the null space slightly negative; variance is not.
        .map(|&v| v.max(0.0))
        .collect();
    eigenvalues.sort_by(|a, b| b.total_cmp(a));
    let held: Vec<f64> = std::iter::once(0.0)
        .chain(eigenvalues.iter().scan(0.0, |sum, &v| {
            *sum += v;
            Some(*sum)
        }))
        .collect();
    // The trace is the total: it equals `n` times the summed column
    // variances, and makes the full-rank prefix exactly 1.
    let total_variance = held[held.len() - 1] / n;
    Ok(dims
        .iter()
        .map(|&dim| {
            let ev = held[dim.min(eigenvalues.len())] / n;
            (dim, round_to((ev / total_variance.max(1e-12)).min(1.0), 4))
        })
        .collect())
}

/// `A·Aᵀ` with rayon over blocks of output rows, each block one GEMM.
fn gram_of_rows(a: ArrayView2<f64>) -> nalgebra::DMatrix<f64> {
    const ROW_BLOCK: usize = 64;
    let m = a.nrows();
    let mut buf = vec![0.0; m * m];
    buf.par_chunks_mut(ROW_BLOCK * m)
        .enumerate()
        .for_each(|(block, out)| {
            let rows = out.len() / m;
            let start = block * ROW_BLOCK;
            let mut out = ndarray::ArrayViewMut2::from_shape((rows, m), out)
                .expect("each chunk holds whole rows of length m");
            ndarray::linalg::general_mat_mul(
                1.0,
                &a.slice(ndarray::s![start..start + rows, ..]),
                &a.t(),
                0.0,
                &mut out,
            );
        });
    nalgebra::DMatrix::from_row_slice(m, m, &buf)
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
        n_features_profiled: geo.n_features,
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

    fn assert_monotone(curve: &[(usize, f64)]) {
        assert!(curve.windows(2).all(|w| w[0].1 <= w[1].1), "{curve:?}");
    }

    #[test]
    fn exact_curve_matches_the_randomized_estimate_on_a_decaying_spectrum() {
        // Independent Gaussian columns with geometrically decaying scale: a
        // well-conditioned spectrum the randomized SVD resolves closely.
        let mut rng = StdRng::seed_from_u64(11);
        let x = Array2::from_shape_fn((400, 40), |(_, j)| {
            let z: f64 = rand::Rng::sample(&mut rng, rand_distr::StandardNormal);
            z * 0.85f64.powi(j as i32)
        });
        let full = profile_numeric_matrix(x.view(), 1000, 42, PcaProbes::DataRelative).unwrap();
        assert_monotone(&full.pca_cumulative_variance);
        assert_eq!(full.pca_cumulative_variance.last(), Some(&(40, 1.0)));
        // Both curves are rounded to 4 digits, so one unit in the last place
        // is the tightest agreement either can show.
        let (exact, randomized) = exact_and_randomized(x.view());
        for ((dim, value), (_, estimate)) in exact.into_iter().zip(randomized) {
            assert!(
                (value - estimate).abs() <= 1e-4 + 1e-12,
                "dim {dim}: {value} vs {estimate}"
            );
        }
    }

    type Curve = Vec<(usize, f64)>;

    /// The exact and randomized curves at Python's probes, which every
    /// matrix here is wide and tall enough to reach.
    fn exact_and_randomized(x: ArrayView2<f64>) -> (Curve, Curve) {
        let dims = PcaProbes::Python.dims(x.ncols().min(x.nrows()));
        let exact = profile_numeric_matrix(x, 1000, 42, PcaProbes::Exact(&dims)).unwrap();
        let randomized = profile_numeric_matrix(x, 1000, 42, PcaProbes::Python).unwrap();
        assert_eq!(
            exact
                .pca_cumulative_variance
                .iter()
                .map(|&(d, _)| d)
                .collect::<Vec<_>>(),
            randomized
                .pca_cumulative_variance
                .iter()
                .map(|&(d, _)| d)
                .collect::<Vec<_>>()
        );
        (
            exact.pca_cumulative_variance,
            randomized.pca_cumulative_variance,
        )
    }

    #[test]
    fn exact_curve_centres_before_measuring() {
        // A +1000 offset on every column: without centring the mean would be
        // one huge direction holding nearly all the "variance", and the curve
        // would read about 1.0 from dimension 2 (the centred columns, scales
        // 0.8^j, put about 0.6 of the variance in two components).
        let mut rng = StdRng::seed_from_u64(5);
        let x = Array2::from_shape_fn((100, 8), |(_, j)| {
            let z: f64 = rand::Rng::sample(&mut rng, rand_distr::StandardNormal);
            1000.0 + z * 0.8f64.powi(j as i32)
        });
        let (exact, randomized) = exact_and_randomized(x.view());
        assert!(exact[0].1 < 0.7, "{exact:?}");
        for ((dim, value), (_, estimate)) in exact.into_iter().zip(randomized) {
            assert!(
                (value - estimate).abs() <= 1e-4 + 1e-12,
                "dim {dim}: {value} vs {estimate}"
            );
        }
        // A population (n) or sample (n - 1) denominator cancels in the
        // ratio; the full-rank prefix pins the total.
        let p = profile_numeric_matrix(x.view(), 1000, 42, PcaProbes::Exact(&[8])).unwrap();
        assert_eq!(p.pca_cumulative_variance, vec![(8, 1.0)]);
    }

    #[test]
    fn the_randomized_estimate_undershoots_a_flat_spectrum() {
        // 80 columns of uniform noise: no gap in the spectrum for the
        // sketch to lock onto, so the randomized prefixes miss variance the
        // exact top components hold.
        let mut rng = StdRng::seed_from_u64(9);
        let x = Array2::from_shape_fn((400, 80), |_| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let (exact, randomized) = exact_and_randomized(x.view());
        for ((dim, value), (_, estimate)) in exact.into_iter().zip(randomized) {
            assert!(estimate < value, "dim {dim}: {value} vs {estimate}");
        }
    }

    #[test]
    fn gram_of_rows_matches_a_plain_product_across_row_blocks() {
        // 150 rows: three output blocks of 64, 64 and 22 rows.
        let mut rng = StdRng::seed_from_u64(13);
        let a = Array2::from_shape_fn((150, 7), |_| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let gram = gram_of_rows(a.view());
        let want = a.dot(&a.t());
        for ((i, j), w) in want.indexed_iter() {
            assert!((gram[(i, j)] - w).abs() < 1e-12, "({i}, {j})");
        }
    }

    #[test]
    fn exact_curve_holds_across_several_gram_blocks() {
        // A gently decaying spectrum on both Gram sides with more than one
        // 64-row block: 300 x 200 forms the 200-wide `XᵀX`, 150 x 300 the
        // 150-wide `XXᵀ`.
        let mut rng = StdRng::seed_from_u64(17);
        for (rows, cols) in [(300, 200), (150, 300)] {
            let x = Array2::from_shape_fn((rows, cols), |(_, j)| {
                let z: f64 = rand::Rng::sample(&mut rng, rand_distr::StandardNormal);
                z * 0.97f64.powi(j as i32)
            });
            let full = profile_numeric_matrix(x.view(), 1000, 42, PcaProbes::DataRelative).unwrap();
            let curve = &full.pca_cumulative_variance;
            assert_monotone(curve);
            let max_dim = cols.min(rows);
            assert_eq!(curve.last(), Some(&(max_dim, 1.0)), "{rows}x{cols}");
            let (exact, randomized) = exact_and_randomized(x.view());
            for ((dim, value), (_, estimate)) in exact.into_iter().zip(randomized) {
                assert!(
                    estimate <= value + 1e-4 && value - estimate < 2e-2,
                    "{rows}x{cols} dim {dim}: {value} vs {estimate}"
                );
            }
        }
    }

    #[test]
    fn exact_curve_saturates_at_the_rank() {
        // Rows outnumber features (the `XᵀX` side): 12 columns, the last six
        // copies of the first six, so rank 6.
        let mut rng = StdRng::seed_from_u64(3);
        let base = Array2::from_shape_fn((200, 6), |_| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let x = ndarray::concatenate![ndarray::Axis(1), base, base];
        let p = profile_numeric_matrix(x.view(), 1000, 42, PcaProbes::DataRelative).unwrap();
        let curve = &p.pca_cumulative_variance;
        assert_eq!(
            curve.iter().map(|&(d, _)| d).collect::<Vec<_>>(),
            (2..=12).collect::<Vec<_>>()
        );
        assert_monotone(curve);
        assert!(curve[3].1 < 1.0, "{curve:?}");
        assert!(curve[4..].iter().all(|&(_, v)| v == 1.0), "{curve:?}");

        // Features outnumber rows (the `XXᵀ` side): centring leaves rank
        // n - 1, and the curve reaches 1 there and stays.
        let wide = Array2::from_shape_fn((30, 80), |_| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let p = profile_numeric_matrix(wide.view(), 1000, 42, PcaProbes::DataRelative).unwrap();
        let curve = &p.pca_cumulative_variance;
        assert_monotone(curve);
        assert_eq!(curve.last(), Some(&(30, 1.0)));
        let below: Vec<f64> = curve
            .iter()
            .filter(|&&(d, _)| d < 29)
            .map(|&(_, v)| v)
            .collect();
        assert!(below.iter().all(|&v| v < 1.0), "{curve:?}");
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
        // Rounding repeats 2 and 3; the two widest gaps (9-11, 11-13) are
        // filled, which here probes every dimension.
        assert_eq!(probes(13), (2..=13).collect::<Vec<_>>());
        // Duplicates refill the top gap: 15-19 gains 17, 23-30 gains 26.
        assert_eq!(probes(19), vec![2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 17, 19]);
        assert_eq!(probes(30), vec![2, 3, 4, 5, 7, 9, 11, 14, 18, 23, 26, 30]);
        assert_eq!(
            probes(200),
            vec![2, 3, 5, 7, 11, 16, 25, 37, 57, 87, 132, 200]
        );
        assert_eq!(
            probes(1000),
            vec![2, 4, 6, 11, 19, 34, 59, 104, 184, 323, 568, 1000]
        );
        for max_dim in 2..=5000 {
            let d = probes(max_dim);
            assert_eq!(d.first(), Some(&2), "{max_dim}");
            assert_eq!(d.last(), Some(&max_dim), "{max_dim}");
            assert_eq!(d.len(), max_dim.min(13) - 1, "{max_dim}: {d:?}");
            assert!(d.windows(2).all(|w| w[0] < w[1]), "{max_dim}: {d:?}");
        }
        // Python's set stops at 20 whatever the width.
        assert_eq!(PcaProbes::Python.dims(58), vec![2, 3, 5, 10, 15, 20]);
        assert_eq!(PcaProbes::Python.dims(8), vec![2, 3, 5]);
    }

    fn noise_table(rows: usize, cols: usize, seed: u64) -> Table {
        let names: Vec<String> = (0..cols).map(|j| format!("f{j}")).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let columns = (0..cols)
            .map(|_| {
                Column::Float(
                    (0..rows)
                        .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
                        .collect(),
                )
            })
            .collect();
        Table::new(names, columns).unwrap()
    }

    #[test]
    fn a_flat_spectrum_finds_its_elbow_below_the_feature_count() {
        // Independent uniform noise: variance spreads nearly evenly, so 90%
        // is crossed a few dimensions short of the width (about 24 of 30
        // and 16 of 19 at 150 rows). The probes must land between that
        // crossing and the width for the elbow, not the width, to set the top.
        for (cols, above) in [(30, 20), (19, 13)] {
            let table = noise_table(150, cols, 7);
            let geo = characterize_dataset(&table, 1000, 42, PcaProbes::DataRelative).unwrap();
            let curve = &geo.pca_cumulative_variance;
            assert_eq!(curve.len(), 12, "{curve:?}");
            assert_eq!(curve.last().map(|&(d, _)| d), Some(cols), "{curve:?}");
            let ceiling = crate::config_build::grid_ceiling(
                curve,
                geo.n_features_profiled,
                geo.n_samples_profiled,
            );
            assert_eq!(
                ceiling.source,
                crate::config_build::CeilingSource::VarianceElbow,
                "{cols}: {ceiling:?} {curve:?}"
            );
            assert!(
                above < ceiling.dim && ceiling.dim < cols,
                "{ceiling:?} {curve:?}"
            );
        }
        // Python's probes stop at 20 on the 30-column table.
        let py =
            characterize_dataset(&noise_table(150, 30, 7), 1000, 42, PcaProbes::Python).unwrap();
        assert_eq!(py.pca_cumulative_variance.last().map(|&(d, _)| d), Some(20));
    }

    #[test]
    fn an_all_missing_column_is_not_a_profiled_feature() {
        let mut table = noise_table(40, 3, 1);
        table.names.push("empty".into());
        table.columns.push(Column::Float(vec![f64::NAN; 40]));
        let geo = characterize_dataset(&table, 1000, 42, PcaProbes::DataRelative).unwrap();
        assert_eq!((geo.n_features, geo.n_features_profiled), (4, 3));
        assert_eq!(geo.pca_cumulative_variance.last(), Some(&(3, 1.0)));
        let (_, n_features, _) = crate::config_build::calibration_curve(&geo, None);
        assert_eq!(n_features, 3);
    }

    #[test]
    fn probe_sets_are_validated_at_the_boundary() {
        // No columns: every probe set refuses it with Python's PCA error.
        let empty = Array2::<f64>::zeros((5, 0));
        let messages: Vec<String> = [
            PcaProbes::Python,
            PcaProbes::DataRelative,
            PcaProbes::Exact(&[1]),
        ]
        .into_iter()
        .map(|probes| {
            let err = profile_numeric_matrix(empty.view(), 1000, 42, probes).unwrap_err();
            assert_eq!(err.code(), "CHARACTERIZATION_FAILED", "{probes:?}");
            err.to_string()
        })
        .collect();
        assert!(
            messages[0].contains("input data has 0 features (columns)"),
            "{messages:?}"
        );
        assert!(messages.iter().all(|m| *m == messages[0]), "{messages:?}");

        // Exact dimensions must lie in 1..=min(features, profiled rows).
        let x = Array2::from_shape_fn((20, 3), |(i, j)| ((i * 3 + j) * 7 % 11) as f64);
        for (dims, bad) in [(&[3, 10][..], 10), (&[0, 2][..], 0)] {
            let err =
                profile_numeric_matrix(x.view(), 1000, 42, PcaProbes::Exact(dims)).unwrap_err();
            assert!(
                matches!(err, CharacterizationError::ProbeOutOfRange { dim, max_dim: 3 } if dim == bad),
                "{err:?}"
            );
            assert_eq!(err.code(), "PROBE_OUT_OF_RANGE");
            assert_eq!(
                err.to_string(),
                format!("Probe dimension {bad} is outside 1..=3; PCA keeps at most min(features, profiled rows) = 3 components here.")
            );
        }
        // Three rows bound four features at 3 components.
        let short = Array2::from_shape_fn((3, 4), |(i, j)| ((i * 4 + j) * 5 % 7) as f64);
        assert!(matches!(
            profile_numeric_matrix(short.view(), 1000, 42, PcaProbes::Exact(&[4])),
            Err(CharacterizationError::ProbeOutOfRange { dim: 4, max_dim: 3 })
        ));
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
