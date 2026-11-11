//! Statistical primitives of the cluster interpreter
//! (`pulsar/mcp/interpreter.py`, lines ~676-1110): the tests and effect sizes
//! behind the feature evidence of a cluster dossier.
//!
//! | Python | Rust |
//! |---|---|
//! | `scipy.stats.rankdata` (average, NaN omitted) | [`average_ranks`] |
//! | `scipy.stats.mannwhitneyu` (auto: exact or asymptotic) | [`mwu_pair_pvalue`], [`mwu_one_vs_rest_pvalue`], [`mwu_exact_two_sided`] |
//! | `scipy.stats.kruskal` | [`kruskal_wallis_pvalue`] |
//! | `scipy.stats.ks_2samp(...).statistic` | [`ks_two_sample_stat`] |
//! | `scipy.stats.wasserstein_distance` | [`wasserstein_distance_1d`] |
//! | `scipy.special.ndtr`, `scipy.stats.chi2.sf` | `statrs` |
//!
//! The exact small-sample Mann-Whitney distribution is counted with the
//! Gaussian-binomial recurrence rather than scipy's sigma-array method; both
//! give the same exact probabilities. Library code is panic-free: degenerate
//! inputs return the same value or error text as the Python.

use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use thiserror::Error;

/// Floor for standard deviations used as divisors.
pub const EPS: f64 = 1e-9;

/// A test that cannot be computed on the given input. The messages match the
/// Python failure strings that dossiers report under `stats_failures`.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StatsError {
    #[error("{0}")]
    Input(String),
}

fn input(msg: &str) -> StatsError {
    StatsError::Input(msg.to_owned())
}

// ---- small helpers ------------------------------------------------------------

/// Benjamini-Hochberg FDR correction; entries that are `None` or not finite
/// stay `None`.
pub fn bh_fdr(p_values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut indexed: Vec<(usize, f64)> = p_values
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.filter(|v| v.is_finite()).map(|v| (i, v)))
        .collect();
    let mut corrected = vec![None; p_values.len()];
    if indexed.is_empty() {
        return corrected;
    }
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let m = indexed.len();
    let mut running = 1.0_f64;
    for (rank0, (idx, p)) in indexed.iter().enumerate().rev() {
        let adjusted = (p * m as f64 / (rank0 + 1) as f64).min(1.0);
        running = running.min(adjusted);
        corrected[*idx] = Some(running);
    }
    corrected
}

/// `np.allclose(values, values[0], equal_nan=True)`.
fn all_close_to_first(values: &[f64]) -> bool {
    let b = values[0];
    values.iter().all(|&a| {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() && b.is_nan();
        }
        a == b || (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
    })
}

/// Empirical percentiles in `[0, 1]`; a constant (or single) column collapses to 0.
pub fn empirical_percentiles(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    if values.len() <= 1 || all_close_to_first(values) {
        return vec![0.0; values.len()];
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let denom = (values.len() - 1).max(1) as f64;
    let mut out = vec![0.0; values.len()];
    for (rank, &i) in order.iter().enumerate() {
        out[i] = rank as f64 / denom;
    }
    out
}

/// `np.sum` of a contiguous float64 array: pairwise summation (8-way unrolled
/// blocks of up to 128 values, halves above that). Reproducing numpy's
/// association order bit for bit matters because scores are ranked, so two
/// values that tie mathematically must tie (or not) exactly as they do there.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for v in a {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r: [f64; 8] = a[..8].try_into().expect("n >= 8");
        let blocks_end = n - (n % 8);
        let mut i = 8;
        while i < blocks_end {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[i..] {
            res += v;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    pairwise_sum(&a[..half]) + pairwise_sum(&a[half..])
}

/// `np.mean`.
pub fn np_mean(values: &[f64]) -> f64 {
    pairwise_sum(values) / values.len() as f64
}

/// `np.var(values, ddof)`: the mean, then the squared deviations, each summed
/// pairwise. NaN for an empty input or `ddof >= len`.
pub fn np_var(values: &[f64], ddof: usize) -> f64 {
    let n = values.len();
    if n == 0 || n <= ddof {
        return f64::NAN;
    }
    let mean = np_mean(values);
    let squares: Vec<f64> = values.iter().map(|v| (v - mean) * (v - mean)).collect();
    pairwise_sum(&squares) / (n - ddof) as f64
}

pub fn safe_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        np_mean(values)
    }
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

fn median_sorted(s: &[f64]) -> f64 {
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

pub fn safe_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        median_sorted(&sorted(values))
    }
}

/// Sample variance (`ddof = 1`); 0 for fewer than two values.
pub fn safe_var(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    np_var(values, 1)
}

pub fn safe_std(values: &[f64]) -> f64 {
    safe_var(values).sqrt()
}

/// Median absolute deviation from the median.
pub fn safe_mad(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let median = safe_median(values);
    let deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
    median_sorted(&sorted(&deviations))
}

/// Pooled standard deviation of two samples, floored at [`EPS`].
pub fn pooled_std(a: &[f64], b: &[f64]) -> f64 {
    if a.len() <= 1 || b.len() <= 1 {
        return safe_std(a).max(safe_std(b)).max(EPS);
    }
    let pooled = ((a.len() - 1) as f64 * safe_var(a) + (b.len() - 1) as f64 * safe_var(b))
        / (a.len() + b.len() - 2).max(1) as f64;
    pooled.max(EPS).sqrt()
}

/// `+1` effects only is `up`, `-1` only is `down`, anything else `mixed`.
pub fn direction_from_effects(effects: &[f64]) -> &'static str {
    let positives = effects.iter().filter(|&&e| e > 0.0).count();
    let negatives = effects.iter().filter(|&&e| e < 0.0).count();
    match (positives > 0, negatives > 0) {
        (true, false) => "up",
        (false, true) => "down",
        _ => "mixed",
    }
}

/// Linear-interpolation quantile of an ascending slice (`np.quantile` default).
fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    let pos = q * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

// ---- ranks and the batched column statistics -------------------------------------

/// Average ranks (1-based, ties share their mean rank) of the non-NaN values,
/// NaN where the input is NaN.
pub fn average_ranks(column: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..column.len()).filter(|&i| !column[i].is_nan()).collect();
    order.sort_by(|&a, &b| column[a].total_cmp(&column[b]));
    let mut ranks = vec![f64::NAN; column.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && column[order[end]] == column[order[start]] {
            end += 1;
        }
        let rank = (start + end + 1) as f64 / 2.0;
        for &i in &order[start..end] {
            ranks[i] = rank;
        }
        start = end;
    }
    ranks
}

/// Kruskal-Wallis tie correction `1 - sum(t^3 - t) / (N^3 - N)` over the sorted
/// non-NaN values; 1 when undefined, 0 when every value is tied.
pub fn column_tie_correction(sorted_values: &[f64]) -> f64 {
    let n = sorted_values.len();
    if n <= 1 {
        return 1.0;
    }
    let (mut tie_cubes, mut run) = (0.0, 1.0_f64);
    for w in sorted_values.windows(2) {
        if w[0] == w[1] {
            run += 1.0;
        } else {
            if run > 1.0 {
                tie_cubes += run * run * run - run;
            }
            run = 1.0;
        }
    }
    if run > 1.0 {
        tie_cubes += run * run * run - run;
    }
    let denom = (n as f64).powi(3) - n as f64;
    if denom == 0.0 {
        return 1.0;
    }
    let correction = 1.0 - tie_cubes / denom;
    if correction <= 0.0 {
        0.0
    } else {
        correction
    }
}

/// Batched per-column statistics for scoring many columns against clusters.
/// Every field is per column. NaN marks a missing value and each column is
/// treated independently: `n_total` is the per-column valid count, which every
/// formula that needs an `N` must use.
#[derive(Debug, Clone)]
pub struct NumericPrecompute {
    /// Column values (`x[j][i]`, NaN where missing).
    pub x: Vec<Vec<f64>>,
    pub valid: Vec<Vec<bool>>,
    pub n_total: Vec<usize>,
    pub col_mean: Vec<f64>,
    /// Population standard deviation (`ddof = 0`), the scale of `global_scale`.
    pub col_std_pop: Vec<f64>,
    /// Sample standard deviation (`ddof = 1`), used row by row.
    pub col_std_sample: Vec<f64>,
    pub col_var_sample: Vec<f64>,
    pub col_median: Vec<f64>,
    pub col_mad: Vec<f64>,
    pub col_iqr: Vec<f64>,
    /// Average ranks, NaN at missing rows.
    pub ranks: Vec<Vec<f64>>,
    pub tie_correction: Vec<f64>,
}

fn nan_moments(values: &[f64], ddof: usize) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    if n <= ddof {
        return (mean, f64::NAN);
    }
    let ss: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    (mean, ss / (n - ddof) as f64)
}

/// Statistics of numeric `columns` (each of the same length, NaN = missing).
pub fn build_numeric_precompute(columns: &[Vec<f64>]) -> NumericPrecompute {
    let k = columns.len();
    let mut pre = NumericPrecompute {
        x: columns.to_vec(),
        valid: Vec::with_capacity(k),
        n_total: Vec::with_capacity(k),
        col_mean: Vec::with_capacity(k),
        col_std_pop: Vec::with_capacity(k),
        col_std_sample: Vec::with_capacity(k),
        col_var_sample: Vec::with_capacity(k),
        col_median: Vec::with_capacity(k),
        col_mad: Vec::with_capacity(k),
        col_iqr: Vec::with_capacity(k),
        ranks: Vec::with_capacity(k),
        tie_correction: Vec::with_capacity(k),
    };
    for col in columns {
        let present: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
        let s = sorted(&present);
        let (mean, var_pop) = nan_moments(&present, 0);
        let (_, var_sample) = nan_moments(&present, 1);
        let median = median_sorted(&s);
        let deviations: Vec<f64> = present.iter().map(|v| (v - median).abs()).collect();
        pre.valid.push(col.iter().map(|v| !v.is_nan()).collect());
        pre.n_total.push(present.len());
        pre.col_mean.push(mean);
        pre.col_std_pop.push(var_pop.sqrt());
        pre.col_std_sample.push(var_sample.sqrt());
        pre.col_var_sample.push(var_sample);
        pre.col_median.push(median);
        pre.col_mad.push(median_sorted(&sorted(&deviations)));
        pre.col_iqr
            .push(quantile_sorted(&s, 0.75) - quantile_sorted(&s, 0.25));
        pre.ranks.push(average_ranks(col));
        pre.tie_correction.push(column_tie_correction(&s));
    }
    pre
}

// ---- Mann-Whitney U ------------------------------------------------------------------

fn std_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("the standard normal is valid")
}

/// Asymptotic two-sided Mann-Whitney p-value with continuity correction;
/// `tie_sum` is `sum(t^3 - t)` over tie groups of the combined sample.
pub fn mwu_asymptotic_two_sided_pvalue(u1: f64, n1: usize, n2: usize, tie_sum: f64) -> f64 {
    let n = n1 + n2;
    if n <= 1 || n1 == 0 || n2 == 0 {
        return 1.0;
    }
    let u = u1.max(n1 as f64 * n2 as f64 - u1);
    let mu = n1 as f64 * n2 as f64 / 2.0;
    let nf = n as f64;
    let variance = n1 as f64 * n2 as f64 / 12.0 * ((nf + 1.0) - tie_sum / (nf * (nf - 1.0)));
    if variance <= 0.0 {
        return 1.0;
    }
    let z = (u - mu - 0.5) / variance.sqrt();
    (std_normal().cdf(-z) * 2.0).clamp(0.0, 1.0)
}

/// `P(U >= u)` for the Mann-Whitney statistic of samples of size `n1` and `n2`
/// without ties, doubled and clipped to 1 (scipy's exact two-sided p-value).
///
/// The counts of arrangements by `U` are the coefficients of the Gaussian
/// binomial `prod_{i=1..m} (1 - x^(n+i)) / (1 - x^i)`; only the lower tail up to
/// `min(u, m*n - u)` is needed, so the polynomial is truncated there.
pub fn mwu_exact_two_sided(u: usize, n1: usize, n2: usize) -> f64 {
    let (m, n) = (n1.min(n2), n1.max(n2));
    let total_u = m * n;
    let tail = u.min(total_u.saturating_sub(u));
    let mut counts = vec![0.0_f64; tail + 1];
    counts[0] = 1.0;
    for i in 1..=m {
        // Multiply by (1 - x^(n+i)), highest degree first so each term reads an old value.
        let shift = n + i;
        if shift <= tail {
            for d in (shift..=tail).rev() {
                counts[d] -= counts[d - shift];
            }
        }
        // Divide by (1 - x^i): a running sum with stride i.
        for d in i..=tail {
            counts[d] += counts[d - i];
        }
    }
    let total: f64 = (1..=m).map(|i| (n + i) as f64 / i as f64).product();
    let lower: f64 = counts.iter().sum::<f64>() / total;
    // `lower` is P(U <= tail). For u at or above the mean, P(U >= u) = P(U <= m*n - u).
    let sf = if u * 2 >= total_u {
        lower
    } else {
        1.0 - lower + counts[tail] / total
    };
    (2.0 * sf).clamp(0.0, 1.0)
}

/// Two-sided p-value of cluster against rest for column `j`, from the
/// precomputed ranks (scipy `mannwhitneyu(method="auto")`).
pub fn mwu_one_vs_rest_pvalue(
    cluster_mask: &[bool],
    j: usize,
    pre: &NumericPrecompute,
) -> Result<f64, StatsError> {
    let valid = &pre.valid[j];
    let in_cluster = |i: usize| cluster_mask[i] && valid[i];
    let n1 = (0..valid.len()).filter(|&i| in_cluster(i)).count();
    let n2 = (0..valid.len())
        .filter(|&i| !cluster_mask[i] && valid[i])
        .count();
    if n1 == 0 || n2 == 0 {
        return Err(input("mannwhitneyu: at least one input has size 0"));
    }
    let tie_correction = pre.tie_correction[j];
    let has_ties = tie_correction < 1.0;
    if (n1 > 8 && n2 > 8) || has_ties {
        let r1: f64 = (0..valid.len())
            .filter(|&i| in_cluster(i))
            .map(|i| pre.ranks[j][i])
            .sum();
        let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
        let n_total = (n1 + n2) as f64;
        let tie_sum = (n_total.powi(3) - n_total) * (1.0 - tie_correction);
        return Ok(mwu_asymptotic_two_sided_pvalue(u1, n1, n2, tie_sum));
    }
    // Small and tie-free: exact, from the ranks of the combined (full) column.
    let r1: f64 = (0..valid.len())
        .filter(|&i| in_cluster(i))
        .map(|i| pre.ranks[j][i])
        .sum();
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u = u1.max((n1 * n2) as f64 - u1);
    Ok(mwu_exact_two_sided(u as usize, n1, n2))
}

/// Two-sided p-value for an arbitrary pair of samples (scipy
/// `mannwhitneyu(x, y, method="auto")`), ranking the combined sample afresh.
pub fn mwu_pair_pvalue(x: &[f64], y: &[f64]) -> Result<f64, StatsError> {
    if x.is_empty() || y.is_empty() {
        return Err(input("mannwhitneyu: at least one input has size 0"));
    }
    let (n1, n2) = (x.len(), y.len());
    let combined: Vec<f64> = x.iter().chain(y).copied().collect();
    let ranks = average_ranks(&combined);
    let s = sorted(&combined);
    let tie_sum = tie_cube_sum(&s);
    let r1: f64 = ranks[..n1].iter().sum();
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    if (n1 > 8 && n2 > 8) || tie_sum > 0.0 {
        return Ok(mwu_asymptotic_two_sided_pvalue(u1, n1, n2, tie_sum));
    }
    let u = u1.max((n1 * n2) as f64 - u1);
    Ok(mwu_exact_two_sided(u as usize, n1, n2))
}

/// `sum(t^3 - t)` over the tie groups of an ascending slice.
fn tie_cube_sum(sorted_values: &[f64]) -> f64 {
    let (mut sum, mut run) = (0.0, 1.0_f64);
    for w in sorted_values.windows(2) {
        if w[0] == w[1] {
            run += 1.0;
        } else {
            sum += run * run * run - run;
            run = 1.0;
        }
    }
    sum + run * run * run - run
}

// ---- Kruskal-Wallis, KS, Wasserstein ----------------------------------------------------

/// Kruskal-Wallis p-value from a column's ranks and group membership masks.
/// NaN when every value in the column is tied.
pub fn kruskal_wallis_pvalue(
    ranks_column: &[f64],
    group_masks: &[Vec<bool>],
    tie_correction: f64,
) -> Result<f64, StatsError> {
    let (mut rank_sums, mut sizes) = (vec![], vec![]);
    for mask in group_masks {
        let ranks: Vec<f64> = mask
            .iter()
            .zip(ranks_column)
            .filter(|(&in_group, _)| in_group)
            .map(|(_, &r)| r)
            .collect();
        if ranks.is_empty() {
            continue;
        }
        rank_sums.push(ranks.iter().sum::<f64>());
        sizes.push(ranks.len() as f64);
    }
    if rank_sums.len() < 2 {
        return Err(input(
            "Kruskal-Wallis requires at least two non-empty groups",
        ));
    }
    let n_total: f64 = sizes.iter().sum();
    let h_uncorrected = (12.0 / (n_total * (n_total + 1.0)))
        * rank_sums
            .iter()
            .zip(&sizes)
            .map(|(rs, gs)| rs * rs / gs)
            .sum::<f64>()
        - 3.0 * (n_total + 1.0);
    if tie_correction == 0.0 {
        return Ok(f64::NAN);
    }
    let h = h_uncorrected / tie_correction;
    Ok(chi2_sf(h, rank_sums.len() - 1))
}

/// `scipy.stats.chi2.sf`.
pub fn chi2_sf(x: f64, df: usize) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    match ChiSquared::new(df as f64) {
        // `sf` keeps precision in the far tail, where `1 - cdf` underflows to 0.
        Ok(d) => d.sf(x),
        Err(_) => f64::NAN,
    }
}

/// Pearson chi-square test of independence on a contingency table
/// (`scipy.stats.chi2_contingency`). With one degree of freedom and
/// `correction`, Yates' continuity correction is applied. Returns
/// `(statistic, p_value, dof)`.
pub fn chi2_contingency(
    table: &[Vec<f64>],
    correction: bool,
) -> Result<(f64, f64, usize), StatsError> {
    let rows = table.len();
    let cols = table.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return Err(input(
            "The table must have at least one row and one column.",
        ));
    }
    let row_sums: Vec<f64> = table.iter().map(|r| r.iter().sum()).collect();
    let col_sums: Vec<f64> = (0..cols)
        .map(|j| table.iter().map(|r| r[j]).sum())
        .collect();
    let total: f64 = row_sums.iter().sum();
    let dof = (rows - 1) * (cols - 1);
    let expected = |i: usize, j: usize| row_sums[i] * col_sums[j] / total;
    for i in 0..rows {
        for j in 0..cols {
            if expected(i, j) == 0.0 {
                return Err(StatsError::Input(format!(
                    "The internally computed table of expected frequencies has a zero element at ({i}, {j})."
                )));
            }
        }
    }
    if dof == 0 {
        return Ok((0.0, 1.0, 0));
    }
    let mut chi2 = 0.0;
    for (i, row) in table.iter().enumerate() {
        for (j, &obs) in row.iter().enumerate() {
            let exp = expected(i, j);
            let adjusted = if dof == 1 && correction {
                let diff = exp - obs;
                obs + diff.signum() * diff.abs().min(0.5)
            } else {
                obs
            };
            chi2 += (adjusted - exp) * (adjusted - exp) / exp;
        }
    }
    Ok((chi2, chi2_sf(chi2, dof), dof))
}

/// `ln(n!)`.
fn ln_factorial(n: usize) -> f64 {
    statrs::function::gamma::ln_gamma(n as f64 + 1.0)
}

/// Two-sided Fisher exact p-value of the 2x2 table `[[a, b], [c, d]]`
/// (`scipy.stats.fisher_exact`): the total probability of the tables, with the
/// same margins, that are no more likely than the observed one.
///
/// scipy finds the boundary of that set by binary search with a relative
/// tolerance of 1e-14; this sums the (small) support directly and treats
/// probabilities within 1e-9 of the observed one as ties, so that tables of
/// equal probability are not split by rounding.
pub fn fisher_exact_two_sided(table: [[u64; 2]; 2]) -> f64 {
    let [[a, b], [c, d]] = table;
    if a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0 {
        return 1.0;
    }
    let (n1, n2, n) = ((a + b) as usize, (c + d) as usize, (a + c) as usize);
    let total = n1 + n2;
    let log_total = ln_factorial(total) - ln_factorial(n) - ln_factorial(total - n);
    let lo = n.saturating_sub(n2);
    let hi = n.min(n1);
    let log_pmf = |x: usize| {
        (ln_factorial(n1) - ln_factorial(x) - ln_factorial(n1 - x))
            + (ln_factorial(n2) - ln_factorial(n - x) - ln_factorial(n2 - (n - x)))
            - log_total
    };
    let observed = log_pmf(a as usize);
    let threshold = observed + 1e-9;
    let p: f64 = (lo..=hi)
        .map(log_pmf)
        .filter(|&lp| lp <= threshold)
        .map(f64::exp)
        .sum();
    p.min(1.0)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Two-sample Kolmogorov-Smirnov statistic. For samples up to 10,000 values the
/// statistic is rationalised to the `lcm(n1, n2)` denominator, as scipy does, so
/// that it is the exact float scipy returns.
pub fn ks_two_sample_stat(a: &[f64], b: &[f64]) -> Result<f64, StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(input("ks_2samp: at least one input has size 0"));
    }
    let (sa, sb) = (sorted(a), sorted(b));
    let (n1, n2) = (sa.len(), sb.len());
    let cdf = |s: &[f64], v: f64| s.partition_point(|&x| x <= v) as f64 / s.len() as f64;
    let mut d = 0.0_f64;
    for &v in sa.iter().chain(&sb) {
        let diff = cdf(&sa, v) - cdf(&sb, v);
        d = d.max(diff.abs().clamp(0.0, 1.0));
    }
    if n1.max(n2) <= 10_000 {
        let lcm = (n1 / gcd(n1, n2)) * n2;
        d = (d * lcm as f64).round() / lcm as f64;
    }
    Ok(d)
}

/// Welch's unequal-variance t-test (`scipy.stats.ttest_ind(equal_var=False)`):
/// `(t, two_sided_p)`. NaN when both samples are constant and equal.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let (v1, v2) = (np_var(a, 1), np_var(b, 1));
    let (vn1, vn2) = (v1 / n1, v2 / n2);
    let mut df = (vn1 + vn2).powi(2) / (vn1 * vn1 / (n1 - 1.0) + vn2 * vn2 / (n2 - 1.0));
    if df.is_nan() {
        df = 1.0;
    }
    let t = (np_mean(a) - np_mean(b)) / (vn1 + vn2).sqrt();
    if t.is_nan() {
        return (t, f64::NAN);
    }
    // P(|T| >= |t|) = I_x(df / 2, 1 / 2) with x = df / (df + t^2): exact in the far tail.
    let x = df / (df + t * t);
    let p = statrs::function::beta::beta_reg(df / 2.0, 0.5, x);
    (t, p.clamp(0.0, 1.0))
}

/// Probability that the two-sample KS statistic of equal-sized samples reaches
/// `h / n` (scipy `_compute_prob_outside_square`).
fn ks_prob_outside_square(n: usize, h: usize) -> f64 {
    let mut p = 0.0;
    let mut k = (n / h) as i64;
    while k >= 0 {
        let mut p1 = 1.0;
        for j in 0..h {
            p1 = ((n as i64 - k * h as i64 - j as i64) as f64) * p1
                / ((n as i64 + k * h as i64 + j as i64 + 1) as f64);
        }
        p = p1 * (1.0 - p);
        k -= 1;
    }
    2.0 * p
}

/// Probability that the KS statistic of samples of sizes `m` and `n` reaches
/// `h / lcm(m, n)`: one minus the probability that a uniformly random lattice
/// path from `(0, 0)` to `(m, n)` stays strictly inside `|n x - m y| < h g`
/// (Hodges' treatment; `g = gcd(m, n)`). The inside probability is propagated
/// step by step with the hypergeometric transition probabilities, so counts
/// never overflow.
fn ks_prob_outside_inside_method(m: usize, n: usize, g: usize, h: usize) -> f64 {
    let (m, n) = if m < n { (n, m) } else { (m, n) };
    let bound = (h * g) as i64;
    let inside = |x: usize, y: usize| ((n * x) as i64 - (m * y) as i64).abs() < bound;
    let mut prev = vec![0.0_f64; n + 1];
    let mut cur = vec![0.0_f64; n + 1];
    for x in 0..=m {
        cur.fill(0.0);
        for y in 0..=n {
            if !inside(x, y) {
                continue;
            }
            let mut q = if x == 0 && y == 0 { 1.0 } else { 0.0 };
            if x > 0 {
                let (rem_x, rem_y) = ((m - x + 1) as f64, (n - y) as f64);
                q += prev[y] * rem_x / (rem_x + rem_y);
            }
            if y > 0 {
                let (rem_x, rem_y) = ((m - x) as f64, (n - y + 1) as f64);
                q += cur[y - 1] * rem_y / (rem_x + rem_y);
            }
            cur[y] = q;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    (1.0 - prev[n]).clamp(0.0, 1.0)
}

/// Two-sided two-sample Kolmogorov-Smirnov p-value (`scipy.stats.ks_2samp`,
/// method `auto`). Exact for up to 10,000 values per sample; above that scipy
/// switches to the finite-sample distribution of the one-sample statistic,
/// which is not ported here: this uses the Kolmogorov limiting distribution
/// instead (accurate when the effective sample size is large).
pub fn ks_two_sample_pvalue(a: &[f64], b: &[f64]) -> Result<f64, StatsError> {
    let d = ks_two_sample_stat(a, b)?;
    let (n1, n2) = (a.len(), b.len());
    let g = gcd(n1, n2);
    let lcm = (n1 / g) * n2;
    let h = (d * lcm as f64).round() as usize;
    if h == 0 {
        return Ok(1.0);
    }
    if n1.max(n2) <= 10_000 {
        let p = if n1 == n2 {
            ks_prob_outside_square(n1, h)
        } else {
            ks_prob_outside_inside_method(n1, n2, g, h)
        };
        if (0.0..=1.0).contains(&p) {
            return Ok(p);
        }
        // scipy rejects a probability that rounds to just above 1 and falls back
        // to the asymptotic value, which is 1 there.
        if p > 1.0 && p < 1.0 + 1e-9 {
            return Ok(1.0);
        }
    }
    let en = (n1 * n2) as f64 / (n1 + n2) as f64;
    let lambda = en.sqrt() * d;
    let series: f64 = (1..=100)
        .map(|k| {
            let k = k as f64;
            (if (k as i64) % 2 == 1 { 2.0 } else { -2.0 }) * (-2.0 * k * k * lambda * lambda).exp()
        })
        .sum();
    Ok(series.clamp(0.0, 1.0))
}

/// 1-D Wasserstein (earth-mover) distance between two samples.
pub fn wasserstein_distance_1d(a: &[f64], b: &[f64]) -> Result<f64, StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(input("wasserstein: at least one input has size 0"));
    }
    let all = sorted(&a.iter().chain(b).copied().collect::<Vec<_>>());
    let (sa, sb) = (sorted(a), sorted(b));
    let cdf = |s: &[f64], v: f64| s.partition_point(|&x| x <= v) as f64 / s.len() as f64;
    let terms: Vec<f64> = all
        .windows(2)
        .map(|w| (cdf(&sa, w[0]) - cdf(&sb, w[0])).abs() * (w[1] - w[0]))
        .collect();
    Ok(pairwise_sum(&terms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bh_fdr_is_monotone_and_keeps_missing() {
        let out = bh_fdr(&[Some(0.01), None, Some(0.04), Some(0.03), Some(f64::NAN)]);
        assert_eq!(out[1], None);
        assert_eq!(out[4], None);
        // Sorted p: .01, .03, .04 with m = 3 -> .03, .045, .04 -> running min from the top.
        assert_eq!(out[0], Some(0.03));
        assert_eq!(out[3], Some(0.04));
        assert_eq!(out[2], Some(0.04));
    }

    #[test]
    fn percentiles_collapse_constants_and_rank_otherwise() {
        assert_eq!(empirical_percentiles(&[2.0, 2.0, 2.0]), [0.0; 3]);
        assert_eq!(empirical_percentiles(&[3.0, 1.0, 2.0]), [1.0, 0.0, 0.5]);
        assert_eq!(empirical_percentiles(&[]), Vec::<f64>::new());
    }

    #[test]
    fn exact_mwu_matches_a_hand_count() {
        // m = n = 2: U takes 0..=4 with counts 1, 1, 2, 1, 1 of 6.
        assert!((mwu_exact_two_sided(4, 2, 2) - 2.0 / 6.0).abs() < 1e-12);
        assert!((mwu_exact_two_sided(3, 2, 2) - 2.0 * 2.0 / 6.0).abs() < 1e-12);
        assert_eq!(mwu_exact_two_sided(2, 2, 2), 1.0);
        // Perfect separation of 3 vs 3: 2 / 20.
        assert!((mwu_exact_two_sided(9, 3, 3) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fisher_matches_the_scipy_documentation_examples() {
        // scipy.stats.fisher_exact docstring values.
        assert!((fisher_exact_two_sided([[6, 2], [1, 4]]) - 0.10256410256410257).abs() < 1e-12);
        assert!((fisher_exact_two_sided([[8, 2], [1, 5]]) - 0.034965034965034975).abs() < 1e-12);
        // A zero margin has no table to test.
        assert_eq!(fisher_exact_two_sided([[0, 0], [3, 4]]), 1.0);
        assert_eq!(fisher_exact_two_sided([[2, 0], [3, 0]]), 1.0);
    }

    #[test]
    fn chi2_contingency_applies_yates_only_with_one_degree_of_freedom() {
        // 2x2 with correction, checked against scipy: chi2 = 0.0, p = 1.0 at independence.
        let (chi2, p, dof) = chi2_contingency(&[vec![10.0, 10.0], vec![20.0, 20.0]], true).unwrap();
        assert_eq!((chi2, p, dof), (0.0, 1.0, 1));
        // A zero row makes an expected frequency zero.
        let e = chi2_contingency(&[vec![1.0, 2.0], vec![0.0, 0.0]], true).unwrap_err();
        assert!(e.to_string().starts_with(
            "The internally computed table of expected frequencies has a zero element"
        ));
        // Far tail keeps its precision instead of rounding to zero.
        assert!(chi2_sf(100.0, 1) > 0.0 && chi2_sf(100.0, 1) < 1e-20);
    }

    #[test]
    fn pairwise_sum_follows_numpy_association_order() {
        // Below 8 values: plain left-to-right from zero.
        assert_eq!(pairwise_sum(&[0.1, 0.2, 0.3]), (0.1 + 0.2) + 0.3);
        // Eight values: ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)).
        let a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        assert_eq!(
            pairwise_sum(&a),
            ((0.1 + 0.2) + (0.3 + 0.4)) + ((0.5 + 0.6) + (0.7 + 0.8))
        );
        // Nine values: the ninth is added after the unrolled eight.
        let b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        assert_eq!(
            pairwise_sum(&b),
            (((0.1 + 0.2) + (0.3 + 0.4)) + ((0.5 + 0.6) + (0.7 + 0.8))) + 0.9
        );
        assert_eq!(pairwise_sum(&[]), 0.0);
    }

    #[test]
    fn ranks_average_ties_and_skip_nan() {
        let r = average_ranks(&[3.0, f64::NAN, 1.0, 3.0, 2.0]);
        assert_eq!(r[0], 3.5);
        assert!(r[1].is_nan());
        assert_eq!((r[2], r[3], r[4]), (1.0, 3.5, 2.0));
    }

    #[test]
    fn degenerate_inputs_return_the_python_answers() {
        assert_eq!(
            mwu_pair_pvalue(&[], &[1.0]).unwrap_err().to_string(),
            "mannwhitneyu: at least one input has size 0"
        );
        assert_eq!(
            ks_two_sample_stat(&[1.0], &[]).unwrap_err().to_string(),
            "ks_2samp: at least one input has size 0"
        );
        assert_eq!(
            wasserstein_distance_1d(&[], &[]).unwrap_err().to_string(),
            "wasserstein: at least one input has size 0"
        );
        assert_eq!(wasserstein_distance_1d(&[1.0, 1.0], &[1.0]).unwrap(), 0.0);
        assert_eq!(direction_from_effects(&[1.0, 2.0]), "up");
        assert_eq!(direction_from_effects(&[-1.0]), "down");
        assert_eq!(direction_from_effects(&[1.0, -1.0]), "mixed");
        assert_eq!(direction_from_effects(&[0.0]), "mixed");
    }
}
