//! Duplicate, `sum_of` and strong-correlation relations between numeric
//! columns (`_compute_relations` and helpers).

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, SeedableRng};

use super::{ColumnRelation, EvidenceOptions, RelationKind, SkippedStep};

const SUM_COEFF_TOL: f64 = 1e-6;
const SUM_OTHER_COEFF_TOL: f64 = 1e-4;
const SUM_RESIDUAL_REL_TOL: f64 = 1e-9;

/// Sample `size` of `n` row indices without replacement, ascending.
fn sample_rows(n: usize, size: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx = rand::seq::index::sample(&mut rng, n, size).into_vec();
    idx.sort_unstable();
    idx
}

/// `(relations, rows_sampled, numeric_columns_considered, steps_skipped)`.
pub(super) fn measure_relations(
    n_rows: usize,
    names: &[String],
    arrays: &[Vec<f64>],
    opts: &EvidenceOptions,
) -> (Vec<ColumnRelation>, usize, usize, Vec<SkippedStep>) {
    let rows_sampled = n_rows.min(opts.sample_rows.max(1));
    let n_num = names.len();
    let skip = |reason: String| {
        vec![SkippedStep {
            step: "relations".into(),
            reason,
        }]
    };

    if n_rows < 2 {
        return (
            vec![],
            rows_sampled,
            0,
            skip("fewer than 2 rows in dataset".into()),
        );
    }
    if n_num > opts.max_relation_columns {
        return (
            vec![],
            rows_sampled,
            0,
            skip(format!(
                "numeric/boolean column count ({n_num}) exceeds max_relation_columns ({})",
                opts.max_relation_columns
            )),
        );
    }
    if n_num < 2 {
        return (
            vec![],
            rows_sampled,
            n_num,
            skip("fewer than 2 numeric or boolean columns".into()),
        );
    }

    let rows: Vec<usize> = if n_rows > rows_sampled {
        sample_rows(n_rows, rows_sampled, opts.seed)
    } else {
        (0..n_rows).collect()
    };
    let x = Array2::from_shape_fn((rows.len(), n_num), |(i, j)| arrays[j][rows[i]]);
    let relations = compute_relations(names, x, opts.max_sum_sources, opts.max_correlation_pairs);
    (relations, rows_sampled, n_num, vec![])
}

/// Non-finite values become NaN; columns with fewer than two observations or
/// no spread are inactive.
fn active_columns(x: &mut Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    x.mapv_inplace(|v| if v.is_finite() { v } else { f64::NAN });
    let mut obs_counts = Vec::with_capacity(x.ncols());
    let mut active = Vec::new();
    for (j, col) in x.axis_iter(Axis(1)).enumerate() {
        let obs: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
        obs_counts.push(obs.len());
        if obs.len() < 2 {
            continue;
        }
        let lo = obs.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = obs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo < hi {
            active.push(j);
        }
    }
    (obs_counts, active)
}

fn find_duplicates(
    names: &[String],
    x: &Array2<f64>,
    obs_counts: &[usize],
    active: &[usize],
) -> (Vec<ColumnRelation>, HashSet<usize>, HashSet<(usize, usize)>) {
    let mut duplicates = Vec::new();
    let mut targets = HashSet::new();
    let mut reported = HashSet::new();
    let mut buckets: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    let nan_bits = f64::NAN.to_bits();

    for &j in active {
        // Canonicalise -0.0 to 0.0 and every NaN to one bit pattern.
        let key: Vec<u64> = x
            .column(j)
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    nan_bits
                } else {
                    (v + 0.0).to_bits()
                }
            })
            .collect();
        match buckets.get_mut(&key) {
            None => {
                buckets.insert(key, vec![j]);
            }
            Some(bucket) => {
                duplicates.push(ColumnRelation {
                    kind: RelationKind::Duplicate,
                    target: names[j].clone(),
                    sources: vec![names[bucket[0]].clone()],
                    value: 0.0,
                    rows_used: obs_counts[j],
                });
                targets.insert(j);
                for &prev in bucket.iter() {
                    reported.insert((prev.min(j), prev.max(j)));
                }
                bucket.push(j);
            }
        }
    }
    (duplicates, targets, reported)
}

/// `(|R|, valid_corr, N)` over pairwise-complete observations, each column
/// pre-standardised on its own observed mean and spread.
fn pairwise_correlation(x: &Array2<f64>) -> (Array2<f64>, Array2<bool>, Array2<f64>) {
    let m: Array2<f64> = x.mapv(|v| if v.is_nan() { 0.0 } else { 1.0 });
    let col_n = m.sum_axis(Axis(0)).mapv(|n| n.max(1.0));
    let x_zero = Array2::from_shape_fn(
        x.dim(),
        |(i, j)| if m[[i, j]] > 0.0 { x[[i, j]] } else { 0.0 },
    );
    let col_mean = x_zero.sum_axis(Axis(0)) / &col_n;
    let x_cent = Array2::from_shape_fn(x.dim(), |(i, j)| {
        if m[[i, j]] > 0.0 {
            x[[i, j]] - col_mean[j]
        } else {
            0.0
        }
    });
    let col_std: Array1<f64> =
        (x_cent.mapv(|v| v * v).sum_axis(Axis(0)) / &col_n).mapv(|v| v.max(1e-30).sqrt());
    let z = Array2::from_shape_fn(x.dim(), |(i, j)| x_cent[[i, j]] / col_std[j]);

    let n_mat = m.t().dot(&m);
    let s_mat = z.t().dot(&m);
    let p_mat = z.t().dot(&z);
    let q_mat = z.mapv(|v| v * v).t().dot(&m);

    let p = x.ncols();
    let mut abs_r = Array2::<f64>::zeros((p, p));
    let mut valid = Array2::from_elem((p, p), false);
    for a in 0..p {
        for b in 0..p {
            let n_safe = n_mat[[a, b]].max(1.0);
            let mean_x = s_mat[[a, b]] / n_safe;
            let mean_y = s_mat[[b, a]] / n_safe;
            let cov = p_mat[[a, b]] - n_mat[[a, b]] * mean_x * mean_y;
            let var_x = q_mat[[a, b]] - n_mat[[a, b]] * mean_x * mean_x;
            let var_y = q_mat[[b, a]] - n_mat[[a, b]] * mean_y * mean_y;
            let denom = (var_x * var_y).max(0.0).sqrt();
            let ok = n_mat[[a, b]] >= 3.0
                && var_x > 1e-12 * n_safe
                && var_y > 1e-12 * n_safe
                && denom > 0.0;
            valid[[a, b]] = ok;
            if ok && a != b {
                abs_r[[a, b]] = (cov / denom).clamp(-1.0, 1.0).abs();
            }
        }
    }
    (abs_r, valid, n_mat)
}

/// Indices of the `k` smallest scores, ties at the k-th value resolved by
/// ascending index (`_k_smallest_with_ties`).
fn k_smallest_with_ties(scores: &[f64], k: usize) -> Vec<usize> {
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    let kth = sorted[k - 1];
    let mut out: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] < kth).collect();
    let need = k - out.len();
    out.extend((0..scores.len()).filter(|&i| scores[i] == kth).take(need));
    out
}

/// Least-squares coefficients of `y ~ cols` (numpy `lstsq`, `rcond=None`).
fn lstsq(cols: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let (m, n) = (y.len(), cols.len());
    let a = DMatrix::from_fn(m, n, |i, j| cols[j][i]);
    let svd = a.svd(true, true);
    let max_sv = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let eps = f64::EPSILON * m.max(n) as f64 * max_sv;
    let beta = svd.solve(&DVector::from_column_slice(y), eps).ok()?;
    Some(beta.iter().copied().collect())
}

fn find_sum_relations(
    names: &[String],
    x_act: &Array2<f64>,
    active: &[usize],
    abs_r: &Array2<f64>,
    duplicate_targets: &HashSet<usize>,
    reported: &mut HashSet<(usize, usize)>,
    max_sum_sources: usize,
) -> Vec<ColumnRelation> {
    let n_act = active.len();
    let mut out = Vec::new();
    let mut eligible: Vec<bool> = active
        .iter()
        .map(|j| !duplicate_targets.contains(j))
        .collect();
    let non_dup: Vec<usize> = (0..n_act).filter(|&p| eligible[p]).collect();
    if non_dup.len() < 3 || max_sum_sources < 2 {
        return out;
    }

    for &t_pos in &non_dup {
        let orig_t = active[t_pos];
        let n_cands = eligible.iter().filter(|&&e| e).count() - usize::from(eligible[t_pos]);
        if n_cands < 2 {
            continue;
        }
        let mut neg: Vec<f64> = (0..n_act)
            .map(|p| if eligible[p] { -abs_r[[t_pos, p]] } else { 1.0 })
            .collect();
        neg[t_pos] = 1.0;
        let k_take = max_sum_sources.min(n_cands);
        let mut pool: Vec<usize> = if n_act > k_take {
            k_smallest_with_ties(&neg, k_take)
        } else {
            (0..n_act).filter(|&p| neg[p] <= 0.0).collect()
        };
        pool.sort_by(|&a, &b| neg[a].total_cmp(&neg[b]).then(a.cmp(&b)));
        let top = pool;
        if top.len() < 2 {
            continue;
        }

        let y = x_act.column(t_pos);
        let joint: Vec<usize> = (0..x_act.nrows())
            .filter(|&i| !y[i].is_nan() && top.iter().all(|&c| !x_act[[i, c]].is_nan()))
            .collect();
        if joint.len() <= 2 * top.len() {
            continue;
        }
        let cols: Vec<Vec<f64>> = top
            .iter()
            .map(|&c| joint.iter().map(|&i| x_act[[i, c]]).collect())
            .collect();
        let y_joint: Vec<f64> = joint.iter().map(|&i| y[i]).collect();
        let Some(coeffs) = lstsq(&cols, &y_joint) else {
            continue;
        };

        let unit: Vec<bool> = coeffs
            .iter()
            .map(|c| (c - 1.0).abs() <= SUM_COEFF_TOL)
            .collect();
        if unit.iter().filter(|&&u| u).count() < 2 {
            continue;
        }
        let other_max = coeffs
            .iter()
            .zip(&unit)
            .filter(|(_, &u)| !u)
            .map(|(c, _)| c.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        if other_max.is_finite() && other_max > SUM_OTHER_COEFF_TOL {
            continue;
        }

        let chosen: Vec<usize> = top
            .iter()
            .zip(&unit)
            .filter(|(_, &u)| u)
            .map(|(&c, _)| c)
            .collect();
        let recheck: Vec<usize> = (0..x_act.nrows())
            .filter(|&i| !y[i].is_nan() && chosen.iter().all(|&c| !x_act[[i, c]].is_nan()))
            .collect();
        if recheck.len() <= 2 * top.len() {
            continue;
        }
        let max_res = recheck
            .iter()
            .map(|&i| (y[i] - chosen.iter().map(|&c| x_act[[i, c]]).sum::<f64>()).abs())
            .fold(0.0, f64::max);
        let max_target = recheck.iter().map(|&i| y[i].abs()).fold(0.0, f64::max);
        if max_res <= SUM_RESIDUAL_REL_TOL * (1.0 + max_target) {
            let mut chosen_orig: Vec<usize> = chosen.iter().map(|&p| active[p]).collect();
            chosen_orig.sort_unstable();
            out.push(ColumnRelation {
                kind: RelationKind::SumOf,
                target: names[orig_t].clone(),
                sources: chosen_orig.iter().map(|&i| names[i].clone()).collect(),
                value: max_res,
                rows_used: recheck.len(),
            });
            eligible[t_pos] = false;
            for &src in &chosen_orig {
                reported.insert((src.min(orig_t), src.max(orig_t)));
            }
        }
    }
    out
}

fn top_correlations(
    names: &[String],
    active: &[usize],
    abs_r: &Array2<f64>,
    valid: &Array2<bool>,
    n_mat: &Array2<f64>,
    reported: &HashSet<(usize, usize)>,
    max_pairs: usize,
) -> Vec<ColumnRelation> {
    if max_pairs == 0 {
        return vec![];
    }
    let n_act = active.len();
    let mut eligible = valid.clone();
    let pos_of: HashMap<usize, usize> = active.iter().enumerate().map(|(p, &o)| (o, p)).collect();
    for &(a, b) in reported {
        if let (Some(&pa), Some(&pb)) = (pos_of.get(&a), pos_of.get(&b)) {
            eligible[[pa, pb]] = false;
            eligible[[pb, pa]] = false;
        }
    }

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_act {
        for j in i + 1..n_act {
            if eligible[[i, j]] {
                pairs.push((i, j));
            }
        }
    }
    if pairs.is_empty() {
        return vec![];
    }
    let neg: Vec<f64> = pairs.iter().map(|&(i, j)| -abs_r[[i, j]]).collect();
    let k = max_pairs.min(pairs.len());
    let mut sel: Vec<usize> = if pairs.len() > k {
        k_smallest_with_ties(&neg, k)
    } else {
        (0..pairs.len()).collect()
    };
    sel.sort_by(|&a, &b| {
        neg[a]
            .total_cmp(&neg[b])
            .then(pairs[a].0.cmp(&pairs[b].0))
            .then(pairs[a].1.cmp(&pairs[b].1))
    });
    sel.into_iter()
        .map(|s| {
            let (a, b) = pairs[s];
            ColumnRelation {
                kind: RelationKind::StrongCorrelation,
                target: names[active[b]].clone(),
                sources: vec![names[active[a]].clone()],
                value: -neg[s],
                rows_used: n_mat[[a, b]] as usize,
            }
        })
        .collect()
}

fn compute_relations(
    names: &[String],
    mut x: Array2<f64>,
    max_sum_sources: usize,
    max_correlation_pairs: usize,
) -> Vec<ColumnRelation> {
    if x.nrows() < 2 || x.ncols() < 2 {
        return vec![];
    }
    let (obs_counts, active) = active_columns(&mut x);
    if active.len() < 2 {
        return vec![];
    }
    let (duplicates, dup_targets, mut reported) = find_duplicates(names, &x, &obs_counts, &active);
    let x_act = x.select(Axis(1), &active);
    let (abs_r, valid, n_mat) = pairwise_correlation(&x_act);
    let sums = find_sum_relations(
        names,
        &x_act,
        &active,
        &abs_r,
        &dup_targets,
        &mut reported,
        max_sum_sources,
    );
    let corrs = top_correlations(
        names,
        &active,
        &abs_r,
        &valid,
        &n_mat,
        &reported,
        max_correlation_pairs,
    );
    let mut all = duplicates;
    all.extend(sums);
    all.extend(corrs);
    all
}
