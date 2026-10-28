//! Per-cluster feature evidence (`build_feature_evidence_index` and the helpers
//! it calls in `pulsar/mcp/interpreter.py`, lines ~1110-2200).
//!
//! For every cluster the index holds, for each numeric column, how the cluster
//! differs from the rest of the data and from its strongest neighbour (effect
//! sizes, KS and Wasserstein distances, Mann-Whitney and Kruskal-Wallis tests
//! with FDR correction) and, for each categorical value, its prevalence, lift,
//! mutual-information contribution and a chi-square or Fisher test. Each row
//! gets an aggregate score (the geometric mean of its metric percentiles) and a
//! signal tier within its cluster. Measurements only: no domain meaning.
//!
//! Known divergences from Python:
//! - Tiers come from the exact optimal 1-D k-means partition of the positive
//!   aggregate scores (dynamic programming); scikit-learn's `KMeans(n_init=10,
//!   random_state=42)` reaches the same optimum on these inputs but is a
//!   randomized search.
//! - Fisher's exact test sums the support directly (see
//!   [`crate::stats::fisher_exact_two_sided`]).
//! - Node `i` of the graph is row `i` of the data, as in a default index.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use indexmap::IndexMap;
use pulsar_data::{Column, Table};
use serde::Serialize;
use serde_json::{json, Value};

use crate::diagnostics::distribution_sparkline;
use crate::stats::{
    bh_fdr, build_numeric_precompute, chi2_contingency, direction_from_effects,
    empirical_percentiles, fisher_exact_two_sided, kruskal_wallis_pvalue, ks_two_sample_stat,
    mwu_one_vs_rest_pvalue, mwu_pair_pvalue, np_mean, np_var, pooled_std, safe_mad, safe_mean,
    safe_median, safe_std, safe_var, wasserstein_distance_1d, NumericPrecompute, EPS,
};

/// Categorical columns with more distinct values than
/// `max(MIN_CARDINALITY_FLOOR, MAX_CARDINALITY_RATIO * rows)` are gated out.
pub const MAX_CARDINALITY_RATIO: f64 = 0.05;
pub const MIN_CARDINALITY_FLOOR: f64 = 10.0;
/// Cochran's rule: chi-square is trusted only when every expected 2x2 cell is
/// at least this large; below it Fisher's exact test is used.
pub const CHI2_MIN_EXPECTED_CELL: f64 = 5.0;
const MISSING_LABEL: &str = "__MISSING__";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvidenceIndexError {
    #[error("Alignment error: clusters({clusters}) != data({rows})")]
    Alignment { clusters: usize, rows: usize },
}

// ---- cluster adjacency ----------------------------------------------------------

/// A neighbouring cluster and the strength of the bridge to it.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct Neighbor {
    pub cluster_id: i64,
    pub bridge_weight: f64,
    pub normalized_weight: f64,
    pub bridge_strength: f64,
}

pub type Adjacency = BTreeMap<i64, Vec<Neighbor>>;

/// Aggregate the cosmic graph's edges `(source, target, weight)` into
/// cluster-to-cluster bridge strengths, strongest first.
pub fn graph_cluster_adjacency(edges: &[(usize, usize, f64)], clusters: &[i64]) -> Adjacency {
    let mut sizes: BTreeMap<i64, usize> = BTreeMap::new();
    for &c in clusters {
        *sizes.entry(c).or_default() += 1;
    }
    let mut internal: BTreeMap<i64, f64> = sizes.keys().map(|&c| (c, 0.0)).collect();
    let mut pairs: IndexMap<(i64, i64), f64> = IndexMap::new();
    for &(s, t, weight) in edges {
        let (Some(&cs), Some(&ct)) = (clusters.get(s), clusters.get(t)) else {
            continue;
        };
        if cs == ct {
            *internal.entry(cs).or_default() += weight;
        } else {
            *pairs.entry((cs.min(ct), cs.max(ct))).or_default() += weight;
        }
    }
    let mut adjacency: Adjacency = sizes.keys().map(|&c| (c, vec![])).collect();
    for (&(a, b), &bridge) in &pairs {
        let size = |c: i64| sizes.get(&c).copied().unwrap_or(1).max(1) as f64;
        let normalized = bridge / (size(a) * size(b)).sqrt().max(1.0);
        let strength = bridge
            / (bridge
                + internal.get(&a).copied().unwrap_or(0.0)
                + internal.get(&b).copied().unwrap_or(0.0))
            .max(EPS);
        let edge = |cluster_id| Neighbor {
            cluster_id,
            bridge_weight: bridge,
            normalized_weight: normalized,
            bridge_strength: strength,
        };
        adjacency.entry(a).or_default().push(edge(b));
        adjacency.entry(b).or_default().push(edge(a));
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_by(|x, y| {
            y.normalized_weight
                .total_cmp(&x.normalized_weight)
                .then(y.bridge_strength.total_cmp(&x.bridge_strength))
                .then(x.cluster_id.cmp(&y.cluster_id))
        });
    }
    adjacency
}

fn strongest_neighbor(adjacency: &Adjacency, cluster_id: i64) -> Option<i64> {
    adjacency.get(&cluster_id)?.first().map(|n| n.cluster_id)
}

// ---- rows ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NumericRow {
    pub column: String,
    pub cluster_id: i64,
    pub direction: String,
    pub mean: f64,
    pub mean_rest: f64,
    pub median: f64,
    pub median_rest: f64,
    pub std_cluster: f64,
    pub std_rest: f64,
    pub mad_cluster: f64,
    pub mad_rest: f64,
    pub global_mean: f64,
    pub global_mean_rest: f64,
    pub z_score: f64,
    pub homogeneity: f64,
    pub effect_mean_std: f64,
    pub effect_median_mad: f64,
    pub ks_stat: f64,
    pub wasserstein_norm: f64,
    pub variance_ratio_log: f64,
    pub concentration_gain: f64,
    pub one_vs_rest_p: f64,
    pub one_vs_rest_q: f64,
    pub global_p: f64,
    pub global_q: f64,
    pub global_cluster_signal: f64,
    pub neighbor_cluster_id: Option<i64>,
    pub neighbor_effect: f64,
    pub neighbor_p: f64,
    pub specificity_score: f64,
    pub aggregate_score: f64,
    pub percentile_score: f64,
    pub signal_tier: String,
    pub sparkline: String,
    pub failure_reasons: Vec<String>,
    pub evidence_vector: IndexMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CategoricalRow {
    pub column: String,
    pub value: String,
    pub cluster_id: i64,
    pub count: usize,
    pub count_cluster: usize,
    pub count_rest: usize,
    pub cluster_size: usize,
    pub global_count: usize,
    pub prevalence_cluster: f64,
    pub prevalence_rest: f64,
    pub prevalence_global: f64,
    pub lift: f64,
    pub log_lift: f64,
    pub specificity: f64,
    pub global_recall: f64,
    pub neighbor_specificity: f64,
    pub mi_contrib: f64,
    pub fisher_p: f64,
    pub fisher_q: f64,
    pub global_p: f64,
    pub global_q: f64,
    pub aggregate_score: f64,
    pub percentile_score: f64,
    pub signal_tier: String,
    pub in_cluster_prevalence: f64,
    pub concentration: f64,
    pub test_method: String,
    pub failure_reasons: Vec<String>,
    pub evidence_vector: IndexMap<String, f64>,
}

/// What the scoring pass needs from either kind of row.
trait Scored {
    const METRICS: &'static [&'static str];
    fn metrics(&self) -> Vec<f64>;
    fn set_scores(&mut self, aggregate: f64, evidence: IndexMap<String, f64>);
    fn set_percentile(&mut self, percentile: f64);
    fn aggregate(&self) -> f64;
    fn set_tier(&mut self, tier: &str);
}

fn neg_log10(q: f64) -> f64 {
    -(q + EPS).log10()
}

impl Scored for NumericRow {
    const METRICS: &'static [&'static str] = &[
        "effect_mean_std",
        "effect_median_mad",
        "ks_stat",
        "wasserstein_norm",
        "concentration_gain",
        "specificity_score",
        "one_vs_rest_sig",
        "global_sig",
        "neighbor_effect",
    ];
    fn metrics(&self) -> Vec<f64> {
        vec![
            self.effect_mean_std.abs(),
            self.effect_median_mad.abs(),
            self.ks_stat,
            self.wasserstein_norm,
            self.concentration_gain.max(0.0),
            self.specificity_score,
            neg_log10(self.one_vs_rest_q),
            neg_log10(self.global_q),
            self.neighbor_effect.abs(),
        ]
    }
    fn set_scores(&mut self, aggregate: f64, evidence: IndexMap<String, f64>) {
        self.aggregate_score = aggregate;
        self.evidence_vector = evidence;
    }
    fn set_percentile(&mut self, p: f64) {
        self.percentile_score = p;
    }
    fn aggregate(&self) -> f64 {
        self.aggregate_score
    }
    fn set_tier(&mut self, tier: &str) {
        self.signal_tier = tier.to_owned();
    }
}

impl Scored for CategoricalRow {
    const METRICS: &'static [&'static str] = &[
        "prevalence_cluster",
        "log_lift",
        "specificity",
        "global_recall",
        "neighbor_specificity",
        "mi_contrib",
        "fisher_sig",
        "global_sig",
    ];
    fn metrics(&self) -> Vec<f64> {
        vec![
            self.prevalence_cluster,
            self.log_lift.abs(),
            self.specificity.abs(),
            self.global_recall,
            self.neighbor_specificity.abs(),
            self.mi_contrib.abs(),
            neg_log10(self.fisher_q),
            neg_log10(self.global_q),
        ]
    }
    fn set_scores(&mut self, aggregate: f64, evidence: IndexMap<String, f64>) {
        self.aggregate_score = aggregate;
        self.evidence_vector = evidence;
    }
    fn set_percentile(&mut self, p: f64) {
        self.percentile_score = p;
    }
    fn aggregate(&self) -> f64 {
        self.aggregate_score
    }
    fn set_tier(&mut self, tier: &str) {
        self.signal_tier = tier.to_owned();
    }
}

/// Python's `max(a, b)`: keeps `a` unless `b` is strictly greater (so NaN stays).
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Fill `aggregate_score`, `evidence_vector` and `percentile_score` on every row:
/// the aggregate is the geometric mean of the row's metric percentiles (offset
/// by a small positive floor); the percentile is the aggregate's own percentile.
fn apply_percentile_aggregate<R: Scored>(rows: &mut [R]) {
    let metrics: Vec<Vec<f64>> = rows.iter().map(Scored::metrics).collect();
    let columns: Vec<Vec<f64>> = (0..R::METRICS.len())
        .map(|m| empirical_percentiles(&metrics.iter().map(|r| r[m]).collect::<Vec<_>>()))
        .collect();
    let mut aggregates = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter_mut().enumerate() {
        let vector: IndexMap<String, f64> = R::METRICS
            .iter()
            .enumerate()
            .map(|(m, name)| ((*name).to_owned(), columns[m][i]))
            .collect();
        let logs: Vec<f64> = vector
            .values()
            .map(|&v| (py_max(v, 0.0) + 1e-6).ln())
            .collect();
        let aggregate = if logs.is_empty() {
            0.0
        } else {
            np_mean(&logs).exp() - 1e-6
        };
        let aggregate = py_max(aggregate, 0.0);
        row.set_scores(aggregate, vector);
        aggregates.push(aggregate);
    }
    for (row, p) in rows.iter_mut().zip(empirical_percentiles(&aggregates)) {
        row.set_percentile(p);
    }
}

/// Labels of the optimal partition of `values` into `k` contiguous groups of
/// sorted values (minimum within-group sum of squares): 0 is the lowest group.
fn kmeans_1d(values: &[f64], k: usize) -> Vec<usize> {
    let mut unique: Vec<f64> = values.to_vec();
    unique.sort_by(f64::total_cmp);
    unique.dedup();
    let m = unique.len();
    let weight = |v: f64| values.iter().filter(|&&x| x == v).count() as f64;
    let w: Vec<f64> = unique.iter().map(|&v| weight(v)).collect();
    let (mut pw, mut ps, mut pq) = (vec![0.0; m + 1], vec![0.0; m + 1], vec![0.0; m + 1]);
    for i in 0..m {
        pw[i + 1] = pw[i] + w[i];
        ps[i + 1] = ps[i] + w[i] * unique[i];
        pq[i + 1] = pq[i] + w[i] * unique[i] * unique[i];
    }
    let cost = |i: usize, j: usize| {
        let (cw, cs) = (pw[j] - pw[i], ps[j] - ps[i]);
        (pq[j] - pq[i]) - cs * cs / cw
    };
    let k = k.min(m).max(1);
    let mut dp = vec![vec![f64::INFINITY; m + 1]; k + 1];
    let mut cut = vec![vec![0usize; m + 1]; k + 1];
    dp[0][0] = 0.0;
    for c in 1..=k {
        for j in c..=m {
            for i in (c - 1)..j {
                let candidate = dp[c - 1][i] + cost(i, j);
                if candidate < dp[c][j] {
                    dp[c][j] = candidate;
                    cut[c][j] = i;
                }
            }
        }
    }
    let mut group_of_unique = vec![0usize; m];
    let mut end = m;
    for c in (1..=k).rev() {
        let start = cut[c][end];
        for g in &mut group_of_unique[start..end] {
            *g = c - 1;
        }
        end = start;
    }
    values
        .iter()
        .map(|v| group_of_unique[unique.partition_point(|u| u < v)])
        .collect()
}

/// Adaptive tiers from the aggregate-score distribution itself: rows with a
/// positive score are split into up to three groups (core, supporting,
/// context, highest score first); the rest are noise. Operates on the rows at
/// `idx` of `rows`.
fn assign_signal_tiers<R: Scored>(rows: &mut [R], idx: &[usize]) {
    for &i in idx {
        rows[i].set_tier("noise");
    }
    let positives: Vec<usize> = idx
        .iter()
        .copied()
        .filter(|&i| rows[i].aggregate() > 0.0)
        .collect();
    if positives.is_empty() {
        return;
    }
    let scores: Vec<f64> = positives.iter().map(|&i| rows[i].aggregate()).collect();
    let mut unique = scores.clone();
    unique.sort_by(f64::total_cmp);
    unique.dedup();
    if unique.len() == 1 {
        for &i in &positives {
            rows[i].set_tier("core");
        }
        return;
    }
    let k = 3.min(unique.len()).min(positives.len());
    let labels = kmeans_1d(&scores, k);
    let groups = labels.iter().max().map_or(0, |m| m + 1);
    const TIERS: [&str; 3] = ["core", "supporting", "context"];
    for (&i, label) in positives.iter().zip(labels) {
        // The highest group is rank 0.
        let rank = groups - 1 - label;
        rows[i].set_tier(TIERS[rank.min(TIERS.len() - 1)]);
    }
}

// ---- numeric rows -------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GlobalNumericStats {
    pub p_value: f64,
    pub effect: f64,
    pub failure_reasons: Vec<String>,
    pub q_value: f64,
}

fn pop_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    np_var(values, 0).sqrt()
}

/// `scipy.stats.iqr`: `q75 - q25` with linear interpolation.
fn iqr(values: &[f64]) -> f64 {
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| {
        let pos = p * (s.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(s.len() - 1);
        s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
    };
    q(0.75) - q(0.25)
}

fn cluster_ids(clusters: &[i64], target: Option<&HashSet<i64>>) -> Vec<i64> {
    let unique: BTreeSet<i64> = clusters.iter().copied().collect();
    unique
        .into_iter()
        .filter(|c| target.is_none_or(|t| t.contains(c)))
        .collect()
}

fn mask_of(clusters: &[i64], cluster: i64) -> Vec<bool> {
    clusters.iter().map(|&c| c == cluster).collect()
}

fn select(x: &[f64], valid: &[bool], mask: &[bool], want: bool) -> Vec<f64> {
    (0..x.len())
        .filter(|&i| valid[i] && mask[i] == want)
        .map(|i| x[i])
        .collect()
}

fn compute_numeric_rows(
    columns: &[(String, Vec<f64>)],
    clusters: &[i64],
    adjacency: &Adjacency,
    target: Option<&HashSet<i64>>,
) -> (Vec<NumericRow>, IndexMap<String, GlobalNumericStats>) {
    if columns.is_empty() {
        return (vec![], IndexMap::new());
    }
    let values: Vec<Vec<f64>> = columns.iter().map(|(_, v)| v.clone()).collect();
    let pre: NumericPrecompute = build_numeric_precompute(&values);
    let cluster_values = cluster_ids(clusters, target);
    let present: Vec<Vec<f64>> = (0..columns.len())
        .map(|j| {
            (0..clusters.len())
                .filter(|&i| pre.valid[j][i])
                .map(|i| pre.x[j][i])
                .collect()
        })
        .collect();

    // Column-level Kruskal-Wallis and the spread of cluster means.
    let mut global: IndexMap<String, GlobalNumericStats> = IndexMap::new();
    for (j, (name, _)) in columns.iter().enumerate() {
        let masks: Vec<Vec<bool>> = cluster_values
            .iter()
            .map(|&c| {
                (0..clusters.len())
                    .map(|i| clusters[i] == c && pre.valid[j][i])
                    .collect()
            })
            .collect();
        let grouped: Vec<Vec<f64>> = masks
            .iter()
            .filter(|m| m.iter().any(|&b| b))
            .map(|m| {
                (0..m.len())
                    .filter(|&i| m[i])
                    .map(|i| pre.x[j][i])
                    .collect()
            })
            .collect();
        let (mut p_value, mut failures) = (1.0, vec![]);
        if grouped.len() >= 2 {
            match kruskal_wallis_pvalue(&pre.ranks[j], &masks, pre.tie_correction[j]) {
                Ok(p) => p_value = p,
                Err(e) => failures.push(format!("kruskal: {e}")),
            }
        }
        let global_scale = if present[j].is_empty() {
            0.0
        } else {
            pop_std(&present[j])
        };
        let mean_dispersion = if grouped.is_empty() {
            0.0
        } else {
            pop_std(&grouped.iter().map(|g| np_mean(g)).collect::<Vec<_>>())
        };
        global.insert(
            name.clone(),
            GlobalNumericStats {
                p_value,
                effect: mean_dispersion / global_scale.max(EPS),
                failure_reasons: failures,
                q_value: 1.0,
            },
        );
    }
    let qs = bh_fdr(&global.values().map(|g| Some(g.p_value)).collect::<Vec<_>>());
    for (stats, q) in global.values_mut().zip(qs) {
        stats.q_value = q.unwrap_or(1.0);
    }

    let global_std: Vec<f64> = present.iter().map(|v| safe_std(v).max(EPS)).collect();
    let global_mad: Vec<f64> = present.iter().map(|v| safe_mad(v).max(EPS)).collect();
    let global_iqr: Vec<f64> = present
        .iter()
        .map(|v| if v.is_empty() { EPS } else { iqr(v).max(EPS) })
        .collect();
    let global_mean: Vec<f64> = present.iter().map(|v| safe_mean(v)).collect();

    let mut rows: Vec<NumericRow> = vec![];
    for &cluster_id in &cluster_values {
        let cluster_mask = mask_of(clusters, cluster_id);
        let neighbor_id = strongest_neighbor(adjacency, cluster_id);
        let neighbor_mask = neighbor_id.map(|n| mask_of(clusters, n));
        for (j, (name, _)) in columns.iter().enumerate() {
            let valid = &pre.valid[j];
            let cl = select(&pre.x[j], valid, &cluster_mask, true);
            let rest = select(&pre.x[j], valid, &cluster_mask, false);
            if cl.is_empty() || rest.is_empty() {
                continue;
            }
            let nb: Vec<f64> = neighbor_mask
                .as_ref()
                .map(|m| select(&pre.x[j], valid, m, true))
                .unwrap_or_default();
            let mut failures = vec![];
            let pooled = pooled_std(&cl, &rest);
            let (gstd, gmad, giqr, gmean) =
                (global_std[j], global_mad[j], global_iqr[j], global_mean[j]);
            let (std_cluster, std_rest) = (safe_std(&cl), safe_std(&rest));
            let (mad_cluster, mad_rest) = (safe_mad(&cl), safe_mad(&rest));
            let (mean_cl, mean_rest) = (safe_mean(&cl), safe_mean(&rest));
            let effect_mean_std = (mean_cl - mean_rest) / pooled.max(EPS);
            let effect_median_mad =
                (safe_median(&cl) - safe_median(&rest)) / (1.4826 * mad_rest.max(gmad)).max(EPS);
            let ks_stat = ks_two_sample_stat(&cl, &rest).unwrap_or_else(|e| {
                failures.push(format!("ks_2samp: {e}"));
                0.0
            });
            let wasserstein_norm = match wasserstein_distance_1d(&cl, &rest) {
                Ok(d) => d / giqr.max(EPS),
                Err(e) => {
                    failures.push(format!("wasserstein: {e}"));
                    0.0
                }
            };
            let variance_ratio_log = ((safe_var(&cl) + EPS) / (safe_var(&rest) + EPS)).ln();
            let concentration_gain = (1.0 - (std_cluster + EPS) / gstd.max(EPS)).max(0.0);
            let one_vs_rest_p =
                mwu_one_vs_rest_pvalue(&cluster_mask, j, &pre).unwrap_or_else(|e| {
                    failures.push(format!("mannwhitneyu(one_vs_rest): {e}"));
                    1.0
                });
            let (mut neighbor_effect, mut neighbor_p) = (0.0, 1.0);
            if !nb.is_empty() {
                neighbor_effect = (mean_cl - safe_mean(&nb)) / pooled_std(&cl, &nb).max(EPS);
                neighbor_p = mwu_pair_pvalue(&cl, &nb).unwrap_or_else(|e| {
                    failures.push(format!("mannwhitneyu(neighbor): {e}"));
                    1.0
                });
            }
            let g = &global[name];
            rows.push(NumericRow {
                column: name.clone(),
                cluster_id,
                direction: direction_from_effects(&[
                    effect_mean_std,
                    effect_median_mad,
                    neighbor_effect,
                ])
                .into(),
                mean: mean_cl,
                mean_rest,
                median: safe_median(&cl),
                median_rest: safe_median(&rest),
                std_cluster,
                std_rest,
                mad_cluster,
                mad_rest,
                global_mean: gmean,
                global_mean_rest: mean_rest,
                z_score: (mean_cl - gmean) / gstd.max(EPS),
                homogeneity: std_cluster / gstd.max(EPS),
                effect_mean_std,
                effect_median_mad,
                ks_stat,
                wasserstein_norm,
                variance_ratio_log,
                concentration_gain,
                one_vs_rest_p,
                one_vs_rest_q: 1.0,
                global_p: g.p_value,
                global_q: g.q_value,
                global_cluster_signal: g.effect,
                neighbor_cluster_id: neighbor_id,
                neighbor_effect,
                neighbor_p,
                specificity_score: (effect_mean_std.abs()
                    + effect_median_mad.abs()
                    + neighbor_effect.abs()
                    + ks_stat
                    + wasserstein_norm)
                    / 5.0,
                aggregate_score: 0.0,
                percentile_score: 0.0,
                signal_tier: "noise".into(),
                sparkline: distribution_sparkline(&cl, 10),
                failure_reasons: failures,
                evidence_vector: IndexMap::new(),
            });
        }
    }
    let qs = bh_fdr(
        &rows
            .iter()
            .map(|r| Some(r.one_vs_rest_p))
            .collect::<Vec<_>>(),
    );
    for (row, q) in rows.iter_mut().zip(qs) {
        row.one_vs_rest_q = q.unwrap_or(1.0);
    }
    apply_percentile_aggregate(&mut rows);
    (rows, global)
}

// ---- categorical rows ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GlobalCategoricalStats {
    pub p_value: f64,
    pub association: f64,
    pub failure_reasons: Vec<String>,
    pub q_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GatedColumn {
    pub column: String,
    pub cardinality: usize,
    pub reason: String,
}

/// `Series.astype(str)` after `fillna("__MISSING__")`.
fn encode_categorical(column: &Column) -> Option<Vec<String>> {
    match column {
        Column::Bool(v) => Some(
            v.iter()
                .map(|b| match b {
                    Some(true) => "True".to_owned(),
                    Some(false) => "False".to_owned(),
                    None => MISSING_LABEL.to_owned(),
                })
                .collect(),
        ),
        Column::Text(v) => Some(
            v.iter()
                .map(|s| s.clone().unwrap_or_else(|| MISSING_LABEL.to_owned()))
                .collect(),
        ),
        _ => None,
    }
}

/// Distinct non-missing values (`Series.nunique()`).
fn nunique(column: &Column) -> usize {
    match column {
        Column::Bool(v) => v.iter().flatten().collect::<HashSet<_>>().len(),
        Column::Text(v) => v.iter().flatten().collect::<HashSet<_>>().len(),
        _ => 0,
    }
}

/// `value_counts()` order: count descending, ties by first appearance.
fn value_counts<'a>(values: impl Iterator<Item = &'a String>) -> Vec<(String, usize)> {
    let mut counts: IndexMap<&String, usize> = IndexMap::new();
    for v in values {
        *counts.entry(v).or_default() += 1;
    }
    let mut out: Vec<(String, usize)> = counts.into_iter().map(|(k, c)| (k.clone(), c)).collect();
    out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    out
}

fn compute_categorical_rows(
    table: &Table,
    names: &[String],
    clusters: &[i64],
    adjacency: &Adjacency,
    target: Option<&HashSet<i64>>,
) -> (
    Vec<CategoricalRow>,
    IndexMap<String, GlobalCategoricalStats>,
    Vec<GatedColumn>,
) {
    let n_total = clusters.len();
    let mut gated = vec![];
    let mut active: Vec<&String> = vec![];
    for name in names {
        let card = nunique(table.column(name).expect("selected from the table"));
        if card as f64 > MIN_CARDINALITY_FLOOR.max(MAX_CARDINALITY_RATIO * n_total as f64) {
            gated.push(GatedColumn {
                column: name.clone(),
                cardinality: card,
                reason: "high_cardinality".into(),
            });
        } else {
            active.push(name);
        }
    }
    let cluster_values = cluster_ids(clusters, target);
    let in_target = |c: i64| target.is_none_or(|t| t.contains(&c));

    let mut encoded: HashMap<&String, Vec<String>> = HashMap::new();
    let mut global_counts: HashMap<&String, HashMap<String, usize>> = HashMap::new();
    let mut global: IndexMap<String, GlobalCategoricalStats> = IndexMap::new();
    for &name in &active {
        let enc = encode_categorical(table.column(name).expect("selected from the table"))
            .expect("categorical columns are boolean or text");
        global_counts.insert(name, value_counts(enc.iter()).into_iter().collect());

        // Cluster x value contingency table over the characterized clusters.
        let values: BTreeSet<&String> = (0..n_total)
            .filter(|&i| in_target(clusters[i]))
            .map(|i| &enc[i])
            .collect();
        let (mut p_value, mut association, mut failures) = (1.0, 0.0, vec![]);
        if cluster_values.len() >= 2 && values.len() >= 2 {
            let table_f: Vec<Vec<f64>> = cluster_values
                .iter()
                .map(|&c| {
                    values
                        .iter()
                        .map(|v| {
                            (0..n_total)
                                .filter(|&i| clusters[i] == c && &&enc[i] == v)
                                .count() as f64
                        })
                        .collect()
                })
                .collect();
            let total: f64 = table_f.iter().flatten().sum();
            match chi2_contingency(&table_f, true) {
                Ok((chi2, p, _)) => {
                    p_value = p;
                    association = chi2 / total.max(1.0);
                }
                Err(e) => failures.push(format!("chi2_contingency: {e}")),
            }
        }
        global.insert(
            name.clone(),
            GlobalCategoricalStats {
                p_value,
                association,
                failure_reasons: failures,
                q_value: 1.0,
            },
        );
        encoded.insert(name, enc);
    }
    let qs = bh_fdr(&global.values().map(|g| Some(g.p_value)).collect::<Vec<_>>());
    for (stats, q) in global.values_mut().zip(qs) {
        stats.q_value = q.unwrap_or(1.0);
    }

    let mut rows: Vec<CategoricalRow> = vec![];
    for &cluster_id in &cluster_values {
        let cluster_size = clusters.iter().filter(|&&c| c == cluster_id).count();
        let rest_size = n_total - cluster_size;
        let neighbor_id = strongest_neighbor(adjacency, cluster_id);
        let neighbor_size = neighbor_id.map_or(0, |n| clusters.iter().filter(|&&c| c == n).count());
        for &name in &active {
            let enc = &encoded[name];
            let counts = value_counts(
                (0..n_total)
                    .filter(|&i| clusters[i] == cluster_id)
                    .map(|i| &enc[i]),
            );
            let neighbor_counts: HashMap<String, usize> = match neighbor_id {
                Some(n) if neighbor_size > 0 => {
                    value_counts((0..n_total).filter(|&i| clusters[i] == n).map(|i| &enc[i]))
                        .into_iter()
                        .collect()
                }
                _ => HashMap::new(),
            };
            for (value, count) in counts {
                let global_count = global_counts[name].get(&value).copied().unwrap_or(0);
                let count_rest = global_count - count;
                let prevalence_cluster = if cluster_size > 0 {
                    count as f64 / cluster_size as f64
                } else {
                    0.0
                };
                let prevalence_rest = if rest_size > 0 {
                    count_rest as f64 / rest_size as f64
                } else {
                    0.0
                };
                let prevalence_global = if n_total > 0 {
                    global_count as f64 / n_total as f64
                } else {
                    0.0
                };
                let lift = prevalence_cluster / prevalence_global.max(EPS);
                let log_lift = lift.max(EPS).ln();
                let neighbor_prevalence = if neighbor_size > 0 {
                    neighbor_counts.get(&value).copied().unwrap_or(0) as f64 / neighbor_size as f64
                } else {
                    0.0
                };
                let denom = n_total.max(1) as f64;
                let (p_cv, p_c, p_v) = (
                    count as f64 / denom,
                    cluster_size as f64 / denom,
                    global_count as f64 / denom,
                );
                let mi_contrib = p_cv * (p_cv.max(EPS) / (p_c * p_v).max(EPS)).ln();

                let table2 = [
                    [count as u64, cluster_size.saturating_sub(count) as u64],
                    [
                        count_rest as u64,
                        rest_size.saturating_sub(count_rest) as u64,
                    ],
                ];
                let (e00, e01, e10, e11) = if n_total > 0 {
                    let (n, g, c, r) = (
                        n_total as f64,
                        global_count as f64,
                        cluster_size as f64,
                        rest_size as f64,
                    );
                    (c * g / n, c * (n - g) / n, r * g / n, r * (n - g) / n)
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };
                let (mut p, mut method, mut failures) = (1.0, "fisher", vec![]);
                if e00.min(e01).min(e10).min(e11) >= CHI2_MIN_EXPECTED_CELL {
                    let as_f = table2
                        .iter()
                        .map(|r| r.iter().map(|&v| v as f64).collect())
                        .collect::<Vec<Vec<f64>>>();
                    if let Ok((_, chi2_p, _)) = chi2_contingency(&as_f, true) {
                        p = chi2_p;
                        method = "chi2";
                    }
                }
                if method == "fisher" {
                    p = fisher_exact_two_sided(table2);
                }
                let _ = &mut failures;

                let g = &global[name.as_str()];
                rows.push(CategoricalRow {
                    column: name.clone(),
                    value,
                    cluster_id,
                    count,
                    count_cluster: count,
                    count_rest,
                    cluster_size,
                    global_count,
                    prevalence_cluster: prevalence_cluster * 100.0,
                    prevalence_rest: prevalence_rest * 100.0,
                    prevalence_global: prevalence_global * 100.0,
                    lift,
                    log_lift,
                    specificity: prevalence_cluster - prevalence_rest,
                    global_recall: if global_count > 0 {
                        count as f64 / global_count as f64 * 100.0
                    } else {
                        0.0
                    },
                    neighbor_specificity: prevalence_cluster - neighbor_prevalence,
                    mi_contrib,
                    fisher_p: p,
                    fisher_q: 1.0,
                    global_p: g.p_value,
                    global_q: g.q_value,
                    aggregate_score: 0.0,
                    percentile_score: 0.0,
                    signal_tier: "noise".into(),
                    in_cluster_prevalence: prevalence_cluster * 100.0,
                    concentration: prevalence_cluster * 100.0,
                    test_method: method.into(),
                    failure_reasons: failures,
                    evidence_vector: IndexMap::new(),
                });
            }
        }
    }
    // Exact (Fisher) and asymptotic (chi-square) p-values are pooled into one
    // FDR correction on purpose: both are valid p-values.
    let qs = bh_fdr(&rows.iter().map(|r| Some(r.fisher_p)).collect::<Vec<_>>());
    for (row, q) in rows.iter_mut().zip(qs) {
        row.fisher_q = q.unwrap_or(1.0);
    }
    apply_percentile_aggregate(&mut rows);
    (rows, global, gated)
}

// ---- the index ------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClusterBundle {
    pub cluster_id: i64,
    pub size: usize,
    pub size_pct: f64,
    pub numeric: Vec<NumericRow>,
    pub categorical: Vec<CategoricalRow>,
    pub topological_neighbors: Vec<Neighbor>,
    pub semantic_name: String,
}

/// Cached feature evidence derived from one clustering assignment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureEvidenceIndex {
    pub cluster_bundles: BTreeMap<i64, ClusterBundle>,
    pub numeric_global_ranking: Vec<String>,
    pub categorical_global_ranking: Vec<Value>,
    pub signal_matrix: Value,
    pub metadata: Value,
    pub working_columns: Vec<String>,
    pub categorical_columns_gated: Vec<GatedColumn>,
}

fn auto_semantic_name(
    numeric: &[NumericRow],
    categorical: &[CategoricalRow],
    cluster_id: i64,
) -> String {
    let mut fragments: Vec<String> = numeric
        .iter()
        .filter(|r| r.signal_tier == "core" || r.signal_tier == "supporting")
        .take(2)
        .map(|r| {
            let direction = if r.z_score > 0.5 {
                "High"
            } else if r.z_score < -0.5 {
                "Low"
            } else {
                "Shifted"
            };
            format!("{direction} {}", r.column)
        })
        .collect();
    if let Some(lead) = categorical.iter().find(|r| r.signal_tier == "core") {
        fragments.push(format!("{} {}", lead.value, lead.column));
    }
    if fragments.is_empty() {
        return format!("Cluster {cluster_id}");
    }
    fragments.truncate(3);
    format!("[Auto] {}", fragments.join(", "))
}

fn assemble_signal_matrix(bundles: &BTreeMap<i64, ClusterBundle>) -> Value {
    let strong = |tier: &str| tier == "core" || tier == "supporting";
    let numeric_columns: BTreeSet<&str> = bundles
        .values()
        .flat_map(|b| &b.numeric)
        .filter(|r| strong(&r.signal_tier))
        .map(|r| r.column.as_str())
        .collect();
    let categorical_pairs: BTreeSet<(&str, &str)> = bundles
        .values()
        .flat_map(|b| &b.categorical)
        .filter(|r| strong(&r.signal_tier))
        .map(|r| (r.column.as_str(), r.value.as_str()))
        .collect();

    let mut numeric_rows = vec![];
    let mut categorical_rows = vec![];
    for (&cluster_id, bundle) in bundles {
        let numeric_map: HashMap<&str, &NumericRow> = bundle
            .numeric
            .iter()
            .map(|r| (r.column.as_str(), r))
            .collect();
        let categorical_map: HashMap<(&str, &str), &CategoricalRow> = bundle
            .categorical
            .iter()
            .map(|r| ((r.column.as_str(), r.value.as_str()), r))
            .collect();
        let values: serde_json::Map<String, Value> = numeric_columns
            .iter()
            .filter_map(|c| numeric_map.get(c).map(|r| (*c, *r)))
            .map(|(c, r)| {
                (
                    c.to_owned(),
                    json!({"value": r.z_score, "z_score": r.z_score, "aggregate_score": r.aggregate_score, "signal_tier": r.signal_tier}),
                )
            })
            .collect();
        numeric_rows.push(json!({"cluster_id": cluster_id, "values": values}));
        let values: serde_json::Map<String, Value> = categorical_pairs
            .iter()
            .filter_map(|p| categorical_map.get(p).map(|r| (*p, *r)))
            .map(|((c, v), r)| {
                (
                    format!("{c}={v}"),
                    json!({"value": r.in_cluster_prevalence, "prevalence": r.in_cluster_prevalence, "aggregate_score": r.aggregate_score, "signal_tier": r.signal_tier}),
                )
            })
            .collect();
        categorical_rows.push(json!({"cluster_id": cluster_id, "values": values}));
    }
    json!({
        "numeric_columns": numeric_columns,
        "categorical_values": categorical_pairs.iter().map(|(c, v)| json!({"column": c, "value": v})).collect::<Vec<_>>(),
        "numeric_rows": numeric_rows,
        "categorical_rows": categorical_rows,
    })
}

fn collect_stats_failures(
    numeric: &[NumericRow],
    categorical: &[CategoricalRow],
    global_numeric: &IndexMap<String, GlobalNumericStats>,
    global_categorical: &IndexMap<String, GlobalCategoricalStats>,
) -> Value {
    let column_level = |pairs: Vec<(&String, &Vec<String>)>| -> serde_json::Map<String, Value> {
        pairs
            .into_iter()
            .filter(|(_, r)| !r.is_empty())
            .map(|(c, r)| (c.clone(), json!(r)))
            .collect()
    };
    json!({
        "numeric_column_level": column_level(global_numeric.iter().map(|(c, g)| (c, &g.failure_reasons)).collect()),
        "categorical_column_level": column_level(global_categorical.iter().map(|(c, g)| (c, &g.failure_reasons)).collect()),
        "numeric_row_level": numeric.iter().filter(|r| !r.failure_reasons.is_empty())
            .map(|r| json!({"cluster_id": r.cluster_id, "column": r.column, "reasons": r.failure_reasons})).collect::<Vec<_>>(),
        "categorical_row_level": categorical.iter().filter(|r| !r.failure_reasons.is_empty())
            .map(|r| json!({"cluster_id": r.cluster_id, "column": r.column, "value": r.value, "reasons": r.failure_reasons})).collect::<Vec<_>>(),
    })
}

/// Distribution-aware evidence for numeric and categorical signals.
///
/// `clusters[i]` is the cluster of row `i` and of graph node `i`; `edges` are
/// the cosmic graph's `(source, target, weight)` triples. With
/// `max_clusters_to_characterize`, only that many of the largest clusters get
/// rows.
pub fn build_feature_evidence_index(
    table: &Table,
    clusters: &[i64],
    edges: &[(usize, usize, f64)],
    exclude_columns: &[String],
    max_clusters_to_characterize: Option<usize>,
) -> Result<FeatureEvidenceIndex, EvidenceIndexError> {
    let n_rows = table.n_rows();
    if clusters.len() != n_rows {
        return Err(EvidenceIndexError::Alignment {
            clusters: clusters.len(),
            rows: n_rows,
        });
    }
    let excluded: HashSet<&str> = exclude_columns.iter().map(String::as_str).collect();
    let working: Vec<(&String, &Column)> = table
        .names
        .iter()
        .zip(&table.columns)
        .filter(|(n, _)| !excluded.contains(n.as_str()))
        .collect();
    let numeric: Vec<(String, Vec<f64>)> = working
        .iter()
        .filter_map(|(n, c)| match c {
            Column::Float(v) => Some(((*n).clone(), v.clone())),
            Column::Int(v) => Some(((*n).clone(), v.iter().map(|&x| x as f64).collect())),
            _ => None,
        })
        .collect();
    let categorical: Vec<String> = working
        .iter()
        .filter(|(_, c)| matches!(c, Column::Bool(_) | Column::Text(_)))
        .map(|(n, _)| (*n).clone())
        .collect();

    let sizes = value_counts_i64(clusters);
    let n_unique = sizes.len();
    let (mut target, mut omitted) = (None, 0);
    if let Some(max) = max_clusters_to_characterize {
        if n_unique > max {
            target = Some(
                sizes
                    .iter()
                    .take(max)
                    .map(|(c, _)| *c)
                    .collect::<HashSet<i64>>(),
            );
            omitted = n_unique - max;
        }
    }

    let adjacency = graph_cluster_adjacency(edges, clusters);
    let (mut numeric_rows, global_numeric) =
        compute_numeric_rows(&numeric, clusters, &adjacency, target.as_ref());
    let working_table = Table {
        names: working.iter().map(|(n, _)| (*n).clone()).collect(),
        columns: working.iter().map(|(_, c)| (*c).clone()).collect(),
    };
    let (mut categorical_rows, global_categorical, gated) = compute_categorical_rows(
        &working_table,
        &categorical,
        clusters,
        &adjacency,
        target.as_ref(),
    );

    let characterized = cluster_ids(clusters, target.as_ref());
    let mut bundles: BTreeMap<i64, ClusterBundle> = BTreeMap::new();
    for &cluster_id in &characterized {
        let mut n_idx: Vec<usize> = (0..numeric_rows.len())
            .filter(|&i| numeric_rows[i].cluster_id == cluster_id)
            .collect();
        n_idx.sort_by(|&a, &b| {
            numeric_rows[b]
                .aggregate_score
                .total_cmp(&numeric_rows[a].aggregate_score)
                .then_with(|| numeric_rows[a].column.cmp(&numeric_rows[b].column))
        });
        let mut c_idx: Vec<usize> = (0..categorical_rows.len())
            .filter(|&i| categorical_rows[i].cluster_id == cluster_id)
            .collect();
        c_idx.sort_by(|&a, &b| {
            let (x, y) = (&categorical_rows[a], &categorical_rows[b]);
            y.aggregate_score
                .total_cmp(&x.aggregate_score)
                .then_with(|| x.column.cmp(&y.column))
                .then_with(|| x.value.cmp(&y.value))
        });
        assign_signal_tiers(&mut numeric_rows, &n_idx);
        assign_signal_tiers(&mut categorical_rows, &c_idx);
        let bundle_numeric: Vec<NumericRow> =
            n_idx.iter().map(|&i| numeric_rows[i].clone()).collect();
        let bundle_categorical: Vec<CategoricalRow> =
            c_idx.iter().map(|&i| categorical_rows[i].clone()).collect();
        let size = clusters.iter().filter(|&&c| c == cluster_id).count();
        let semantic_name = auto_semantic_name(&bundle_numeric, &bundle_categorical, cluster_id);
        bundles.insert(
            cluster_id,
            ClusterBundle {
                cluster_id,
                size,
                size_pct: size as f64 / working_table.n_rows() as f64 * 100.0,
                numeric: bundle_numeric,
                categorical: bundle_categorical,
                topological_neighbors: adjacency.get(&cluster_id).cloned().unwrap_or_default(),
                semantic_name,
            },
        );
    }

    let signal_matrix = assemble_signal_matrix(&bundles);
    let stats_failures = collect_stats_failures(
        &numeric_rows,
        &categorical_rows,
        &global_numeric,
        &global_categorical,
    );
    let working_columns = working_table.names.clone();

    let mut scores: IndexMap<&str, f64> = IndexMap::new();
    for r in &numeric_rows {
        let e = scores.entry(r.column.as_str()).or_insert(0.0);
        *e = py_max(*e, r.aggregate_score);
    }
    let mut ranked: Vec<(&str, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let numeric_global_ranking = ranked.into_iter().map(|(c, _)| c.to_owned()).collect();

    let mut by_score: Vec<&CategoricalRow> = categorical_rows.iter().collect();
    by_score.sort_by(|a, b| {
        b.aggregate_score
            .total_cmp(&a.aggregate_score)
            .then_with(|| a.column.cmp(&b.column))
            .then_with(|| a.value.cmp(&b.value))
            .then(a.cluster_id.cmp(&b.cluster_id))
    });
    let categorical_global_ranking = by_score
        .iter()
        .map(|r| json!({"column": r.column, "value": r.value, "cluster_id": r.cluster_id, "aggregate_score": r.aggregate_score, "signal_tier": r.signal_tier}))
        .collect();

    let metadata = json!({
        "n_total": working_table.n_rows(),
        "n_clusters": n_unique,
        "n_characterized": characterized.len(),
        "omitted_clusters_count": omitted,
        "max_clusters_to_characterize": max_clusters_to_characterize,
        "columns": working_columns,
        "excluded_columns": exclude_columns,
        "numeric_features_screened": numeric.len(),
        "categorical_columns_screened": categorical.len() - gated.len(),
        "categorical_values_screened": categorical_rows.len(),
        "categorical_columns_gated": gated,
        "tiering_method": "adaptive_kmeans_on_aggregate_percentiles",
        "neighbor_contrast_enabled": true,
        "global_numeric_stats": global_numeric,
        "global_categorical_stats": global_categorical,
        "cluster_adjacency": adjacency,
        "stats_failures": stats_failures,
    });
    Ok(FeatureEvidenceIndex {
        cluster_bundles: bundles,
        numeric_global_ranking,
        categorical_global_ranking,
        signal_matrix,
        metadata,
        working_columns,
        categorical_columns_gated: gated,
    })
}

/// `clusters.value_counts()` order: count descending, ties by first appearance.
fn value_counts_i64(clusters: &[i64]) -> Vec<(i64, usize)> {
    let mut counts: IndexMap<i64, usize> = IndexMap::new();
    for &c in clusters {
        *counts.entry(c).or_default() += 1;
    }
    let mut out: Vec<(i64, usize)> = counts.into_iter().collect();
    out.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_1d_finds_the_optimal_split() {
        // Three obvious groups.
        let labels = kmeans_1d(&[0.1, 0.12, 0.5, 0.52, 0.9, 0.95, 0.11], 3);
        assert_eq!(labels, [0, 0, 1, 1, 2, 2, 0]);
        // Two values, one group each.
        assert_eq!(kmeans_1d(&[1.0, 2.0], 2), [0, 1]);
        // Duplicates stay together.
        assert_eq!(kmeans_1d(&[1.0, 1.0, 5.0, 5.0, 5.0], 2), [0, 0, 1, 1, 1]);
    }

    #[test]
    fn tiers_rank_groups_by_score() {
        let mk = |agg: f64| NumericRow {
            aggregate_score: agg,
            ..blank_numeric()
        };
        let mut rows = vec![mk(0.9), mk(0.0), mk(0.5), mk(0.52), mk(0.1), mk(0.88)];
        assign_signal_tiers(&mut rows, &[0, 1, 2, 3, 4, 5]);
        let tiers: Vec<&str> = rows.iter().map(|r| r.signal_tier.as_str()).collect();
        assert_eq!(
            tiers,
            [
                "core",
                "noise",
                "supporting",
                "supporting",
                "context",
                "core"
            ]
        );
        // A single distinct positive score is all core.
        let mut rows = vec![mk(0.3), mk(0.3), mk(0.0)];
        assign_signal_tiers(&mut rows, &[0, 1, 2]);
        assert_eq!(rows[0].signal_tier, "core");
        assert_eq!(rows[2].signal_tier, "noise");
    }

    #[test]
    fn adjacency_ranks_bridges() {
        // Clusters 0 (nodes 0,1), 1 (node 2), 2 (node 3).
        let clusters = [0, 0, 1, 2];
        let edges = [(0, 1, 2.0), (1, 2, 1.0), (2, 3, 0.5), (0, 2, 1.0)];
        let adj = graph_cluster_adjacency(&edges, &clusters);
        assert_eq!(adj[&0][0].cluster_id, 1);
        assert_eq!(adj[&0][0].bridge_weight, 2.0);
        assert_eq!(adj[&1].len(), 2);
        assert_eq!(strongest_neighbor(&adj, 2), Some(1));
        assert!(adj[&2][0].bridge_strength > 0.0);
    }

    #[test]
    fn mismatched_clusters_are_rejected_with_the_python_message() {
        let table = Table::new(vec!["a".into()], vec![Column::Float(vec![1.0, 2.0])]).unwrap();
        let e = build_feature_evidence_index(&table, &[0], &[], &[], None).unwrap_err();
        assert_eq!(e.to_string(), "Alignment error: clusters(1) != data(2)");
    }

    fn blank_numeric() -> NumericRow {
        NumericRow {
            column: "c".into(),
            cluster_id: 0,
            direction: "mixed".into(),
            mean: 0.0,
            mean_rest: 0.0,
            median: 0.0,
            median_rest: 0.0,
            std_cluster: 0.0,
            std_rest: 0.0,
            mad_cluster: 0.0,
            mad_rest: 0.0,
            global_mean: 0.0,
            global_mean_rest: 0.0,
            z_score: 0.0,
            homogeneity: 0.0,
            effect_mean_std: 0.0,
            effect_median_mad: 0.0,
            ks_stat: 0.0,
            wasserstein_norm: 0.0,
            variance_ratio_log: 0.0,
            concentration_gain: 0.0,
            one_vs_rest_p: 1.0,
            one_vs_rest_q: 1.0,
            global_p: 1.0,
            global_q: 1.0,
            global_cluster_signal: 0.0,
            neighbor_cluster_id: None,
            neighbor_effect: 0.0,
            neighbor_p: 1.0,
            specificity_score: 0.0,
            aggregate_score: 0.0,
            percentile_score: 0.0,
            signal_tier: "noise".into(),
            sparkline: String::new(),
            failure_reasons: vec![],
            evidence_vector: IndexMap::new(),
        }
    }
}
