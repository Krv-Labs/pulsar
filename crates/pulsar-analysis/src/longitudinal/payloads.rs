//! Measurement payloads over the two longitudinal representations: surface
//! diagnosis, the threshold ladder, trajectory archetypes, cross-time
//! neighbours, trajectory classification and the deterministic entity
//! sample.
//!
//! Known divergences from Python:
//! - Ranked lists (archetypes by population, neighbours by weight, classes by
//!   count) break ties by first appearance. numpy's default sort is not
//!   stable, so Python's order inside a tie was unspecified.
//! - `cross_time_payload` takes an `obs_id` the caller has range-checked.

use std::collections::{BTreeMap, HashMap};

use kodama::{linkage, Method};
use ndarray::Array2;
use pulsar_core::TemporalCosmicGraph;
use pulsar_data::{Column, Key, Table};
use pulsar_pipeline::{CosmicTrajectory, Graph, Observation};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::payloads::{bounded_list, size_summary};
use crate::plan::py_repr;
use crate::pyround::round_to;

/// Self-scaled cut that makes the six aggregations comparable.
const AGGREGATION_CUT_QUANTILE: f64 = 0.90;
pub const ARCHETYPE_SWEEP_THRESHOLDS: [f64; 5] = [0.0, 0.1, 0.25, 0.5, 0.75];
/// `fcluster(..., t=4, criterion="maxclust")` of the sequence classifiers.
const SEQUENCE_COHORTS: usize = 4;

/// What each temporal aggregation measures and when to reach for it.
pub fn aggregation_guidance() -> Vec<(&'static str, Value)> {
    vec![
        (
            "persistence",
            json!({
                "value_range": "[0,1]",
                "measures": "fraction of time steps where the pair exceeds the threshold",
                "why": "Counts how often a relationship holds, ignoring its magnitude.",
                "best_for": ["stable_cohorts", "report_ready"],
                "avoid_for": ["change_detection"],
            }),
        ),
        (
            "mean",
            json!({
                "value_range": "[0,1]",
                "measures": "average affinity across time",
                "why": "General-purpose summary; a strong brief relationship and a weak persistent one can score alike.",
                "best_for": ["general_structure"],
                "avoid_for": ["change_detection", "outlier_mining"],
            }),
        ),
        (
            "recency",
            json!({
                "value_range": "[0,1]",
                "measures": "exponentially decayed affinity, most recent step weighted 1",
                "why": "Answers what the cohort looks like now rather than on average.",
                "best_for": ["current_state_cohorts"],
                "avoid_for": ["stable_cohorts"],
            }),
        ),
        (
            "volatility",
            json!({
                "value_range": "unbounded>=0",
                "measures": "population variance of the pair's affinity across time",
                "why": "High values mark relationships that come and go.",
                "best_for": ["unstable_pairs", "detail_seeking"],
                "avoid_for": ["report_ready"],
            }),
        ),
        (
            "trend",
            json!({
                "value_range": "signed",
                "measures": "OLS slope of affinity against time",
                "why": "Sign carries the meaning: positive converging, negative diverging.",
                "best_for": ["converging_pairs", "diverging_pairs"],
                "avoid_for": ["stable_cohorts"],
            }),
        ),
        (
            "change_point",
            json!({
                "value_range": "unbounded>=0",
                "measures": "largest single step-to-step jump in affinity",
                "why": "Isolates abrupt transitions that averages smooth away.",
                "best_for": ["abrupt_transitions", "outlier_mining"],
                "avoid_for": ["report_ready"],
            }),
        ),
    ]
}

// ---- numeric helpers -----------------------------------------------------------------

/// `np.quantile(values, q)` (linear method), including numpy's two-sided
/// interpolation formula, on an ascending slice.
fn np_quantile_sorted(s: &[f64], q: f64) -> f64 {
    let virtual_index = q * (s.len() - 1) as f64;
    let lo = virtual_index.floor();
    let gamma = virtual_index - lo;
    let lo = lo as usize;
    let hi = (lo + 1).min(s.len() - 1);
    let (a, b) = (s[lo], s[hi]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    s
}

/// Off-diagonal upper-triangle values, row-major (`np.triu_indices(k=1)`).
fn upper_values(m: &Array2<f64>) -> Vec<f64> {
    let n = m.nrows();
    (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .map(|(i, j)| m[[i, j]])
        .collect()
}

fn weight_distribution(values: &[f64]) -> Value {
    if values.is_empty() {
        return json!({"count": 0, "p25": null, "p50": null, "p95": null, "max": null});
    }
    let s = sorted(values);
    let q = |p: f64| round_to(np_quantile_sorted(&s, p), 6);
    json!({
        "count": values.len(),
        "p25": q(0.25),
        "p50": q(0.5),
        "p95": q(0.95),
        "max": round_to(s[s.len() - 1], 6),
    })
}

/// Component morphology of a graph (`connected_components` + `np.unique`).
fn component_summary(g: &Graph) -> Value {
    let labels = g.component_labels();
    let n_components = labels.iter().max().map_or(0, |m| m + 1);
    let mut counts = vec![0_usize; n_components];
    for &l in &labels {
        counts[l] += 1;
    }
    let total = labels.len();
    let largest = counts.iter().copied().max().unwrap_or(0);
    json!({
        "component_count": n_components,
        "largest_component_fraction": if total > 0 { round_to(largest as f64 / total as f64, 4) } else { 0.0 },
        "singleton_count": counts.iter().filter(|&&c| c == 1).count(),
        "component_sizes": size_summary(&counts, 10),
    })
}

/// `value_counts()`: distinct items by count, descending, ties by first appearance.
fn value_counts<T: Clone + Eq + std::hash::Hash>(items: &[T]) -> Vec<(T, usize)> {
    let mut index: HashMap<&T, usize> = HashMap::new();
    let mut out: Vec<(T, usize)> = Vec::new();
    for item in items {
        match index.get(item) {
            Some(&k) => out[k].1 += 1,
            None => {
                index.insert(item, out.len());
                out.push((item.clone(), 1));
            }
        }
    }
    out.sort_by_key(|item| std::cmp::Reverse(item.1));
    out
}

// ---- trajectory sequences --------------------------------------------------------------

/// `obs.assign(cluster=labels).pivot(index="entity_id", columns="t")`:
/// entities ascending, one label (or `None` where unobserved) per time step.
struct Sequences {
    entities: Vec<Key>,
    rows: Vec<Vec<Option<usize>>>,
    n_times: usize,
}

impl Sequences {
    fn new(ct: &CosmicTrajectory, labels: &[usize]) -> Self {
        Self::from_obs(&ct.obs, labels)
    }

    fn from_obs(obs: &[Observation], labels: &[usize]) -> Self {
        let mut times: Vec<usize> = obs.iter().map(|o| o.t).collect();
        times.sort_unstable();
        times.dedup();
        let mut entities: Vec<Key> = obs.iter().map(|o| o.entity_id.clone()).collect();
        entities.sort();
        entities.dedup();
        let e_index: HashMap<&Key, usize> =
            entities.iter().enumerate().map(|(k, e)| (e, k)).collect();
        let t_index: HashMap<usize, usize> =
            times.iter().enumerate().map(|(k, &t)| (t, k)).collect();
        let mut rows = vec![vec![None; times.len()]; entities.len()];
        for (o, &label) in obs.iter().zip(labels) {
            rows[e_index[&o.entity_id]][t_index[&o.t]] = Some(label);
        }
        Self {
            entities,
            rows,
            n_times: times.len(),
        }
    }

    fn observed(row: &[Option<usize>]) -> Vec<usize> {
        row.iter().flatten().copied().collect()
    }
}

fn transitions(seq: &[usize]) -> usize {
    seq.windows(2).filter(|w| w[0] != w[1]).count()
}

/// Component and archetype counts across a threshold ladder: one reading of
/// a steep curve is not ground truth.
pub fn threshold_sweep(ct: &CosmicTrajectory) -> Value {
    let rows: Vec<Value> = ARCHETYPE_SWEEP_THRESHOLDS
        .iter()
        .map(|&threshold| {
            let labels = ct.cluster_labels(threshold);
            let seqs = Sequences::new(ct, &labels);
            let counts = value_counts(&seqs.rows);
            let n_clusters = labels.iter().max().map_or(0, |m| m + 1);
            json!({
                "threshold": threshold,
                "observation_clusters": n_clusters,
                "distinct_trajectories": counts.len(),
                "largest_trajectory_group": counts.first().map_or(0, |c| c.1),
            })
        })
        .collect();
    json!({
        "basis": "connected components on CO_SIMILAR above each threshold",
        "note": "Cluster counts move steeply with threshold; compare rows rather than trusting any single one.",
        "rows": rows,
    })
}

/// Measurement payload for the observation-centric surface.
pub fn trajectory_diagnosis(ct: &CosmicTrajectory, detail: &str) -> Value {
    // The symmetric CSR stores every edge twice.
    let weights: Vec<f64> = ct
        .similarity
        .edges
        .iter()
        .flat_map(|&(_, _, w)| [w, w])
        .collect();
    let total_edges = ct.similarity.edges.len();
    let cross_edges = ct.cross_time(0.0).edges.len();

    let mut cover = Map::new();
    cover.insert("n_balls".into(), json!(ct.balls.len()));
    if !ct.balls.is_empty() {
        let mut by_eps: Vec<(f64, usize)> = Vec::new();
        for ball in &ct.balls {
            match by_eps.iter_mut().find(|(e, _)| *e == ball.eps) {
                Some((_, c)) => *c += 1,
                None => by_eps.push((ball.eps, 1)),
            }
        }
        by_eps.sort_by(|a, b| a.0.total_cmp(&b.0));
        let by_eps: Map<String, Value> = by_eps
            .into_iter()
            .map(|(e, c)| (pulsar_data::preprocess::py_float_repr(e), json!(c)))
            .collect();
        cover.insert("balls_by_eps".into(), Value::Object(by_eps));
        let sizes: Vec<usize> = ct.balls.iter().map(|b| b.members.len()).collect();
        cover.insert("ball_sizes".into(), size_summary(&sizes, 10));
        let memberships: Vec<f64> = ct.memberships().into_iter().map(|m| m as f64).collect();
        cover.insert(
            "memberships_per_observation".into(),
            weight_distribution(&memberships),
        );
    }

    let mut payload = json!({
        "representation": "trajectory",
        "node_identity": "observation (entity, t)",
        "panel": {
            "n_observations": ct.n_observations(),
            "n_entities": ct.n_entities(),
            "n_times": ct.n_times(),
        },
        "similarity_surface": {
            "surface": "CO_SIMILAR on the pooled observation graph",
            "construction": ct.meta["construction"],
            "similarity_threshold_applied": ct.meta["similarity_threshold"],
            "n_edges": total_edges,
            "cross_time_edges": cross_edges,
            "cross_time_fraction": if total_edges > 0 { round_to(cross_edges as f64 / total_edges as f64, 4) } else { 0.0 },
            "weight_distribution": weight_distribution(&weights),
        },
        "cover": cover,
        "threshold_sweep": threshold_sweep(ct),
        "next_tools": ["get_trajectory_archetypes", "get_cross_time_neighbors"],
    });
    if detail == "full" {
        payload["component_morphology"] = component_summary(&ct.similarity);
        payload["meta"] = ct.meta.clone();
    }
    payload
}

/// Per-aggregation comparison so the agent picks the right lens. Each
/// aggregation is cut at its own q90 magnitude: their ranges are not
/// comparable.
pub fn temporal_diagnosis(tcg: &TemporalCosmicGraph, detail: &str) -> Value {
    let n = tcg.n();
    let aggregations: Vec<Value> = aggregation_guidance()
        .into_iter()
        .map(|(name, guidance)| {
            let matrix = match name {
                "persistence" => tcg.persistence_graph(None),
                "mean" => tcg.mean_graph(),
                "recency" => tcg.recency_graph(0.9).expect("0.9 is in (0, 1)"),
                "volatility" => tcg.volatility_graph(),
                "trend" => tcg.trend_graph(),
                _ => tcg.change_point_graph(),
            };
            let values = upper_values(&matrix);
            let magnitude: Vec<f64> = values.iter().map(|v| v.abs()).collect();
            let cut = if magnitude.is_empty() {
                0.0
            } else {
                np_quantile_sorted(&sorted(&magnitude), AGGREGATION_CUT_QUANTILE)
            };
            // Discrete-valued aggregations put q90 exactly on the max, where a
            // strict `>` would select nothing: the cut is inclusive.
            let selected = |v: f64| {
                if cut > 0.0 {
                    v.abs() >= cut
                } else {
                    v.abs() > 0.0
                }
            };
            let mut edges = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    if selected(matrix[[i, j]]) {
                        edges.push((i, j, 1.0));
                    }
                }
            }
            let components = Graph { n, edges }.n_components();
            let edges_at_cut = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .filter(|&(i, j)| i != j && selected(matrix[[i, j]]))
                .count()
                / 2;
            let mut entry = Map::new();
            entry.insert("aggregation".into(), json!(name));
            if let Value::Object(g) = guidance {
                entry.extend(g);
            }
            entry.insert("distribution".into(), weight_distribution(&magnitude));
            entry.insert("self_scaled_cut".into(), json!(round_to(cut, 6)));
            entry.insert("edges_at_cut".into(), json!(edges_at_cut));
            entry.insert("components_at_cut".into(), json!(components));
            if name == "trend" {
                let (conv, div) = if cut > 0.0 {
                    (
                        values.iter().filter(|&&v| v >= cut).count(),
                        values.iter().filter(|&&v| v <= -cut).count(),
                    )
                } else {
                    (0, 0)
                };
                entry.insert("converging_pairs".into(), json!(conv));
                entry.insert("diverging_pairs".into(), json!(div));
            }
            Value::Object(entry)
        })
        .collect();
    let bytes = tcg.nbytes();
    let (a, b, c) = tcg.shape();
    let mut payload = json!({
        "representation": "temporal",
        "node_identity": "entity",
        "panel": {"n_entities": n, "n_times": tcg.T()},
        "tensor": {
            "shape": [a, b, c],
            "bytes": bytes,
            "mb": round_to(bytes as f64 / (1024.0 * 1024.0), 2),
        },
        "cut_basis": format!(
            "each aggregation cut at its own q{} magnitude; ranges are not comparable across aggregations",
            (AGGREGATION_CUT_QUANTILE * 100.0) as i64
        ),
        "aggregations": aggregations,
        "next_tools": ["diagnose_longitudinal_graph"],
    });
    if detail == "full" {
        payload["detail"] = json!("full");
        payload["threshold"] = json!(tcg.threshold());
    }
    payload
}

/// Distinct cluster sequences and their populations.
pub fn archetype_payload(
    ct: &CosmicTrajectory,
    threshold: f64,
    max_archetypes: usize,
    max_entities_per_archetype: usize,
) -> Value {
    let labels = ct.cluster_labels(threshold);
    let seqs = Sequences::new(ct, &labels);
    let counts = value_counts(&seqs.rows);
    let n_entities = seqs.rows.len();
    let archetypes: Vec<Value> = counts
        .iter()
        .take(max_archetypes)
        .map(|(sequence, population)| {
            let members: Vec<String> = seqs
                .rows
                .iter()
                .zip(&seqs.entities)
                .filter(|(row, _)| *row == sequence)
                .map(|(_, e)| e.to_string())
                .collect();
            json!({
                "sequence": sequence,
                "n_entities": population,
                "fraction": round_to(*population as f64 / n_entities as f64, 4),
                "transitions": transitions(&Sequences::observed(sequence)),
                "entities": bounded_list(&members, max_entities_per_archetype),
            })
        })
        .collect();
    json!({
        "representation": "trajectory",
        "interpretation_threshold": threshold,
        "threshold_role": "interpretation (edge filter), not construction",
        "n_entities": n_entities,
        "n_times": seqs.n_times,
        "distinct_trajectories": counts.len(),
        "archetypes_returned": archetypes.len(),
        "archetypes_omitted": counts.len().saturating_sub(archetypes.len()),
        "archetypes": archetypes,
        "threshold_sweep": threshold_sweep(ct),
        "next_tools": ["get_cross_time_neighbors", "diagnose_longitudinal_graph"],
    })
}

/// Ranked cross-time lookalikes of observation `obs_id` (`< n_observations`).
/// `direction` is `"forward"`, `"backward"` or anything else for both.
pub fn cross_time_payload(
    ct: &CosmicTrajectory,
    obs_id: usize,
    threshold: f64,
    max_neighbors: usize,
    direction: &str,
) -> Value {
    let source = &ct.obs[obs_id];
    let t0 = source.t as i64;
    let mut row: Vec<(usize, f64)> = ct
        .cross_time(threshold)
        .edges
        .iter()
        .filter_map(|&(i, j, w)| {
            if i == obs_id {
                Some((j, w))
            } else if j == obs_id {
                Some((i, w))
            } else {
                None
            }
        })
        .collect();
    row.sort_by_key(|&(col, _)| col);
    let kept: Vec<(usize, f64, i64)> = row
        .into_iter()
        .map(|(col, w)| (col, w, ct.obs[col].t as i64 - t0))
        .filter(|&(_, _, d)| match direction {
            "forward" => d > 0,
            "backward" => d < 0,
            _ => true,
        })
        .collect();
    let mut order: Vec<usize> = (0..kept.len()).collect();
    order.sort_by(|&a, &b| kept[b].1.total_cmp(&kept[a].1));
    let neighbors: Vec<Value> = order
        .iter()
        .take(max_neighbors)
        .map(|&k| {
            let (col, w, d) = kept[k];
            let o = &ct.obs[col];
            json!({
                "obs_id": col,
                "entity_id": o.entity_id.to_string(),
                "t": o.t,
                "time_label": o.timestamp,
                "delta_t": d,
                "weight": round_to(w, 6),
            })
        })
        .collect();
    json!({
        "representation": "trajectory",
        "surface": "CO_SIMILAR restricted to cross-time pairs",
        "interpretation_threshold": threshold,
        "direction": direction,
        "source": {
            "obs_id": obs_id,
            "entity_id": source.entity_id.to_string(),
            "t": source.t,
            "time_label": source.timestamp,
        },
        "neighbors_returned": neighbors.len(),
        "neighbors_omitted": kept.len().saturating_sub(neighbors.len()),
        "cross_time_degree": kept.len(),
        "neighbors": neighbors,
        "next_tools": ["get_trajectory_archetypes"],
    })
}

// ---- trajectory classification -------------------------------------------------------

/// Minimum insertions, deletions and substitutions turning `a` into `b`.
pub fn levenshtein_distance(a: &[usize], b: &[usize]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            let cost = usize::from(x != y);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Dynamic-time-warping distance with a 0/1 match cost.
pub fn dtw_distance(a: &[usize], b: &[usize]) -> f64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return n.max(m) as f64;
    }
    let cost = |i: usize, j: usize| if a[i] == b[j] { 0.0 } else { 1.0 };
    let mut dp = vec![vec![f64::INFINITY; m]; n];
    dp[0][0] = cost(0, 0);
    for i in 1..n {
        dp[i][0] = dp[i - 1][0] + cost(i, 0);
    }
    for j in 1..m {
        dp[0][j] = dp[0][j - 1] + cost(0, j);
    }
    for i in 1..n {
        for j in 1..m {
            dp[i][j] = cost(i, j) + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }
    dp[n - 1][m - 1]
}

/// `fcluster(linkage(condensed, "complete"), t=k, criterion="maxclust")`:
/// the lowest merge height leaving at most `k` flat clusters, labelled
/// 1.. in the left-first order of the dendrogram walk.
fn complete_linkage_maxclust(mut condensed: Vec<f64>, n: usize, k: usize) -> Vec<usize> {
    if n == 1 {
        return vec![1];
    }
    let dendrogram = linkage(&mut condensed, n, Method::Complete);
    let steps = dendrogram.steps();
    // Heights are non-decreasing; the root (index n - 2) always yields 1 cluster.
    let threshold = if n <= k {
        f64::NEG_INFINITY
    } else {
        steps
            .iter()
            .map(|st| st.dissimilarity)
            .find(|&h| n - steps.iter().filter(|st| st.dissimilarity <= h).count() <= k)
            .unwrap_or(f64::INFINITY)
    };
    // scipy's `cluster_monocrit` walk: from the root, descend into internal
    // children (left first) before labelling a node's leaf children; the
    // first node at or below the threshold leads one flat cluster.
    let mut labels = vec![0; n];
    let mut visited = vec![false; 2 * n - 1];
    let mut stack = vec![2 * n - 2];
    let mut n_cluster = 0;
    let mut leader: Option<usize> = None;
    while let Some(&node) = stack.last() {
        let step = &steps[node - n];
        let (lc, rc) = (step.cluster1, step.cluster2);
        if leader.is_none() && step.dissimilarity <= threshold {
            leader = Some(node);
            n_cluster += 1;
        }
        if lc >= n && !visited[lc] {
            visited[lc] = true;
            stack.push(lc);
            continue;
        }
        if rc >= n && !visited[rc] {
            visited[rc] = true;
            stack.push(rc);
            continue;
        }
        for leaf in [lc, rc] {
            if leaf < n {
                if leader.is_none() {
                    n_cluster += 1;
                }
                labels[leaf] = n_cluster;
            }
        }
        if leader == Some(node) {
            leader = None;
        }
        stack.pop();
    }
    labels
}

fn sequence_cohorts(
    seqs: &[Vec<usize>],
    distance: impl Fn(&[usize], &[usize]) -> f64,
) -> Result<Vec<usize>, String> {
    let n = seqs.len();
    if n < 2 {
        return Err(
            "The number of observations cannot be determined on an empty distance matrix.".into(),
        );
    }
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            condensed.push(distance(&seqs[i], &seqs[j]));
        }
    }
    Ok(complete_linkage_maxclust(condensed, n, SEQUENCE_COHORTS))
}

/// Shannon entropy (bits) of the visit counts, summed in `value_counts` order.
fn entropy(seq: &[usize]) -> f64 {
    let counts = value_counts(seq);
    if counts.len() <= 1 {
        return 0.0;
    }
    let len = seq.len() as f64;
    -counts
        .iter()
        .map(|&(_, c)| {
            let p = c as f64 / len;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Classify every entity's trajectory by `method`: `complexity`,
/// `transition`, `levenshtein`, `dtw` or `sequence`.
pub fn classify_trajectories_payload(
    ct: &CosmicTrajectory,
    method: &str,
    threshold: f64,
) -> Result<Value, String> {
    classify_labels(&ct.obs, &ct.cluster_labels(threshold), method, threshold)
}

/// [`classify_trajectories_payload`] over given per-observation labels.
pub fn classify_labels(
    obs: &[Observation],
    labels: &[usize],
    method: &str,
    threshold: f64,
) -> Result<Value, String> {
    let seqs = Sequences::from_obs(obs, labels);
    let observed: Vec<Vec<usize>> = seqs.rows.iter().map(|r| Sequences::observed(r)).collect();
    let mut classification: Vec<(String, Value)> = Vec::with_capacity(observed.len());
    let name = |e: &Key| e.to_string();
    match method {
        "complexity" => {
            for (e, seq) in seqs.entities.iter().zip(&observed) {
                if seq.is_empty() {
                    classification.push((name(e), json!({"class": "Unknown", "entropy": 0.0, "transitions": 0, "length": 0, "sequence": []})));
                    continue;
                }
                let h = entropy(seq);
                let tr = transitions(seq);
                let class = if seq.len() <= 1 {
                    "Singleton (1 visit)"
                } else if h == 0.0 {
                    "Stable (0 entropy)"
                } else if tr < 2 {
                    "Gradual Transition"
                } else {
                    "Volatile / Refractory"
                };
                classification.push((
                    name(e),
                    json!({"class": class, "entropy": round_to(h, 4), "transitions": tr, "length": seq.len(), "sequence": seq}),
                ));
            }
        }
        "transition" => {
            for (e, seq) in seqs.entities.iter().zip(&observed) {
                if seq.len() <= 1 {
                    classification.push((name(e), json!({"class": "Insufficient Visits (<2)", "self_retention_rate": 1.0, "length": seq.len(), "sequence": seq})));
                    continue;
                }
                let loops = seq.windows(2).filter(|w| w[0] == w[1]).count();
                let rate = loops as f64 / (seq.len() - 1) as f64;
                let class = if rate >= 0.8 {
                    "Highly Stable (>=80% retention)"
                } else if rate >= 0.4 {
                    "Transitioning (40%-80% retention)"
                } else {
                    "Highly Volatile (<40% retention)"
                };
                classification.push((
                    name(e),
                    json!({"class": class, "self_retention_rate": round_to(rate, 4), "length": seq.len(), "sequence": seq}),
                ));
            }
        }
        "levenshtein" | "dtw" => {
            let cohorts = if method == "levenshtein" {
                sequence_cohorts(&observed, |a, b| levenshtein_distance(a, b) as f64)?
            } else {
                sequence_cohorts(&observed, dtw_distance)?
            };
            for ((e, seq), cohort) in seqs.entities.iter().zip(&observed).zip(cohorts) {
                classification.push((
                    name(e),
                    json!({"class": format!("Cohort Class {cohort}"), "length": seq.len(), "sequence": seq}),
                ));
            }
        }
        "sequence" => {
            for (e, seq) in seqs.entities.iter().zip(&observed) {
                let mut states = seq.clone();
                states.sort_unstable();
                states.dedup();
                let class = if seq.len() <= 1 {
                    "Singleton Path".to_owned()
                } else if states.len() == 1 {
                    "Monostate Path".to_owned()
                } else {
                    format!("Multistate Transitioning Path ({} states)", states.len())
                };
                classification.push((
                    name(e),
                    json!({"class": class, "unique_states": states.len(), "length": seq.len(), "sequence": seq}),
                ));
            }
        }
        other => return Err(format!("Unknown classification method '{other}'")),
    }

    // Python keyed a dict by str(entity_id): a repeated name keeps its first
    // position and its last value.
    let mut by_name: Vec<(String, Value)> = Vec::new();
    let mut position: HashMap<String, usize> = HashMap::new();
    for (k, v) in classification {
        match position.get(&k) {
            Some(&p) => by_name[p].1 = v,
            None => {
                position.insert(k.clone(), by_name.len());
                by_name.push((k, v));
            }
        }
    }
    let classes: Vec<String> = by_name
        .iter()
        .map(|(_, v)| v["class"].as_str().unwrap_or_default().to_owned())
        .collect();
    let total = by_name.len();
    let summary: Map<String, Value> = value_counts(&classes)
        .into_iter()
        .map(|(class, count)| {
            (
                class,
                json!({"count": count, "fraction": round_to(count as f64 / total as f64, 4)}),
            )
        })
        .collect();
    let sorted_classification: BTreeMap<String, Value> = by_name.into_iter().collect();
    Ok(json!({
        "representation": "trajectory",
        "method": method,
        "interpretation_threshold": threshold,
        "n_entities": total,
        "classes_summary": summary,
        "classification": sorted_classification,
        "next_tools": ["get_trajectory_archetypes", "diagnose_longitudinal_graph"],
    }))
}

// ---- entity sampling -------------------------------------------------------------------

/// `f"{type(e).__module__}.{type(e).__qualname__}:{e!r}"`: the identity the
/// sample ranks entities by.
fn python_identity(key: &Key) -> String {
    match key {
        Key::Bool(b) => format!("builtins.bool:{}", if *b { "True" } else { "False" }),
        Key::Int(i) => format!("builtins.int:{i}"),
        Key::Float(x) => format!(
            "builtins.float:{}",
            pulsar_data::preprocess::py_float_repr(*x)
        ),
        Key::Text(s) => format!("builtins.str:{}", py_repr(s)),
    }
}

fn key_at(col: &Column, i: usize) -> Option<Key> {
    match col {
        Column::Float(v) => (!v[i].is_nan()).then_some(Key::Float(v[i])),
        Column::Int(v) => Some(Key::Int(v[i])),
        Column::Bool(v) => v[i].map(Key::Bool),
        Column::Text(v) => v[i].clone().map(Key::Text),
    }
}

/// A deterministic entity sample: entities with at least `min_observations`
/// rows (and `min_time_points` distinct times), ranked by the SHA-256 of
/// their Python identity, first `max_entities` kept. Returns the selected
/// entities, the eligible count and the rows to keep (table order).
pub fn select_longitudinal_entities(
    table: &Table,
    entity_column: &str,
    max_entities: usize,
    min_observations: usize,
    time_column: Option<&str>,
    min_time_points: Option<usize>,
) -> (Vec<Key>, usize, Vec<usize>) {
    let Some(entities) = table.column(entity_column) else {
        return (vec![], 0, vec![]);
    };
    let times = time_column.and_then(|t| table.column(t));
    let mut rows_of: HashMap<Key, (usize, Vec<Key>)> = HashMap::new();
    for i in 0..table.n_rows() {
        let Some(e) = key_at(entities, i) else {
            continue;
        };
        let entry = rows_of.entry(e).or_default();
        entry.0 += 1;
        if let Some(t) = times.and_then(|c| key_at(c, i)) {
            entry.1.push(t);
        }
    }
    let mut eligible: Vec<Key> = rows_of
        .iter_mut()
        .filter(|(_, (count, _))| *count >= min_observations)
        .filter_map(|(e, (_, ts))| {
            if let Some(min_t) = min_time_points {
                ts.sort();
                ts.dedup();
                if ts.len() < min_t {
                    return None;
                }
            }
            Some(e.clone())
        })
        .collect();
    let digest = |e: &Key| Sha256::digest(python_identity(e).as_bytes());
    eligible.sort_by_cached_key(digest);
    let eligible_count = eligible.len();
    eligible.truncate(max_entities);
    let chosen: std::collections::HashSet<&Key> = eligible.iter().collect();
    let rows = (0..table.n_rows())
        .filter(|&i| key_at(entities, i).is_some_and(|e| chosen.contains(&e)))
        .collect();
    (eligible, eligible_count, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_quantile_uses_the_two_sided_lerp() {
        let s = [0.0, 1.0, 2.0, 3.0, 10.0];
        assert_eq!(np_quantile_sorted(&s, 0.5), 2.0);
        assert_eq!(np_quantile_sorted(&s, 0.9), 7.200000000000001); // numpy 2.x
        assert_eq!(np_quantile_sorted(&[4.0], 0.9), 4.0);
    }

    #[test]
    fn edit_distances_have_known_answers() {
        assert_eq!(levenshtein_distance(&[1, 2, 3], &[1, 3]), 1);
        assert_eq!(levenshtein_distance(&[], &[4, 4]), 2);
        assert_eq!(levenshtein_distance(&[0, 1, 2], &[2, 1, 0]), 2);
        assert_eq!(dtw_distance(&[0, 0, 1], &[0, 1, 1]), 0.0);
        assert_eq!(dtw_distance(&[0, 1], &[2]), 2.0);
        assert_eq!(dtw_distance(&[], &[1, 2, 3]), 3.0);
    }

    #[test]
    fn maxclust_matches_scipy_on_probed_cases() {
        // Observed with scipy 1.18: fcluster(linkage(pdist(x), "complete"), 4, "maxclust").
        let line = |xs: &[f64]| {
            let mut c = vec![];
            for i in 0..xs.len() {
                for j in i + 1..xs.len() {
                    c.push((xs[i] - xs[j]).abs());
                }
            }
            complete_linkage_maxclust(c, xs.len(), 4)
        };
        assert_eq!(line(&[0., 1., 3., 6., 10.]), vec![1, 1, 2, 3, 4]);
        assert_eq!(line(&[0., 1., 3., 6.]), vec![1, 2, 3, 4]);
        assert_eq!(line(&[0., 1.]), vec![1, 2]);
        assert_eq!(line(&[0., 0., 0., 5., 5.]), vec![1, 1, 1, 2, 2]);
        assert_eq!(
            line(&[0., 1., 2., 3., 4., 5., 6., 7.]),
            vec![1, 1, 2, 2, 3, 3, 4, 4]
        );
    }

    #[test]
    fn value_counts_breaks_ties_by_first_appearance() {
        assert_eq!(
            value_counts(&["b", "a", "a", "c", "b", "d"]),
            vec![("b", 2), ("a", 2), ("c", 1), ("d", 1)]
        );
    }

    #[test]
    fn python_identities_match_repr() {
        assert_eq!(
            python_identity(&Key::Text("p'1".into())),
            "builtins.str:\"p'1\""
        );
        assert_eq!(python_identity(&Key::Int(-3)), "builtins.int:-3");
        assert_eq!(python_identity(&Key::Float(2.0)), "builtins.float:2.0");
    }
}
