//! The tiered cluster dossier (`build_dossier` and `dossier_to_markdown` in
//! `pulsar/mcp/interpreter.py`): one profile per cluster with its strongest
//! numeric and categorical evidence, its neighbours and the three most central
//! rows, plus global context. Measurements only: the agent names the clusters.
//!
//! Central rows are the three highest-PageRank nodes of the cluster's subgraph,
//! with networkx's power iteration (`alpha = 0.85`, `tol = 1e-6`, at most 100
//! iterations); if it does not converge the first three rows of the cluster are
//! used. Nodes with equal PageRank (isolated nodes, say) are ordered as networkx
//! orders them: by the iteration order of the Python set of the cluster's node
//! ids, which [`cpython_set_order`] reproduces.

use std::collections::HashMap;

use pulsar_data::{Column, Table};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::dossier_payloads::{detail_categorical_rows, detail_numeric_rows, to_values};
use crate::feature_evidence::{FeatureEvidenceIndex, Neighbor};

/// A dossier with more feature-cluster rows than this warns the reader to use
/// the targeted profiling tools instead.
pub const OVERSIZE_ROW_WARNING: usize = 500;
const PAGERANK_ALPHA: f64 = 0.85;
const PAGERANK_TOL: f64 = 1.0e-6;
const PAGERANK_MAX_ITER: usize = 100;
const CENTRAL_ROWS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DossierError {
    #[error("detail must be 'summary', 'standard', or 'full'")]
    Detail,
}

/// Statistical profile of one cluster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterProfile {
    pub cluster_id: i64,
    pub size: usize,
    pub size_pct: f64,
    pub semantic_name: String,
    pub numeric_features: Vec<Value>,
    pub categorical_features: Vec<Value>,
    pub central_rows: Vec<Value>,
    pub numeric_tiers: Value,
    pub categorical_tiers: Value,
    pub topological_neighbors: Vec<Neighbor>,
}

/// All cluster profiles and the global context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologicalDossier {
    pub n_total: usize,
    pub n_clusters: usize,
    pub clusters: Vec<ClusterProfile>,
    pub global_stats: Value,
}

// ---- PageRank --------------------------------------------------------------------------

/// PageRank of an undirected weighted graph on `n` nodes (networkx
/// `pagerank`, SciPy variant): power iteration from the uniform vector with a
/// uniform teleport and dangling-node distribution, stopping when the L1
/// change drops below `n * tol`. `None` if it has not converged in
/// `PAGERANK_MAX_ITER` iterations.
pub fn pagerank(n: usize, edges: &[(usize, usize, f64)]) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(vec![]);
    }
    let mut out_weight = vec![0.0_f64; n];
    for &(s, t, w) in edges {
        out_weight[s] += w;
        if s != t {
            out_weight[t] += w;
        }
    }
    let p = vec![1.0 / n as f64; n];
    let mut x = p.clone();
    for _ in 0..PAGERANK_MAX_ITER {
        let dangling: f64 = (0..n).filter(|&i| out_weight[i] == 0.0).map(|i| x[i]).sum();
        let mut walk = vec![0.0_f64; n];
        for &(s, t, w) in edges {
            walk[t] += x[s] * w / out_weight[s];
            if s != t {
                walk[s] += x[t] * w / out_weight[t];
            }
        }
        let next: Vec<f64> = (0..n)
            .map(|i| PAGERANK_ALPHA * (walk[i] + dangling * p[i]) + (1.0 - PAGERANK_ALPHA) * p[i])
            .collect();
        let err: f64 = next.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
        x = next;
        if err < n as f64 * PAGERANK_TOL {
            return Some(x);
        }
    }
    None
}

// ---- CPython set order -------------------------------------------------------------------

const SET_MIN_SIZE: usize = 8;
const SET_LINEAR_PROBES: usize = 9;
const SET_PERTURB_SHIFT: u32 = 5;

/// The iteration order of `set(items)` for distinct non-negative ints inserted
/// in the given order (CPython 3.12's open-addressing table: ints hash to
/// themselves, nine linear probes before a perturbed jump, growth to a power of
/// two above four times the element count once the table is three-fifths full).
///
/// networkx builds a subgraph's node set this way, and orders nodes with equal
/// scores by that iteration order.
fn cpython_set_order(items: &[usize]) -> Vec<usize> {
    fn slot_for(table: &[Option<usize>], hash: usize) -> usize {
        let mask = table.len() - 1;
        let (mut i, mut perturb) = (hash & mask, hash);
        loop {
            if table[i].is_none() {
                return i;
            }
            if i + SET_LINEAR_PROBES <= mask {
                if let Some(j) = (1..=SET_LINEAR_PROBES).find(|&j| table[i + j].is_none()) {
                    return i + j;
                }
            }
            perturb >>= SET_PERTURB_SHIFT;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }
    let mut table: Vec<Option<usize>> = vec![None; SET_MIN_SIZE];
    let mut used = 0;
    for &key in items {
        let slot = slot_for(&table, key);
        table[slot] = Some(key);
        used += 1;
        let mask = table.len() - 1;
        if used * 5 >= mask * 3 {
            let mut size = SET_MIN_SIZE;
            while size <= used * 4 {
                size <<= 1;
            }
            let mut grown: Vec<Option<usize>> = vec![None; size];
            for old in table.iter().flatten() {
                let slot = slot_for(&grown, *old);
                grown[slot] = Some(*old);
            }
            table = grown;
        }
    }
    table.into_iter().flatten().collect()
}

// ---- central rows -----------------------------------------------------------------------

fn cell(column: &Column, i: usize) -> Value {
    match column {
        Column::Float(v) => {
            if v[i].is_nan() {
                Value::Null
            } else {
                json!(v[i])
            }
        }
        Column::Int(v) => json!(v[i]),
        Column::Bool(v) => v[i].map_or(Value::Null, Value::Bool),
        Column::Text(v) => v[i].as_ref().map_or(Value::Null, |s| json!(s)),
    }
}

fn record(table: &Table, row: usize, columns: &[String]) -> Value {
    Value::Object(
        columns
            .iter()
            .filter_map(|c| table.column(c).map(|col| (c.clone(), cell(col, row))))
            .collect::<Map<_, _>>(),
    )
}

/// The columns shown for a cluster's central rows: the first ten numeric and
/// five categorical feature columns of its evidence, without repeats.
fn top_columns(numeric: &[Value], categorical: &[Value]) -> Vec<String> {
    let mut columns: Vec<String> = vec![];
    let named = |rows: &[Value], limit: usize| -> Vec<String> {
        rows.iter()
            .take(limit)
            .filter_map(|r| r["column"].as_str().map(str::to_owned))
            .collect()
    };
    for c in named(numeric, 10).into_iter().chain(named(categorical, 5)) {
        if !columns.contains(&c) {
            columns.push(c);
        }
    }
    columns
}

fn central_rows(
    table: &Table,
    clusters: &[i64],
    edges: &[(usize, usize, f64)],
    cluster_id: i64,
    columns: &[String],
) -> Vec<Value> {
    let members: Vec<usize> = (0..clusters.len())
        .filter(|&i| clusters[i] == cluster_id)
        .collect();
    // networkx iterates a subgraph's nodes in the order of a set of the member
    // ids when that is under half the graph, else in graph order.
    let nodes = if 2 * members.len() < clusters.len() {
        cpython_set_order(&members)
    } else {
        members
    };
    let local: HashMap<usize, usize> = nodes.iter().enumerate().map(|(k, &n)| (n, k)).collect();
    let sub_edges: Vec<(usize, usize, f64)> = edges
        .iter()
        .filter_map(|&(s, t, w)| Some((*local.get(&s)?, *local.get(&t)?, w)))
        .collect();
    let chosen: Vec<usize> = match pagerank(nodes.len(), &sub_edges) {
        Some(scores) => {
            let mut order: Vec<usize> = (0..nodes.len()).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            order
                .into_iter()
                .take(CENTRAL_ROWS)
                .map(|k| nodes[k])
                .collect()
        }
        None => nodes.iter().copied().take(CENTRAL_ROWS).collect(),
    };
    chosen
        .into_iter()
        .map(|row| record(table, row, columns))
        .collect()
}

// ---- build ------------------------------------------------------------------------------

/// Build a tiered dossier from the cached feature evidence. `graph_metrics` is
/// the graph diagnosis of the model, carried through as is.
pub fn build_dossier(
    index: &FeatureEvidenceIndex,
    table: &Table,
    clusters: &[i64],
    edges: &[(usize, usize, f64)],
    detail: &str,
    graph_metrics: Value,
) -> Result<TopologicalDossier, DossierError> {
    if !matches!(detail, "summary" | "standard" | "full") {
        return Err(DossierError::Detail);
    }
    let excluded: Vec<&str> = index.metadata["excluded_columns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let working: Vec<String> = if index.working_columns.is_empty() {
        table
            .names
            .iter()
            .filter(|c| !excluded.contains(&c.as_str()))
            .cloned()
            .collect()
    } else {
        index.working_columns.clone()
    };

    let mut profiles = vec![];
    for (&cluster_id, bundle) in &index.cluster_bundles {
        let numeric_rows = to_values(&bundle.numeric);
        let categorical_rows = to_values(&bundle.categorical);
        let tiers = |rows: &[Value]| -> Value {
            let of = |tier: &str| -> Vec<Value> {
                rows.iter()
                    .filter(|r| r["signal_tier"] == tier)
                    .cloned()
                    .collect()
            };
            json!({"core": of("core"), "supporting": of("supporting"), "context": of("context")})
        };
        let mut top = top_columns(&numeric_rows, &categorical_rows);
        if top.is_empty() {
            top = working.iter().take(10).cloned().collect();
        }
        profiles.push(ClusterProfile {
            cluster_id,
            size: bundle.size,
            size_pct: bundle.size_pct,
            semantic_name: bundle.semantic_name.clone(),
            numeric_features: detail_numeric_rows(&numeric_rows, detail),
            categorical_features: detail_categorical_rows(&categorical_rows, detail),
            central_rows: central_rows(table, clusters, edges, cluster_id, &top),
            numeric_tiers: tiers(&numeric_rows),
            categorical_tiers: tiers(&categorical_rows),
            topological_neighbors: bundle.topological_neighbors.clone(),
        });
    }

    let meta = &index.metadata;
    let numeric_effects: Map<String, Value> = meta["global_numeric_stats"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(column, stats)| (column.clone(), stats["effect"].clone()))
        .collect();
    const EVIDENCE_KEYS: [&str; 8] = [
        "numeric_features_screened",
        "categorical_columns_screened",
        "categorical_values_screened",
        "categorical_columns_gated",
        "tiering_method",
        "neighbor_contrast_enabled",
        "excluded_columns",
        "stats_failures",
    ];
    let evidence_metadata: Map<String, Value> = meta
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(k, _)| EVIDENCE_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok(TopologicalDossier {
        n_total: meta["n_total"].as_u64().unwrap_or(0) as usize,
        n_clusters: meta["n_clusters"].as_u64().unwrap_or(0) as usize,
        clusters: profiles,
        global_stats: json!({
            "numeric": numeric_effects,
            "columns": meta["columns"],
            "graph_metrics": graph_metrics,
            "evidence_metadata": evidence_metadata,
            "cluster_adjacency": meta["cluster_adjacency"],
            "signal_matrix": index.signal_matrix,
            "numeric_global_ranking": index.numeric_global_ranking,
            "categorical_global_ranking": index.categorical_global_ranking,
            "detail": detail,
        }),
    })
}

// ---- Markdown ---------------------------------------------------------------------------

/// A horizontal bar for `value` out of `max_value` (`generate_proportion_bar`).
pub fn proportion_bar(value: f64, max_value: f64, length: usize) -> String {
    if max_value <= 0.0 {
        return "░".repeat(length);
    }
    let fraction = (value / max_value).clamp(0.0, 1.0);
    let filled = (fraction * length as f64) as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(length - filled))
}

/// `f"{x:.{digits}f}"` of a stored number (null is NaN).
fn fixed(value: &Value, digits: usize) -> String {
    match value.as_f64() {
        None => "nan".into(),
        Some(x) if x.is_nan() => "nan".into(),
        Some(x) if x.is_infinite() => {
            if x > 0.0 {
                "inf".into()
            } else {
                "-inf".into()
            }
        }
        Some(x) => format!("{x:.digits$}"),
    }
}

/// `str(value)` of a table cell as pandas hands it over (missing is `nan`).
fn cell_str(value: &Value) -> String {
    match value {
        Value::Null => "nan".into(),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_f64() => {
            pulsar_data::preprocess::py_float_repr(n.as_f64().expect("is_f64"))
        }
        other => other.to_string(),
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => cell_str(other),
    }
}

/// Render the dossier as Markdown. `max_clusters` keeps the largest clusters;
/// `feature_preview_limit` caps the feature rows shown per cluster.
pub fn dossier_to_markdown(
    dossier: &TopologicalDossier,
    max_clusters: Option<usize>,
    feature_preview_limit: Option<usize>,
) -> String {
    let mut profiles: Vec<&ClusterProfile> = dossier.clusters.iter().collect();
    if let Some(max) = max_clusters {
        profiles.sort_by_key(|p| std::cmp::Reverse(p.size));
        profiles.truncate(max);
    }
    let omitted = dossier.clusters.len().saturating_sub(profiles.len());
    let total_rows: usize = dossier
        .clusters
        .iter()
        .map(|p| p.numeric_features.len() + p.categorical_features.len())
        .sum();

    let mut md = vec![
        "# Topological Analysis Dossier".to_owned(),
        format!("**Dataset Size**: {} points", dossier.n_total),
        format!("**Clusters Found**: {}", dossier.n_clusters),
        String::new(),
    ];
    if total_rows >= OVERSIZE_ROW_WARNING {
        md.extend([
            "> [!WARNING]".to_owned(),
            format!("> This dossier contains {total_rows} feature-cluster rows, which is extremely large and may cause substantial context bloat."),
            "> It is highly recommended to use targeted profiling tools like `get_cluster_profile` or `get_feature_signal` to inspect specific clusters or features of interest instead of analyzing this full report.".to_owned(),
            String::new(),
        ]);
    }
    let gated = dossier.global_stats["evidence_metadata"]["categorical_columns_gated"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if !gated.is_empty() {
        let described: Vec<String> = gated
            .iter()
            .map(|g| {
                format!(
                    "{} (cardinality {})",
                    text(&g["column"]),
                    text(&g["cardinality"])
                )
            })
            .collect();
        md.push(format!(
            "**Categorical columns gated as high-cardinality noise**: {}",
            described.join(", ")
        ));
        md.push(String::new());
    }
    md.extend(["## Cluster Profiles".to_owned(), String::new()]);

    let limit = |rows: &[Value]| -> Vec<Value> {
        rows.iter()
            .take(feature_preview_limit.unwrap_or(rows.len()))
            .cloned()
            .collect()
    };
    for p in profiles {
        let bar = proportion_bar(p.size as f64, dossier.n_total as f64, 12);
        md.push(format!("### {}", p.semantic_name));
        md.push(format!(
            "- **Size**: {} points ({:.1}%) {bar}",
            p.size, p.size_pct
        ));
        md.push("\n#### Top Defining Numeric Features (Shifted & Homogeneous)".into());
        md.push(
            "| Feature | Distribution | Cluster Mean | Global Mean | Z-Score | Homogeneity |"
                .into(),
        );
        md.push("|---|:---|---|---|---|---|".into());
        for f in limit(&p.numeric_features) {
            let h = f["homogeneity"].as_f64().unwrap_or(f64::NAN);
            let description = if h < 0.5 {
                "Tight"
            } else if h > 1.5 {
                "Divergent"
            } else {
                "Normal"
            };
            md.push(format!(
                "| {} | `{}` | {} | {} | {} | {description} ({}) |",
                text(&f["column"]),
                f["sparkline"].as_str().unwrap_or(""),
                fixed(&f["mean"], 3),
                fixed(&f["global_mean"], 3),
                fixed(&f["z_score"], 2),
                fixed(&f["homogeneity"], 2),
            ));
        }
        let categorical = limit(&p.categorical_features);
        if !categorical.is_empty() {
            md.push("\n#### Distinctive Categories (Concentration)".into());
            md.push("| Feature | Value | In-cluster prevalence | Global recall |".into());
            md.push("|---|---|---|---|".into());
            for f in &categorical {
                let prevalence = f
                    .get("in_cluster_prevalence")
                    .or_else(|| f.get("concentration"))
                    .cloned()
                    .unwrap_or(json!(0.0));
                let count = f.get("count").cloned().unwrap_or(json!(0));
                let cluster_size = f.get("cluster_size").cloned().unwrap_or(json!(p.size));
                let global_count = f
                    .get("global_count")
                    .cloned()
                    .unwrap_or_else(|| count.clone());
                md.push(format!(
                    "| {} | {} | {}% ({}/{}) | {}% ({}/{}) |",
                    text(&f["column"]),
                    text(&f["value"]),
                    fixed(&prevalence, 1),
                    text(&count),
                    text(&cluster_size),
                    fixed(f.get("global_recall").unwrap_or(&json!(0.0)), 1),
                    text(&count),
                    text(&global_count),
                ));
            }
        }
        md.push("\n#### Central Representative Rows".into());
        for (i, row) in p.central_rows.iter().enumerate() {
            let fields: Vec<String> = row
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, v)| format!("{k}: {}", cell_str(v)))
                .collect();
            md.push(format!("{}. {}...", i + 1, fields.join(", ")));
        }
        md.push("\n---\n".into());
    }
    if omitted > 0 {
        md.push(format!(
            "_... and {omitted} smaller cluster(s) omitted (increase `max_clusters` or use `get_cluster_profile` to inspect)._"
        ));
    }
    md.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_order_follows_the_hash_table_not_the_insertion_order() {
        // CPython: list({5, 100, 33, 8}) == [8, 33, 100, 5] (slots 0, 1, 4, 5 of 8).
        assert_eq!(cpython_set_order(&[5, 100, 33, 8]), [8, 33, 100, 5]);
        // Small ints in a larger table come out ascending: list(set(range(10))).
        assert_eq!(
            cpython_set_order(&(0..10).collect::<Vec<_>>()),
            (0..10).collect::<Vec<_>>()
        );
        // A colliding id wraps into a low slot: 9 and 1 share slot 1 of 8.
        assert_eq!(cpython_set_order(&[1, 9]), [1, 9]);
        assert_eq!(cpython_set_order(&[]), Vec::<usize>::new());
    }

    #[test]
    fn proportion_bars_fill_in_proportion() {
        assert_eq!(proportion_bar(5.0, 10.0, 12), "██████░░░░░░");
        assert_eq!(proportion_bar(0.0, 10.0, 4), "░░░░");
        assert_eq!(proportion_bar(3.0, 0.0, 4), "░░░░");
        assert_eq!(proportion_bar(30.0, 10.0, 3), "███");
    }

    #[test]
    fn an_unknown_detail_is_rejected_with_the_python_message() {
        let e = DossierError::Detail.to_string();
        assert_eq!(e, "detail must be 'summary', 'standard', or 'full'");
    }
}
