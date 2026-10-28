//! Agent-facing payloads built from the feature evidence index
//! (`cluster_profile_payload`, `feature_signal_payload`,
//! `signal_matrix_payload` and the row helpers in `pulsar/mcp/interpreter.py`).
//!
//! All are summary-first and bounded: tiered rows (core and supporting in full,
//! context compacted, noise only at `detail="full"`), a cap on features per
//! cluster, and omitted-cluster telemetry instead of unbounded lists.
//!
//! Known behaviors kept from Python so the two agree: a `column=value` feature
//! also selects every value of that column in `feature_signal_payload`; ties
//! between equal-scored clusters resolve by ascending cluster id (Python
//! iterates a set of small ints, which is ascending in practice).

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

use crate::feature_evidence::{ClusterBundle, FeatureEvidenceIndex};
use crate::pyround::round_to;

/// Numeric and categorical features shown in a signal matrix.
pub const MAX_SIGNAL_MATRIX_NUMERIC: usize = 10;
pub const MAX_SIGNAL_MATRIX_CATEGORICAL: usize = 5;
/// `max_signal` at or above this earns an omitted cluster a targeted-inspection note.
const RECOMMEND_INSPECTION_SIGNAL: f64 = 2.0;

/// The cluster id is not in the index.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ClusterNotFound(pub i64);

// ---- row helpers --------------------------------------------------------------------

pub(crate) fn to_values<T: serde::Serialize>(rows: &[T]) -> Vec<Value> {
    rows.iter()
        .map(|r| serde_json::to_value(r).expect("rows are representable as JSON"))
        .collect()
}

/// `row.get(key, default)` as a float (a stored NaN, written as null, stays NaN).
fn get_f(row: &Value, key: &str, default: f64) -> f64 {
    match row.get(key) {
        None => default,
        Some(v) => v.as_f64().unwrap_or(f64::NAN),
    }
}

fn get_s<'a>(row: &'a Value, key: &str) -> &'a str {
    row.get(key).and_then(Value::as_str).unwrap_or("")
}

fn pick(row: &Value, keys: &[&str]) -> Value {
    Value::Object(
        keys.iter()
            .map(|k| ((*k).to_owned(), row[*k].clone()))
            .collect::<Map<_, _>>(),
    )
}

const NUMERIC_COMPACT: [&str; 16] = [
    "column",
    "direction",
    "mean",
    "global_mean",
    "mean_rest",
    "median",
    "median_rest",
    "z_score",
    "homogeneity",
    "aggregate_score",
    "percentile_score",
    "signal_tier",
    "one_vs_rest_q",
    "global_q",
    "neighbor_effect",
    "sparkline",
];

const CATEGORICAL_COMPACT: [&str; 17] = [
    "column",
    "value",
    "count",
    "prevalence_cluster",
    "prevalence_rest",
    "global_recall",
    "lift",
    "aggregate_score",
    "percentile_score",
    "signal_tier",
    "fisher_q",
    "global_q",
    "neighbor_specificity",
    "concentration",
    "in_cluster_prevalence",
    "cluster_size",
    "global_count",
];

fn compact_numeric_row(row: &Value) -> Value {
    pick(row, &NUMERIC_COMPACT)
}

fn compact_categorical_row(row: &Value) -> Value {
    let mut compact = Map::new();
    for key in CATEGORICAL_COMPACT {
        // `concentration` is reported as the in-cluster prevalence.
        let source = if key == "concentration" {
            "in_cluster_prevalence"
        } else {
            key
        };
        compact.insert(key.to_owned(), row[source].clone());
    }
    Value::Object(compact)
}

/// Rows by detail: `summary` is core and supporting compacted; `standard` is
/// core and supporting in full plus compacted context; `full` is every row.
fn tier_filter(rows: &[Value], detail: &str, compact: fn(&Value) -> Value) -> Vec<Value> {
    let of = |tier: &str| -> Vec<Value> {
        rows.iter()
            .filter(|r| get_s(r, "signal_tier") == tier)
            .cloned()
            .collect()
    };
    let (core, supporting, context) = (of("core"), of("supporting"), of("context"));
    match detail {
        "summary" => core.iter().chain(&supporting).map(compact).collect(),
        "standard" => core
            .into_iter()
            .chain(supporting)
            .chain(context.iter().map(compact))
            .collect(),
        _ => rows.to_vec(),
    }
}

pub(crate) fn detail_numeric_rows(rows: &[Value], detail: &str) -> Vec<Value> {
    tier_filter(rows, detail, compact_numeric_row)
}

pub(crate) fn detail_categorical_rows(rows: &[Value], detail: &str) -> Vec<Value> {
    tier_filter(rows, detail, compact_categorical_row)
}

/// Negated scores, then column and value: ascending order is strongest first.
type SortKey = (Vec<f64>, String, String);

/// Sort key of a feature row: descending aggregate, then the kind's secondary
/// effects, then column and value. Compact rows lack some effect keys; those
/// default to 0 exactly as in Python.
fn signal_sort_key(row: &Value, numeric: bool) -> SortKey {
    let aggregate = get_f(row, "aggregate_score", 0.0).abs();
    let secondary = if numeric {
        [
            get_f(row, "effect_mean_std", 0.0).abs(),
            get_f(row, "z_score", 0.0).abs(),
            get_f(row, "neighbor_effect", 0.0).abs(),
        ]
    } else {
        [
            get_f(row, "log_lift", 0.0).abs(),
            get_f(row, "neighbor_specificity", 0.0).abs(),
            (get_f(row, "lift", 1.0) - 1.0).abs(),
        ]
    };
    let value = match row.get("value") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => crate::plan::py_str(other),
        None => String::new(),
    };
    (
        std::iter::once(-aggregate)
            .chain(secondary.iter().map(|v| -v))
            .collect(),
        get_s(row, "column").to_owned(),
        value,
    )
}

fn cmp_keys(a: &SortKey, b: &SortKey) -> std::cmp::Ordering {
    for (x, y) in a.0.iter().zip(&b.0) {
        let o = x.total_cmp(y);
        if o != std::cmp::Ordering::Equal {
            return o;
        }
    }
    a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2))
}

/// The strongest `max_features` rows across both kinds, with the counts of what
/// was returned and omitted.
fn limit_cluster_feature_rows(
    numeric: Vec<Value>,
    categorical: Vec<Value>,
    max_features: usize,
) -> (Vec<Value>, Vec<Value>, Map<String, Value>) {
    let (n_numeric, n_categorical) = (numeric.len(), categorical.len());
    let sorted_by_kind = |rows: Vec<Value>, is_numeric: bool| {
        let mut keyed: Vec<_> = rows
            .into_iter()
            .map(|r| (signal_sort_key(&r, is_numeric), r))
            .collect();
        keyed.sort_by(|a, b| cmp_keys(&a.0, &b.0));
        keyed
    };
    let mut ranked: Vec<(bool, SortKey, Value)> = sorted_by_kind(numeric, true)
        .into_iter()
        .map(|(k, r)| (true, k, r))
        .chain(
            sorted_by_kind(categorical, false)
                .into_iter()
                .map(|(k, r)| (false, k, r)),
        )
        .collect();
    ranked.sort_by(|a, b| cmp_keys(&a.1, &b.1));
    ranked.truncate(max_features);

    let limited_numeric: Vec<Value> = ranked.iter().filter(|r| r.0).map(|r| r.2.clone()).collect();
    let limited_categorical: Vec<Value> = ranked
        .iter()
        .filter(|r| !r.0)
        .map(|r| r.2.clone())
        .collect();
    let omitted_numeric = n_numeric.saturating_sub(limited_numeric.len());
    let omitted_categorical = n_categorical.saturating_sub(limited_categorical.len());
    let mut meta = Map::new();
    meta.insert("feature_limit".into(), json!(max_features));
    meta.insert(
        "features_returned".into(),
        json!({"numeric": limited_numeric.len(), "categorical": limited_categorical.len(), "total": ranked.len()}),
    );
    meta.insert(
        "features_omitted".into(),
        json!({"numeric": omitted_numeric, "categorical": omitted_categorical, "total": omitted_numeric + omitted_categorical}),
    );
    (limited_numeric, limited_categorical, meta)
}

// ---- cluster profile -----------------------------------------------------------------

/// One cluster's profile: tiered feature rows capped at `max_features`, its
/// strongest topological neighbours and tier counts.
pub fn cluster_profile_payload(
    index: &FeatureEvidenceIndex,
    cluster_id: i64,
    detail: &str,
    max_features: usize,
) -> Result<Value, ClusterNotFound> {
    let bundle = index
        .cluster_bundles
        .get(&cluster_id)
        .ok_or(ClusterNotFound(cluster_id))?;
    let numeric_rows = to_values(&bundle.numeric);
    let categorical_rows = to_values(&bundle.categorical);
    let (numeric, categorical, limit) = limit_cluster_feature_rows(
        detail_numeric_rows(&numeric_rows, detail),
        detail_categorical_rows(&categorical_rows, detail),
        max_features,
    );
    let tier_counts = |rows: &[Value]| -> Value {
        let count = |tier: &str| {
            rows.iter()
                .filter(|r| get_s(r, "signal_tier") == tier)
                .count()
        };
        json!({"core": count("core"), "supporting": count("supporting"), "context": count("context"), "noise": count("noise")})
    };
    let mut payload = Map::new();
    payload.insert("cluster_id".into(), json!(cluster_id));
    payload.insert("size".into(), json!(bundle.size));
    payload.insert("size_pct".into(), json!(bundle.size_pct));
    payload.insert("semantic_name".into(), json!(bundle.semantic_name));
    payload.insert(
        "topological_neighbors".into(),
        json!(bundle.topological_neighbors),
    );
    payload.insert("numeric_features".into(), Value::Array(numeric));
    payload.insert("categorical_features".into(), Value::Array(categorical));
    payload.extend(limit);
    payload.insert(
        "tier_counts".into(),
        json!({"numeric": tier_counts(&numeric_rows), "categorical": tier_counts(&categorical_rows)}),
    );
    Ok(Value::Object(payload))
}

// ---- feature signal ------------------------------------------------------------------

/// How the named features behave across clusters. A `column=value` name picks a
/// categorical value; a bare name picks the column (and every value of a
/// categorical column).
pub fn feature_signal_payload(
    index: &FeatureEvidenceIndex,
    feature_names: &[String],
    cluster_ids: Option<&[i64]>,
    detail: &str,
    max_clusters: Option<usize>,
) -> Value {
    let requested: BTreeSet<i64> = match cluster_ids {
        Some(ids) => ids.iter().copied().collect(),
        None => index.cluster_bundles.keys().copied().collect(),
    };
    let features: Vec<&String> = feature_names.iter().filter(|f| !f.is_empty()).collect();
    let mut pairs: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut bare: BTreeSet<&str> = BTreeSet::new();
    for feature in &features {
        match feature.split_once('=') {
            Some((column, value)) => {
                pairs.insert((column, value));
                bare.insert(column);
            }
            None => {
                bare.insert(feature.as_str());
            }
        }
    }

    struct Raw<'a> {
        bundle: &'a ClusterBundle,
        signal: f64,
        numeric: Vec<Value>,
        categorical: Vec<Value>,
    }
    let mut raw: Vec<Raw> = vec![];
    for cluster_id in &requested {
        let Some(bundle) = index.cluster_bundles.get(cluster_id) else {
            continue;
        };
        let numeric: Vec<Value> = to_values(
            &bundle
                .numeric
                .iter()
                .filter(|r| bare.contains(r.column.as_str()))
                .cloned()
                .collect::<Vec<_>>(),
        );
        let categorical: Vec<Value> = to_values(
            &bundle
                .categorical
                .iter()
                .filter(|r| {
                    bare.contains(r.column.as_str())
                        || pairs.contains(&(r.column.as_str(), r.value.as_str()))
                })
                .cloned()
                .collect::<Vec<_>>(),
        );
        if numeric.is_empty() && categorical.is_empty() {
            continue;
        }
        let signal = numeric
            .iter()
            .chain(&categorical)
            .map(|r| get_f(r, "aggregate_score", 0.0).abs())
            .fold(0.0_f64, |a, b| if b > a { b } else { a });
        raw.push(Raw {
            bundle,
            signal,
            numeric,
            categorical,
        });
    }

    let (kept, omitted) = match (cluster_ids, max_clusters) {
        (None, Some(max)) => {
            raw.sort_by(|a, b| b.signal.total_cmp(&a.signal));
            let omitted = raw.len().saturating_sub(max);
            raw.truncate(max);
            raw.sort_by_key(|c| c.bundle.cluster_id);
            (raw, omitted)
        }
        _ => (raw, 0),
    };
    let clusters: Vec<Value> = kept
        .iter()
        .map(|c| {
            json!({
                "cluster_id": c.bundle.cluster_id,
                "semantic_name": c.bundle.semantic_name,
                "cluster_size": c.bundle.size,
                "numeric_features": detail_numeric_rows(&c.numeric, detail),
                "categorical_features": detail_categorical_rows(&c.categorical, detail),
            })
        })
        .collect();
    json!({
        "feature_names": features,
        "detail": detail,
        "clusters_returned": clusters.len(),
        "clusters_omitted": omitted,
        "clusters": clusters,
    })
}

// ---- signal matrix -------------------------------------------------------------------

fn rank_matrix_features(scores: &BTreeMap<String, f64>, limit: usize) -> Vec<String> {
    let mut ranked: Vec<(&String, f64)> = scores.iter().map(|(k, v)| (k, *v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked
        .into_iter()
        .take(limit)
        .map(|(k, _)| k.clone())
        .collect()
}

/// A matrix cell as Python prints it: floats to three places, anything else as text.
fn cell(value: Option<&Value>) -> String {
    match value {
        None => "-".to_owned(),
        Some(Value::Number(n)) if n.is_f64() => {
            let f = n.as_f64().expect("is_f64");
            format!("{f:.3}")
        }
        Some(Value::Null) => "nan".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => crate::plan::py_str(other),
    }
}

fn render_markdown_table(rows: &[Value], feature_order: &[String]) -> String {
    if rows.is_empty() {
        return "*No signals found in the selected tiers.*".into();
    }
    let available: BTreeSet<&str> = rows
        .iter()
        .flat_map(|r| {
            r["values"]
                .as_object()
                .into_iter()
                .flat_map(|m| m.keys().map(String::as_str))
        })
        .collect();
    let features: Vec<&String> = feature_order
        .iter()
        .filter(|f| available.contains(f.as_str()))
        .collect();
    if features.is_empty() {
        return "*No features available in selected tiers.*".into();
    }
    let mut out = format!(
        "| Cluster ID | {} |\n",
        features
            .iter()
            .map(|f| f.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    );
    out.push_str(&format!(
        "| --- |{} |\n",
        vec!["---"; features.len()].join(" | ")
    ));
    for row in rows {
        let cells: Vec<String> = features
            .iter()
            .map(|f| cell(row["values"].get(f.as_str())))
            .collect();
        out.push_str(&format!(
            "| {} | {} |\n",
            row["cluster_id"],
            cells.join(" | ")
        ));
    }
    out
}

/// Which clusters (strongest signal first, capped) and features show in the
/// cluster-by-feature matrix, as JSON or a Markdown report.
pub fn signal_matrix_payload(
    index: &FeatureEvidenceIndex,
    cluster_ids: Option<&[i64]>,
    include_context_tier: bool,
    max_clusters: Option<usize>,
    return_markdown: bool,
) -> Value {
    let requested: BTreeSet<i64> = match cluster_ids {
        Some(ids) => ids.iter().copied().collect(),
        None => index.cluster_bundles.keys().copied().collect(),
    };
    let score_of = |cid: i64| -> f64 {
        index.cluster_bundles.get(&cid).map_or(0.0, |b| {
            let rows = to_values(&b.numeric)
                .into_iter()
                .chain(to_values(&b.categorical));
            rows.map(|r| get_f(&r, "aggregate_score", 0.0).abs())
                .fold(0.0, |a, v| if v > a { v } else { a })
        })
    };
    let scores: BTreeMap<i64, f64> = requested.iter().map(|&c| (c, score_of(c))).collect();
    let mut ranked: Vec<i64> = requested.iter().copied().collect();
    ranked.sort_by(|a, b| scores[b].total_cmp(&scores[a]));
    let (kept, omitted_ids): (BTreeSet<i64>, Vec<i64>) = match max_clusters {
        Some(max) if ranked.len() > max => (
            ranked[..max].iter().copied().collect(),
            ranked[max..].to_vec(),
        ),
        _ => (ranked.iter().copied().collect(), vec![]),
    };
    let omitted_telemetry: Vec<Value> = omitted_ids
        .iter()
        .map(|c| json!({"cluster_id": c, "max_signal": round_to(scores[c], 3)}))
        .collect();

    let matrix = &index.signal_matrix;
    let by_cluster = |key: &str| -> BTreeMap<i64, Vec<&Value>> {
        let mut map: BTreeMap<i64, Vec<&Value>> = BTreeMap::new();
        for row in matrix[key].as_array().into_iter().flatten() {
            if let Some(cid) = row["cluster_id"].as_i64() {
                map.entry(cid).or_default().push(row);
            }
        }
        map
    };
    let (numeric_by, categorical_by) = (by_cluster("numeric_rows"), by_cluster("categorical_rows"));
    let allowed = |tier: Option<&str>| match tier {
        Some("core" | "supporting") => true,
        Some("context") => include_context_tier,
        _ => false,
    };

    let feature_scores = |by: &BTreeMap<i64, Vec<&Value>>| -> BTreeMap<String, f64> {
        let mut scores: BTreeMap<String, f64> = BTreeMap::new();
        for cid in &kept {
            for row in by.get(cid).into_iter().flatten() {
                for (key, value) in row["values"].as_object().into_iter().flatten() {
                    if !allowed(value["signal_tier"].as_str()) {
                        continue;
                    }
                    let score = get_f(value, "aggregate_score", 0.0).abs();
                    let entry = scores.entry(key.clone()).or_insert(0.0);
                    if score > *entry {
                        *entry = score;
                    }
                }
            }
        }
        scores
    };
    let (numeric_scores, categorical_scores) =
        (feature_scores(&numeric_by), feature_scores(&categorical_by));
    let selected_numeric = rank_matrix_features(&numeric_scores, MAX_SIGNAL_MATRIX_NUMERIC);
    let selected_categorical =
        rank_matrix_features(&categorical_scores, MAX_SIGNAL_MATRIX_CATEGORICAL);
    let numeric_omitted = numeric_scores.len().saturating_sub(selected_numeric.len());
    let categorical_omitted = categorical_scores
        .len()
        .saturating_sub(selected_categorical.len());

    let filtered_rows =
        |by: &BTreeMap<i64, Vec<&Value>>, selected: &[String], numeric: bool| -> Vec<Value> {
            let mut out = vec![];
            for cid in &kept {
                for row in by.get(cid).into_iter().flatten() {
                    let values: Map<String, Value> = row["values"]
                        .as_object()
                        .into_iter()
                        .flatten()
                        .filter(|(k, v)| selected.contains(k) && allowed(v["signal_tier"].as_str()))
                        .map(|(k, v)| {
                            // `safe_float` for numeric cells; categorical cells pass through.
                            let cell = if numeric {
                                json!(v["value"].as_f64().unwrap_or(f64::NAN))
                            } else {
                                v["value"].clone()
                            };
                            (k.clone(), cell)
                        })
                        .collect();
                    if !values.is_empty() {
                        out.push(json!({"cluster_id": cid, "values": values}));
                    }
                }
            }
            out
        };
    let numeric_rows = filtered_rows(&numeric_by, &selected_numeric, true);
    let categorical_rows = filtered_rows(&categorical_by, &selected_categorical, false);

    if return_markdown {
        let mut report = String::from("## Topological Signal Matrix\n\n### Telemetry\n");
        report.push_str(&format!("- Clusters returned: {}\n", kept.len()));
        report.push_str(&format!("- Clusters omitted: {}\n", omitted_ids.len()));
        report.push_str(&format!(
            "- Numeric columns shown: {} ({numeric_omitted} omitted)\n",
            selected_numeric.len()
        ));
        report.push_str(&format!(
            "- Categorical values shown: {} ({categorical_omitted} omitted)\n",
            selected_categorical.len()
        ));
        report.push_str(
            "- Cell meanings: numeric = z-score; categorical = in-cluster prevalence percent.\n",
        );
        if !omitted_telemetry.is_empty() {
            report.push_str("\n#### Omitted Clusters\n");
            for item in &omitted_telemetry {
                let score = item["max_signal"].as_f64().unwrap_or(0.0);
                let rec = if score >= RECOMMEND_INSPECTION_SIGNAL {
                    "Recommended for targeted inspection"
                } else {
                    "Low Signal: Ignorable"
                };
                report.push_str(&format!(
                    "- Cluster {}: max signal {}; {rec}\n",
                    item["cluster_id"],
                    crate::plan::py_str(&item["max_signal"])
                ));
            }
        }
        report.push('\n');
        report.push_str("### Numeric Feature Matrix\n");
        report.push_str(&render_markdown_table(&numeric_rows, &selected_numeric));
        report.push_str("\n\n### Categorical Feature Matrix\n");
        report.push_str(&render_markdown_table(
            &categorical_rows,
            &selected_categorical,
        ));
        report.push('\n');
        return json!({
            "clusters_returned": kept.len(),
            "clusters_omitted": omitted_ids.len(),
            "omitted_clusters": omitted_telemetry,
            "numeric_features_returned": selected_numeric.len(),
            "numeric_features_omitted": numeric_omitted,
            "categorical_features_returned": selected_categorical.len(),
            "categorical_features_omitted": categorical_omitted,
            "markdown_report": report,
        });
    }

    let selected_set: BTreeSet<&str> = selected_categorical.iter().map(String::as_str).collect();
    let categorical_values: Vec<Value> = matrix["categorical_values"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| {
            selected_set
                .contains(format!("{}={}", get_s(item, "column"), get_s(item, "value")).as_str())
        })
        .cloned()
        .collect();
    json!({
        "numeric_columns": selected_numeric,
        "categorical_values": categorical_values,
        "clusters_returned": kept.len(),
        "clusters_omitted": omitted_ids.len(),
        "omitted_clusters": omitted_telemetry,
        "numeric_features_omitted": numeric_omitted,
        "categorical_features_omitted": categorical_omitted,
        "numeric_rows": numeric_rows,
        "categorical_rows": categorical_rows,
    })
}
