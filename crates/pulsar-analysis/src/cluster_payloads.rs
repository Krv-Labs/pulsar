//! Cluster-facing payload builders (`pulsar/mcp/payloads.py` and the note in
//! `pulsar/mcp/tools/clustering.py`): the cluster header with fragmentation and
//! readiness advice, the cluster-selection disclosure, the dossier and summary
//! evidence payloads, the threshold surface, and the Markdown views.
//!
//! All are bounded and summary-first: size collections become
//! `{total, preview, omitted}`, previews are capped, and the readiness block is
//! advisory (`not a hard quality threshold`), never a verdict.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::dossier::TopologicalDossier;
use crate::dossier_payloads::to_values;
use crate::feature_evidence::FeatureEvidenceIndex;
use crate::payloads::{bounded_list, size_summary};
use crate::plan::py_str;
use crate::pyround::round_to;

const CLUSTER_ID_STABILITY_NOTE: &str = "Cluster IDs are stable across tools; subsets differ only by ranking basis. ID lists are capped previews with exact omitted counts.";
const TIERS: [&str; 4] = ["core", "supporting", "context", "noise"];

// ---- formatting helpers ----------------------------------------------------------------

/// `f"{x:.{digits}f}"`.
fn fixed(x: f64, digits: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        }
    } else {
        format!("{x:.digits$}")
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// `str(row.get(key, default))`: a missing key takes the default, a stored null prints `None`.
fn text_or(row: &Value, key: &str, default: &str) -> String {
    match row.get(key) {
        None => default.to_owned(),
        Some(Value::Null) => "None".into(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => py_str(other),
    }
}

fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        other => py_str(other),
    }
}

fn size_values(meta: &Value) -> Vec<usize> {
    meta["cluster_sizes"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|row| row["n"].as_i64().unwrap_or(0).max(0) as usize)
        .collect()
}

// ---- cluster header --------------------------------------------------------------------

/// The cluster header with bounded size telemetry; the full size list only on request.
pub fn compact_cluster_result_payload(meta: &Value, include_full_sizes: bool) -> Value {
    let mut out: Map<String, Value> = meta.as_object().cloned().unwrap_or_default();
    out.insert(
        "cluster_sizes_summary".into(),
        size_summary(&size_values(meta), 20),
    );
    if !include_full_sizes {
        out.shift_remove("cluster_sizes");
        if out.get("stability_plateaus").is_none_or(Value::is_null) {
            out.shift_remove("stability_plateaus");
        }
    }
    Value::Object(out)
}

/// A uniform disclosure of which clusters a truncated view shows, so that a
/// partial view is never mistaken for a different partition.
pub fn build_cluster_selection(meta: &Value, shown_ids: &[i64], ranked_by: &str) -> Value {
    let all_ids: Vec<i64> = meta["cluster_sizes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["cluster_id"].as_i64())
        .collect();
    let total = meta["n_clusters"]
        .as_u64()
        .map_or(all_ids.len(), |n| n as usize);
    let mut shown: Vec<i64> = shown_ids.to_vec();
    shown.sort_unstable();
    shown.dedup();
    let mut omitted: Vec<i64> = all_ids
        .into_iter()
        .filter(|id| !shown.contains(id))
        .collect();
    omitted.sort_unstable();
    json!({
        "total_clusters": total,
        "ranked_by": ranked_by,
        "shown_ids": bounded_list(&shown, 20),
        "omitted_ids": bounded_list(&omitted, 20),
        "note": format!("Showing {} of {total} clusters, ranked by {ranked_by}. {CLUSTER_ID_STABILITY_NOTE}", shown.len()),
    })
}

/// A compact, prominent Markdown header for a cluster-selection disclosure.
pub fn cluster_selection_to_markdown(selection: &Value) -> String {
    let (shown, omitted) = (&selection["shown_ids"], &selection["omitted_ids"]);
    let list = |v: &Value| {
        if v.is_null() {
            "[]".to_owned()
        } else {
            py_str(v)
        }
    };
    [
        "## Cluster Selection".to_owned(),
        String::new(),
        format!(
            "- Showing {} of {} clusters, ranked by `{}`",
            shown["total"].as_u64().unwrap_or(0),
            selection["total_clusters"].as_u64().unwrap_or(0),
            selection["ranked_by"].as_str().unwrap_or("unknown")
        ),
        format!(
            "- Shown IDs: {} (+{} more)",
            list(&shown["preview"]),
            shown["omitted"].as_u64().unwrap_or(0)
        ),
        format!(
            "- Omitted IDs: {} (+{} more)",
            list(&omitted["preview"]),
            omitted["omitted"].as_u64().unwrap_or(0)
        ),
        format!("- {}", selection["note"].as_str().unwrap_or("")),
        String::new(),
    ]
    .join("\n")
}

/// How fragmented a partition is, from its cluster sizes (largest first).
fn cluster_fragmentation_payload(sizes_desc: &[usize], total: usize) -> Value {
    if total == 0 || sizes_desc.is_empty() {
        return json!({
            "n_clusters": 0, "singleton_cluster_count": 0, "singleton_cluster_ratio": 0.0,
            "singleton_point_fraction": 0.0, "small_cluster_count": 0, "small_cluster_point_fraction": 0.0,
            "nontrivial_cluster_count": 0, "nontrivial_point_fraction": 0.0, "significant_cluster_count": 0,
            "largest_non_giant_cluster_pct": 0.0, "reference_nontrivial_min_size": 0,
            "reference_significant_min_size": 0,
        });
    }
    let t = total as f64;
    let nontrivial_min = 10.max((t * 0.01).ceil() as usize);
    let significant_min = 25.max((t * 0.03).ceil() as usize);
    let singleton_count = sizes_desc.iter().filter(|&&v| v == 1).count();
    let small: Vec<usize> = sizes_desc
        .iter()
        .copied()
        .filter(|&v| v < nontrivial_min)
        .collect();
    let nontrivial: Vec<usize> = sizes_desc
        .iter()
        .copied()
        .filter(|&v| v >= nontrivial_min)
        .collect();
    let significant = sizes_desc.iter().filter(|&&v| v >= significant_min).count();
    let tail_max = sizes_desc.iter().skip(1).copied().max().unwrap_or(0);
    json!({
        "n_clusters": sizes_desc.len(),
        "singleton_cluster_count": singleton_count,
        "singleton_cluster_ratio": round_to(singleton_count as f64 / sizes_desc.len() as f64, 4),
        "singleton_point_fraction": round_to(singleton_count as f64 / t, 4),
        "small_cluster_count": small.len(),
        "small_cluster_point_fraction": round_to(small.iter().sum::<usize>() as f64 / t, 4),
        "nontrivial_cluster_count": nontrivial.len(),
        "nontrivial_point_fraction": round_to(nontrivial.iter().sum::<usize>() as f64 / t, 4),
        "significant_cluster_count": significant,
        "largest_non_giant_cluster_pct": round_to(tail_max as f64 / t, 4),
        "reference_nontrivial_min_size": nontrivial_min,
        "reference_significant_min_size": significant_min,
    })
}

/// Advice on whether the labels are ready to interpret, from relative cluster mass.
fn interpretation_readiness(f: &Value) -> Value {
    let (singletons, nontrivial) = (
        f["singleton_cluster_count"].as_u64().unwrap_or(0),
        f["nontrivial_cluster_count"].as_u64().unwrap_or(0),
    );
    let (singleton_fraction, small_fraction, tail_pct) = (
        num(&f["singleton_point_fraction"]),
        num(&f["small_cluster_point_fraction"]),
        num(&f["largest_non_giant_cluster_pct"]),
    );
    let mut codes: Vec<&str> = vec![];
    if singletons > 0 {
        codes.push("SINGLETON_TAIL_PRESENT");
    }
    if singletons > nontrivial && singleton_fraction > tail_pct {
        codes.push("SINGLETON_TAIL_DOMINATES_NON_GIANT_STRUCTURE");
    }
    if small_fraction > tail_pct && f["small_cluster_count"].as_u64().unwrap_or(0) > 1 {
        codes.push("TINY_CLUSTER_MASS_DOMINATES_TAIL");
    }
    if f["n_clusters"].as_u64().unwrap_or(0) > 1 && nontrivial <= 1 {
        codes.push("LOW_NONTRIVIAL_CLUSTER_COUNT");
    }
    let (status, message, avoid): (&str, &str, Vec<&str>) = if codes
        .contains(&"SINGLETON_TAIL_DOMINATES_NON_GIANT_STRUCTURE")
    {
        (
            "review_required",
            "Cluster labels are dominated by singleton/tiny tail structure. Treat the labels as exploratory until threshold stability or sweep history shows coherent non-giant mass.",
            vec!["cluster naming before reviewing stability", "full dossier first"],
        )
    } else if !codes.is_empty() {
        (
            "caution",
            "Cluster labels include a small-cluster tail. Interpret only after checking whether the tail is stable signal or ice-chipping.",
            vec!["cluster naming before checking small-cluster evidence"],
        )
    } else {
        (
            "ready",
            "No singleton-heavy interpretation warning detected.",
            vec![],
        )
    };
    json!({
        "status": status,
        "reason_codes": codes,
        "basis": "advisory_from_relative_cluster_mass; not a hard quality threshold",
        "message": message,
        "allowed_next_steps": ["get_sweep_history", "get_threshold_stability_curve", "get_cluster_profile", "get_feature_signal"],
        "avoid": avoid,
    })
}

/// The cluster header of a clustering result: sizes, fragmentation, readiness
/// advice and provenance.
pub fn cluster_result_payload(
    labels: &[i64],
    method_used: &str,
    n_clusters: usize,
    silhouette_score: Option<f64>,
    interpretation_threshold: f64,
    stability_plateaus: &Value,
    failure_reason: Option<&str>,
) -> Value {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for &l in labels {
        *counts.entry(l).or_default() += 1;
    }
    let total = labels.len();
    let mut desc: Vec<usize> = counts.values().copied().collect();
    desc.sort_unstable_by(|a, b| b.cmp(a));
    let fragmentation = cluster_fragmentation_payload(&desc, total);
    let readiness = interpretation_readiness(&fragmentation);
    let sizes: Vec<Value> = counts
        .iter()
        .map(|(cid, n)| json!({"cluster_id": cid, "n": n, "pct": round_to(*n as f64 / total as f64 * 100.0, 1)}))
        .collect();
    json!({
        "method_used": method_used,
        "n_clusters": n_clusters,
        "cluster_sizes": sizes,
        "cluster_fragmentation": fragmentation,
        "interpretation_readiness": readiness,
        "silhouette_score": silhouette_score,
        "interpretation_edge_weight_threshold_applied": interpretation_threshold,
        "stability_plateaus": stability_plateaus,
        "failure_reason": failure_reason,
    })
}

// ---- evidence payloads -------------------------------------------------------------------

fn tier_counts_of_tiers(tiers: &Value) -> Value {
    Value::Object(
        tiers
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, rows)| (k.clone(), json!(rows.as_array().map_or(0, Vec::len))))
            .collect(),
    )
}

fn tier_counts_from_rows(rows: &[Value]) -> Value {
    Value::Object(
        TIERS
            .iter()
            .map(|t| {
                (
                    (*t).to_owned(),
                    json!(rows.iter().filter(|r| r["signal_tier"] == *t).count()),
                )
            })
            .collect(),
    )
}

fn feature_preview_row(row: &Value, numeric: bool) -> Value {
    let get = |k: &str| row.get(k).cloned().unwrap_or(Value::Null);
    let mut out = Map::new();
    out.insert(
        "kind".into(),
        json!(if numeric { "numeric" } else { "categorical" }),
    );
    for key in ["column", "signal_tier", "aggregate_score"] {
        out.insert(key.into(), get(key));
    }
    let extra: &[&str] = if numeric {
        &["direction", "mean", "mean_rest"]
    } else {
        &["value", "lift", "prevalence_cluster"]
    };
    for key in extra {
        out.insert((*key).into(), get(key));
    }
    Value::Object(out)
}

/// The strongest core and supporting rows across both kinds, capped.
fn feature_preview(
    numeric: &[Value],
    categorical: &[Value],
    limit: usize,
    total_available: usize,
) -> Map<String, Value> {
    let strong = |r: &&Value| matches!(r["signal_tier"].as_str(), Some("core" | "supporting"));
    let mut ranked: Vec<(bool, &Value)> =
        numeric.iter().filter(strong).map(|r| (true, r)).collect();
    ranked.extend(categorical.iter().filter(strong).map(|r| (false, r)));
    let score = |r: &Value| r.get("aggregate_score").map_or(0.0, num).abs();
    ranked.sort_by(|a, b| (-score(a.1)).total_cmp(&(-score(b.1))));
    let preview: Vec<Value> = ranked
        .iter()
        .take(limit)
        .map(|(n, r)| feature_preview_row(r, *n))
        .collect();
    let mut out = Map::new();
    out.insert("feature_preview".into(), Value::Array(preview.clone()));
    out.insert("feature_preview_limit".into(), json!(limit));
    out.insert("features_previewed".into(), json!(preview.len()));
    out.insert(
        "features_omitted".into(),
        json!(total_available.saturating_sub(preview.len())),
    );
    out
}

fn evidence_metadata_summary(metadata: &Value) -> Value {
    let counts: Map<String, Value> = metadata["stats_failures"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, v)| {
            (
                k.clone(),
                json!(v
                    .as_array()
                    .map(Vec::len)
                    .or_else(|| v.as_object().map(Map::len))
                    .unwrap_or(0)),
            )
        })
        .collect();
    json!({
        "stats_failure_counts": counts,
        "numeric_features_screened": metadata.get("numeric_features_screened").cloned().unwrap_or(json!(0)),
        "categorical_columns_screened": metadata.get("categorical_columns_screened").cloned().unwrap_or(json!(0)),
        "categorical_columns_gated": metadata.get("categorical_columns_gated").cloned().unwrap_or(json!([])),
    })
}

fn signal_matrix_summary(matrix: &Value) -> Value {
    let len = |k: &str| matrix[k].as_array().map_or(0, Vec::len);
    json!({
        "n_numeric_columns": len("numeric_columns"),
        "n_categorical_values": len("categorical_values"),
        "n_numeric_rows": len("numeric_rows"),
        "n_categorical_rows": len("categorical_rows"),
    })
}

fn cluster_payloads_from_dossier(
    dossier: &TopologicalDossier,
    detail: &str,
    limit: usize,
) -> Vec<Map<String, Value>> {
    dossier
        .clusters
        .iter()
        .map(|p| {
            let mut entry = Map::new();
            entry.insert("cluster_id".into(), json!(p.cluster_id));
            entry.insert("size".into(), json!(p.size));
            entry.insert("size_pct".into(), json!(p.size_pct));
            entry.insert("semantic_name".into(), json!(p.semantic_name));
            entry.insert(
                "topological_neighbors".into(),
                json!(p.topological_neighbors),
            );
            match detail {
                "summary" => {
                    entry.insert(
                        "numeric_tier_counts".into(),
                        tier_counts_of_tiers(&p.numeric_tiers),
                    );
                    entry.insert(
                        "categorical_tier_counts".into(),
                        tier_counts_of_tiers(&p.categorical_tiers),
                    );
                    let total = [&p.numeric_tiers, &p.categorical_tiers]
                        .iter()
                        .flat_map(|t| t.as_object().into_iter().flat_map(|m| m.values()))
                        .map(|rows| rows.as_array().map_or(0, Vec::len))
                        .sum();
                    entry.extend(feature_preview(
                        &p.numeric_features,
                        &p.categorical_features,
                        limit,
                        total,
                    ));
                }
                "standard" => {
                    entry.insert(
                        "numeric_tier_counts".into(),
                        tier_counts_of_tiers(&p.numeric_tiers),
                    );
                    entry.insert(
                        "categorical_tier_counts".into(),
                        tier_counts_of_tiers(&p.categorical_tiers),
                    );
                    entry.insert(
                        "numeric_features".into(),
                        json!(p.numeric_features.iter().take(limit).collect::<Vec<_>>()),
                    );
                    entry.insert(
                        "categorical_features".into(),
                        json!(p
                            .categorical_features
                            .iter()
                            .take(limit)
                            .collect::<Vec<_>>()),
                    );
                    entry.insert("central_rows".into(), json!(p.central_rows));
                }
                _ => {
                    entry.insert("numeric_features".into(), json!(p.numeric_features));
                    entry.insert("categorical_features".into(), json!(p.categorical_features));
                    entry.insert("numeric_tiers".into(), p.numeric_tiers.clone());
                    entry.insert("categorical_tiers".into(), p.categorical_tiers.clone());
                    entry.insert("central_rows".into(), json!(p.central_rows));
                }
            }
            entry
        })
        .collect()
}

/// The dossier as an agent payload: clusters (largest first, capped), the
/// graph metrics and the evidence context, bounded unless `detail` is `full`.
pub fn build_evidence_payload(
    dossier: &TopologicalDossier,
    cluster_meta: &Value,
    detail: &str,
    max_clusters: Option<usize>,
    feature_preview_limit: usize,
) -> Value {
    let mut clusters = cluster_payloads_from_dossier(dossier, detail, feature_preview_limit);
    clusters.sort_by_key(|c| std::cmp::Reverse(c["size"].as_u64().unwrap_or(0)));
    let mut omitted = 0;
    if let Some(max) = max_clusters {
        if clusters.len() > max {
            omitted = clusters.len() - max;
            clusters.truncate(max);
        }
    }
    let stats = &dossier.global_stats;
    let mut graph_metrics: Map<String, Value> = stats["graph_metrics"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    if detail != "full" {
        if let Some(sizes) = graph_metrics.shift_remove("component_sizes") {
            let sizes: Vec<usize> = sizes
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|s| s.as_u64().map(|v| v as usize))
                .collect();
            graph_metrics.insert("component_sizes_summary".into(), size_summary(&sizes, 20));
        }
    }
    let mut payload = Map::new();
    payload.insert("status".into(), json!("ok"));
    payload.insert(
        "cluster_result".into(),
        compact_cluster_result_payload(cluster_meta, detail == "full"),
    );
    payload.insert(
        "detail".into(),
        stats.get("detail").cloned().unwrap_or(json!("standard")),
    );
    payload.insert("graph_metrics".into(), Value::Object(graph_metrics));
    payload.insert("clusters_returned".into(), json!(clusters.len()));
    payload.insert("clusters_omitted".into(), json!(omitted));
    payload.insert(
        "clusters".into(),
        Value::Array(clusters.into_iter().map(Value::Object).collect()),
    );
    let head = |key: &str, n: usize| -> Value {
        Value::Array(
            stats[key]
                .as_array()
                .into_iter()
                .flatten()
                .take(n)
                .cloned()
                .collect(),
        )
    };
    if detail != "full" {
        payload.insert(
            "evidence_metadata_summary".into(),
            evidence_metadata_summary(&stats["evidence_metadata"]),
        );
        payload.insert(
            "numeric_global_ranking_preview".into(),
            head("numeric_global_ranking", 10),
        );
        payload.insert(
            "categorical_global_ranking_preview".into(),
            head("categorical_global_ranking", 10),
        );
        payload.insert(
            "signal_matrix_summary".into(),
            signal_matrix_summary(&stats["signal_matrix"]),
        );
    } else {
        payload.insert(
            "evidence_metadata".into(),
            stats["evidence_metadata"].clone(),
        );
        payload.insert(
            "numeric_global_ranking".into(),
            stats["numeric_global_ranking"].clone(),
        );
        payload.insert(
            "categorical_global_ranking".into(),
            stats["categorical_global_ranking"].clone(),
        );
        payload.insert("signal_matrix".into(), stats["signal_matrix"].clone());
    }
    Value::Object(payload)
}

/// The summary payload built straight from the evidence index (no dossier), so
/// the first look at a clustering stays cheap.
pub fn build_summary_evidence_payload(
    index: &FeatureEvidenceIndex,
    cluster_meta: &Value,
    graph_metrics: &Value,
    max_clusters: Option<usize>,
    feature_preview_limit: usize,
    global_preview_limit: usize,
) -> Value {
    let mut bundles: Vec<_> = index.cluster_bundles.values().collect();
    bundles.sort_by_key(|b| std::cmp::Reverse(b.size));
    let mut omitted = 0;
    if let Some(max) = max_clusters {
        if bundles.len() > max {
            omitted = bundles.len() - max;
            bundles.truncate(max);
        }
    }
    let clusters: Vec<Value> = bundles
        .iter()
        .map(|b| {
            let (numeric, categorical) = (to_values(&b.numeric), to_values(&b.categorical));
            let total = numeric
                .iter()
                .chain(&categorical)
                .filter(|r| {
                    matches!(
                        r["signal_tier"].as_str(),
                        Some("core" | "supporting" | "context")
                    )
                })
                .count();
            let mut entry = Map::new();
            entry.insert("cluster_id".into(), json!(b.cluster_id));
            entry.insert("size".into(), json!(b.size));
            entry.insert("size_pct".into(), json!(b.size_pct));
            entry.insert("semantic_name".into(), json!(b.semantic_name));
            entry.insert(
                "topological_neighbors".into(),
                json!(b.topological_neighbors),
            );
            entry.insert(
                "numeric_tier_counts".into(),
                tier_counts_from_rows(&numeric),
            );
            entry.insert(
                "categorical_tier_counts".into(),
                tier_counts_from_rows(&categorical),
            );
            entry.extend(feature_preview(
                &numeric,
                &categorical,
                feature_preview_limit,
                total,
            ));
            Value::Object(entry)
        })
        .collect();
    let head = |items: &[Value]| -> Value {
        Value::Array(items.iter().take(global_preview_limit).cloned().collect())
    };
    let numeric_ranking: Vec<Value> = index
        .numeric_global_ranking
        .iter()
        .map(|c| json!(c))
        .collect();
    json!({
        "status": "ok",
        "cluster_result": compact_cluster_result_payload(cluster_meta, false),
        "detail": "summary",
        "graph_metrics": graph_metrics,
        "clusters_returned": clusters.len(),
        "clusters_omitted": omitted,
        "clusters": clusters,
        "evidence_metadata_summary": evidence_metadata_summary(&index.metadata),
        "numeric_global_ranking_preview": head(&numeric_ranking),
        "categorical_global_ranking_preview": head(&index.categorical_global_ranking),
        "signal_matrix_summary": signal_matrix_summary(&index.signal_matrix),
    })
}

// ---- Markdown ------------------------------------------------------------------------------

fn format_tier_counts(counts: &Value) -> String {
    TIERS
        .iter()
        .map(|t| format!("{t}={}", counts.get(t).and_then(Value::as_i64).unwrap_or(0)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn score_text(row: &Value) -> String {
    match row.get("aggregate_score") {
        Some(v) if !v.is_null() => format!(", score={}", fixed(num(v), 3)),
        _ => String::new(),
    }
}

fn format_feature_preview(row: &Value) -> String {
    let score = score_text(row);
    if row["kind"] == "numeric" {
        format!(
            "{}: {} ({}{score})",
            text(&row["column"]),
            text_or(row, "direction", "mixed"),
            text_or(row, "signal_tier", "unknown")
        )
    } else {
        format!(
            "{}={}: {} signal{score}",
            text(&row["column"]),
            text(&row["value"]),
            text_or(row, "signal_tier", "unknown")
        )
    }
}

fn format_numeric_signal(row: &Value) -> String {
    let z = match row.get("z_score") {
        Some(v) if !v.is_null() => format!(", z={}", fixed(num(v), 2)),
        _ => String::new(),
    };
    format!(
        "{}: {} ({}{z}, score={})",
        text(&row["column"]),
        text_or(row, "direction", "mixed"),
        text_or(row, "signal_tier", "unknown"),
        fixed(row.get("aggregate_score").map_or(0.0, num), 3)
    )
}

fn format_categorical_signal(row: &Value) -> String {
    let prevalence = row
        .get("prevalence_cluster")
        .or_else(|| row.get("in_cluster_prevalence"));
    let prevalence_text = match prevalence {
        Some(v) if !v.is_null() => format!(", prevalence={}%", fixed(num(v), 1)),
        _ => String::new(),
    };
    format!(
        "{}={}: {} (lift={}{prevalence_text}, score={})",
        text(&row["column"]),
        text(&row["value"]),
        text_or(row, "signal_tier", "unknown"),
        fixed(row.get("lift").map_or(0.0, num), 2),
        fixed(row.get("aggregate_score").map_or(0.0, num), 3)
    )
}

/// Markdown of a summary evidence payload.
pub fn summary_evidence_payload_to_markdown(payload: &Value) -> String {
    let result = &payload["cluster_result"];
    let readiness = &result["interpretation_readiness"];
    let mut lines = vec![
        "# Cluster Dossier Summary".to_owned(),
        String::new(),
        format!(
            "- Method: {}",
            result["method_used"].as_str().unwrap_or("unknown")
        ),
        format!("- Clusters: {}", result["n_clusters"].as_u64().unwrap_or(0)),
        format!(
            "- Returned: {}",
            payload["clusters_returned"].as_u64().unwrap_or(0)
        ),
        format!(
            "- Omitted: {}",
            payload["clusters_omitted"].as_u64().unwrap_or(0)
        ),
    ];
    if readiness.as_object().is_some_and(|m| !m.is_empty()) {
        lines.push(format!(
            "- Readiness: {}",
            readiness["status"].as_str().unwrap_or("unknown")
        ));
        let codes: Vec<&str> = readiness["reason_codes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if !codes.is_empty() {
            lines.push(format!("- Readiness reasons: {}", codes.join(", ")));
        }
        if let Some(message) = readiness["message"].as_str().filter(|m| !m.is_empty()) {
            lines.push(format!("- Guidance: {message}"));
        }
    }
    let signal = &payload["signal_matrix_summary"];
    lines.extend([
        String::new(),
        "## Signal Inventory".into(),
        String::new(),
        format!(
            "- Numeric signal columns: {}",
            signal["n_numeric_columns"].as_u64().unwrap_or(0)
        ),
        format!(
            "- Categorical signal values: {}",
            signal["n_categorical_values"].as_u64().unwrap_or(0)
        ),
    ]);
    let gated = payload["evidence_metadata_summary"]["categorical_columns_gated"]
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
        lines.push(format!(
            "- Categorical columns gated as high-cardinality noise: {}",
            described.join(", ")
        ));
    }
    lines.extend([String::new(), "## Clusters".into(), String::new()]);
    for cluster in payload["clusters"].as_array().into_iter().flatten() {
        lines.push(format!(
            "### Cluster {}: {}",
            text(&cluster["cluster_id"]),
            text(&cluster["semantic_name"])
        ));
        lines.push(format!(
            "- Size: {} ({}%)",
            text(&cluster["size"]),
            fixed(num(&cluster["size_pct"]), 1)
        ));
        lines.push(format!(
            "- Numeric tiers: {}",
            format_tier_counts(&cluster["numeric_tier_counts"])
        ));
        lines.push(format!(
            "- Categorical tiers: {}",
            format_tier_counts(&cluster["categorical_tier_counts"])
        ));
        let preview = cluster["feature_preview"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if preview.is_empty() {
            lines.push("- Feature preview: none in selected tiers".into());
        } else {
            lines.push("- Feature preview (top capped signals):".into());
            lines.extend(
                preview
                    .iter()
                    .map(|row| format!("  - {}", format_feature_preview(row))),
            );
        }
        lines.push(format!(
            "- Feature rows omitted: {}",
            cluster["features_omitted"].as_u64().unwrap_or(0)
        ));
        lines.push(String::new());
    }
    lines.extend([
        "## Next Tools".into(),
        String::new(),
        "- `get_cluster_profile` for one cluster's capped evidence rows.".into(),
        "- `get_feature_signal` for selected feature columns across clusters.".into(),
        "- `get_cluster_signal_matrix` for a compact cross-cluster matrix.".into(),
        "- `export_html_report` for a human-facing full report outside agent context.".into(),
    ]);
    lines.join("\n").trim().to_owned()
}

/// Markdown of a `{cluster, detail}` profile payload.
pub fn cluster_profile_payload_to_markdown(payload: &Value) -> String {
    let cluster = &payload["cluster"];
    let neighbors: Vec<String> = cluster["topological_neighbors"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|n| text(&n["cluster_id"]))
        .collect();
    let neighbor_text = if neighbors.is_empty() {
        "None".to_owned()
    } else {
        neighbors.join(", ")
    };
    let mut lines = vec![
        "# Cluster Profile".to_owned(),
        String::new(),
        format!(
            "## Cluster {}: {}",
            text(&cluster["cluster_id"]),
            text(&cluster["semantic_name"])
        ),
        format!(
            "- Size: {} ({}%)",
            text(&cluster["size"]),
            fixed(num(&cluster["size_pct"]), 1)
        ),
        format!("- Topological neighbors: {neighbor_text}"),
        format!(
            "- Detail: {}",
            payload["detail"].as_str().unwrap_or("standard")
        ),
        format!(
            "- Feature rows returned: {}",
            text(&cluster["features_returned"]["total"])
        ),
        format!(
            "- Feature rows omitted: {}",
            text(&cluster["features_omitted"]["total"])
        ),
        String::new(),
        "### Numeric signals".into(),
    ];
    let numeric = cluster["numeric_features"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if numeric.is_empty() {
        lines.push("- None in selected tiers.".into());
    }
    lines.extend(
        numeric
            .iter()
            .map(|r| format!("- {}", format_numeric_signal(r))),
    );
    lines.extend([String::new(), "### Categorical signals".into()]);
    let categorical = cluster["categorical_features"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if categorical.is_empty() {
        lines.push("- None in selected tiers.".into());
    }
    lines.extend(
        categorical
            .iter()
            .map(|r| format!("- {}", format_categorical_signal(r))),
    );
    lines.extend([
        String::new(),
        "### Next Tools".into(),
        "- Use `get_feature_signal` to compare a listed feature across clusters.".into(),
        "- Use `compare_clusters` for pairwise statistical comparison.".into(),
    ]);
    lines.join("\n").trim().to_owned()
}

/// Markdown of a feature-signal payload.
pub fn feature_signal_payload_to_markdown(signals: &Value) -> String {
    let names: Vec<&str> = signals["feature_names"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut lines = vec![
        "# Feature Signal Summary".to_owned(),
        String::new(),
        format!(
            "- Features: {}",
            if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(", ")
            }
        ),
        format!(
            "- Detail: {}",
            signals["detail"].as_str().unwrap_or("summary")
        ),
        format!(
            "- Clusters returned: {}",
            signals["clusters_returned"].as_u64().unwrap_or(0)
        ),
        format!(
            "- Clusters omitted: {}",
            signals["clusters_omitted"].as_u64().unwrap_or(0)
        ),
        String::new(),
    ];
    if signals["clusters_omitted"].as_u64().unwrap_or(0) > 0 {
        lines.extend([
            "> More clusters had matching signals. Pass explicit `cluster_ids` or raise `max_clusters` only if you need the tail.".to_owned(),
            String::new(),
        ]);
    }
    for cluster in signals["clusters"].as_array().into_iter().flatten() {
        lines.push(format!(
            "## Cluster {}: {}",
            text(&cluster["cluster_id"]),
            text(&cluster["semantic_name"])
        ));
        lines.push(format!("- Size: {}", text(&cluster["cluster_size"])));
        let numeric = cluster["numeric_features"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let categorical = cluster["categorical_features"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !numeric.is_empty() {
            lines.push("- Numeric:".into());
            lines.extend(
                numeric
                    .iter()
                    .map(|r| format!("  - {}", format_numeric_signal(r))),
            );
        }
        if !categorical.is_empty() {
            lines.push("- Categorical:".into());
            lines.extend(
                categorical
                    .iter()
                    .map(|r| format!("  - {}", format_categorical_signal(r))),
            );
        }
        if numeric.is_empty() && categorical.is_empty() {
            lines.push("- No selected feature signals.".into());
        }
        lines.push(String::new());
    }
    lines.join("\n").trim().to_owned()
}

// ---- threshold surface ------------------------------------------------------------------------

/// Which graph surface the clustering was computed on, relative to the
/// persisted graph: the two thresholds are separate levers.
pub fn threshold_surface_payload(
    construction_threshold: f64,
    interpretation_threshold: f64,
    threshold_inherited: bool,
    threshold_source: &str,
) -> Value {
    // `np.isclose(interpretation, construction)`.
    let close = (interpretation_threshold - construction_threshold).abs()
        <= 1e-8 + 1e-5 * construction_threshold.abs();
    let mismatch = "Cluster labels may not match topological neighbor and centrality evidence from the persisted graph.";
    let (status, guidance) = if threshold_source == "spectral_default_full_affinity" {
        (
            "full_affinity_spectral",
            "Spectral clustering used the full retained weighted affinity matrix. The persisted diagnostic graph was not rebuilt.".to_owned(),
        )
    } else if threshold_inherited {
        (
            "matched",
            "Interpretation inherited the constructed graph surface.".to_owned(),
        )
    } else if close {
        (
            "matched_explicit",
            "Interpretation explicitly matches the constructed graph surface.".to_owned(),
        )
    } else if interpretation_threshold < construction_threshold {
        ("looser_than_construction", format!("Interpretation uses a looser slice of the retained weighted matrix than the persisted diagnostic graph. {mismatch}"))
    } else {
        ("stricter_than_construction", format!("Interpretation uses a stricter slice of the retained weighted matrix than the persisted diagnostic graph. {mismatch}"))
    };
    json!({
        "status": status,
        "construction_threshold": construction_threshold,
        "interpretation_edge_weight_threshold": interpretation_threshold,
        "threshold_inherited": threshold_inherited,
        "threshold_source": threshold_source,
        "guidance": guidance,
    })
}

/// A Markdown header describing the threshold surface, before `markdown`.
pub fn prepend_threshold_markdown(markdown: &str, surface: &Value) -> String {
    let header = [
        "# Threshold Surface".to_owned(),
        String::new(),
        format!(
            "- Construction threshold: {}",
            fixed(num(&surface["construction_threshold"]), 4)
        ),
        format!(
            "- Interpretation edge weight threshold: {}",
            fixed(num(&surface["interpretation_edge_weight_threshold"]), 4)
        ),
        format!("- Status: {}", text(&surface["status"])),
        format!("- Guidance: {}", text(&surface["guidance"])),
        String::new(),
    ]
    .join("\n");
    format!("{header}\n{markdown}")
}

/// The short "two levers" clarification prepended to dossier Markdown.
pub fn two_lever_markdown_note(
    construction_threshold: f64,
    interpretation_threshold: f64,
) -> String {
    [
        "## Two Levers (do not conflate)".to_owned(),
        String::new(),
        format!("- Graph-level stats (components, density, singletons) describe the persisted cosmic graph @ construction_threshold ({:.4}).", construction_threshold),
        format!("- Cluster tables and per-cluster evidence describe the interpretation slice @ interpretation_edge_weight_threshold ({:.4}) of the [0,1]-normalized weighted adjacency (sparsified only when enabled) \u{2014} not a dense affinity matrix.", interpretation_threshold),
        String::new(),
    ]
    .join("\n")
}
