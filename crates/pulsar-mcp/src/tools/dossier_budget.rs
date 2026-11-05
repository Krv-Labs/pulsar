//! Fitting cluster results to the response budget, shared by
//! `generate_cluster_dossier` and `get_cluster_detail`.
//!
//! Clusters are ranked by size (largest first, ties by id): the canonical
//! order both tools page through, so a dossier's `next` cursor continues
//! exactly where its sections stop. A fitted result always keeps its header
//! and an index of every cluster; per-cluster sections shrink (shorter
//! lists, then no central rows) before whole sections are dropped from the
//! end. `truncation` says what was shown and gives the call that continues.

use std::collections::BTreeMap;

use pulsar_analysis::disclosure::label;
use pulsar_analysis::feature_evidence::FeatureEvidenceIndex;
use pulsar_store::ClusterAssignmentId;
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use crate::budget::{self, json_chars, pair_chars};
use crate::envelope::ok_result;
use crate::phirewall;
use crate::release;
use crate::state::ServerState;

/// The drill-down tool every `next` names.
pub const DETAIL_TOOL: &str = "get_cluster_detail";

/// Per-cluster list sections, at the same paths in both tools.
pub const LIST_SECTIONS: [&str; 5] = [
    "feature_preview",
    "numeric_features",
    "categorical_features",
    "topological_neighbors",
    "central_rows",
];

/// Ids of `omitted_cluster_ids` listed in a `truncation` (the rest are
/// reached through `next`).
const OMITTED_LISTED: usize = 50;

/// `(cluster_id, size)` largest first, ties by id.
pub fn size_order_of_labels(labels: &[i64]) -> Vec<(i64, usize)> {
    let mut order: Vec<(i64, usize)> = pulsar_analysis::clustering::merge::label_sizes(labels)
        .into_iter()
        .collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    order
}

/// The canonical order from a `cluster_result_payload`'s `cluster_sizes`.
pub fn size_order_of_meta(meta: &Value) -> Vec<(i64, usize)> {
    let mut order: Vec<(i64, usize)> = meta["cluster_sizes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| Some((c["cluster_id"].as_i64()?, c["n"].as_u64()? as usize)))
        .collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    order
}

/// One-line labels of the characterized clusters.
pub fn labels_of(evidence: &FeatureEvidenceIndex) -> BTreeMap<i64, String> {
    evidence
        .cluster_bundles
        .iter()
        .map(|(id, b)| (*id, label(&b.semantic_name).0))
        .collect()
}

/// Sorted ids as compact ranges: `0-3, 7, 9-12`.
pub fn ranges(ids: &mut [i64]) -> String {
    ids.sort_unstable();
    let mut out: Vec<String> = vec![];
    let mut i = 0;
    while i < ids.len() {
        let start = ids[i];
        let mut end = start;
        while i + 1 < ids.len() && ids[i + 1] == end + 1 {
            i += 1;
            end = ids[i];
        }
        out.push(if start == end {
            start.to_string()
        } else {
            format!("{start}-{end}")
        });
        i += 1;
    }
    out.join(", ")
}

/// An index of every cluster: `{cluster_id, size, label}` in canonical
/// order; when that would pass `max_chars`, the largest are listed and the
/// rest grouped by size under `grouped`, their ids as ranges.
pub fn cluster_index(
    order: &[(i64, usize)],
    labels: &BTreeMap<i64, String>,
    max_chars: usize,
) -> Value {
    let entry = |&(id, size): &(i64, usize)| {
        let mut e = Map::new();
        e.insert("cluster_id".into(), json!(id));
        e.insert("size".into(), json!(size));
        if let Some(l) = labels.get(&id) {
            e.insert("label".into(), json!(l));
        }
        Value::Object(e)
    };
    let all: Vec<Value> = order.iter().map(entry).collect();
    let full = json!({"total": order.len(), "clusters": all});
    if json_chars(&full) <= max_chars {
        return full;
    }
    // The largest clusters listed while they take half the room; the rest
    // grouped by size.
    let mut listed: Vec<Value> = vec![];
    let mut used = 0;
    for item in order {
        let e = entry(item);
        used += json_chars(&e) + 1;
        if used > max_chars / 2 {
            break;
        }
        listed.push(e);
    }
    let mut by_size: BTreeMap<std::cmp::Reverse<usize>, Vec<i64>> = BTreeMap::new();
    for (id, size) in &order[listed.len()..] {
        by_size
            .entry(std::cmp::Reverse(*size))
            .or_default()
            .push(*id);
    }
    let range_chars = (max_chars / 2 / by_size.len().max(1)).max(60);
    let grouped: Vec<Value> = by_size
        .into_iter()
        .map(|(size, mut ids)| {
            let count = ids.len();
            let text = ranges(&mut ids);
            let shown = budget::cut_text(&text, range_chars);
            let ids_text = if shown.len() < text.len() {
                format!("{shown}…")
            } else {
                text
            };
            json!({"size": size.0, "count": count, "cluster_ids": ids_text})
        })
        .collect();
    json!({
        "total": order.len(),
        "clusters": listed,
        "grouped": grouped,
        "note": "Smaller clusters are grouped by size; get_cluster_detail pages through every cluster.",
    })
}

/// Cap every list section of a cluster entry at `cap` (central rows go
/// below 3), recording what was cut under `sections_trimmed`.
pub fn cap_sections(entry: &mut Value, cap: usize) {
    let Value::Object(map) = entry else {
        return;
    };
    let mut trimmed = Map::new();
    for key in LIST_SECTIONS {
        let drop_all = key == "central_rows" && cap < 3;
        if let Some(Value::Array(items)) = map.get_mut(key) {
            let keep = if drop_all { 0 } else { cap };
            if items.len() > keep {
                trimmed.insert(key.into(), json!({"shown": keep, "total": items.len()}));
                items.truncate(keep);
            }
        }
    }
    for key in ["numeric_tiers", "categorical_tiers"] {
        if let Some(Value::Object(tiers)) = map.get_mut(key) {
            for (tier, rows) in tiers.iter_mut() {
                if let Value::Array(items) = rows {
                    if items.len() > cap {
                        trimmed.insert(
                            format!("{key}.{tier}"),
                            json!({"shown": cap, "total": items.len()}),
                        );
                        items.truncate(cap);
                    }
                }
            }
        }
    }
    if let Some(Value::Array(preview)) = map.get("feature_preview") {
        let n = preview.len();
        if let Some(t) = trimmed.get("feature_preview") {
            let cut = t["total"].as_u64().unwrap_or(0) as usize - n;
            let omitted = map["features_omitted"].as_u64().unwrap_or(0) as usize + cut;
            map.insert("features_previewed".into(), json!(n));
            map.insert("features_omitted".into(), json!(omitted));
        }
    }
    if !trimmed.is_empty() {
        map.insert("sections_trimmed".into(), Value::Object(trimmed));
    }
}

/// The `get_cluster_detail` call that continues at `cursor` (an offset in
/// the canonical order), as a call string and as `{tool, arguments}`.
pub fn next_page(
    assignment_id: &str,
    cursor: usize,
    extra: &Map<String, Value>,
) -> (String, Value) {
    let mut arguments = Map::new();
    arguments.insert("cluster_assignment_id".into(), json!(assignment_id));
    arguments.insert("cursor".into(), json!(cursor.to_string()));
    arguments.extend(extra.clone());
    call_pair(arguments)
}

/// The `get_cluster_detail` call for exactly `ids`.
pub fn next_ids(assignment_id: &str, ids: &[i64], extra: &Map<String, Value>) -> (String, Value) {
    let mut arguments = Map::new();
    arguments.insert("cluster_assignment_id".into(), json!(assignment_id));
    arguments.insert("cluster_ids".into(), json!(ids));
    arguments.extend(extra.clone());
    call_pair(arguments)
}

fn call_pair(arguments: Map<String, Value>) -> (String, Value) {
    let args: Vec<String> = arguments.iter().map(|(k, v)| format!("{k}={v}")).collect();
    (
        format!("{DETAIL_TOOL}({})", args.join(", ")),
        json!({"tool": DETAIL_TOOL, "arguments": arguments}),
    )
}

/// The `truncation` object.
pub fn truncation(
    shown: usize,
    total: usize,
    omitted: &[i64],
    sections_trimmed: bool,
    budget: usize,
    next: Option<(String, Value)>,
) -> Value {
    let (next, next_call) = match next {
        Some((n, c)) => (json!(n), c),
        None => (Value::Null, Value::Null),
    };
    json!({
        "truncated": shown < total || sections_trimmed,
        "shown": shown,
        "total": total,
        "omitted_cluster_ids": omitted.iter().take(OMITTED_LISTED).collect::<Vec<_>>(),
        "omitted_count": omitted.len(),
        "sections_trimmed": sections_trimmed,
        "budget_chars": budget,
        "next": next,
        "next_call": next_call,
    })
}

/// The one plain line mirroring `truncation` in the text view.
pub fn truncation_line(t: &Value) -> String {
    let shown = t["shown"].as_u64().unwrap_or(0);
    let total = t["total"].as_u64().unwrap_or(0);
    let trimmed = if t["sections_trimmed"] == true {
        ", per-cluster sections shortened"
    } else {
        ""
    };
    match t["next"].as_str() {
        Some(next) => format!(
            "Truncated: {shown} of {total} cluster sections shown{trimmed} (budget {} chars). Next: {next}",
            t["budget_chars"]
        ),
        None => format!("All {total} cluster sections shown."),
    }
}

fn fmt_num(v: &Value, dp: usize) -> String {
    v.as_f64()
        .map_or_else(|| "?".into(), |x| format!("{x:.dp$}"))
}

fn feature_line(row: &Value) -> String {
    let tier = row["signal_tier"].as_str().unwrap_or("?");
    if row.get("value").is_some_and(|v| !v.is_null()) {
        format!(
            "{}={} ({tier}, lift {})",
            row["column"].as_str().unwrap_or("?"),
            row["value"]
                .as_str()
                .map_or_else(|| row["value"].to_string(), str::to_owned),
            fmt_num(&row["lift"], 2)
        )
    } else {
        let z = if row["z_score"].is_number() {
            format!(", z {}", fmt_num(&row["z_score"], 2))
        } else {
            String::new()
        };
        format!(
            "{}: {} ({tier}{z})",
            row["column"].as_str().unwrap_or("?"),
            row["direction"].as_str().unwrap_or("mixed"),
        )
    }
}

/// Markdown of one cluster section.
pub fn section_markdown(c: &Value) -> String {
    let mut lines = vec![
        format!(
            "### Cluster {}: {}",
            c["cluster_id"],
            c["semantic_name"].as_str().unwrap_or("")
        ),
        format!("- Size: {} ({}%)", c["size"], fmt_num(&c["size_pct"], 1)),
    ];
    if let Some(n) = c["topological_neighbors"]
        .as_array()
        .filter(|n| !n.is_empty())
    {
        let list: Vec<String> = n
            .iter()
            .map(|x| {
                format!(
                    "{} ({})",
                    x["cluster_id"],
                    fmt_num(&x["bridge_strength"], 2)
                )
            })
            .collect();
        lines.push(format!("- Neighbours: {}", list.join(", ")));
    }
    for (key, title) in [
        ("feature_preview", "Feature preview"),
        ("numeric_features", "Numeric features"),
        ("categorical_features", "Categorical features"),
    ] {
        if let Some(rows) = c[key].as_array().filter(|r| !r.is_empty()) {
            lines.push(format!("- {title}:"));
            lines.extend(rows.iter().map(|r| format!("  - {}", feature_line(r))));
        }
    }
    if let Some(rows) = c["central_rows"].as_array().filter(|r| !r.is_empty()) {
        lines.push(format!(
            "- Central rows: {} (structured content)",
            rows.len()
        ));
    }
    if let Some(t) = c["sections_trimmed"].as_object() {
        let cut: Vec<String> = t
            .iter()
            .map(|(k, v)| format!("{k} {}/{}", v["shown"], v["total"]))
            .collect();
        lines.push(format!("- Shortened: {}", cut.join(", ")));
    }
    lines.join("\n")
}

/// Markdown of a cluster index.
pub fn index_markdown(index: &Value) -> String {
    let mut lines = vec![
        format!("## Cluster Index ({} clusters)", index["total"]),
        String::new(),
    ];
    for c in index["clusters"].as_array().into_iter().flatten() {
        let label = c["label"]
            .as_str()
            .map(|l| format!(": {l}"))
            .unwrap_or_default();
        lines.push(format!("- {} (n={}){label}", c["cluster_id"], c["size"]));
    }
    for g in index["grouped"].as_array().into_iter().flatten() {
        lines.push(format!(
            "- {} clusters of size {}: {}",
            g["count"],
            g["size"],
            g["cluster_ids"].as_str().unwrap_or("")
        ));
    }
    lines.join("\n")
}

/// The dossier header in Markdown, from the payload.
fn dossier_header(p: &Value) -> String {
    [
        "# Cluster Dossier".to_owned(),
        String::new(),
        format!(
            "- cluster_assignment_id: `{}` (the session's live labels)",
            p["cluster_assignment_id"].as_str().unwrap_or("")
        ),
        format!("- run_id: `{}`", p["run_id"].as_str().unwrap_or("")),
        format!(
            "- Method: {}; clusters: {}; interpretation threshold {} (construction {})",
            p["method_applied"].as_str().unwrap_or("?"),
            p["cluster_count"],
            fmt_num(&p["interpretation_edge_weight_threshold_applied"], 4),
            fmt_num(&p["construction_threshold"], 4),
        ),
        format!(
            "- Readiness: {}",
            p["cluster_result"]["interpretation_readiness"]["status"]
                .as_str()
                .unwrap_or("unknown")
        ),
    ]
    .join("\n")
}

/// Keys of a dossier payload that are bulk, not header: trimmed before any
/// cluster section is dropped.
const DOSSIER_BULK: [&str; 9] = [
    "signal_matrix",
    "numeric_global_ranking",
    "categorical_global_ranking",
    "evidence_metadata",
    "graph_metrics",
    "surface_labels",
    "profiled_features",
    // detail='full' carries every cluster's size here; the index has them.
    "cluster_result",
    "cluster_selection",
];

/// Fit a finished dossier `(payload, text)` whose assignment has been
/// published: the clustering's sizes and labels come from the session (or,
/// if the session moved on, from the store). Any tool that publishes an
/// assignment can return through this.
pub fn fit_published(state: &ServerState, payload: Value, text: Option<String>) -> CallToolResult {
    let budget = budget::render_budget(state.options.max_response_bytes);
    let id = payload["cluster_assignment_id"]
        .as_str()
        .unwrap_or_default();
    let published = state
        .session()
        .clusters
        .clone()
        .filter(|c| c.cluster_assignment_id.as_str() == id);
    match published {
        Some(c) => fit_dossier(payload, text, &c.meta, &c.evidence, budget),
        None => {
            let labels = ClusterAssignmentId::parse(id)
                .and_then(|id| state.store.cluster_labels(&id))
                .unwrap_or_default();
            let meta = json!({"cluster_sizes": size_order_of_labels(&labels)
                .iter()
                .map(|(id, n)| json!({"cluster_id": id, "n": n}))
                .collect::<Vec<_>>()});
            fit_dossier(
                payload,
                text,
                &meta,
                &FeatureEvidenceIndex::default(),
                budget,
            )
        }
    }
}

/// Fit a finished dossier `(payload, text)` to `budget` agent-visible
/// characters. `meta` is the clustering's `cluster_result_payload` (the
/// payload only keeps a bounded size summary); `evidence` labels the
/// index. `text` is the Markdown view, or `None` for JSON.
pub fn fit_dossier(
    mut payload: Value,
    text: Option<String>,
    meta: &Value,
    evidence: &FeatureEvidenceIndex,
    budget: usize,
) -> CallToolResult {
    let order = size_order_of_meta(meta);
    let total = order.len();
    let assignment_id = payload["cluster_assignment_id"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    // Sections in canonical order: the longest prefix of it the dossier
    // rendered, so the next cursor is exact.
    let mut by_id: BTreeMap<i64, Value> = payload["clusters"]
        .as_array_mut()
        .map(std::mem::take)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|c| Some((c["cluster_id"].as_i64()?, c)))
        .collect();
    let sections: Vec<Value> = order.iter().map_while(|(id, _)| by_id.remove(id)).collect();
    let labels = labels_of(evidence);
    // The index is in both views; it gets an eighth of the budget per view.
    payload["cluster_index"] = cluster_index(&order, &labels, budget / 8);

    let render =
        |payload: &Value, sections: &[Value], n: usize, trimmed: bool, original: Option<&str>| {
            let mut p = payload.clone();
            p["clusters"] = json!(sections[..n]);
            p["clusters_returned"] = json!(n);
            p["clusters_omitted"] = json!(total - n);
            let omitted: Vec<i64> = order[n..].iter().map(|(id, _)| *id).collect();
            let next = if n < total {
                Some(next_page(&assignment_id, n, &Map::new()))
            } else if trimmed && n > 0 {
                let ids: Vec<i64> = order[..n.min(8)].iter().map(|(id, _)| *id).collect();
                Some(next_ids(&assignment_id, &ids, &Map::new()))
            } else {
                None
            };
            let t = truncation(n, total, &omitted, trimmed, budget, next);
            let line = truncation_line(&t);
            p["truncation"] = t;
            let text = match (&text, original) {
                (Some(_), Some(original)) => {
                    format!(
                        "{original}\n\n{}\n\n{line}",
                        index_markdown(&p["cluster_index"])
                    )
                }
                (Some(_), None) => {
                    let body: Vec<String> = p["clusters"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(section_markdown)
                        .collect();
                    format!(
                        "{}\n\n{line}\n\n{}\n\n## Clusters\n\n{}",
                        dossier_header(&p),
                        index_markdown(&p["cluster_index"]),
                        body.join("\n\n")
                    )
                }
                (None, _) => serde_json::to_string(&p).unwrap_or_default(),
            };
            (p, text)
        };

    // 1. As rendered, with the index.
    let n_all = sections.len();
    let (p, t) = render(&payload, &sections, n_all, false, text.as_deref());
    if pair_chars(&p, &t) <= budget {
        return ok_result(p, Some(t));
    }
    // 2. Bulk keys trimmed, then shorter sections, then fewer of them.
    for key in DOSSIER_BULK {
        if let Some(v) = payload.get_mut(key) {
            budget::trim_value(v, budget / 40);
        }
    }
    let mut trimmed_sections = sections.clone();
    for cap in [5, 3, 1] {
        for s in &mut trimmed_sections {
            cap_sections(s, cap);
        }
        let (p, t) = render(&payload, &trimmed_sections, n_all, true, None);
        if pair_chars(&p, &t) <= budget {
            return ok_result(p, Some(t));
        }
    }
    let (mut lo, mut hi) = (0, n_all);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        let (p, t) = render(&payload, &trimmed_sections, mid, true, None);
        if pair_chars(&p, &t) <= budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let (p, t) = render(&payload, &trimmed_sections, lo, true, None);
    if pair_chars(&p, &t) <= budget {
        return ok_result(p, Some(t));
    }
    // 3. Header and index only, the index grouped harder; then whatever
    //    else is left is trimmed (the middleware is the last backstop).
    payload["cluster_index"] = cluster_index(&order, &labels, budget / 16);
    let (mut p, _) = render(&payload, &trimmed_sections, 0, true, None);
    let keep = budget / 2 - 200;
    if budget::structured_chars(&p) > keep {
        for (k, v) in p.as_object_mut().into_iter().flatten() {
            if !matches!(
                k.as_str(),
                "truncation" | "cluster_index" | release::KEY | phirewall::KEY
            ) {
                budget::trim_value(v, budget / 40);
            }
        }
    }
    let (mut p, mut t) = render(&p, &trimmed_sections, 0, true, None);
    if pair_chars(&p, &t) > budget {
        // Only the truncation is sacred now; the middleware stays the
        // backstop for anything still over.
        let truncation = p["truncation"].clone();
        budget::trim_value(&mut p, budget / 3);
        p["truncation"] = truncation;
        t = match &text {
            Some(_) => format!(
                "{}\n\n{}",
                dossier_header(&p),
                truncation_line(&p["truncation"])
            ),
            None => serde_json::to_string(&p).unwrap_or_default(),
        };
    }
    ok_result(p, Some(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_compress_runs() {
        assert_eq!(ranges(&mut [3, 1, 2, 7, 9, 10]), "1-3, 7, 9-10");
    }

    #[test]
    fn the_index_groups_unlabelled_clusters_when_large() {
        let order: Vec<(i64, usize)> = (0..5_000)
            .map(|i| (i, if i < 3 { 100 } else { 1 }))
            .collect();
        let labels = BTreeMap::from([(0, "High a".to_owned())]);
        let index = cluster_index(&order, &labels, 2_000);
        assert!(json_chars(&index) < 2_000, "{}", json_chars(&index));
        assert_eq!(index["total"], 5_000);
        let listed = index["clusters"].as_array().unwrap().len();
        assert!(listed >= 3, "{index}");
        assert_eq!(index["clusters"][0]["label"], "High a");
        let grouped: u64 = index["grouped"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["count"].as_u64().unwrap())
            .sum();
        assert_eq!(listed as u64 + grouped, 5_000);
        assert_eq!(index["grouped"][0]["cluster_ids"], format!("{listed}-4999"));
    }

    #[test]
    fn capping_records_what_was_cut() {
        let mut c =
            json!({"cluster_id": 1, "numeric_features": [1, 2, 3, 4], "central_rows": [1, 2, 3]});
        cap_sections(&mut c, 2);
        assert_eq!(c["numeric_features"], json!([1, 2]));
        assert_eq!(c["central_rows"], json!([]));
        assert_eq!(c["sections_trimmed"]["numeric_features"]["total"], 4);
    }
}
