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

use super::dossier_render;

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
    cap_lists(map, cap, &mut trimmed);
    cap_tiers(map, cap, &mut trimmed);
    count_preview_cut(map, &trimmed);
    if !trimmed.is_empty() {
        map.insert("sections_trimmed".into(), Value::Object(trimmed));
    }
}

/// Keep the first `keep` of `items`, recording a cut under `name`.
fn cut(items: &mut Vec<Value>, keep: usize, name: String, trimmed: &mut Map<String, Value>) {
    if items.len() > keep {
        trimmed.insert(name, json!({"shown": keep, "total": items.len()}));
        items.truncate(keep);
    }
}

fn cap_lists(map: &mut Map<String, Value>, cap: usize, trimmed: &mut Map<String, Value>) {
    for key in LIST_SECTIONS {
        let keep = if key == "central_rows" && cap < 3 {
            0
        } else {
            cap
        };
        if let Some(Value::Array(items)) = map.get_mut(key) {
            cut(items, keep, key.into(), trimmed);
        }
    }
}

fn cap_tiers(map: &mut Map<String, Value>, cap: usize, trimmed: &mut Map<String, Value>) {
    for key in ["numeric_tiers", "categorical_tiers"] {
        let Some(Value::Object(tiers)) = map.get_mut(key) else {
            continue;
        };
        for (tier, rows) in tiers.iter_mut() {
            if let Value::Array(items) = rows {
                cut(items, cap, format!("{key}.{tier}"), trimmed);
            }
        }
    }
}

/// A cut preview moves its rows into `features_omitted`.
fn count_preview_cut(map: &mut Map<String, Value>, trimmed: &Map<String, Value>) {
    let (Some(Value::Array(preview)), Some(t)) =
        (map.get("feature_preview"), trimmed.get("feature_preview"))
    else {
        return;
    };
    let n = preview.len();
    let cut = t["total"].as_u64().unwrap_or(0) as usize - n;
    let omitted = map["features_omitted"].as_u64().unwrap_or(0) as usize + cut;
    map.insert("features_previewed".into(), json!(n));
    map.insert("features_omitted".into(), json!(omitted));
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

/// Section list caps tried in turn before sections are dropped.
const SECTION_CAPS: [usize; 3] = [5, 3, 1];

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
///
/// The pipeline: rank the sections, measure the dossier as rendered, then
/// shrink sections, drop clusters from the end, and finally keep only the
/// header, the index and the truncation.
pub fn fit_dossier(
    mut payload: Value,
    text: Option<String>,
    meta: &Value,
    evidence: &FeatureEvidenceIndex,
    budget: usize,
) -> CallToolResult {
    let order = size_order_of_meta(meta);
    let mut sections = rank_sections(&mut payload, &order);
    let labels = labels_of(evidence);
    // The index is in both views; it gets an eighth of the budget per view.
    payload["cluster_index"] = cluster_index(&order, &labels, budget / 8);
    let fit = Fit {
        order: &order,
        assignment_id: payload["cluster_assignment_id"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        markdown: text.is_some(),
        budget,
    };

    // 1. As rendered, with the index.
    if let Some(r) =
        fit.within(fit.render(&payload, &sections, sections.len(), false, text.as_deref()))
    {
        return r;
    }
    // 2. Bulk keys trimmed, then shorter sections, then fewer of them.
    trim_bulk(&mut payload, budget);
    if let Some(r) = fit.shrink_sections(&payload, &mut sections) {
        return r;
    }
    if let Some(r) = fit.drop_clusters(&payload, &sections) {
        return r;
    }
    // 3. Header and index only, the index grouped harder.
    payload["cluster_index"] = cluster_index(&order, &labels, budget / 16);
    fit.header_only(&payload, &sections)
}

/// The rendered sections in canonical order: the longest prefix of it the
/// dossier rendered, so the next cursor is exact.
fn rank_sections(payload: &mut Value, order: &[(i64, usize)]) -> Vec<Value> {
    let mut by_id: BTreeMap<i64, Value> = payload["clusters"]
        .as_array_mut()
        .map(std::mem::take)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|c| Some((c["cluster_id"].as_i64()?, c)))
        .collect();
    order.iter().map_while(|(id, _)| by_id.remove(id)).collect()
}

/// Trim the dossier's bulk keys to a fortieth of the budget each.
fn trim_bulk(payload: &mut Value, budget: usize) {
    for key in DOSSIER_BULK {
        if let Some(v) = payload.get_mut(key) {
            budget::trim_value(v, budget / 40);
        }
    }
}

/// One dossier being fitted: what every rendering of it shares.
struct Fit<'a> {
    order: &'a [(i64, usize)],
    assignment_id: String,
    markdown: bool,
    budget: usize,
}

impl Fit<'_> {
    /// The dossier with its first `n` sections, its truncation attached,
    /// and its text view (`original` kept when given).
    fn render(
        &self,
        payload: &Value,
        sections: &[Value],
        n: usize,
        trimmed: bool,
        original: Option<&str>,
    ) -> (Value, String) {
        let total = self.order.len();
        let mut p = payload.clone();
        p["clusters"] = json!(sections[..n]);
        p["clusters_returned"] = json!(n);
        p["clusters_omitted"] = json!(total - n);
        let t = self.truncation(n, trimmed);
        let line = truncation_line(&t);
        p["truncation"] = t;
        let text = if self.markdown {
            dossier_render::dossier_markdown(&p, &line, original)
        } else {
            serde_json::to_string(&p).unwrap_or_default()
        };
        (p, text)
    }

    /// The `truncation` of a rendering that keeps `n` sections.
    fn truncation(&self, n: usize, trimmed: bool) -> Value {
        let omitted: Vec<i64> = self.order[n..].iter().map(|(id, _)| *id).collect();
        let next = if n < self.order.len() {
            Some(next_page(&self.assignment_id, n, &Map::new()))
        } else if trimmed && n > 0 {
            let ids: Vec<i64> = self.order[..n.min(8)].iter().map(|(id, _)| *id).collect();
            Some(next_ids(&self.assignment_id, &ids, &Map::new()))
        } else {
            None
        };
        truncation(n, self.order.len(), &omitted, trimmed, self.budget, next)
    }

    /// The result, if the rendering fits the budget.
    fn within(&self, (p, t): (Value, String)) -> Option<CallToolResult> {
        (pair_chars(&p, &t) <= self.budget).then(|| ok_result(p, Some(t)))
    }

    /// Cap every section harder in turn; the first rendering that fits.
    fn shrink_sections(&self, payload: &Value, sections: &mut [Value]) -> Option<CallToolResult> {
        for cap in SECTION_CAPS {
            for s in sections.iter_mut() {
                cap_sections(s, cap);
            }
            let fitted = self.within(self.render(payload, sections, sections.len(), true, None));
            if fitted.is_some() {
                return fitted;
            }
        }
        None
    }

    /// The most sections that fit, if any count does.
    fn drop_clusters(&self, payload: &Value, sections: &[Value]) -> Option<CallToolResult> {
        let (mut lo, mut hi) = (0, sections.len());
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let (p, t) = self.render(payload, sections, mid, true, None);
            if pair_chars(&p, &t) <= self.budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        self.within(self.render(payload, sections, lo, true, None))
    }

    /// Header, index and truncation only; whatever else is left is trimmed
    /// (the middleware is the last backstop).
    fn header_only(&self, payload: &Value, sections: &[Value]) -> CallToolResult {
        let (mut p, _) = self.render(payload, sections, 0, true, None);
        let keep = self.budget / 2 - 200;
        if budget::structured_chars(&p) > keep {
            trim_all_but_sacred(&mut p, self.budget);
        }
        let (mut p, mut t) = self.render(&p, sections, 0, true, None);
        if pair_chars(&p, &t) > self.budget {
            t = self.last_resort(&mut p);
        }
        ok_result(p, Some(t))
    }

    /// Only the truncation is sacred now; the middleware stays the backstop
    /// for anything still over. Returns the text view.
    fn last_resort(&self, p: &mut Value) -> String {
        let truncation = p["truncation"].clone();
        budget::trim_value(p, self.budget / 3);
        p["truncation"] = truncation;
        if self.markdown {
            format!(
                "{}\n\n{}",
                dossier_render::dossier_header(p),
                truncation_line(&p["truncation"])
            )
        } else {
            serde_json::to_string(p).unwrap_or_default()
        }
    }
}

/// Trim every key of a rendered dossier but the truncation, the index and
/// the Phirewall blocks.
fn trim_all_but_sacred(p: &mut Value, budget: usize) {
    for (k, v) in p.as_object_mut().into_iter().flatten() {
        if !matches!(
            k.as_str(),
            "truncation" | "cluster_index" | release::KEY | phirewall::KEY
        ) {
            budget::trim_value(v, budget / 40);
        }
    }
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
