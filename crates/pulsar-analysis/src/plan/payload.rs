//! The structured payloads and Markdown views of `draft_preprocessing_plan`
//! and `edit_preprocessing_plan` (`pulsar/mcp/preprocessing.py`).
//!
//! Summary-first: lists are capped to previews with omitted counts, and the
//! full object (every entry, the YAML) appears only with `Detail::Full`.
//!
//! `preprocessing_yaml_chars` is the length of the YAML text this build
//! renders (`serde_yaml_ng`); PyYAML renders the same plan with different
//! quoting, so that one number can differ from Python's.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

use super::model::{PlanQuestion, QuestionStatus};
use super::plan::PreprocessingPlan;
use super::proxies::{DryRun, ProcessedSpace};
use super::render::{
    build_marker_legend, plan_fingerprint, plan_to_preprocessing_yaml, py_canonical_json,
    py_json_dumps, summarize_plan_counts,
};
use super::text::py_str;
use crate::evidence::{ColumnKind, PreprocessingEvidence};

pub const HOW_TO_USE: &str = "Plan contains measured evidence and labeled defaults, not a verdict. \
Resolve questions you can infer from context via edit_preprocessing_plan(source='agent'); ask the user in their own domain words only when the answer cannot be inferred and stakes are material, then record with edit_preprocessing_plan(source='user'). \
To hold out outcome columns, pass edits=[{'column': '<col>', 'role': 'target'}] to edit_preprocessing_plan (or target_columns=[...] to draft_preprocessing_plan). \
To remove TARGET_PROXY_COLUMNS, set indicator='none' or role='drop' on the source columns in edits. \
For code_columns that are labels rather than amounts, pass edits=[{'column': '<col>', 'encode': 'one_hot'}] or role='drop'.";

pub const EVENT_FLAG_HINT_DESCRIPTION: &str = "complete 0/1 or count columns whose value exactly explains other columns' blanks (often events or exposures; only you can say whether any is an outcome)";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Detail {
    #[default]
    Summary,
    Full,
}

/// How many items the summary shows of each list.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub question_limit: usize,
    pub evidence_limit: usize,
    pub change_limit: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            question_limit: 8,
            evidence_limit: 12,
            change_limit: 20,
        }
    }
}

/// The first `limit` items and how many were left out (`None` = all).
fn cap_list<T: Clone>(items: &[T], limit: Option<usize>) -> (Vec<T>, usize) {
    match limit {
        None => (items.to_vec(), 0),
        Some(n) => {
            let kept: Vec<T> = items.iter().take(n).cloned().collect();
            let omitted = items.len() - kept.len();
            (kept, omitted)
        }
    }
}

/// Insert `<preview_key or key>`, an optional `<count_key>` and
/// `<key>_omitted` for a capped list.
fn capped(
    out: &mut Map<String, Value>,
    key: &str,
    items: Vec<Value>,
    cap: Option<usize>,
    preview_key: Option<&str>,
    count_key: Option<&str>,
) {
    let total = items.len();
    let (kept, omitted) = cap_list(&items, cap);
    out.insert(preview_key.unwrap_or(key).into(), Value::Array(kept));
    if let Some(c) = count_key {
        out.insert(c.into(), json!(total));
    }
    out.insert(format!("{key}_omitted"), json!(omitted));
}

fn capped_simple(out: &mut Map<String, Value>, key: &str, items: Vec<Value>, cap: Option<usize>) {
    capped(out, key, items, cap, None, None);
}

fn plan_header(plan: &PreprocessingPlan) -> Map<String, Value> {
    let mut h = Map::new();
    h.insert("status".into(), json!("ok"));
    h.insert("plan_id".into(), json!(plan.plan_id));
    h.insert("dataset_id".into(), json!(plan.dataset_id));
    h.insert("revision".into(), json!(plan.revision));
    h.insert("plan_fingerprint".into(), json!(plan_fingerprint(plan)));
    h
}

/// One warning per target whose proxy columns overlap it exactly.
fn target_proxy_warnings(plan: &PreprocessingPlan) -> Vec<Value> {
    let mut by_target: Vec<(String, Vec<String>)> = Vec::new();
    for p in plan
        .target_proxies
        .iter()
        .filter(|p| p.share_with_target == 1.0)
    {
        match by_target.iter_mut().find(|(t, _)| *t == p.target) {
            Some((_, cols)) => cols.push(p.column.clone()),
            None => by_target.push((p.target.clone(), vec![p.column.clone()])),
        }
    }
    by_target
        .into_iter()
        .map(|(target, columns)| {
            json!({
                "code": "TARGET_PROXY_COLUMNS",
                "target": target,
                "message": format!("these columns are 1 only for rows where {target} is set; they would let the map encode the outcome"),
                "columns": columns,
            })
        })
        .collect()
}

fn question_item(q: &PlanQuestion, col_preview: Option<usize>) -> Value {
    let (preview, _) = cap_list(&q.applies_to, col_preview);
    json!({
        "question_id": q.question_id,
        "topic": q.topic,
        "text": q.text,
        "applies_to_preview": preview,
        "applies_to_count": q.applies_to.len(),
        "rows_affected": q.rows_affected,
        "options": q.options.iter().map(|o| json!({
            "option_id": o.option_id, "label": o.label, "effect": o.effect,
        })).collect::<Vec<_>>(),
        "default_option_id": q.default_option_id,
        "stakes": q.stakes,
    })
}

fn open_question_items(plan: &PreprocessingPlan, col_preview: Option<usize>) -> Vec<Value> {
    plan.questions
        .values()
        .filter(|q| q.status == QuestionStatus::Open)
        .map(|q| question_item(q, col_preview))
        .collect()
}

fn target_proxy_fields(plan: &PreprocessingPlan, cap: Option<usize>) -> Map<String, Value> {
    let mut out = Map::new();
    let items = plan
        .target_proxies
        .iter()
        .map(|p| serde_json::to_value(p).expect("proxy"))
        .collect();
    capped_simple(&mut out, "target_proxies", items, cap);
    out
}

fn evidence_lists(
    plan: &PreprocessingPlan,
    ev_cap: Option<usize>,
    col_cap: Option<usize>,
) -> Map<String, Value> {
    let ev: &PreprocessingEvidence = &plan.evidence;
    let by_col = |name: &str| ev.columns.iter().find(|c| c.name == name);

    let missing_groups = ev
        .missing_groups
        .iter()
        .map(|g| {
            json!({
                "group_id": g.group_id,
                "columns_preview": cap_list(&g.columns, col_cap).0,
                "columns_count": g.columns.len(),
                "n_missing": g.n_missing,
                "zero_free_columns_preview": cap_list(&g.zero_free_columns, col_cap).0,
            })
        })
        .collect();
    let mask_dependencies = ev
        .mask_dependencies
        .iter()
        .map(|m| {
            json!({
                "column": m.column,
                "partner": m.partner,
                "relation": m.relation,
                "n_missing": by_col(&m.column).map_or(0, |c| c.n_missing),
                "n_other_partners": m.n_other_partners,
            })
        })
        .collect();
    let relations = ev
        .relations
        .iter()
        .map(|r| {
            json!({
                "kind": r.kind, "target": r.target, "sources": r.sources,
                "value": r.value, "rows_used": r.rows_used,
            })
        })
        .collect();

    let mut rare: Vec<(&str, f64, f64)> = ev
        .columns
        .iter()
        .filter_map(|c| Some((c.name.as_str(), c.minority_fraction?, c.max_abs_z?)))
        .collect();
    rare.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(b.0)));
    let rare = rare
        .into_iter()
        .map(|(c, frac, z)| json!({"column": c, "minority_fraction": frac, "max_abs_z": z}))
        .collect();

    let constants: Vec<Value> = ev
        .columns
        .iter()
        .filter(|c| c.is_constant)
        .map(|c| json!(c.name))
        .collect();
    let dates: Vec<Value> = ev
        .columns
        .iter()
        .filter(|c| c.kind == ColumnKind::DatetimeLike)
        .map(|c| json!(c.name))
        .collect();
    let integer_coded = ev
        .columns
        .iter()
        .filter(|c| {
            c.kind == ColumnKind::Numeric
                && c.integer_valued == Some(true)
                && !c.is_constant
                && c.n_unique > 2
        })
        .map(|c| json!({"column": c.name, "n_unique": c.n_unique}))
        .collect();

    let mut most_missing: Vec<(&str, f64, Value)> = ev
        .columns
        .iter()
        .filter(|c| c.n_missing > 0 && plan.entries.contains_key(&c.name))
        .map(|c| {
            let entry = &plan.entries[&c.name];
            let item = json!({
                "column": c.name,
                "missing_fraction": c.missing_fraction,
                "n_missing": c.n_missing,
                "missing_group": c.missing_group,
                "plan_role": entry.role,
                "plan_impute_method": entry.impute_method(),
            });
            (c.name.as_str(), c.missing_fraction, item)
        })
        .collect();
    most_missing.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let most_missing = most_missing.into_iter().map(|(_, _, v)| v).collect();

    let mut out = Map::new();
    capped_simple(&mut out, "missing_groups", missing_groups, ev_cap);
    capped_simple(&mut out, "mask_dependencies", mask_dependencies, ev_cap);
    capped_simple(&mut out, "relations", relations, ev_cap);
    capped_simple(&mut out, "rare_binary_leverage", rare, ev_cap);
    capped(
        &mut out,
        "constant_columns",
        constants,
        ev_cap,
        Some("constant_columns_preview"),
        Some("constant_columns_count"),
    );
    capped(
        &mut out,
        "datetime_columns",
        dates,
        ev_cap,
        Some("datetime_columns_preview"),
        Some("datetime_columns_count"),
    );
    capped(
        &mut out,
        "integer_coded",
        integer_coded,
        ev_cap,
        Some("integer_coded_preview"),
        None,
    );
    out.extend(target_proxy_fields(plan, ev_cap));
    capped_simple(&mut out, "most_missing_columns", most_missing, ev_cap);
    out
}

fn space_value(space: Option<&ProcessedSpace>) -> Value {
    space.map_or(Value::Null, |s| serde_json::to_value(s).expect("space"))
}

/// The payload of `draft_preprocessing_plan`.
pub fn build_draft_plan_payload(
    plan: &PreprocessingPlan,
    processed_space: Option<&ProcessedSpace>,
    detail: Detail,
    limits: Limits,
) -> Value {
    let full = detail == Detail::Full;
    let q_cap = (!full).then_some(limits.question_limit.max(1));
    let ev_cap = (!full).then_some(limits.evidence_limit.max(1));
    let col_cap = (!full).then_some(5);

    let legend = build_marker_legend(plan, if full { 10_000 } else { 5 });
    let mut partners: Vec<Value> = Vec::new();
    for m in &plan.evidence.mask_dependencies {
        if !partners.iter().any(|p| p == m.partner.as_str()) {
            partners.push(json!(m.partner));
        }
    }
    let yaml = plan_to_preprocessing_yaml(plan, true);

    let mut p = plan_header(plan);
    p.insert(
        "detail".into(),
        json!(if full { "full" } else { "summary" }),
    );
    p.insert("counts".into(), summarize_plan_counts(plan));
    p.insert("processed_space".into(), space_value(processed_space));
    capped_simple(&mut p, "marker_legend", legend, ev_cap);
    capped_simple(
        &mut p,
        "questions",
        open_question_items(plan, col_cap),
        q_cap,
    );
    p.insert(
        "evidence".into(),
        Value::Object(evidence_lists(plan, ev_cap, col_cap)),
    );
    p.insert("warnings".into(), Value::Array(target_proxy_warnings(plan)));
    let mut hints = Map::new();
    capped_simple(&mut hints, "event_flag_columns", partners, ev_cap);
    hints.insert("description".into(), json!(EVENT_FLAG_HINT_DESCRIPTION));
    p.insert("weak_hints".into(), Value::Object(hints));
    p.insert(
        "preprocessing_yaml_chars".into(),
        json!(yaml.chars().count()),
    );
    p.insert("cost".into(), Value::Object(plan.cost.clone()));
    p.insert("how_to_use".into(), json!(HOW_TO_USE));
    p.insert(
        "next_tools".into(),
        json!(["edit_preprocessing_plan", "create_config", "probe_columns"]),
    );
    if full {
        p.insert("preprocessing_yaml".into(), json!(yaml));
        p.insert(
            "entries".into(),
            serde_json::to_value(&plan.entries).expect("entries"),
        );
    }
    Value::Object(p)
}

/// The payload of `edit_preprocessing_plan`. `changes` are the entry diffs
/// of this call (`diff_plan_entries`).
pub fn build_edit_plan_payload(
    plan: &PreprocessingPlan,
    changes: &[Value],
    processed_space: Option<&ProcessedSpace>,
    validation: &DryRun,
    limits: Limits,
) -> Value {
    let field = |c: &Value| c["field"].as_str().unwrap_or("").to_string();
    let effective: Vec<Value> = changes
        .iter()
        .filter(|c| {
            matches!(
                field(c).as_str(),
                "role" | "impute" | "impute_candidates" | "indicator" | "encode"
            )
        })
        .cloned()
        .collect();

    // One row per distinct (field, was, now), with how many columns changed that way.
    let mut groups: Vec<(String, Map<String, Value>)> = Vec::new();
    for c in &effective {
        let key = format!(
            "{}|{}|{}",
            field(c),
            py_canonical_json(&c["was"]),
            py_canonical_json(&c["now"])
        );
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, g)) => {
                let n = g["count"].as_u64().unwrap_or(0) + 1;
                g.insert("count".into(), json!(n));
            }
            None => {
                let mut g = Map::new();
                g.insert("field".into(), json!(field(c)));
                g.insert("was".into(), c["was"].clone());
                g.insert("now".into(), c["now"].clone());
                g.insert("count".into(), json!(1));
                groups.push((key, g));
            }
        }
    }
    let summary: Vec<Value> = groups.into_iter().map(|(_, g)| Value::Object(g)).collect();

    let open = open_question_items(plan, Some(5));
    let legend = build_marker_legend(plan, 5);
    let changed_cols: HashSet<&str> = effective
        .iter()
        .filter_map(|c| c["column"].as_str())
        .collect();
    let provenance_cols: HashSet<&str> = changes
        .iter()
        .filter(|c| matches!(field(c).as_str(), "status" | "decided_by"))
        .filter_map(|c| c["column"].as_str())
        .collect();
    let mut counts = summarize_plan_counts(plan);
    counts["confirmed_without_change"] = json!(provenance_cols.difference(&changed_cols).count());

    let (change_cap, ev_cap) = (
        Some(limits.change_limit.max(1)),
        Some(limits.evidence_limit.max(1)),
    );
    let yaml = plan_to_preprocessing_yaml(plan, true);
    let next_tools = if open.is_empty() {
        json!(["create_config", "validate_preprocessing_config"])
    } else {
        json!(["edit_preprocessing_plan", "create_config"])
    };

    let mut p = plan_header(plan);
    capped_simple(&mut p, "changes", effective, change_cap);
    capped_simple(&mut p, "change_summary", summary, change_cap);
    p.insert("counts".into(), counts);
    p.insert("processed_space".into(), space_value(processed_space));
    capped_simple(&mut p, "marker_legend", legend, ev_cap);
    p.insert(
        "evidence".into(),
        Value::Object(target_proxy_fields(plan, ev_cap)),
    );
    p.insert("warnings".into(), Value::Array(target_proxy_warnings(plan)));
    p.insert(
        "validation".into(),
        json!({"valid": validation.valid, "error": validation.error}),
    );
    capped(
        &mut p,
        "open_questions",
        open,
        Some(limits.question_limit.max(1)),
        Some("open_questions_preview"),
        None,
    );
    p.insert(
        "preprocessing_yaml_chars".into(),
        json!(yaml.chars().count()),
    );
    p.insert("next_tools".into(), next_tools);
    Value::Object(p)
}

// ---- Markdown ----------------------------------------------------------------

fn s(v: &Value) -> String {
    py_str(v)
}

fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or(0)
}

fn warning_lines(payload: &Value) -> Vec<String> {
    payload["warnings"]
        .as_array()
        .map(|ws| {
            ws.iter()
                .map(|w| {
                    let cols: Vec<String> = w["columns"]
                        .as_array()
                        .map(|c| c.iter().map(|c| format!("`{}`", s(c))).collect())
                        .unwrap_or_default();
                    format!(
                        "- Warning `{}` (`{}`): {} ({})",
                        s(&w["code"]),
                        s(&w["target"]),
                        s(&w["message"]),
                        cols.join(", ")
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn question_lines(q: &Value) -> Vec<String> {
    let stakes = &q["stakes"];
    let stakes_str = match stakes["neighbors_kept"].as_f64() {
        Some(kept) => format!(
            "neighbors_kept={kept:.2}, feature_delta={:+}",
            stakes["feature_count_delta"].as_i64().unwrap_or(0)
        ),
        None => "stakes=n/a".into(),
    };
    let mut lines = vec![
        format!(
            "### `{}` [{}] ({} rows; {stakes_str})",
            s(&q["question_id"]),
            s(&q["topic"]),
            s(&q["rows_affected"])
        ),
        s(&q["text"]),
    ];
    for opt in q["options"].as_array().into_iter().flatten() {
        let default_tag = if opt["option_id"] == q["default_option_id"] {
            " *(default)*"
        } else {
            ""
        };
        lines.push(format!(
            "- `{}` \u{2014} **{}**{default_tag}: {}",
            s(&opt["option_id"]),
            s(&opt["label"]),
            s(&opt["effect"])
        ));
    }
    lines.push(String::new());
    lines
}

fn open_questions_markdown(payload: &Value) -> Vec<String> {
    let questions = payload["questions"].as_array().cloned().unwrap_or_default();
    let mut lines = vec![String::new(), "## Open Questions".into(), String::new()];
    if questions.is_empty() {
        lines.push("- None. All entries are resolved.".into());
        return lines;
    }
    for q in &questions {
        lines.extend(question_lines(q));
    }
    let omitted = int(&payload["questions_omitted"]);
    if omitted > 0 {
        lines.push(format!(
            "*... +{omitted} additional open question(s) omitted.*"
        ));
    }
    lines
}

fn len_of(v: &Value) -> usize {
    v.as_array().map_or(0, Vec::len)
}

/// A compact Markdown view of the draft payload.
pub fn draft_plan_payload_to_markdown(payload: &Value) -> String {
    let counts = &payload["counts"];
    let proc = &payload["processed_space"];
    let mut lines = vec![
        format!(
            "# Preprocessing Plan `{}` (rev {})",
            s(&payload["plan_id"]),
            s(&payload["revision"])
        ),
        String::new(),
        format!(
            "- Dataset: `{}` | Fingerprint: `{}`",
            s(&payload["dataset_id"]),
            s(&payload["plan_fingerprint"])
        ),
        format!(
            "- Columns: {} (feature columns: {}, dropped/held-out: {}, open questions: {})",
            int(&counts["columns"]),
            int(&counts["feature_columns"]),
            int(&counts["dropped"]),
            int(&counts["open_questions"])
        ),
    ];
    if proc.is_null() {
        lines.push("- Processed space: dry-run unavailable".into());
    } else {
        lines.push(format!(
            "- Processed space: {} features ({} missing markers, {} dummy columns)",
            int(&proc["feature_count"]),
            int(&proc["missing_marker_count"]),
            int(&proc["dummy_column_count"])
        ));
    }
    lines.extend(warning_lines(payload));
    lines.extend(open_questions_markdown(payload));

    let ev = &payload["evidence"];
    lines.extend([String::new(), "## Evidence Headline".into(), String::new()]);
    for (label, key) in [
        ("Shared missing groups", "missing_groups"),
        ("Conditional mask dependencies", "mask_dependencies"),
        ("Exact relations (duplicate/sum_of)", "relations"),
        ("Most-missing columns", "most_missing_columns"),
    ] {
        lines.push(format!(
            "- {label}: {} (+{} omitted)",
            len_of(&ev[key]),
            int(&ev[format!("{key}_omitted")])
        ));
    }
    lines.push(format!(
        "- Constant columns dropped: {}",
        int(&ev["constant_columns_count"])
    ));
    let hints = payload["weak_hints"]["event_flag_columns"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if !hints.is_empty() {
        let omitted = int(&payload["weak_hints"]["event_flag_columns_omitted"]);
        let suffix = if omitted > 0 {
            format!(" (+{omitted} omitted)")
        } else {
            String::new()
        };
        let names: Vec<String> = hints.iter().map(|c| format!("`{}`", s(c))).collect();
        lines.push(format!(
            "- Event-flag weak hints: {}{suffix}",
            names.join(", ")
        ));
    }

    if payload.get("preprocessing_yaml").is_some() {
        lines.extend([
            String::new(),
            "## Preprocessing YAML".into(),
            String::new(),
            "```yaml".into(),
            s(&payload["preprocessing_yaml"]).trim().to_string(),
            "```".into(),
        ]);
    } else {
        lines.push(format!(
            "- Preprocessing YAML: {} chars (pass `detail='full'` or call `create_config` to view)",
            int(&payload["preprocessing_yaml_chars"])
        ));
    }
    lines.extend([String::new(), HOW_TO_USE.to_string()]);
    lines.join("\n").trim().to_string()
}

/// A compact Markdown view of the edit payload.
pub fn edit_plan_payload_to_markdown(payload: &Value) -> String {
    let counts = &payload["counts"];
    let val = &payload["validation"];
    let mut lines = vec![
        format!(
            "# Updated Preprocessing Plan `{}` (rev {})",
            s(&payload["plan_id"]),
            s(&payload["revision"])
        ),
        String::new(),
        format!(
            "- Fingerprint: `{}` | Valid: `{}`",
            s(&payload["plan_fingerprint"]),
            s(&val["valid"])
        ),
        format!(
            "- Counts: {} feature columns, {} dropped/held-out, {} open questions",
            int(&counts["feature_columns"]),
            int(&counts["dropped"]),
            int(&counts["open_questions"])
        ),
        format!(
            "- Preprocessing YAML: {} chars",
            int(&payload["preprocessing_yaml_chars"])
        ),
    ];
    let preview = payload["open_questions_preview"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if !preview.is_empty() {
        let tags: Vec<String> = preview
            .iter()
            .map(|q| format!("`{}` [{}]", s(&q["question_id"]), s(&q["topic"])))
            .collect();
        let mut tag_str = tags.join(", ");
        let omitted = int(&payload["open_questions_omitted"]);
        if omitted > 0 {
            tag_str.push_str(&format!(" (+{omitted} more)"));
        }
        lines.push(format!("- Open questions: {tag_str}"));
    }
    if let Some(err) = val["error"].as_str().filter(|e| !e.is_empty()) {
        lines.push(format!("- Validation error: `{err}`"));
    }
    lines.extend(warning_lines(payload));

    let changes = payload["changes"].as_array().cloned().unwrap_or_default();
    lines.extend([String::new(), "## Changes".into(), String::new()]);
    if changes.is_empty() {
        lines.push("- No effective entry changes.".into());
    } else {
        lines.extend([
            "| Column | Field | Was | Now |".to_string(),
            "|---|---|---|---|".to_string(),
        ]);
        for ch in &changes {
            lines.push(format!(
                "| `{}` | `{}` | `{}` | `{}` |",
                s(&ch["column"]),
                s(&ch["field"]),
                py_json_dumps(&ch["was"]),
                py_json_dumps(&ch["now"])
            ));
        }
        let omitted = int(&payload["changes_omitted"]);
        if omitted > 0 {
            lines.push(format!("\n*... +{omitted} additional change(s) omitted.*"));
        }
    }
    lines.join("\n").trim().to_string()
}
