//! Applying question answers and column edits to a plan, producing a new
//! revision, and diffing revisions (`preprocessing_plan/edits.py`).
//!
//! Inputs are loose JSON because they come straight from agent tool calls;
//! every check Python makes with `isinstance` is reproduced, and error
//! messages use Python `repr` so they read the same to an agent. All work
//! happens on copies: a failed call leaves the original plan untouched.
//!
//! `entry.impute` is always the first of `entry.impute_candidates` when
//! candidates exist (in Python they are the same dict object); [`sync`]
//! restores that after any change to the candidates.

mod answers;
mod guard;
mod impute;

use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

pub use guard::HeldOneHot;

use self::answers::{apply_single_answer, sync_question_statuses};
use self::guard::{confirm_or_hold, tagged_with_question};
use self::impute::edit_impute;
use super::model::{DecidedBy, EntryStatus, PlanEditError, PlanEntry, PlanQuestion, Role};
use super::plan::PreprocessingPlan;
use super::text::{py_repr, py_repr_list, py_repr_value, py_str};
use crate::evidence::{ColumnEvidence, ColumnKind};

pub const VALID_ROLES: [&str; 6] = ["drop", "entity", "feature", "id", "target", "time"];
pub const VALID_IMPUTE_METHODS: [&str; 6] = [
    "fill_mean",
    "fill_median",
    "fill_mode",
    "fill_value",
    "sample_categorical",
    "sample_normal",
];
pub const VALID_ENCODE_METHODS: [&str; 1] = ["one_hot"];
const EDIT_ALLOWED_KEYS: [&str; 7] = [
    "column",
    "encode",
    "impute",
    "impute_method",
    "impute_value",
    "indicator",
    "role",
];

type Entries = IndexMap<String, PlanEntry>;
type Groups = IndexMap<String, Vec<String>>;
type Evidence<'a> = HashMap<&'a str, &'a ColumnEvidence>;

fn err(message: String, details: Vec<(&str, Value)>) -> PlanEditError {
    PlanEditError::new(
        "INVALID_ARGUMENT",
        message,
        details
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

fn is_protected(role: Role) -> bool {
    matches!(role, Role::Target | Role::Entity | Role::Time)
}

fn parse_role(s: &str) -> Option<Role> {
    match s {
        "feature" => Some(Role::Feature),
        "drop" => Some(Role::Drop),
        "id" => Some(Role::Id),
        "entity" => Some(Role::Entity),
        "time" => Some(Role::Time),
        "target" => Some(Role::Target),
        _ => None,
    }
}

fn source_str(source: DecidedBy) -> &'static str {
    match source {
        DecidedBy::Default => "default",
        DecidedBy::Agent => "agent",
        DecidedBy::User => "user",
        DecidedBy::Phirewall => "phirewall",
    }
}

/// Keep `impute` equal to the first candidate when candidates exist.
fn sync(entry: &mut PlanEntry) {
    if let Some(first) = entry.impute_candidates.as_ref().and_then(|c| c.first()) {
        entry.impute = Some(first.clone());
    }
}

// ---- indicator -------------------------------------------------------------

fn normalize_indicator(
    indicator: &str,
    col: &str,
    groups: &Groups,
) -> Result<String, PlanEditError> {
    let mut known: Vec<&String> = groups.keys().collect();
    known.sort();
    let known = json!(known);
    let ind = indicator.trim();
    if ind == "own" {
        return Ok("own".to_string());
    }
    if let Some(rest) = ind.strip_prefix("group:") {
        let gid = rest.trim();
        if gid.is_empty() || (!groups.contains_key(gid) && gid != col) {
            return Err(err(
                format!(
                    "Invalid indicator {} for column {}: group must be an evidence missingness group or {}.",
                    py_repr(indicator),
                    py_repr(col),
                    py_repr(col)
                ),
                vec![
                    ("column", json!(col)),
                    ("indicator", json!(indicator)),
                    ("known_groups", known),
                ],
            ));
        }
        return Ok(format!("group:{gid}"));
    }
    Err(err(
        format!(
            "Invalid indicator {} for column {}. Expected 'none', 'own', or 'group:<gid>'.",
            py_repr(indicator),
            py_repr(col)
        ),
        vec![
            ("column", json!(col)),
            ("indicator", json!(indicator)),
            ("known_groups", known),
        ],
    ))
}

fn validate_indicator(
    indicator: &str,
    col: &str,
    ev: Option<&ColumnEvidence>,
    groups: &Groups,
) -> Result<String, PlanEditError> {
    if indicator.trim() == "none" {
        return Ok("none".into());
    }
    let normalized = normalize_indicator(indicator, col, groups)?;
    if let Some(ev) = ev {
        let mixed_text = ev.kind == ColumnKind::NumericText && ev.n_non_numeric.unwrap_or(0) > 0;
        if ev.n_missing == 0 && !mixed_text {
            return Err(err(
                format!(
                    "Column {} has no blanks; indicator must be 'none'.",
                    py_repr(col)
                ),
                vec![
                    ("column", json!(col)),
                    ("indicator", json!(indicator)),
                    ("n_missing", json!(0)),
                ],
            ));
        }
    }
    Ok(normalized)
}

// ---- role and encode -------------------------------------------------------

fn edit_role(
    entry: &mut PlanEntry,
    edit: &Map<String, Value>,
    col: &str,
) -> Result<(), PlanEditError> {
    let Some(raw) = edit.get("role").filter(|v| !v.is_null()) else {
        return Ok(());
    };
    let role_val = py_str(raw);
    match parse_role(&role_val) {
        Some(role) => {
            entry.role = role;
            Ok(())
        }
        None => Err(err(
            format!(
                "Invalid role {} for column {}; valid roles are {} (for outcomes use role 'target').",
                py_repr(&role_val),
                py_repr(col),
                py_repr_list(&VALID_ROLES)
            ),
            vec![
                ("column", json!(col)),
                ("role", json!(role_val)),
                ("valid_roles", json!(VALID_ROLES)),
            ],
        )),
    }
}

fn edit_encode(
    entry: &mut PlanEntry,
    edit: &Map<String, Value>,
    col: &str,
) -> Result<(), PlanEditError> {
    let Some(enc) = edit.get("encode") else {
        return Ok(());
    };
    if enc.is_null() || enc == "none" || enc == "" {
        entry.encode = None;
        return Ok(());
    }
    let enc_str = py_str(enc);
    if !VALID_ENCODE_METHODS.contains(&enc_str.as_str()) {
        return Err(err(
            format!(
                "Invalid encode method {} for column {}.",
                py_repr(&enc_str),
                py_repr(col)
            ),
            vec![
                ("column", json!(col)),
                ("encode", json!(enc_str)),
                ("valid_methods", json!(VALID_ENCODE_METHODS)),
            ],
        ));
    }
    entry.encode = Some(enc_str);
    Ok(())
}

fn edit_indicator(
    entry: &mut PlanEntry,
    edit: &Map<String, Value>,
    col: &str,
    ev: Option<&ColumnEvidence>,
    groups: &Groups,
) -> Result<(), PlanEditError> {
    let Some(raw) = edit.get("indicator").filter(|v| !v.is_null()) else {
        return Ok(());
    };
    entry.indicator = validate_indicator(&py_str(raw), col, ev, groups)?;
    if let Some(cands) = entry.impute_candidates.as_mut() {
        for spec in cands {
            spec.indicator = None;
        }
    }
    if let Some(imp) = entry.impute.as_mut() {
        imp.indicator = None;
    }
    Ok(())
}

// ---- one edit --------------------------------------------------------------

fn validate_edit_fields(edit: &Map<String, Value>) -> Result<(), PlanEditError> {
    let mut unknown: Vec<&String> = edit
        .keys()
        .filter(|k| !EDIT_ALLOWED_KEYS.contains(&k.as_str()))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort();
    let col_repr = py_repr_value(edit.get("column").unwrap_or(&Value::Null));
    Err(err(
        format!(
            "Unsupported edit field(s) for column {col_repr}: {}.",
            py_repr_list(&unknown)
        ),
        vec![
            ("column", edit.get("column").cloned().unwrap_or(Value::Null)),
            ("unsupported_fields", json!(unknown)),
            ("allowed_fields", json!(EDIT_ALLOWED_KEYS)),
        ],
    ))
}

fn clear_non_feature_settings(
    entry: &mut PlanEntry,
    edit: &Map<String, Value>,
    has_impute_edit: bool,
) {
    if entry.role == Role::Feature {
        return;
    }
    if !has_impute_edit {
        entry.impute = None;
        entry.impute_candidates = None;
    }
    if !edit.contains_key("encode") {
        entry.encode = None;
    }
    if !edit.contains_key("indicator") {
        entry.indicator = "none".into();
    }
}

fn apply_single_edit(
    entries: &mut Entries,
    edit: &Value,
    source: DecidedBy,
    groups: &Groups,
    evidence: &Evidence,
) -> Result<(), PlanEditError> {
    let Some(edit) = edit.as_object() else {
        return Err(err(
            "Each edit must be a mapping.".into(),
            vec![("edit", edit.clone())],
        ));
    };
    validate_edit_fields(edit)?;
    let col_value = edit.get("column").cloned().unwrap_or(Value::Null);
    let col = match col_value.as_str() {
        Some(c) if entries.contains_key(c) => c.to_string(),
        _ => {
            return Err(err(
                format!("Unknown column: {}.", py_repr_value(&col_value)),
                vec![("column", col_value)],
            ))
        }
    };
    let ev = evidence.get(col.as_str()).copied();
    let entry = entries.get_mut(&col).expect("checked above");

    edit_role(entry, edit, &col)?;
    edit_encode(entry, edit, &col)?;
    let has_impute_edit = edit_impute(entry, edit, &col, ev)?;
    edit_indicator(entry, edit, &col, ev, groups)?;

    // A column set to a non-feature role without explicit impute/encode/
    // indicator drops its feature-only settings, so held-out columns do not
    // linger in impute and indicator groups.
    clear_non_feature_settings(entry, edit, has_impute_edit);
    entry.decided_by = source;
    entry.status = EntryStatus::Resolved;
    Ok(())
}

// ---- batch entrypoints -----------------------------------------------------

/// Apply question `answers` and column `edits`, returning a new revision.
/// The input plan is never modified, including when this fails.
pub fn apply_answers_and_edits(
    plan: &PreprocessingPlan,
    answers: &[Value],
    edits: &[Value],
    source: DecidedBy,
) -> Result<PreprocessingPlan, PlanEditError> {
    apply_inner(plan, answers, edits, source, None).map(|(plan, _)| plan)
}

/// Like [`apply_answers_and_edits`], but an answer or edit that would one-hot a
/// column above the categorical threshold leaves that column unchanged and is
/// returned in the held list, unless the column is named in `confirmed`
/// (applied, with the confirmation recorded in the entry's reason). A held
/// answer is held whole, so the question stays open.
pub fn apply_answers_and_edits_guarded(
    plan: &PreprocessingPlan,
    answers: &[Value],
    edits: &[Value],
    source: DecidedBy,
    confirmed: &[String],
) -> Result<(PreprocessingPlan, Vec<HeldOneHot>), PlanEditError> {
    apply_inner(plan, answers, edits, source, Some(confirmed))
}

fn validate_apply_inputs(
    answers: &[Value],
    edits: &[Value],
    source: DecidedBy,
) -> Result<(), PlanEditError> {
    if source == DecidedBy::Default {
        return Err(err(
            format!(
                "Invalid source {}; expected 'agent' or 'user'.",
                py_repr(source_str(source))
            ),
            vec![
                ("source", json!(source_str(source))),
                ("valid_sources", json!(["agent", "user"])),
            ],
        ));
    }
    if answers.is_empty() && edits.is_empty() {
        return Err(err(
            "Nothing to change: provide at least one answer or edit.".into(),
            vec![("answers", json!([])), ("edits", json!([]))],
        ));
    }
    Ok(())
}

fn incoming_protected_roles(edits: &[Value]) -> HashMap<String, Role> {
    edits
        .iter()
        .filter_map(|e| {
            let col = e.get("column")?.as_str()?;
            let role = parse_role(e.get("role")?.as_str()?).filter(|r| is_protected(*r))?;
            Some((col.to_string(), role))
        })
        .collect()
}

struct ApplyContext<'a> {
    groups: &'a Groups,
    evidence: Evidence<'a>,
    threshold: f64,
    source: DecidedBy,
    confirmed: Option<&'a [String]>,
}

fn apply_answers_batch(
    ctx: &ApplyContext,
    answers: &[Value],
    questions: &mut IndexMap<String, PlanQuestion>,
    entries: &mut Entries,
    incoming: &HashMap<String, Role>,
    held: &mut Vec<HeldOneHot>,
) -> Result<(), PlanEditError> {
    for ans in answers {
        let snapshot = ctx.confirmed.map(|_| (entries.clone(), questions.clone()));
        apply_single_answer(
            ans,
            questions,
            entries,
            incoming,
            ctx.source,
            ctx.groups,
            &ctx.evidence,
        )?;
        if let (Some(confirmed), Some((old_entries, old_questions))) = (ctx.confirmed, snapshot) {
            let unconfirmed = confirm_or_hold(
                &old_entries,
                entries,
                &ctx.evidence,
                ctx.threshold,
                confirmed,
            );
            if !unconfirmed.is_empty() {
                *entries = old_entries;
                *questions = old_questions;
                held.extend(tagged_with_question(unconfirmed, ans));
            }
        }
    }
    Ok(())
}

fn apply_edits_batch(
    ctx: &ApplyContext,
    edits: &[Value],
    entries: &mut Entries,
    held: &mut Vec<HeldOneHot>,
) -> Result<bool, PlanEditError> {
    let mut target_role_added = false;
    for edit in edits {
        let snapshot = ctx.confirmed.map(|_| entries.clone());
        apply_single_edit(entries, edit, ctx.source, ctx.groups, &ctx.evidence)?;
        if let (Some(confirmed), Some(old_entries)) = (ctx.confirmed, snapshot) {
            for h in confirm_or_hold(
                &old_entries,
                entries,
                &ctx.evidence,
                ctx.threshold,
                confirmed,
            ) {
                entries.insert(h.column.clone(), old_entries[&h.column].clone());
                held.push(h);
            }
        }
        if edit.get("role").is_some_and(|r| r == "target") {
            target_role_added = true;
        }
    }
    Ok(target_role_added)
}

fn apply_inner(
    plan: &PreprocessingPlan,
    answers: &[Value],
    edits: &[Value],
    source: DecidedBy,
    confirmed: Option<&[String]>,
) -> Result<(PreprocessingPlan, Vec<HeldOneHot>), PlanEditError> {
    validate_apply_inputs(answers, edits, source)?;
    let ctx = ApplyContext {
        groups: &plan.missing_groups,
        evidence: plan
            .evidence
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect(),
        threshold: (plan.evidence.n_rows.max(1) as f64).sqrt(),
        source,
        confirmed,
    };
    let mut entries = plan.entries.clone();
    let mut questions = plan.questions.clone();
    let incoming = incoming_protected_roles(edits);
    let mut held: Vec<HeldOneHot> = Vec::new();

    apply_answers_batch(
        &ctx,
        answers,
        &mut questions,
        &mut entries,
        &incoming,
        &mut held,
    )?;
    let target_role_added = apply_edits_batch(&ctx, edits, &mut entries, &mut held)?;
    sync_question_statuses(&mut questions, &entries, target_role_added, source);

    Ok((
        PreprocessingPlan {
            plan_id: plan.plan_id.clone(),
            dataset_id: plan.dataset_id.clone(),
            revision: plan.revision + 1,
            entries,
            questions,
            missing_groups: plan.missing_groups.clone(),
            evidence: plan.evidence.clone(),
            created_at: plan.created_at.clone(),
            cost: plan.cost.clone(),
            target_proxies: plan.target_proxies.clone(),
            n_imputations: plan.n_imputations,
            imported: plan.imported.clone(),
            phirewall_release: plan.phirewall_release,
        },
        held,
    ))
}

// ---- diff ------------------------------------------------------------------

fn tracked(entry: &PlanEntry, field: &str) -> Value {
    match field {
        "role" => json!(entry.role),
        "impute" => json!(entry.impute),
        "impute_candidates" => json!(entry.impute_candidates),
        "indicator" => json!(entry.indicator),
        "encode" => json!(entry.encode),
        "status" => json!(entry.status),
        _ => json!(entry.decided_by),
    }
}

/// `{column, field, was, now}` for every tracked field that differs between
/// two revisions of the same plan.
pub fn diff_plan_entries(before: &PreprocessingPlan, after: &PreprocessingPlan) -> Vec<Value> {
    const FIELDS: [&str; 7] = [
        "role",
        "impute",
        "impute_candidates",
        "indicator",
        "encode",
        "status",
        "decided_by",
    ];
    let mut changes = Vec::new();
    for (col, old) in &before.entries {
        let Some(new) = after.entries.get(col) else {
            continue;
        };
        for field in FIELDS {
            let (was, now) = (tracked(old, field), tracked(new, field));
            if was != now {
                changes.push(json!({"column": col, "field": field, "was": was, "now": now}));
            }
        }
    }
    changes
}
