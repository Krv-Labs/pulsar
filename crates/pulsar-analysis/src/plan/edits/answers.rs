//! Question answer validation, option application, and multi-option merging.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use super::super::model::{
    DecidedBy, EntryStatus, ImputeSpec, PlanEditError, PlanEntry, PlanQuestion, QuestionOption,
    QuestionStatus, QuestionTopic, Role,
};
use super::super::text::{py_repr, py_repr_value};
use super::{apply_single_edit, err, is_protected, source_str, Entries, Evidence, Groups};

fn collect_group_indicators(entries: &[PlanEntry]) -> Vec<&String> {
    let mut groups: Vec<&String> = Vec::new();
    for e in entries {
        if e.indicator.starts_with("group:") && !groups.contains(&&e.indicator) {
            groups.push(&e.indicator);
        }
    }
    groups
}

fn entries_conflict(primary: &PlanEntry, rest: &[PlanEntry]) -> bool {
    rest.iter().any(|e| {
        e.role != primary.role
            || e.encode != primary.encode
            || e.impute.is_none() != primary.impute.is_none()
    })
}

fn validate_compatible_entries(
    col: &str,
    entries: &[PlanEntry],
) -> Result<Option<String>, PlanEditError> {
    let group_indicators = collect_group_indicators(entries);
    let has_own = entries.iter().any(|e| e.indicator == "own");
    let incompatible = group_indicators.len() > 1
        || (!group_indicators.is_empty() && has_own)
        || entries_conflict(&entries[0], &entries[1..]);
    if incompatible {
        return Err(err(
            format!(
                "Selected options for column {} have incompatible roles, encoding, or missingness groups.",
                py_repr(col)
            ),
            vec![("column", json!(col))],
        ));
    }
    Ok(group_indicators.first().map(|s| (*s).clone()))
}

fn entry_specs(entry: &PlanEntry) -> Vec<&ImputeSpec> {
    match &entry.impute_candidates {
        Some(c) => c.iter().collect(),
        None => entry.impute.iter().collect(),
    }
}

fn merge_impute_candidates(entries: &[PlanEntry], has_group: bool) -> Vec<ImputeSpec> {
    let mut candidates: Vec<ImputeSpec> = Vec::new();
    for entry in entries {
        for raw in entry_specs(entry) {
            let mut spec = raw.clone();
            if spec.indicator.is_none() {
                spec.indicator = Some(!has_group && entry.indicator == "own");
            }
            if has_group {
                spec.indicator = Some(false);
            }
            if !candidates.contains(&spec) {
                candidates.push(spec);
            }
        }
    }
    candidates
}

/// Combine compatible treatments chosen by several options of one question
/// without losing the candidates' indicators.
pub(super) fn merge_answer_entries(
    col: &str,
    entries: Vec<PlanEntry>,
) -> Result<PlanEntry, PlanEditError> {
    let group_ind = validate_compatible_entries(col, &entries)?;
    let mut primary = entries[0].clone();
    if primary.impute.is_none() {
        return Ok(primary);
    }
    let candidates = merge_impute_candidates(&entries, group_ind.is_some());
    primary.impute = candidates.first().cloned();
    primary.impute_candidates = (candidates.len() > 1).then(|| candidates.clone());
    primary.indicator = match group_ind {
        Some(g) => g,
        None if candidates.iter().any(|s| s.indicator == Some(true)) => "own".into(),
        None => "none".into(),
    };
    Ok(primary)
}

fn resolve_question_id(
    ans: &Map<String, Value>,
    questions: &IndexMap<String, PlanQuestion>,
) -> Result<String, PlanEditError> {
    let qid_value = ans.get("question_id").cloned().unwrap_or(Value::Null);
    match qid_value.as_str() {
        Some(q) if questions.contains_key(q) => Ok(q.to_string()),
        _ => Err(err(
            format!("Unknown question_id: {}.", py_repr_value(&qid_value)),
            vec![
                ("question_id", qid_value),
                (
                    "known_question_ids",
                    json!(questions.keys().collect::<Vec<_>>()),
                ),
            ],
        )),
    }
}

fn resolve_options<'a>(
    ans: &Map<String, Value>,
    qid: &str,
    q: &'a PlanQuestion,
) -> Result<(Vec<Value>, Vec<&'a QuestionOption>), PlanEditError> {
    let raw_oid = ans
        .get("option_ids")
        .or_else(|| ans.get("option_id"))
        .cloned()
        .unwrap_or(Value::Null);
    let valid_options: Vec<&String> = q.options.iter().map(|o| &o.option_id).collect();
    let option_ids: Vec<Value> = match raw_oid {
        Value::Array(items) => items,
        other => vec![other],
    };
    if option_ids.is_empty() {
        return Err(err(
            format!(
                "option_id list for question {} must not be empty.",
                py_repr(qid)
            ),
            vec![
                ("question_id", json!(qid)),
                ("valid_options", json!(valid_options)),
            ],
        ));
    }
    let mut matched: Vec<&QuestionOption> = Vec::new();
    for oid in &option_ids {
        let Some(opt) = q.options.iter().find(|o| oid == o.option_id.as_str()) else {
            return Err(err(
                format!(
                    "Unknown option_id {} for question {}.",
                    py_repr_value(oid),
                    py_repr(qid)
                ),
                vec![
                    ("question_id", json!(qid)),
                    ("topic", json!(q.topic)),
                    ("option_id", oid.clone()),
                    ("valid_options", json!(valid_options)),
                ],
            ));
        };
        matched.push(opt);
    }
    Ok((option_ids, matched))
}

fn effective_role(col: &str, entries: &Entries, incoming: &HashMap<String, Role>) -> Role {
    incoming.get(col).copied().unwrap_or(entries[col].role)
}

fn describe_roles(pairs: &[(String, Role)]) -> String {
    pairs
        .iter()
        .map(|(c, r)| format!("{c}={}", r.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn stale_question_roles(
    effective: &[(String, Role)],
    entries: &Entries,
    incoming: &HashMap<String, Role>,
) -> String {
    let roles_str = describe_roles(effective);
    if !roles_str.is_empty() {
        return roles_str;
    }
    let protected: Vec<(String, Role)> = entries
        .iter()
        .map(|(c, e)| (c.clone(), incoming.get(c).copied().unwrap_or(e.role)))
        .filter(|(_, r)| is_protected(*r))
        .collect();
    let protected_str = describe_roles(&protected);
    if !protected_str.is_empty() {
        return protected_str;
    }
    "resolved".into()
}

fn check_question_applicable(
    qid: &str,
    q: &PlanQuestion,
    entries: &Entries,
    incoming: &HashMap<String, Role>,
) -> Result<(), PlanEditError> {
    let effective: Vec<(String, Role)> = q
        .applies_to
        .iter()
        .filter(|c| entries.contains_key(c.as_str()))
        .map(|c| (c.clone(), effective_role(c, entries, incoming)))
        .collect();
    let all_protected = !effective.is_empty() && effective.iter().all(|(_, r)| is_protected(*r));
    if q.status == QuestionStatus::Open && !all_protected {
        return Ok(());
    }
    let roles_str = stale_question_roles(&effective, entries, incoming);
    let columns: Map<String, Value> = effective
        .iter()
        .map(|(c, r)| (c.clone(), json!(r.as_str())))
        .collect();
    Err(err(
        format!("question {qid} no longer applies (columns now: {roles_str})"),
        vec![
            ("question_id", json!(qid)),
            ("topic", json!(q.topic)),
            ("status", json!(q.status)),
            ("columns", Value::Object(columns)),
        ],
    ))
}

fn is_col_protected(
    col_name: Option<&str>,
    entries: &Entries,
    incoming: &HashMap<String, Role>,
) -> bool {
    let Some(c) = col_name.filter(|c| entries.contains_key(*c)) else {
        return false;
    };
    is_protected(effective_role(c, entries, incoming))
}

fn apply_option_edits(
    opt: &QuestionOption,
    target: &mut Entries,
    incoming: &HashMap<String, Role>,
    source: DecidedBy,
    groups: &Groups,
    evidence: &Evidence,
) -> Result<Vec<String>, PlanEditError> {
    let mut edited: Vec<String> = Vec::new();
    for edit_spec in &opt.edits {
        let col_name = edit_spec.get("column").and_then(Value::as_str);
        if is_col_protected(col_name, target, incoming) {
            continue;
        }
        apply_single_edit(
            target,
            &Value::Object(edit_spec.clone()),
            source,
            groups,
            evidence,
        )?;
        if let Some(c) = col_name {
            if !edited.iter().any(|e| e == c) {
                edited.push(c.to_string());
            }
        }
    }
    Ok(edited)
}

fn apply_matched_options(
    matched: &[&QuestionOption],
    entries: &mut Entries,
    incoming: &HashMap<String, Role>,
    source: DecidedBy,
    groups: &Groups,
    evidence: &Evidence,
) -> Result<(), PlanEditError> {
    if matched.len() == 1 {
        apply_option_edits(matched[0], entries, incoming, source, groups, evidence)?;
        return Ok(());
    }
    let mut by_col: IndexMap<String, Vec<PlanEntry>> = IndexMap::new();
    for opt in matched {
        // Every selected option starts from the same revision, so one option's
        // edits cannot become another option's implicit defaults.
        let mut scratch = entries.clone();
        let edited = apply_option_edits(opt, &mut scratch, incoming, source, groups, evidence)?;
        for c in edited {
            by_col
                .entry(c.clone())
                .or_default()
                .push(scratch[&c].clone());
        }
    }
    for (col, versions) in by_col {
        entries.insert(col.clone(), merge_answer_entries(&col, versions)?);
    }
    Ok(())
}

fn resolve_question_columns(
    applies_to: &[String],
    entries: &mut Entries,
    incoming: &HashMap<String, Role>,
    source: DecidedBy,
) {
    for col in applies_to {
        let Some(role) = entries
            .get(col)
            .map(|e| incoming.get(col).copied().unwrap_or(e.role))
        else {
            continue;
        };
        if !is_protected(role) {
            let e = entries.get_mut(col).expect("checked above");
            e.decided_by = source;
            e.status = EntryStatus::Resolved;
        }
    }
}

pub(super) fn apply_single_answer(
    ans: &Value,
    questions: &mut IndexMap<String, PlanQuestion>,
    entries: &mut Entries,
    incoming: &HashMap<String, Role>,
    source: DecidedBy,
    groups: &Groups,
    evidence: &Evidence,
) -> Result<(), PlanEditError> {
    let Some(ans_map) = ans.as_object() else {
        return Err(err(
            "Each answer must be a mapping with 'question_id' and 'option_id'.".into(),
            vec![("answer", ans.clone())],
        ));
    };
    let qid = resolve_question_id(ans_map, questions)?;
    let q = questions[&qid].clone();
    let (option_ids, matched) = resolve_options(ans_map, &qid, &q)?;
    check_question_applicable(&qid, &q, entries, incoming)?;
    apply_matched_options(&matched, entries, incoming, source, groups, evidence)?;
    resolve_question_columns(&q.applies_to, entries, incoming, source);

    let answered = questions.get_mut(&qid).expect("checked above");
    answered.status = QuestionStatus::Answered;
    let option_id = if option_ids.len() == 1 {
        option_ids[0].clone()
    } else {
        Value::Array(option_ids)
    };
    answered.answer = Some(json!({"option_id": option_id, "decided_by": source_str(source)}));
    Ok(())
}

fn is_resolved_by_edits(q: &PlanQuestion, entries: &Entries) -> bool {
    !q.applies_to.is_empty()
        && q.applies_to
            .iter()
            .filter_map(|c| entries.get(c))
            .all(|e| e.status == EntryStatus::Resolved && e.decided_by != DecidedBy::Default)
}

/// Questions whose columns were all resolved by direct edits are answered.
pub(super) fn sync_question_statuses(
    questions: &mut IndexMap<String, PlanQuestion>,
    entries: &Entries,
    target_role_added: bool,
    source: DecidedBy,
) {
    for q in questions
        .values_mut()
        .filter(|q| q.status == QuestionStatus::Open)
    {
        let by_target = q.topic == QuestionTopic::OutcomeColumns && target_role_added;
        if by_target || is_resolved_by_edits(q, entries) {
            q.status = QuestionStatus::Answered;
            q.answer = Some(json!({"option_id": "custom_edit", "decided_by": source_str(source)}));
        }
    }
}
