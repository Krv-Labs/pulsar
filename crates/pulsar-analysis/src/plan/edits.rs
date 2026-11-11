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

use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use super::model::{
    DecidedBy, EntryStatus, ImputeSpec, PlanEditError, PlanEntry, PlanQuestion, QuestionOption,
    QuestionStatus, Role,
};
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
    }
}

/// Keep `impute` equal to the first candidate when candidates exist.
fn sync(entry: &mut PlanEntry) {
    if let Some(first) = entry.impute_candidates.as_ref().and_then(|c| c.first()) {
        entry.impute = Some(first.clone());
    }
}

// ---- indicator -------------------------------------------------------------

fn validate_indicator(
    indicator: &str,
    col: &str,
    ev: Option<&ColumnEvidence>,
    groups: &Groups,
) -> Result<String, PlanEditError> {
    let mut known: Vec<&String> = groups.keys().collect();
    known.sort();
    let known = json!(known);
    let ind = indicator.trim();
    let normalized = if ind == "none" {
        return Ok("none".into());
    } else if ind == "own" {
        "own".to_string()
    } else if let Some(rest) = ind.strip_prefix("group:") {
        let gid = rest.trim();
        if gid.is_empty() || (!groups.contains_key(gid) && gid != col) {
            return Err(err(
                format!(
                    "Invalid indicator {} for column {}: group must be an evidence missingness group or {}.",
                    py_repr(indicator),
                    py_repr(col),
                    py_repr(col)
                ),
                vec![("column", json!(col)), ("indicator", json!(indicator)), ("known_groups", known)],
            ));
        }
        format!("group:{gid}")
    } else {
        return Err(err(
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
        ));
    };
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

// ---- role and encode ---------------------------------------------------------

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
            vec![("column", json!(col)), ("role", json!(role_val)), ("valid_roles", json!(VALID_ROLES))],
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

// ---- impute ----------------------------------------------------------------

fn invalid_method(label: &str, method: &Value, col: &str) -> PlanEditError {
    err(
        format!(
            "Invalid {label} {} for column {}.",
            py_repr_value(method),
            py_repr(col)
        ),
        vec![
            ("column", json!(col)),
            ("impute_method", method.clone()),
            ("valid_methods", json!(VALID_IMPUTE_METHODS)),
        ],
    )
}

fn validate_single_impute(
    imp: &mut ImputeSpec,
    encoded: bool,
    col: &str,
    ev: Option<&ColumnEvidence>,
) -> Result<(), PlanEditError> {
    let kind = ev.map(|e| e.kind);
    let text_mode = kind.is_some_and(|k| {
        matches!(k, ColumnKind::Text | ColumnKind::DatetimeLike)
            || (k == ColumnKind::NumericText && encoded)
    });
    let numeric_mode = kind.is_some_and(|k| {
        matches!(k, ColumnKind::Numeric | ColumnKind::Boolean)
            || (k == ColumnKind::NumericText && !encoded)
    });
    let kind_str = kind.map_or("", ColumnKind::as_str);
    let method = imp.method.clone();

    if method == "fill_value" {
        let Some(val) = imp.value.clone().filter(|v| !v.is_null()) else {
            return Err(err(
                format!(
                    "impute_method 'fill_value' for column {} requires 'impute_value'.",
                    py_repr(col)
                ),
                vec![("column", json!(col))],
            ));
        };
        if val.as_f64().is_some_and(|f| !f.is_finite()) {
            return Err(err(
                format!(
                    "impute_value for column {} must be finite, got {}.",
                    py_repr(col),
                    py_repr_value(&val)
                ),
                vec![
                    ("column", json!(col)),
                    ("impute_value", json!(py_str(&val))),
                ],
            ));
        }
        if ev.is_some() {
            if numeric_mode {
                if !(val.is_number() || val.is_boolean()) {
                    return Err(err(
                        format!(
                            "Column {} is {kind_str}; impute_value for 'fill_value' must be numeric, got {}.",
                            py_repr(col),
                            py_repr_value(&val)
                        ),
                        vec![("column", json!(col)), ("kind", json!(kind_str)), ("impute_value", val)],
                    ));
                }
            } else if text_mode && val.is_number() {
                imp.value = Some(json!(py_str(&val)));
            }
        }
    } else if imp.value.is_some() {
        return Err(err(
            format!(
                "impute_value is only valid with impute_method='fill_value' (column {}).",
                py_repr(col)
            ),
            vec![("column", json!(col)), ("impute_method", json!(method))],
        ));
    }
    if ev.is_some() {
        if text_mode
            && matches!(
                method.as_str(),
                "fill_median" | "fill_mean" | "sample_normal"
            )
        {
            return Err(err(
                format!(
                    "Column {} is {kind_str}; numeric impute_method {} cannot be used on text.",
                    py_repr(col),
                    py_repr(&method)
                ),
                vec![
                    ("column", json!(col)),
                    ("kind", json!(kind_str)),
                    ("impute_method", json!(method)),
                ],
            ));
        }
        if numeric_mode && matches!(method.as_str(), "fill_mode" | "sample_categorical") {
            return Err(err(
                format!(
                    "Column {} is {kind_str}; categorical impute_method {} cannot be used on numbers.",
                    py_repr(col),
                    py_repr(&method)
                ),
                vec![("column", json!(col)), ("kind", json!(kind_str)), ("impute_method", json!(method))],
            ));
        }
    }
    Ok(())
}

fn validate_impute_spec(
    entry: &mut PlanEntry,
    col: &str,
    ev: Option<&ColumnEvidence>,
) -> Result<(), PlanEditError> {
    let encoded = entry.encode.is_some();
    if let Some(imp) = entry.impute.as_mut() {
        validate_single_impute(imp, encoded, col, ev)?;
    } else {
        return Ok(());
    }
    if let Some(cands) = entry.impute_candidates.as_mut() {
        for cand in cands {
            validate_single_impute(cand, encoded, col, ev)?;
        }
    }
    Ok(())
}

/// One impute mapping (or `null`/`"none"` for no imputation).
fn parse_impute_mapping(raw: &Value, col: &str) -> Result<Option<ImputeSpec>, PlanEditError> {
    if raw.is_null() {
        return Ok(None);
    }
    let Some(map) = raw.as_object() else {
        return Err(err(
            format!(
                "impute for column {} must be a mapping, list of mappings, or null.",
                py_repr(col)
            ),
            vec![("column", json!(col)), ("impute", raw.clone())],
        ));
    };
    let method = map.get("method").filter(|m| !m.is_null());
    let Some(method) = method else {
        return Ok(None);
    };
    if method == "none" {
        return Ok(None);
    }
    let method_str = method.as_str().filter(|m| VALID_IMPUTE_METHODS.contains(m));
    let Some(method_str) = method_str else {
        return Err(invalid_method("impute method", method, col));
    };
    let mut spec = ImputeSpec::method(method_str);
    if let Some(v) = map.get("value").filter(|v| !v.is_null()) {
        if v.is_boolean() {
            return Err(err(
                format!(
                    "impute_value for column {} cannot be a boolean.",
                    py_repr(col)
                ),
                vec![("column", json!(col)), ("impute_value", v.clone())],
            ));
        }
        spec.value = Some(v.clone());
    }
    if let Some(seed) = map.get("seed").filter(|v| !v.is_null()) {
        match seed.as_i64() {
            Some(i) if !seed.is_boolean() => spec.seed = Some(i),
            _ => {
                return Err(err(
                    format!(
                        "impute seed for column {} must be an integer.",
                        py_repr(col)
                    ),
                    vec![("column", json!(col)), ("seed", seed.clone())],
                ))
            }
        }
    }
    if let Some(ind) = map.get("indicator") {
        match ind.as_bool() {
            Some(b) => spec.indicator = Some(b),
            None => {
                return Err(err(
                    format!(
                        "Impute candidate indicator for column {} must be a boolean.",
                        py_repr(col)
                    ),
                    vec![("column", json!(col)), ("indicator", ind.clone())],
                ))
            }
        }
    }
    Ok(Some(spec))
}

type Parsed = (Option<ImputeSpec>, Option<Vec<ImputeSpec>>);

fn collapse(mut candidates: Vec<ImputeSpec>) -> Parsed {
    let first = candidates.first().cloned();
    let many = (candidates.len() > 1).then(|| std::mem::take(&mut candidates));
    (first, many)
}

/// `impute:` as a mapping, a list of mappings, or a mapping whose `method`
/// is a list.
fn parse_impute_edit_value(raw: &Value, col: &str) -> Result<Parsed, PlanEditError> {
    if let Some(items) = raw.as_array() {
        if items.is_empty() {
            return Err(err(
                format!(
                    "impute candidate list for column {} must not be empty.",
                    py_repr(col)
                ),
                vec![("column", json!(col))],
            ));
        }
        let mut candidates = Vec::new();
        for item in items {
            match parse_impute_mapping(item, col)? {
                Some(spec) => candidates.push(spec),
                None => {
                    return Err(err(
                        format!(
                            "impute candidate list for column {} cannot contain null entries.",
                            py_repr(col)
                        ),
                        vec![("column", json!(col)), ("impute", raw.clone())],
                    ))
                }
            }
        }
        return Ok(collapse(candidates));
    }
    if let Some(methods) = raw.get("method").and_then(Value::as_array) {
        if methods.is_empty() {
            return Err(err(
                format!(
                    "impute method list for column {} must not be empty.",
                    py_repr(col)
                ),
                vec![("column", json!(col))],
            ));
        }
        let mut candidates = Vec::new();
        for m in methods {
            let mut spec = raw.as_object().cloned().unwrap_or_default();
            spec.insert("method".into(), m.clone());
            if m != "fill_value" {
                spec.shift_remove("value");
            }
            match parse_impute_mapping(&Value::Object(spec), col)? {
                Some(s) => candidates.push(s),
                None => {
                    return Err(err(
                        format!(
                            "impute method list for column {} cannot contain 'none'.",
                            py_repr(col)
                        ),
                        vec![("column", json!(col))],
                    ))
                }
            }
        }
        return Ok(collapse(candidates));
    }
    Ok((parse_impute_mapping(raw, col)?, None))
}

fn edit_impute(
    entry: &mut PlanEntry,
    edit: &Map<String, Value>,
    col: &str,
    ev: Option<&ColumnEvidence>,
) -> Result<bool, PlanEditError> {
    let mut has_impute_edit = false;
    if let Some(raw) = edit.get("impute") {
        has_impute_edit = true;
        (entry.impute, entry.impute_candidates) = parse_impute_edit_value(raw, col)?;
    }

    if let Some(method) = edit.get("impute_method") {
        has_impute_edit = true;
        let prev_val = entry.impute.as_ref().and_then(|i| i.value.clone());
        let with_prev = |m: &str| {
            let mut spec = ImputeSpec::method(m);
            if m == "fill_value" {
                spec.value = prev_val.clone();
            }
            spec
        };
        if method.is_null() || method == "none" || method == "" {
            entry.impute = None;
            entry.impute_candidates = None;
        } else if let Some(list) = method.as_array() {
            if list.is_empty() {
                return Err(err(
                    format!(
                        "impute_method list for column {} must not be empty.",
                        py_repr(col)
                    ),
                    vec![("column", json!(col))],
                ));
            }
            let mut candidates = Vec::new();
            for m in list {
                let m_str = py_str(m);
                if !VALID_IMPUTE_METHODS.contains(&m_str.as_str()) {
                    return Err(invalid_method("impute_method", &json!(m_str), col));
                }
                candidates.push(with_prev(&m_str));
            }
            (entry.impute, entry.impute_candidates) = collapse(candidates);
        } else {
            let m_str = py_str(method);
            if !VALID_IMPUTE_METHODS.contains(&m_str.as_str()) {
                return Err(invalid_method("impute_method", &json!(m_str), col));
            }
            entry.impute = Some(with_prev(&m_str));
            entry.impute_candidates = None;
        }
    }

    if let Some(val) = edit.get("impute_value").filter(|v| !v.is_null()) {
        has_impute_edit = true;
        if val.is_boolean() {
            return Err(err(
                format!(
                    "impute_value for column {} cannot be a boolean.",
                    py_repr(col)
                ),
                vec![("column", json!(col)), ("impute_value", val.clone())],
            ));
        }
        if entry.impute.is_none() {
            let mut spec = ImputeSpec::method("fill_value");
            spec.value = Some(val.clone());
            entry.impute = Some(spec);
        }
        let only_one = [entry.impute.as_mut().expect("set above")];
        let mut specs: Vec<&mut ImputeSpec> = match entry.impute_candidates.as_mut() {
            Some(c) => c.iter_mut().collect(),
            None => only_one.into_iter().collect(),
        };
        let mut constants = 0;
        for spec in specs.iter_mut().filter(|s| s.method == "fill_value") {
            spec.value = Some(val.clone());
            constants += 1;
        }
        if constants == 0 {
            return Err(err(
                format!(
                    "impute_value is only valid with impute_method='fill_value' (column {}).",
                    py_repr(col)
                ),
                vec![("column", json!(col))],
            ));
        }
        sync(entry);
    }

    if has_impute_edit && entry.impute.is_some() {
        if let Some(ev) = ev {
            let mixed_unencoded = ev.kind == ColumnKind::NumericText
                && ev.n_non_numeric.unwrap_or(0) > 0
                && entry.encode.is_none();
            if ev.n_missing == 0 && !mixed_unencoded {
                return Err(err(
                    format!(
                        "Column {} has no blanks; impute is not applicable.",
                        py_repr(col)
                    ),
                    vec![("column", json!(col)), ("n_missing", json!(0))],
                ));
            }
        }
    }

    validate_impute_spec(entry, col, ev)?;
    sync(entry);
    Ok(has_impute_edit)
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

// ---- one edit ----------------------------------------------------------------

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
    let mut unknown: Vec<&String> = edit
        .keys()
        .filter(|k| !EDIT_ALLOWED_KEYS.contains(&k.as_str()))
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        let col_repr = py_repr_value(edit.get("column").unwrap_or(&Value::Null));
        return Err(err(
            format!(
                "Unsupported edit field(s) for column {col_repr}: {}.",
                py_repr_list(&unknown)
            ),
            vec![
                ("column", edit.get("column").cloned().unwrap_or(Value::Null)),
                ("unsupported_fields", json!(unknown)),
                ("allowed_fields", json!(EDIT_ALLOWED_KEYS)),
            ],
        ));
    }
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
    if entry.role != Role::Feature {
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
    entry.decided_by = source;
    entry.status = EntryStatus::Resolved;
    Ok(())
}

// ---- answers -----------------------------------------------------------------

/// Combine compatible treatments chosen by several options of one question
/// without losing the candidates' indicators.
fn merge_answer_entries(col: &str, entries: Vec<PlanEntry>) -> Result<PlanEntry, PlanEditError> {
    let mut group_indicators: Vec<&String> = Vec::new();
    for e in &entries {
        if e.indicator.starts_with("group:") && !group_indicators.contains(&&e.indicator) {
            group_indicators.push(&e.indicator);
        }
    }
    let has_own = entries.iter().any(|e| e.indicator == "own");
    let primary = &entries[0];
    let incompatible = group_indicators.len() > 1
        || (!group_indicators.is_empty() && has_own)
        || entries[1..].iter().any(|e| {
            e.role != primary.role
                || e.encode != primary.encode
                || e.impute.is_none() != primary.impute.is_none()
        });
    if incompatible {
        return Err(err(
            format!(
                "Selected options for column {} have incompatible roles, encoding, or missingness groups.",
                py_repr(col)
            ),
            vec![("column", json!(col))],
        ));
    }
    let group_ind = group_indicators.first().map(|s| (*s).clone());
    let mut primary = entries[0].clone();
    if primary.impute.is_none() {
        return Ok(primary);
    }
    let mut candidates: Vec<ImputeSpec> = Vec::new();
    for entry in &entries {
        let specs: Vec<&ImputeSpec> = match &entry.impute_candidates {
            Some(c) => c.iter().collect(),
            None => entry.impute.iter().collect(),
        };
        for raw in specs {
            let mut spec = raw.clone();
            if spec.indicator.is_none() {
                spec.indicator = Some(group_ind.is_none() && entry.indicator == "own");
            }
            if group_ind.is_some() {
                spec.indicator = Some(false);
            }
            if !candidates.contains(&spec) {
                candidates.push(spec);
            }
        }
    }
    primary.impute = candidates.first().cloned();
    primary.impute_candidates = (candidates.len() > 1).then(|| candidates.clone());
    primary.indicator = match group_ind {
        Some(g) => g,
        None if candidates.iter().any(|s| s.indicator == Some(true)) => "own".into(),
        None => "none".into(),
    };
    Ok(primary)
}

fn apply_single_answer(
    ans: &Value,
    questions: &mut IndexMap<String, PlanQuestion>,
    entries: &mut Entries,
    incoming: &HashMap<String, Role>,
    source: DecidedBy,
    groups: &Groups,
    evidence: &Evidence,
) -> Result<(), PlanEditError> {
    let Some(ans) = ans.as_object() else {
        return Err(err(
            "Each answer must be a mapping with 'question_id' and 'option_id'.".into(),
            vec![("answer", ans.clone())],
        ));
    };
    let qid_value = ans.get("question_id").cloned().unwrap_or(Value::Null);
    let raw_oid = ans
        .get("option_ids")
        .or_else(|| ans.get("option_id"))
        .cloned()
        .unwrap_or(Value::Null);
    let qid = match qid_value.as_str() {
        Some(q) if questions.contains_key(q) => q.to_string(),
        _ => {
            return Err(err(
                format!("Unknown question_id: {}.", py_repr_value(&qid_value)),
                vec![
                    ("question_id", qid_value),
                    (
                        "known_question_ids",
                        json!(questions.keys().collect::<Vec<_>>()),
                    ),
                ],
            ))
        }
    };
    let q = questions[&qid].clone();
    let valid_options: Vec<&String> = q.options.iter().map(|o| &o.option_id).collect();
    let option_ids: Vec<Value> = match raw_oid {
        Value::Array(items) => items,
        other => vec![other],
    };
    if option_ids.is_empty() {
        return Err(err(
            format!(
                "option_id list for question {} must not be empty.",
                py_repr(&qid)
            ),
            vec![
                ("question_id", json!(qid)),
                ("valid_options", json!(valid_options)),
            ],
        ));
    }
    let mut matched: Vec<&QuestionOption> = Vec::new();
    for oid in &option_ids {
        match q.options.iter().find(|o| oid == o.option_id.as_str()) {
            Some(opt) => matched.push(opt),
            None => {
                return Err(err(
                    format!(
                        "Unknown option_id {} for question {}.",
                        py_repr_value(oid),
                        py_repr(&qid)
                    ),
                    vec![
                        ("question_id", json!(qid)),
                        ("topic", json!(q.topic)),
                        ("option_id", oid.clone()),
                        ("valid_options", json!(valid_options)),
                    ],
                ))
            }
        }
    }

    let role_of = |c: &str, entries: &Entries| incoming.get(c).copied().unwrap_or(entries[c].role);
    let effective: Vec<(String, Role)> = q
        .applies_to
        .iter()
        .filter(|c| entries.contains_key(c.as_str()))
        .map(|c| (c.clone(), role_of(c, entries)))
        .collect();
    let all_protected = !effective.is_empty() && effective.iter().all(|(_, r)| is_protected(*r));
    if q.status != QuestionStatus::Open || all_protected {
        let describe = |pairs: Vec<(String, Role)>| {
            pairs
                .iter()
                .map(|(c, r)| format!("{c}={}", r.as_str()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut roles_str = describe(effective.clone());
        if roles_str.is_empty() {
            roles_str = describe(
                entries
                    .iter()
                    .map(|(c, e)| (c.clone(), incoming.get(c).copied().unwrap_or(e.role)))
                    .filter(|(_, r)| is_protected(*r))
                    .collect(),
            );
        }
        if roles_str.is_empty() {
            roles_str = "resolved".into();
        }
        let columns: Map<String, Value> = effective
            .iter()
            .map(|(c, r)| (c.clone(), json!(r.as_str())))
            .collect();
        return Err(err(
            format!("question {qid} no longer applies (columns now: {roles_str})"),
            vec![
                ("question_id", json!(qid)),
                ("topic", json!(q.topic)),
                ("status", json!(q.status)),
                ("columns", Value::Object(columns)),
            ],
        ));
    }

    let multiple = matched.len() > 1;
    let mut by_col: IndexMap<String, Vec<PlanEntry>> = IndexMap::new();
    for opt in &matched {
        // Every selected option starts from the same revision, so one option's
        // edits cannot become another option's implicit defaults.
        let mut scratch = multiple.then(|| entries.clone());
        let mut edited: Vec<String> = Vec::new();
        for edit_spec in &opt.edits {
            let col_name = edit_spec.get("column").and_then(Value::as_str);
            if let Some(c) = col_name.filter(|c| entries.contains_key(*c)) {
                if is_protected(incoming.get(c).copied().unwrap_or(entries[c].role)) {
                    continue;
                }
            }
            let target = scratch.as_mut().unwrap_or(&mut *entries);
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
        if let Some(scratch) = scratch {
            for c in edited {
                by_col
                    .entry(c.clone())
                    .or_default()
                    .push(scratch[&c].clone());
            }
        }
    }
    for (col, versions) in by_col {
        entries.insert(col.clone(), merge_answer_entries(&col, versions)?);
    }

    for col in &q.applies_to {
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

/// Questions whose columns were all resolved by direct edits are answered.
fn sync_question_statuses(
    questions: &mut IndexMap<String, PlanQuestion>,
    entries: &Entries,
    target_role_added: bool,
    source: DecidedBy,
) {
    for q in questions
        .values_mut()
        .filter(|q| q.status == QuestionStatus::Open)
    {
        let by_target = q.topic == super::model::QuestionTopic::OutcomeColumns && target_role_added;
        let by_edits = !by_target
            && !q.applies_to.is_empty()
            && q.applies_to
                .iter()
                .filter_map(|c| entries.get(c))
                .all(|e| e.status == EntryStatus::Resolved && e.decided_by != DecidedBy::Default);
        if by_target || by_edits {
            q.status = QuestionStatus::Answered;
            q.answer = Some(json!({"option_id": "custom_edit", "decided_by": source_str(source)}));
        }
    }
}

/// Apply question `answers` and column `edits`, returning a new revision.
/// The input plan is never modified, including when this fails.
pub fn apply_answers_and_edits(
    plan: &PreprocessingPlan,
    answers: &[Value],
    edits: &[Value],
    source: DecidedBy,
) -> Result<PreprocessingPlan, PlanEditError> {
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

    let evidence: Evidence = plan
        .evidence
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();
    let mut entries = plan.entries.clone();
    let mut questions = plan.questions.clone();

    // Columns explicitly given a protected role by this call's edits.
    let incoming: HashMap<String, Role> = edits
        .iter()
        .filter_map(|e| {
            let col = e.get("column")?.as_str()?;
            let role = parse_role(e.get("role")?.as_str()?).filter(|r| is_protected(*r))?;
            Some((col.to_string(), role))
        })
        .collect();

    for ans in answers {
        apply_single_answer(
            ans,
            &mut questions,
            &mut entries,
            &incoming,
            source,
            &plan.missing_groups,
            &evidence,
        )?;
    }
    let mut target_role_added = false;
    for edit in edits {
        apply_single_edit(&mut entries, edit, source, &plan.missing_groups, &evidence)?;
        if edit.get("role").is_some_and(|r| r == "target") {
            target_role_added = true;
        }
    }
    sync_question_statuses(&mut questions, &entries, target_role_added, source);

    Ok(PreprocessingPlan {
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
    })
}

// ---- diff --------------------------------------------------------------------

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
