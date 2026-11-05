//! Parsing and validation of imputation specs on plan entries.

use serde_json::{json, Map, Value};

use super::super::model::{ImputeSpec, PlanEditError, PlanEntry};
use super::super::text::{py_repr, py_repr_value, py_str};
use super::{err, sync, VALID_IMPUTE_METHODS};
use crate::evidence::{ColumnEvidence, ColumnKind};

type Parsed = (Option<ImputeSpec>, Option<Vec<ImputeSpec>>);

fn col_err(col: &str, message: String, mut extra: Vec<(&str, Value)>) -> PlanEditError {
    let mut details = vec![("column", json!(col))];
    details.append(&mut extra);
    err(message, details)
}

fn invalid_method(label: &str, method: &Value, col: &str) -> PlanEditError {
    col_err(
        col,
        format!(
            "Invalid {label} {} for column {}.",
            py_repr_value(method),
            py_repr(col)
        ),
        vec![
            ("impute_method", method.clone()),
            ("valid_methods", json!(VALID_IMPUTE_METHODS)),
        ],
    )
}

fn ensure_non_empty(items: &[Value], label: &str, col: &str) -> Result<(), PlanEditError> {
    if items.is_empty() {
        return Err(col_err(
            col,
            format!(
                "{label} list for column {} must not be empty.",
                py_repr(col)
            ),
            vec![],
        ));
    }
    Ok(())
}

fn reject_bool_value(val: &Value, col: &str) -> Result<(), PlanEditError> {
    if val.is_boolean() {
        return Err(col_err(
            col,
            format!(
                "impute_value for column {} cannot be a boolean.",
                py_repr(col)
            ),
            vec![("impute_value", val.clone())],
        ));
    }
    Ok(())
}

fn is_text_mode(kind: ColumnKind, encoded: bool) -> bool {
    matches!(kind, ColumnKind::Text | ColumnKind::DatetimeLike)
        || (kind == ColumnKind::NumericText && encoded)
}

fn is_numeric_mode(kind: ColumnKind, encoded: bool) -> bool {
    matches!(kind, ColumnKind::Numeric | ColumnKind::Boolean)
        || (kind == ColumnKind::NumericText && !encoded)
}

fn validate_fill_value_kind(
    imp: &mut ImputeSpec,
    val: Value,
    encoded: bool,
    col: &str,
    ev: &ColumnEvidence,
) -> Result<(), PlanEditError> {
    let kind_str = ev.kind.as_str();
    if is_numeric_mode(ev.kind, encoded) {
        if !(val.is_number() || val.is_boolean()) {
            return Err(col_err(
                col,
                format!(
                    "Column {} is {kind_str}; impute_value for 'fill_value' must be numeric, got {}.",
                    py_repr(col),
                    py_repr_value(&val)
                ),
                vec![("kind", json!(kind_str)), ("impute_value", val)],
            ));
        }
    } else if is_text_mode(ev.kind, encoded) && val.is_number() {
        imp.value = Some(json!(py_str(&val)));
    }
    Ok(())
}

fn validate_fill_value(
    imp: &mut ImputeSpec,
    encoded: bool,
    col: &str,
    ev: Option<&ColumnEvidence>,
) -> Result<(), PlanEditError> {
    let Some(val) = imp.value.clone().filter(|v| !v.is_null()) else {
        return Err(col_err(
            col,
            format!(
                "impute_method 'fill_value' for column {} requires 'impute_value'.",
                py_repr(col)
            ),
            vec![],
        ));
    };
    if val.as_f64().is_some_and(|f| !f.is_finite()) {
        return Err(col_err(
            col,
            format!(
                "impute_value for column {} must be finite, got {}.",
                py_repr(col),
                py_repr_value(&val)
            ),
            vec![("impute_value", json!(py_str(&val)))],
        ));
    }
    if let Some(ev) = ev {
        validate_fill_value_kind(imp, val, encoded, col, ev)?;
    }
    Ok(())
}

fn validate_method_for_kind(
    method: &str,
    encoded: bool,
    col: &str,
    ev: &ColumnEvidence,
) -> Result<(), PlanEditError> {
    let kind_str = ev.kind.as_str();
    let bad_target = if is_text_mode(ev.kind, encoded)
        && matches!(method, "fill_median" | "fill_mean" | "sample_normal")
    {
        Some(("numeric", "text"))
    } else if is_numeric_mode(ev.kind, encoded)
        && matches!(method, "fill_mode" | "sample_categorical")
    {
        Some(("categorical", "numbers"))
    } else {
        None
    };
    if let Some((family, domain)) = bad_target {
        return Err(col_err(
            col,
            format!(
                "Column {} is {kind_str}; {family} impute_method {} cannot be used on {domain}.",
                py_repr(col),
                py_repr(method)
            ),
            vec![("kind", json!(kind_str)), ("impute_method", json!(method))],
        ));
    }
    Ok(())
}

pub(super) fn validate_single_impute(
    imp: &mut ImputeSpec,
    encoded: bool,
    col: &str,
    ev: Option<&ColumnEvidence>,
) -> Result<(), PlanEditError> {
    let method = imp.method.clone();
    if method == "fill_value" {
        validate_fill_value(imp, encoded, col, ev)?;
    } else if imp.value.is_some() {
        return Err(col_err(
            col,
            format!(
                "impute_value is only valid with impute_method='fill_value' (column {}).",
                py_repr(col)
            ),
            vec![("impute_method", json!(method))],
        ));
    }
    if let Some(ev) = ev {
        validate_method_for_kind(&method, encoded, col, ev)?;
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

fn parse_impute_fields(
    spec: &mut ImputeSpec,
    map: &Map<String, Value>,
    col: &str,
) -> Result<(), PlanEditError> {
    if let Some(v) = map.get("value").filter(|v| !v.is_null()) {
        reject_bool_value(v, col)?;
        spec.value = Some(v.clone());
    }
    if let Some(seed) = map.get("seed").filter(|v| !v.is_null()) {
        match seed.as_i64() {
            Some(i) if !seed.is_boolean() => spec.seed = Some(i),
            _ => {
                return Err(col_err(
                    col,
                    format!(
                        "impute seed for column {} must be an integer.",
                        py_repr(col)
                    ),
                    vec![("seed", seed.clone())],
                ))
            }
        }
    }
    if let Some(ind) = map.get("indicator") {
        match ind.as_bool() {
            Some(b) => spec.indicator = Some(b),
            None => {
                return Err(col_err(
                    col,
                    format!(
                        "Impute candidate indicator for column {} must be a boolean.",
                        py_repr(col)
                    ),
                    vec![("indicator", ind.clone())],
                ))
            }
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
        return Err(col_err(
            col,
            format!(
                "impute for column {} must be a mapping, list of mappings, or null.",
                py_repr(col)
            ),
            vec![("impute", raw.clone())],
        ));
    };
    let Some(method) = map.get("method").filter(|m| !m.is_null()) else {
        return Ok(None);
    };
    if method == "none" {
        return Ok(None);
    }
    let Some(method_str) = method.as_str().filter(|m| VALID_IMPUTE_METHODS.contains(m)) else {
        return Err(invalid_method("impute method", method, col));
    };
    let mut spec = ImputeSpec::method(method_str);
    parse_impute_fields(&mut spec, map, col)?;
    Ok(Some(spec))
}

fn collapse(mut candidates: Vec<ImputeSpec>) -> Parsed {
    let first = candidates.first().cloned();
    let many = (candidates.len() > 1).then(|| std::mem::take(&mut candidates));
    (first, many)
}

fn parse_impute_array(items: &[Value], raw: &Value, col: &str) -> Result<Parsed, PlanEditError> {
    ensure_non_empty(items, "impute candidate", col)?;
    let mut candidates = Vec::new();
    for item in items {
        let Some(spec) = parse_impute_mapping(item, col)? else {
            return Err(col_err(
                col,
                format!(
                    "impute candidate list for column {} cannot contain null entries.",
                    py_repr(col)
                ),
                vec![("impute", raw.clone())],
            ));
        };
        candidates.push(spec);
    }
    Ok(collapse(candidates))
}

fn parse_impute_method_array(
    methods: &[Value],
    raw: &Value,
    col: &str,
) -> Result<Parsed, PlanEditError> {
    ensure_non_empty(methods, "impute method", col)?;
    let mut candidates = Vec::new();
    for m in methods {
        let mut spec = raw.as_object().cloned().unwrap_or_default();
        spec.insert("method".into(), m.clone());
        if m != "fill_value" {
            spec.shift_remove("value");
        }
        let Some(s) = parse_impute_mapping(&Value::Object(spec), col)? else {
            return Err(col_err(
                col,
                format!(
                    "impute method list for column {} cannot contain 'none'.",
                    py_repr(col)
                ),
                vec![],
            ));
        };
        candidates.push(s);
    }
    Ok(collapse(candidates))
}

/// `impute:` as a mapping, a list of mappings, or a mapping whose `method`
/// is a list.
fn parse_impute_edit_value(raw: &Value, col: &str) -> Result<Parsed, PlanEditError> {
    if let Some(items) = raw.as_array() {
        return parse_impute_array(items, raw, col);
    }
    if let Some(methods) = raw.get("method").and_then(Value::as_array) {
        return parse_impute_method_array(methods, raw, col);
    }
    Ok((parse_impute_mapping(raw, col)?, None))
}

fn spec_with_prev(method: &str, prev_val: &Option<Value>) -> ImputeSpec {
    let mut spec = ImputeSpec::method(method);
    if method == "fill_value" {
        spec.value.clone_from(prev_val);
    }
    spec
}

fn parse_method_list(
    list: &[Value],
    prev_val: &Option<Value>,
    col: &str,
) -> Result<Parsed, PlanEditError> {
    ensure_non_empty(list, "impute_method", col)?;
    let mut candidates = Vec::new();
    for m in list {
        let m_str = py_str(m);
        if !VALID_IMPUTE_METHODS.contains(&m_str.as_str()) {
            return Err(invalid_method("impute_method", &json!(m_str), col));
        }
        candidates.push(spec_with_prev(&m_str, prev_val));
    }
    Ok(collapse(candidates))
}

fn apply_impute_method_edit(
    entry: &mut PlanEntry,
    method: &Value,
    col: &str,
) -> Result<(), PlanEditError> {
    if method.is_null() || method == "none" || method == "" {
        entry.impute = None;
        entry.impute_candidates = None;
        return Ok(());
    }
    let prev_val = entry.impute.as_ref().and_then(|i| i.value.clone());
    if let Some(list) = method.as_array() {
        (entry.impute, entry.impute_candidates) = parse_method_list(list, &prev_val, col)?;
        return Ok(());
    }
    let m_str = py_str(method);
    if !VALID_IMPUTE_METHODS.contains(&m_str.as_str()) {
        return Err(invalid_method("impute_method", &json!(m_str), col));
    }
    entry.impute = Some(spec_with_prev(&m_str, &prev_val));
    entry.impute_candidates = None;
    Ok(())
}

fn apply_impute_value_edit(
    entry: &mut PlanEntry,
    val: &Value,
    col: &str,
) -> Result<(), PlanEditError> {
    reject_bool_value(val, col)?;
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
        return Err(col_err(
            col,
            format!(
                "impute_value is only valid with impute_method='fill_value' (column {}).",
                py_repr(col)
            ),
            vec![],
        ));
    }
    sync(entry);
    Ok(())
}

fn check_column_has_blanks(
    entry: &PlanEntry,
    col: &str,
    ev: Option<&ColumnEvidence>,
) -> Result<(), PlanEditError> {
    let Some(ev) = ev else {
        return Ok(());
    };
    let mixed_unencoded = ev.kind == ColumnKind::NumericText
        && ev.n_non_numeric.unwrap_or(0) > 0
        && entry.encode.is_none();
    if ev.n_missing == 0 && !mixed_unencoded {
        return Err(col_err(
            col,
            format!(
                "Column {} has no blanks; impute is not applicable.",
                py_repr(col)
            ),
            vec![("n_missing", json!(0))],
        ));
    }
    Ok(())
}

pub(super) fn edit_impute(
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
        apply_impute_method_edit(entry, method, col)?;
    }
    if let Some(val) = edit.get("impute_value").filter(|v| !v.is_null()) {
        has_impute_edit = true;
        apply_impute_value_edit(entry, val, col)?;
    }
    if has_impute_edit && entry.impute.is_some() {
        check_column_has_blanks(entry, col, ev)?;
    }
    validate_impute_spec(entry, col, ev)?;
    sync(entry);
    Ok(has_impute_edit)
}
