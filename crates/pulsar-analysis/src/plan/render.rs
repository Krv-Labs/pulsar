//! Plan → `preprocessing:` config block, fingerprint, marker legend and
//! counts (`preprocessing_plan/render.py`).
//!
//! The YAML text is rendered by `serde_yaml_ng`, not PyYAML, so its quoting
//! style can differ from Python's; the two are semantically identical
//! (parity compares them parsed). The fingerprint does not depend on YAML: it
//! hashes Python's canonical `json.dumps(sort_keys=True, separators=(",",":"))`,
//! reproduced exactly (ASCII escapes, float repr).

use std::collections::HashSet;
use std::fmt::Write as _;

use indexmap::IndexMap;
use pulsar_data::preprocess::py_float_repr;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::model::{DecidedBy, EntryStatus, ImputeSpec, Role};
use super::plan::PreprocessingPlan;

/// `[(internal group id, readable yaml key, referencing columns)]`.
///
/// Only groups referenced by at least one feature (or a dropped column that
/// keeps a blank marker) are emitted, listing only the referencing columns.
fn group_references(plan: &PreprocessingPlan) -> Vec<(String, String, Vec<String>)> {
    let mut by_gid: IndexMap<String, Vec<String>> = IndexMap::new();
    for (col, entry) in &plan.entries {
        let Some(gid) = entry.indicator.strip_prefix("group:") else {
            continue;
        };
        if !matches!(entry.role, Role::Feature | Role::Drop) || gid.is_empty() {
            continue;
        }
        by_gid.entry(gid.to_string()).or_default().push(col.clone());
    }
    let mut used: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for (gid, cols) in by_gid {
        let base = if cols.len() == 1 {
            cols[0].clone()
        } else {
            format!("{}_and_{}_more", cols[0], cols.len() - 1)
        };
        let key = if used.contains(&base) {
            format!("{base}_{gid}")
        } else {
            base
        };
        used.insert(key.clone());
        out.push((gid, key, cols));
    }
    out
}

/// One entry per emitted blank-marker column: which columns it covers.
pub fn build_marker_legend(plan: &PreprocessingPlan, preview_limit: usize) -> Vec<Value> {
    let mut legend = Vec::new();
    let mut grouped: HashSet<String> = HashSet::new();
    for (_gid, key, cols) in group_references(plan) {
        grouped.extend(cols.iter().cloned());
        legend.push(json!({
            "marker_column": format!("{key}_was_missing"),
            "columns_preview": cols.iter().take(preview_limit).collect::<Vec<_>>(),
            "columns_count": cols.len(),
        }));
    }
    for (col, entry) in &plan.entries {
        let Some(impute) = &entry.impute else {
            continue;
        };
        if entry.role != Role::Feature || grouped.contains(col) {
            continue;
        }
        let own = entry.indicator == "own";
        let specs: Vec<&ImputeSpec> = match &entry.impute_candidates {
            Some(c) if !c.is_empty() => c.iter().collect(),
            _ => vec![impute],
        };
        if specs.iter().any(|s| s.indicator.unwrap_or(own)) {
            legend.push(json!({
                "marker_column": format!("{col}_was_missing"),
                "columns_preview": [col],
                "columns_count": 1,
            }));
        }
    }
    legend
}

fn format_impute_spec(spec: &ImputeSpec, indicator_mode: &str) -> Value {
    let mut out = Map::new();
    out.insert("method".into(), json!(spec.method));
    if let Some(v) = spec.value.as_ref().filter(|v| !v.is_null()) {
        out.insert("value".into(), v.clone());
    }
    if let Some(seed) = spec.seed {
        out.insert("seed".into(), json!(seed));
    }
    match spec.indicator {
        Some(i) => {
            out.insert("indicator".into(), json!(i));
        }
        None if indicator_mode != "own" => {
            out.insert("indicator".into(), json!(false));
        }
        None => {}
    }
    Value::Object(out)
}

/// The canonical `preprocessing:` dict the config loader consumes.
pub fn plan_to_preprocessing_dict(plan: &PreprocessingPlan) -> Map<String, Value> {
    let drop_columns: Vec<&String> = plan
        .entries
        .iter()
        .filter(|(_, e)| e.role != Role::Feature)
        .map(|(c, _)| c)
        .collect();

    let groups: Map<String, Value> = group_references(plan)
        .into_iter()
        .map(|(_, key, cols)| (key, json!(cols)))
        .collect();

    let (mut impute, mut encode) = (Map::new(), Map::new());
    for (col, entry) in &plan.entries {
        if entry.role != Role::Feature {
            continue;
        }
        match (&entry.impute_candidates, &entry.impute) {
            (Some(cands), _) if cands.len() > 1 => {
                let specs: Vec<Value> = cands
                    .iter()
                    .map(|c| format_impute_spec(c, &entry.indicator))
                    .collect();
                impute.insert(col.clone(), Value::Array(specs));
            }
            (_, Some(spec)) => {
                impute.insert(col.clone(), format_impute_spec(spec, &entry.indicator));
            }
            _ => {}
        }
        if let Some(method) = &entry.encode {
            encode.insert(col.clone(), json!({ "method": method }));
        }
    }

    let mut pre = Map::new();
    pre.insert("drop_columns".into(), json!(drop_columns));
    if plan.n_imputations > 1 {
        pre.insert("n_imputations".into(), json!(plan.n_imputations));
    }
    for (key, map) in [
        ("missing_indicator_groups", groups),
        ("impute", impute),
        ("encode", encode),
    ] {
        if !map.is_empty() {
            pre.insert(key.into(), Value::Object(map));
        }
    }
    pre
}

/// `json.dumps(value, sort_keys=True, separators=(",", ":"))` as Python
/// writes it: non-ASCII escaped as `\uXXXX` (surrogate pairs above the BMP),
/// floats by `repr`.
pub fn py_canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(
        value,
        &mut out,
        &Style {
            sort: true,
            item: ",",
            key: ":",
        },
    );
    out
}

/// `json.dumps(value)` with Python's defaults: insertion order, `", "` and
/// `": "` separators, ASCII-escaped.
pub fn py_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(
        value,
        &mut out,
        &Style {
            sort: false,
            item: ", ",
            key: ": ",
        },
    );
    out
}

struct Style {
    sort: bool,
    item: &'static str,
    key: &'static str,
}

fn write_canonical(value: &Value, out: &mut String, style: &Style) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => out.push_str(&i.to_string()),
            (_, Some(u)) => out.push_str(&u.to_string()),
            _ => out.push_str(&py_float_repr(n.as_f64().unwrap_or(f64::NAN))),
        },
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item);
                }
                write_canonical(item, out, style);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            if style.sort {
                keys.sort();
            }
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item);
                }
                write_string(key, out);
                out.push_str(style.key);
                write_canonical(&map[key.as_str()], out, style);
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c if (c as u32) > 0xFFFF => {
                let v = c as u32 - 0x10000;
                let _ = write!(
                    out,
                    "\\u{:04x}\\u{:04x}",
                    0xD800 + (v >> 10),
                    0xDC00 + (v & 0x3FF)
                );
            }
            c => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
        }
    }
    out.push('"');
}

/// 12 hex characters of the SHA-256 of the canonical preprocessing dict.
pub fn plan_fingerprint(plan: &PreprocessingPlan) -> String {
    let canonical = py_canonical_json(&Value::Object(plan_to_preprocessing_dict(plan)));
    let digest = Sha256::digest(canonical.as_bytes());
    digest.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

/// The `preprocessing:` YAML block for the plan, optionally with a header
/// naming the plan, revision and fingerprint.
pub fn plan_to_preprocessing_yaml(plan: &PreprocessingPlan, include_header: bool) -> String {
    let block = json!({ "preprocessing": plan_to_preprocessing_dict(plan) });
    let rendered = serde_yaml_ng::to_string(&block).expect("a JSON value serializes to YAML");
    let rendered = rendered.trim_end();
    if include_header {
        format!(
            "# preprocessing plan {} revision {} ({})\n{rendered}",
            plan.plan_id,
            plan.revision,
            plan_fingerprint(plan)
        )
    } else {
        rendered.to_string()
    }
}

/// Column, role and provenance counts for the plan header.
pub fn summarize_plan_counts(plan: &PreprocessingPlan) -> Value {
    let (mut default, mut agent, mut user) = (0, 0, 0);
    let (mut features, mut dropped, mut open_entries) = (0, 0, 0);
    for entry in plan.entries.values() {
        match entry.decided_by {
            DecidedBy::Default => default += 1,
            DecidedBy::Agent => agent += 1,
            DecidedBy::User => user += 1,
        }
        if entry.role == Role::Feature {
            features += 1;
        } else {
            dropped += 1;
        }
        if entry.status == EntryStatus::Open {
            open_entries += 1;
        }
    }
    let open_questions = plan
        .questions
        .values()
        .filter(|q| q.status == super::model::QuestionStatus::Open)
        .count();
    json!({
        "columns": plan.entries.len(),
        "feature_columns": features,
        "dropped": dropped,
        "open_entries": open_entries,
        "open_questions": open_questions,
        "decided_by": {"default": default, "agent": agent, "user": user},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_matches_pythons_ascii_escaping_and_ordering() {
        let v = json!({"b": 1, "a": ["é", "x\"y", "\u{1F600}", "\u{7f}", "\n"], "c": 0.5, "d": [true, null]});
        // Verified against CPython's json.dumps(..., sort_keys=True,
        // separators=(",", ":")). Built from a backslash so no escape
        // sequence appears literally in the source.
        let bs = |s: &str| format!("{}{s}", '\\');
        let expected = format!(
            r#"{{"a":["{}","x\"y","{}{}","{}","\n"],"b":1,"c":0.5,"d":[true,null]}}"#,
            bs("u00e9"),
            bs("ud83d"),
            bs("ude00"),
            bs("u007f"),
        );
        assert_eq!(py_canonical_json(&v), expected);
    }

    #[test]
    fn canonical_json_writes_floats_like_python_repr() {
        assert_eq!(py_canonical_json(&json!(1.0)), "1.0");
        assert_eq!(py_canonical_json(&json!(1e-5)), "1e-05");
        assert_eq!(py_canonical_json(&json!(1e16)), "1e+16");
        assert_eq!(py_canonical_json(&json!(0)), "0");
    }
}
