//! Wording helpers shared by plan defaults and questions
//! (`preprocessing_plan/text.py`) plus the Python formatting they lean on.

use std::collections::HashMap;

use indexmap::IndexMap;

use crate::evidence::{ColumnEvidence, ColumnKind};

/// `format(n, ",")`: thousands separators.
pub fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `repr(str)`: single quotes unless the text has a `'` and no `"`.
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr(list[str])`: `['a', 'b']`.
pub fn py_repr_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| py_repr(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

/// `repr(value)` for a JSON-shaped Python value: `None`, `True`, numbers,
/// quoted strings, `[...]` and `{...}` in insertion order.
pub fn py_repr_value(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (_, Some(u)) => u.to_string(),
            _ => pulsar_data::preprocess::py_float_repr(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => py_repr(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr_value).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_repr_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// `str(value)`: a string is itself, everything else its repr.
pub fn py_str(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => py_repr_value(other),
    }
}

/// `singular` when `n == 1`, else `plural`.
pub fn plural<'a>(n: usize, singular: &'a str, plural: &'a str) -> &'a str {
    if n == 1 {
        singular
    } else {
        plural
    }
}

/// A numeric count-like column with blanks and no recorded 0.
pub fn is_count_like_zero_free(ev: &ColumnEvidence) -> bool {
    ev.kind == ColumnKind::Numeric
        && ev.n_missing > 0
        && ev.integer_valued == Some(true)
        && ev.nonnegative == Some(true)
        && ev.n_zero == Some(0)
}

/// Column names in backticks, capped at `limit` names:
/// `` `a` ``, `` `a` and `b` ``, `` `a`, `b`, and `c` ``, `` `a`, `b`, and 2 more ``.
pub fn format_col_list<S: AsRef<str>>(cols: &[S], limit: usize) -> String {
    let items: Vec<String> = cols.iter().map(|c| format!("`{}`", c.as_ref())).collect();
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 if limit >= 2 => format!("{} and {}", items[0], items[1]),
        n if n <= limit => format!("{}, and {}", items[..n - 1].join(", "), items[n - 1]),
        n => format!("{}, and {} more", items[..limit].join(", "), n - limit),
    }
}

/// Why a column is blank, from its evidence.
pub fn format_missing_reason(
    ev: &ColumnEvidence,
    missing_groups: &IndexMap<String, Vec<String>>,
    ev_by_col: &HashMap<&str, &ColumnEvidence>,
) -> String {
    let mut parts = vec![format!(
        "{} {}",
        thousands(ev.n_missing),
        plural(ev.n_missing, "blank", "blanks")
    )];
    if ev.n_zero == Some(0) && ev.nonnegative == Some(true) {
        parts.push("no recorded 0".into());
    }
    if let Some(group) = ev
        .missing_group
        .as_ref()
        .and_then(|g| missing_groups.get(g))
    {
        let n_others = group.len().saturating_sub(1);
        if n_others > 0 {
            parts.push(format!(
                "blank on the same rows as {n_others} other {} ({})",
                plural(n_others, "column", "columns"),
                ev.missing_group.as_deref().unwrap_or_default()
            ));
        }
    }
    if let Some(md) = &ev.mask_dependency {
        let partner = ev_by_col.get(md.partner.as_str());
        if partner.is_none_or(|p| p.kind != ColumnKind::Boolean) {
            parts.push(format!(
                "blank on the exact {} where {}",
                plural(ev.n_missing, "row", "rows"),
                condition(&md.partner, md.relation.as_str())
            ));
        }
    }
    parts.join("; ")
}

/// `"<partner> is 0"` / `"<partner> is not 0"` for a mask relation.
pub fn condition(partner: &str, relation: &str) -> String {
    if relation == "missing_when_partner_zero" {
        format!("{partner} is 0")
    } else {
        format!("{partner} is not 0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_matches_python_format() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    #[test]
    fn py_repr_matches_python() {
        assert_eq!(py_repr("abc"), "'abc'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_repr("tab\there\\"), "'tab\\there\\\\'");
        assert_eq!(py_repr("\u{1}"), "'\\x01'");
        assert_eq!(py_repr("é"), "'é'");
        assert_eq!(py_repr_list(&["a", "b c"]), "['a', 'b c']");
    }

    #[test]
    fn col_list_wording() {
        let l = |v: &[&str]| format_col_list(v, 3);
        assert_eq!(l(&[]), "");
        assert_eq!(l(&["a"]), "`a`");
        assert_eq!(l(&["a", "b"]), "`a` and `b`");
        assert_eq!(l(&["a", "b", "c"]), "`a`, `b`, and `c`");
        assert_eq!(l(&["a", "b", "c", "d", "e"]), "`a`, `b`, `c`, and 2 more");
    }
}
