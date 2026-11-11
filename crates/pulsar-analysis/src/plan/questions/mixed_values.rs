//! `mixed_values`: numeric text columns that also hold non-numeric tokens.

use pulsar_data::preprocess::py_float_repr;
use pulsar_data::Column;
use serde_json::{json, Value};

use super::group_or;
use crate::evidence::kind::{strip_thousands, to_numeric};
use crate::plan::defaults::DraftContext;
use crate::plan::model::{edit, PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic};
use crate::plan::text::{format_col_list, plural, py_repr, thousands};

/// The one number all numeric tokens across `cols` share, if they all agree
/// (`(min, max)` per column must collapse to the same value).
fn single_numeric_value(ctx: &DraftContext, cols: &[String]) -> Option<f64> {
    let ranges: Vec<Option<(f64, f64)>> = cols
        .iter()
        .map(|c| {
            let Some(Column::Text(values)) = ctx.table.column(c) else {
                return None;
            };
            values
                .iter()
                .flatten()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .filter_map(|s| to_numeric(&strip_thousands(s)))
                .filter(|x| x.is_finite())
                .fold(None, |acc, x| match acc {
                    None => Some((x, x)),
                    Some((lo, hi)) => Some((lo.min(x), hi.max(x))),
                })
        })
        .collect();
    let first = ranges.first().copied().flatten()?;
    ranges
        .iter()
        .all(|r| r.is_some_and(|(lo, hi)| lo == hi && lo == first.0))
        .then_some(first.0)
}

struct Facts<'a> {
    cols: &'a [String],
    c_top: &'a str,
    top_val: &'a str,
    top_n: usize,
    total_non_num: usize,
    single_num: Option<f64>,
    mostly_num_all: bool,
}

fn question_text(f: &Facts) -> String {
    let ask = "Should those values count as their own category, or as unreadable entries treated like blanks?";
    let val = py_repr(f.top_val);
    let num = f.single_num.map(|v| {
        if v.fract() == 0.0 {
            format!("{}", v as i64)
        } else {
            py_float_repr(v)
        }
    });
    let n = f.cols.len();
    if n == 1 {
        let rows = format!("{} {}", thousands(f.top_n), plural(f.top_n, "row", "rows"));
        let lead = match (&num, f.mostly_num_all) {
            (Some(num), _) => format!("`{}` holds the number {num} and the value {val}", f.c_top),
            (None, true) => format!(
                "`{}` mostly holds numbers but also the value {val}",
                f.c_top
            ),
            (None, false) => format!("`{}` holds numbers and the value {val}", f.c_top),
        };
        return format!("{lead} ({rows}). {ask}");
    }
    let entries = format!(
        "{} {}",
        thousands(f.total_non_num),
        plural(f.total_non_num, "entry", "entries")
    );
    let e_g = format!("e.g. `{}`: {}", f.c_top, thousands(f.top_n));
    let lead = match (&num, f.mostly_num_all) {
        (Some(num), _) => format!("{n} columns hold the number {num} and the value {val}"),
        (None, true) => format!("{n} columns mostly hold numbers but also the value {val}"),
        (None, false) => format!("{n} columns hold numbers and the value {val}"),
    };
    format!("{lead} ({entries} in total; {e_g}). {ask}")
}

fn options(
    ctx: &DraftContext,
    cols: &[String],
    total_non_num: usize,
    top_val: &str,
) -> Vec<QuestionOption> {
    let entry_word = plural(total_non_num, "entry", "entries");
    let verb = if total_non_num == 1 {
        "becomes a blank and is estimated"
    } else {
        "become blanks and are estimated"
    };
    let as_categories = cols
        .iter()
        .map(|c| {
            let has_blanks = ctx.ev_by_col[c.as_str()].n_missing > 0;
            let mut e = edit([
                ("column", json!(c)),
                ("role", json!("feature")),
                (
                    "impute_method",
                    if has_blanks {
                        json!("fill_value")
                    } else {
                        Value::Null
                    },
                ),
                ("indicator", json!("none")),
                ("encode", json!("one_hot")),
            ]);
            if has_blanks {
                e.insert("impute_value".into(), json!("(blank)"));
            }
            e
        })
        .collect();
    let as_blank = cols
        .iter()
        .map(|c| {
            edit([
                ("column", json!(c)),
                ("role", json!("feature")),
                ("impute_method", json!("fill_median")),
                ("indicator", json!(group_or(ctx, c, "own"))),
                ("encode", Value::Null),
            ])
        })
        .collect();
    let leave_out = cols
        .iter()
        .map(|c| {
            edit([
                ("column", json!(c)),
                ("role", json!("drop")),
                ("impute_method", Value::Null),
                ("indicator", json!("none")),
                ("encode", Value::Null),
            ])
        })
        .collect();
    vec![
        QuestionOption::new(
            "as_categories",
            "Treat as categories",
            "Each distinct value (including non-numeric text) gets its own 0/1 category column; blanks become '(blank)'",
            as_categories,
        ),
        QuestionOption::new(
            "as_blank",
            "Treat non-numeric text as blank",
            format!(
                "{} {} {entry_word} {verb} with the median of observed numbers, plus a 0/1 blank marker",
                thousands(total_non_num),
                py_repr(top_val)
            ),
            as_blank,
        ),
        QuestionOption::new(
            "leave_out",
            "Leave out",
            format!("Leaves {} out of the map", format_col_list(cols, 3)),
            leave_out,
        ),
    ]
}

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let cols = ctx.mixed_value_cols.clone();
    if cols.is_empty() {
        return vec![];
    }
    let ev = |c: &String| ctx.ev_by_col[c.as_str()];
    let non_num = |c: &String| ev(c).n_non_numeric.unwrap_or(0);
    let total_non_num: usize = cols.iter().map(non_num).sum();
    // First maximal column, as Python's `max` (not `Iterator::max_by_key`, which keeps the last).
    let c_top = cols.iter().fold(
        &cols[0],
        |best, c| if non_num(c) > non_num(best) { c } else { best },
    );
    let ev_top = ev(c_top);
    let top_val = ev_top
        .non_numeric_values
        .as_ref()
        .and_then(|v| v.first())
        .map_or("", |(t, _)| t.as_str());
    let top_n = non_num(c_top);
    let mostly_num_all = cols.iter().all(|c| {
        let e = ev(c);
        e.n_rows - e.n_missing - non_num(c) > non_num(c)
    });
    let facts = Facts {
        cols: &cols,
        c_top,
        top_val,
        top_n,
        total_non_num,
        single_num: single_numeric_value(ctx, &cols),
        mostly_num_all,
    };
    let rows_affected = cols
        .iter()
        .map(|c| non_num(c) + ev(c).n_missing)
        .max()
        .unwrap_or(0);
    let default = if cols
        .iter()
        .all(|c| ev(c).n_unique as f64 <= ctx.cat_threshold)
    {
        "as_categories"
    } else {
        "as_blank"
    };
    let question = PlanQuestion::new(
        QuestionTopic::MixedValues,
        question_text(&facts),
        cols.clone(),
        rows_affected,
        options(ctx, &cols, total_non_num, top_val),
        default,
    );
    vec![(question, cols)]
}
