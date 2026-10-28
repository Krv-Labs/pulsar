//! `mostly_blank`: numeric columns blank for more than half the rows with no
//! blank-meaning explanation in the data.

use serde_json::{json, Value};

use super::group_or;
use crate::plan::defaults::DraftContext;
use crate::plan::model::{
    edit, DecidedBy, EntryStatus, PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic,
    Role,
};
use crate::plan::text::format_col_list;
use crate::pyround::round_int;

/// `(question text, "has value" effect)`.
fn copy(cols: &[String], pct0: i64) -> (String, String) {
    let c0 = &cols[0];
    if cols.len() == 1 {
        return (
            format!(
                "`{c0}` is blank for {pct0}% of rows, and the data doesn't show why. Leave it out, keep only whether it has a value, or estimate the missing values?"
            ),
            format!("Leaves `{c0}` out of the map and adds a 0/1 marker for whether it has a value"),
        );
    }
    let others = &cols[1..];
    let other_clause = match others {
        [a] => format!("and 1 other column, `{a}`, is blank for more than half of rows"),
        [a, b] => {
            format!("and 2 other columns are blank for more than half of rows: `{a}` and `{b}`")
        }
        _ => format!(
            "and {} other columns are blank for more than half of rows, including `{}` and `{}`",
            others.len(),
            others[0],
            others[1]
        ),
    };
    (
        format!(
            "`{c0}` is blank for {pct0}% of rows ({other_clause}), and the data doesn't show why. Leave them out, keep only whether each has a value, or estimate the missing values?"
        ),
        format!(
            "Leaves {} out of the map and adds 0/1 markers for whether values are filled in",
            format_col_list(cols, 3)
        ),
    )
}

fn options(
    ctx: &DraftContext,
    cols: &[String],
    preview: &str,
    has_value_effect: String,
    est_pct: i64,
) -> Vec<QuestionOption> {
    let drop = |indicator: &dyn Fn(&String) -> String| -> Vec<_> {
        cols.iter()
            .map(|c| {
                edit([
                    ("column", json!(c)),
                    ("role", json!("drop")),
                    ("impute_method", Value::Null),
                    ("indicator", json!(indicator(c))),
                ])
            })
            .collect()
    };
    let estimate = cols
        .iter()
        .map(|c| {
            edit([
                ("column", json!(c)),
                ("role", json!("feature")),
                ("impute_method", json!("fill_median")),
                ("indicator", json!(group_or(ctx, c, "own"))),
            ])
        })
        .collect();
    vec![
        QuestionOption::new(
            "leave_out",
            "Leave out",
            format!("Leaves {preview} out of the map"),
            drop(&|_| "none".into()),
        ),
        QuestionOption::new(
            "has_value",
            "Keep only whether value is filled in",
            has_value_effect,
            drop(&|c| group_or(ctx, c, &format!("group:{c}"))),
        ),
        QuestionOption::new(
            "estimate_and_mark",
            "Fill with median and mark blanks",
            format!(
                "Keeps in the map with blanks filled by the median of observed rows and a 0/1 blank marker (about {est_pct}% of these values would be estimates)"
            ),
            estimate,
        ),
    ]
}

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let frac = |c: &String| ctx.ev_by_col[c.as_str()].missing_fraction;
    let mut raw: Vec<String> = Vec::new();
    for (_gid, cols) in ctx.grouped_7b.iter() {
        if ctx.blank_group_cols.contains(&cols[0]) {
            continue;
        }
        let first = ctx.ev_by_col[cols[0].as_str()];
        if first.n_missing > ctx.n_rows - first.n_missing {
            raw.extend(cols.iter().cloned());
        }
    }
    raw.sort_by(|a, b| frac(b).total_cmp(&frac(a)).then_with(|| a.cmp(b)));
    let cols = raw;
    if cols.is_empty() {
        return vec![];
    }

    for c in &cols {
        if let Some(e) = ctx.entries.get_mut(c) {
            e.role = Role::Drop;
            e.impute = None;
            e.indicator = "none".into();
            e.encode = None;
            e.decided_by = DecidedBy::Default;
            e.status = EntryStatus::Open;
        }
    }

    let pct0 = 51.max(round_int(100.0 * frac(&cols[0])));
    let (text, has_value_effect) = copy(&cols, pct0);
    let preview = format_col_list(&cols, 3);
    let rows_affected = cols
        .iter()
        .map(|c| ctx.ev_by_col[c.as_str()].n_missing)
        .max()
        .unwrap_or(0);
    let mean = cols.iter().map(&frac).sum::<f64>() / cols.len() as f64;
    let est_pct = 51.max(round_int(100.0 * mean));
    let question = PlanQuestion::new(
        QuestionTopic::MostlyBlank,
        text,
        cols.clone(),
        rows_affected,
        options(ctx, &cols, &preview, has_value_effect, est_pct),
        "leave_out",
    );
    vec![(question, cols)]
}
