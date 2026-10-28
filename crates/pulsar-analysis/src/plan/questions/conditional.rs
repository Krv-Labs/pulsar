//! `conditional_column`: columns that only have values when a 0/1 partner
//! has one particular value.

use serde_json::{json, Value};

use super::group_or;
use crate::plan::defaults::DraftContext;
use crate::plan::model::{PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic};
use crate::plan::text::format_col_list;

fn options(
    ctx: &DraftContext,
    partner: &str,
    cols: &[String],
    preview: &str,
) -> Vec<QuestionOption> {
    let per_col = |f: &dyn Fn(&String) -> Vec<(&'static str, Value)>| {
        cols.iter().map(|c| edit_from(f(c))).collect::<Vec<_>>()
    };
    vec![
        QuestionOption::new(
            "leave_out",
            "Leave out",
            format!("Leaves {preview} out of the map because the blank pattern largely repeats `{partner}`"),
            per_col(&|c| {
                vec![
                    ("column", json!(c)),
                    ("role", json!("drop")),
                    ("impute_method", Value::Null),
                    ("indicator", json!("none")),
                ]
            }),
        ),
        QuestionOption::new(
            "blank_as_zero",
            "Treat blanks as 0",
            format!("Keeps in the map with blanks set to 0 and no blank marker (largely repeats `{partner}`)"),
            per_col(&|c| {
                vec![
                    ("column", json!(c)),
                    ("role", json!("feature")),
                    ("impute_method", json!("fill_value")),
                    ("impute_value", json!(0)),
                    ("indicator", json!("none")),
                ]
            }),
        ),
        QuestionOption::new(
            "estimate_and_mark",
            "Fill with median and mark blanks",
            format!(
                "Keeps in the map with blanks filled by the median and adds a 0/1 blank marker that largely repeats `{partner}`"
            ),
            per_col(&|c| {
                vec![
                    ("column", json!(c)),
                    ("role", json!("feature")),
                    ("impute_method", json!("fill_median")),
                    ("indicator", json!(group_or(ctx, c, "own"))),
                ]
            }),
        ),
    ]
}

fn edit_from(pairs: Vec<(&'static str, Value)>) -> crate::plan::model::Edit {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let mut pending = Vec::new();
    for (partner, cols) in ctx.conditional_by_partner.clone() {
        let first = ctx.ev_by_col[cols[0].as_str()]
            .mask_dependency
            .as_ref()
            .expect("a conditional column has a mask dependency");
        // The relation says when the column is *missing*; the question says when it has a value.
        let when = if first.relation.as_str() == "missing_when_partner_zero" {
            format!("when `{partner}` is not 0")
        } else {
            format!("when `{partner}` is 0")
        };
        let preview = format_col_list(&cols, 3);
        let text = if cols.len() == 1 {
            format!(
                "`{}` only has a value {when} \u{2014} it is blank for everyone else. Should the map use it?",
                cols[0]
            )
        } else {
            format!(
                "{} columns ({preview}) only have values {when} \u{2014} they are blank for everyone else. Should the map use them?",
                cols.len()
            )
        };
        let rows_affected = cols
            .iter()
            .map(|c| ctx.ev_by_col[c.as_str()].n_missing)
            .max()
            .unwrap_or(0);
        let question = PlanQuestion::new(
            QuestionTopic::ConditionalColumn,
            text,
            cols.clone(),
            rows_affected,
            options(ctx, &partner, &cols, &preview),
            "leave_out",
        );
        pending.push((question, cols));
    }
    pending
}
