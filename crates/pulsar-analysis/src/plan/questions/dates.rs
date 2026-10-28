//! `date_columns`: what to do with date-like text columns.

use std::collections::BTreeSet;

use serde_json::json;

use super::group_or;
use crate::plan::defaults::DraftContext;
use crate::plan::model::{edit, PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic};
use crate::plan::text::format_col_list;

fn complete_clause(n: usize) -> String {
    if n == 1 {
        "1 date column has no blanks and stays out".into()
    } else {
        format!("{n} date columns have no blanks and stay out")
    }
}

fn has_date_effect(with_blanks: &[String], n_complete: usize, partners: &[String]) -> String {
    if with_blanks.is_empty() {
        return format!(
            "Leaves raw dates out of the map ({})",
            complete_clause(n_complete)
        );
    }
    let mut effect = if partners.is_empty() {
        "Leaves raw dates out of the map and adds a 0/1 marker for whether each date is filled in"
            .to_string()
    } else {
        format!(
            "Leaves raw dates out of the map and adds a 0/1 blank marker (note: this marker would repeat {})",
            format_col_list(partners, 3)
        )
    };
    if n_complete > 0 {
        effect = format!("{effect}; {}", complete_clause(n_complete));
    }
    effect
}

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let date_cols = ctx.date_cols.clone();
    if date_cols.is_empty() {
        return vec![];
    }
    let ev = |c: &String| ctx.ev_by_col[c.as_str()];
    let with_blanks: Vec<String> = date_cols
        .iter()
        .filter(|c| 0 < ev(c).n_missing && ev(c).n_missing < ev(c).n_rows)
        .cloned()
        .collect();
    let n_complete = date_cols.len() - with_blanks.len();
    let rows_affected = date_cols
        .iter()
        .map(|c| {
            if ev(c).n_missing > 0 {
                ev(c).n_missing
            } else {
                ev(c).n_rows
            }
        })
        .max()
        .expect("date_cols is non-empty");
    let partners: Vec<String> = with_blanks
        .iter()
        .filter_map(|c| ev(c).mask_dependency.as_ref().map(|m| m.partner.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let ignore = date_cols
        .iter()
        .map(|c| {
            edit([
                ("column", json!(c)),
                ("role", json!("drop")),
                ("indicator", json!("none")),
            ])
        })
        .collect();
    let has_date = with_blanks
        .iter()
        .map(|c| {
            edit([
                ("column", json!(c)),
                ("role", json!("drop")),
                ("indicator", json!(group_or(ctx, c, &format!("group:{c}")))),
            ])
        })
        .collect();

    let question = PlanQuestion::new(
        QuestionTopic::DateColumns,
        format!(
            "Some columns hold dates ({}). Should the map ignore them, or use only whether each date is filled in?",
            format_col_list(&date_cols, 3)
        ),
        date_cols.clone(),
        rows_affected,
        vec![
            QuestionOption::new("ignore", "Ignore dates", "Leaves date columns out of the map", ignore),
            QuestionOption::new(
                "has_date",
                "Use whether date is filled in",
                has_date_effect(&with_blanks, n_complete, &partners),
                has_date,
            ),
        ],
        "ignore",
    );
    vec![(question, date_cols)]
}
