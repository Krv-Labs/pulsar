//! The question builders of `draft_plan`
//! (`preprocessing_plan/questions/`), run in a fixed order because some read
//! what earlier ones recorded (`blank_group_cols`) or changed (`entries`).

mod blank_meaning;
mod code_columns;
mod conditional;
mod dates;
mod duplicates;
mod mixed_values;
mod mostly_blank;
mod outcome;

use super::defaults::DraftContext;
use super::model::PendingQuestion;

/// Run every builder in canonical order.
pub fn build_pending_questions(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let mut pending = Vec::new();
    pending.extend(dates::build(ctx));
    pending.extend(mixed_values::build(ctx));
    pending.extend(conditional::build(ctx));
    pending.extend(blank_meaning::build(ctx));
    pending.extend(mostly_blank::build(ctx));
    pending.extend(duplicates::build(ctx));
    pending.extend(code_columns::build(ctx));
    pending.extend(outcome::build(ctx));
    pending
}

/// `group:<gid>` when the column is in a missing group, else `fallback`.
fn group_or(ctx: &DraftContext, col: &str, fallback: &str) -> String {
    match &ctx.ev_by_col[col].missing_group {
        Some(g) => format!("group:{g}"),
        None => fallback.to_string(),
    }
}
