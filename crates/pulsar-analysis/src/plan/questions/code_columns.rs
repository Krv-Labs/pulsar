//! `code_columns`: low-cardinality whole-number columns that might be labels
//! rather than amounts.

use std::collections::HashSet;

use crate::evidence::{ColumnKind, RelationKind};
use crate::plan::defaults::DraftContext;
use crate::plan::model::{PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic, Role};
use crate::plan::text::is_count_like_zero_free;

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let additive: HashSet<&str> = ctx
        .evidence
        .relations
        .iter()
        .filter(|r| matches!(r.kind, RelationKind::SumOf | RelationKind::Duplicate))
        .flat_map(|r| {
            std::iter::once(r.target.as_str()).chain(r.sources.iter().map(String::as_str))
        })
        .collect();

    let mut cols: Vec<String> = ctx
        .entries
        .iter()
        .filter(|(col, entry)| {
            let ev = ctx.ev_by_col[col.as_str()];
            entry.role == Role::Feature
                && ev.kind == ColumnKind::Numeric
                && ev.integer_valued == Some(true)
                && ev.nonnegative == Some(true)
                && ev.n_unique >= 3
                && ev.n_unique as f64 <= ctx.cat_threshold
                && !is_count_like_zero_free(ev)
                && !ctx.blank_group_cols.contains(col.as_str())
                && !additive.contains(col.as_str())
        })
        .map(|(col, _)| col.clone())
        .collect();
    cols.sort_by(|a, b| {
        let (ua, ub) = (
            ctx.ev_by_col[a.as_str()].n_unique,
            ctx.ev_by_col[b.as_str()].n_unique,
        );
        ub.cmp(&ua).then_with(|| a.cmp(b))
    });
    if cols.is_empty() {
        return vec![];
    }

    let preview = cols
        .iter()
        .take(3)
        .map(|c| {
            format!(
                "`{c}`: {} distinct values",
                ctx.ev_by_col[c.as_str()].n_unique
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let paren = if cols.len() == 1 {
        format!("e.g. {preview}")
    } else {
        format!("e.g. {preview}; {} columns", cols.len())
    };
    let question = PlanQuestion::new(
        QuestionTopic::CodeColumns,
        format!(
            "Some whole-number columns might be labels rather than amounts ({paren}). For labels, bigger numbers don't mean 'more'. Are any of these labels or codes?"
        ),
        cols,
        ctx.n_rows,
        vec![QuestionOption::new(
            "all_amounts",
            "All are amounts",
            "Keeps all as numeric amounts in the map",
            vec![],
        )],
        "all_amounts",
    );
    vec![(question, vec![])]
}
