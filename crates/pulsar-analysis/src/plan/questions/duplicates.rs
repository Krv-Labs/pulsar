//! `duplicate_signal`: exact sums and duplicate columns that would count one
//! signal more than once.

use std::collections::HashSet;

use serde_json::json;

use crate::evidence::{ColumnRelation, RelationKind};
use crate::plan::defaults::DraftContext;
use crate::plan::model::{
    edit, Edit, PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic, Role,
};
use crate::plan::text::format_col_list;

struct Accepted<'a> {
    sum_targets: Vec<String>,
    sum_sources: Vec<String>,
    sums: Vec<&'a ColumnRelation>,
    dup_kept: Vec<String>,
    dup_dropped: Vec<String>,
    dups: Vec<&'a ColumnRelation>,
}

/// Keep each column in at most one role: a sum target is never a source or
/// a source of another sum, a duplicate is dropped only once.
fn collect<'a>(ctx: &DraftContext, relations: &'a [ColumnRelation]) -> Accepted<'a> {
    let mut a = Accepted {
        sum_targets: vec![],
        sum_sources: vec![],
        sums: vec![],
        dup_kept: vec![],
        dup_dropped: vec![],
        dups: vec![],
    };
    let (mut targets, mut sources): (HashSet<&str>, HashSet<&str>) = Default::default();
    let (mut kept, mut dropped): (HashSet<&str>, HashSet<&str>) = Default::default();

    for rel in relations {
        if !matches!(rel.kind, RelationKind::Duplicate | RelationKind::SumOf) {
            continue;
        }
        let all_features = std::iter::once(&rel.target)
            .chain(&rel.sources)
            .all(|c| ctx.entries.get(c).is_some_and(|e| e.role == Role::Feature));
        if !all_features {
            continue;
        }
        if rel.kind == RelationKind::SumOf {
            if targets.contains(rel.target.as_str())
                || sources.contains(rel.target.as_str())
                || rel.sources.iter().any(|s| targets.contains(s.as_str()))
            {
                continue;
            }
            targets.insert(&rel.target);
            a.sum_targets.push(rel.target.clone());
            for s in &rel.sources {
                if sources.insert(s) {
                    a.sum_sources.push(s.clone());
                }
            }
            a.sums.push(rel);
        } else {
            let (col_a, col_b) = (rel.sources[0].as_str(), rel.target.as_str());
            if dropped.contains(col_a) || kept.contains(col_b) || dropped.contains(col_b) {
                continue;
            }
            if kept.insert(col_a) {
                a.dup_kept.push(col_a.to_string());
            }
            dropped.insert(col_b);
            a.dup_dropped.push(col_b.to_string());
            a.dups.push(rel);
        }
    }
    a
}

fn role_edits(cols: &[String], role: &str) -> Vec<Edit> {
    cols.iter()
        .map(|c| edit([("column", json!(c)), ("role", json!(role))]))
        .collect()
}

fn sum_question(a: &Accepted) -> PlanQuestion {
    let r0 = a.sums[0];
    let rhs = if r0.sources.len() <= 2 {
        r0.sources
            .iter()
            .map(|s| format!("`{s}`"))
            .collect::<Vec<_>>()
            .join(" + ")
    } else {
        format!(
            "{} + \u{2026}",
            r0.sources[..2]
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(" + ")
        )
    };
    let formula = format!("`{}` = {rhs}", r0.target);
    let tail = "Counting both gives those signals extra weight. Keep everything, keep only the totals, or keep only the parts?";
    let text = if a.sums.len() == 1 {
        format!("1 total in this table is an exact sum of other columns ({formula}). {tail}")
    } else {
        format!(
            "{} totals in this table are exact sums of other columns (e.g. {formula}). {tail}",
            a.sums.len()
        )
    };
    let applies: Vec<String> = a
        .sum_targets
        .iter()
        .chain(&a.sum_sources)
        .cloned()
        .collect();
    let rows = a.sums.iter().map(|r| r.rows_used).max().unwrap_or(0);
    let targets = format_col_list(&a.sum_targets, 3);
    let sources = format_col_list(&a.sum_sources, 3);
    let mixed = |t: &str, s: &str| {
        let mut edits = role_edits(&a.sum_targets, t);
        edits.extend(role_edits(&a.sum_sources, s));
        edits
    };
    PlanQuestion::new(
        QuestionTopic::DuplicateSignal,
        text,
        applies.clone(),
        rows,
        vec![
            QuestionOption::new(
                "keep_all",
                "Keep everything",
                "Keeps both the total columns and the part columns in the map",
                role_edits(&applies, "feature"),
            ),
            QuestionOption::new(
                "keep_totals",
                "Keep only the totals",
                format!("Keeps {targets} and leaves {sources} out of the map"),
                mixed("feature", "drop"),
            ),
            QuestionOption::new(
                "keep_parts",
                "Keep only the parts",
                format!("Keeps {sources} and leaves {targets} out of the map"),
                mixed("drop", "feature"),
            ),
        ],
        "keep_all",
    )
}

fn duplicate_question(a: &Accepted) -> PlanQuestion {
    let applies: Vec<String> = a.dup_kept.iter().chain(&a.dup_dropped).cloned().collect();
    let rows = a.dups.iter().map(|r| r.rows_used).max().unwrap_or(0);
    let (text, keep_both, keep_one) = if a.dups.len() == 1 {
        let (col_a, col_b) = (&a.dups[0].sources[0], &a.dups[0].target);
        (
            format!(
                "`{col_b}` has identical values to `{col_a}`, so the map counts that signal twice. Keep both columns, or keep only `{col_a}`?"
            ),
            format!("Keeps both `{col_a}` and `{col_b}` in the map"),
            format!("Keeps `{col_a}` and leaves duplicate `{col_b}` out of the map"),
        )
    } else {
        let r0 = a.dups[0];
        (
            format!(
                "{} columns have identical values to another column (e.g. `{}` = `{}`), so the map counts those signals twice. Keep all of them, or leave out the duplicates?",
                a.dups.len(),
                r0.target,
                r0.sources[0]
            ),
            "Keeps all duplicate columns in the map".to_string(),
            format!(
                "Keeps {} and leaves duplicate {} out of the map",
                format_col_list(&a.dup_kept, 3),
                format_col_list(&a.dup_dropped, 3)
            ),
        )
    };
    let mut one_edits = role_edits(&a.dup_kept, "feature");
    one_edits.extend(role_edits(&a.dup_dropped, "drop"));
    PlanQuestion::new(
        QuestionTopic::DuplicateSignal,
        text,
        applies.clone(),
        rows,
        vec![
            QuestionOption::new(
                "keep_both",
                "Keep both",
                keep_both,
                role_edits(&applies, "feature"),
            ),
            QuestionOption::new("keep_one", "Keep only one", keep_one, one_edits),
        ],
        "keep_both",
    )
}

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let evidence = ctx.evidence;
    let a = collect(ctx, &evidence.relations);
    let mut pending = Vec::new();
    if !a.sums.is_empty() {
        pending.push((sum_question(&a), vec![]));
    }
    if !a.dups.is_empty() {
        pending.push((duplicate_question(&a), vec![]));
    }
    pending
}
