//! `blank_meaning`: blank counts in a missing group that never records a 0,
//! where a blank could mean "none" or "unknown".

use serde_json::json;

use super::group_or;
use crate::plan::defaults::DraftContext;
use crate::plan::model::{
    edit, EntryStatus, PendingQuestion, PlanQuestion, QuestionOption, QuestionTopic,
};
use crate::plan::text::{format_col_list, is_count_like_zero_free, plural, thousands};

/// `(question text, "none" effect, "unknown" effect)`.
fn copy(
    cols: &[String],
    zero_free: &[String],
    rows: usize,
    n_with_zero: usize,
    partner_sentence: &str,
) -> (String, String, String) {
    let row_word = plural(rows, "row", "rows");
    let ask = "Does a blank mean none (zero), or that the value is unknown?";
    if cols.len() == 1 {
        return (
            format!(
                "`{}` is blank for {} {row_word} and never records a 0.{partner_sentence} {ask}",
                cols[0],
                thousands(rows)
            ),
            "Blanks become 0; no extra marker column".into(),
            "Blanks are filled with the median of observed rows, plus a 0/1 marker column for which rows were blank".into(),
        );
    }
    let zf = format_col_list(zero_free, 3);
    let verb = if zero_free.len() == 1 {
        "never records a 0"
    } else {
        "never record a 0"
    };
    let same_rows = if cols.len() == 2 {
        format!(
            "Both are blank for the same {} {row_word}.",
            thousands(rows)
        )
    } else {
        format!(
            "All {} are blank for the same {} {row_word}.",
            cols.len(),
            thousands(rows)
        )
    };
    let text = if n_with_zero > 0 {
        let other = if n_with_zero == 1 {
            "1 other column sometimes does".to_string()
        } else {
            format!("the other {n_with_zero} columns sometimes do")
        };
        format!("{zf} {verb} ({other}). {same_rows}{partner_sentence} {ask}")
    } else {
        format!("{zf} {verb}. {same_rows}{partner_sentence} {ask}")
    };
    let none = if cols.len() == 2 {
        "Both columns get 0 where blank; no extra marker column".to_string()
    } else {
        format!(
            "All {} columns get 0 where blank; no extra marker column",
            cols.len()
        )
    };
    (
        text,
        none,
        "Blanks are filled with the median of observed rows, plus one shared 0/1 marker column for the blank pattern".into(),
    )
}

fn options(
    ctx: &DraftContext,
    cols: &[String],
    none: String,
    unknown: String,
) -> Vec<QuestionOption> {
    let none_edits = cols
        .iter()
        .map(|c| {
            edit([
                ("column", json!(c)),
                ("role", json!("feature")),
                ("impute_method", json!("fill_value")),
                ("impute_value", json!(0)),
                ("indicator", json!("none")),
            ])
        })
        .collect();
    let unknown_edits = cols
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
        QuestionOption::new("none", "Blank means zero (none)", none, none_edits),
        QuestionOption::new("unknown", "Blank means unknown", unknown, unknown_edits),
    ]
}

pub(super) fn build(ctx: &mut DraftContext) -> Vec<PendingQuestion> {
    let mut pending = Vec::new();
    for (_gid, cols) in ctx.grouped_7b.clone() {
        let count_like: Vec<String> = cols
            .iter()
            .filter(|c| is_count_like_zero_free(ctx.ev_by_col[c.as_str()]))
            .cloned()
            .collect();
        if count_like.is_empty() {
            continue;
        }
        ctx.blank_group_cols.extend(cols.iter().cloned());
        for c in &cols {
            if let Some(e) = ctx.entries.get_mut(c) {
                e.status = EntryStatus::Open;
            }
        }
        let rows = ctx.ev_by_col[cols[0].as_str()].n_missing;
        let mut zero_free = count_like.clone();
        zero_free.extend(
            cols.iter()
                .filter(|c| ctx.ev_by_col[c.as_str()].n_zero == Some(0) && !count_like.contains(c))
                .cloned(),
        );
        let n_with_zero = cols.len() - zero_free.len();

        let partner_sentence = cols
            .iter()
            .find_map(|c| ctx.ev_by_col[c.as_str()].mask_dependency.as_ref())
            .map(|md| {
                let cond = if md.relation.as_str() == "missing_when_partner_zero" {
                    "0"
                } else {
                    "not 0"
                };
                let subj = if cols.len() == 1 { "It is" } else { "They are" };
                format!(" {subj} blank exactly where `{}` is {cond}.", md.partner)
            })
            .unwrap_or_default();

        let (text, none, unknown) = copy(&cols, &zero_free, rows, n_with_zero, &partner_sentence);
        let question = PlanQuestion::new(
            QuestionTopic::BlankMeaning,
            text,
            cols.clone(),
            rows,
            options(ctx, &cols, none, unknown),
            "unknown",
        );
        pending.push((question, cols));
    }
    pending
}
