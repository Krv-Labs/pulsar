//! One-hot cardinality guard (`confirm_one_hot`) for answers and column edits.

use serde_json::Value;

use super::super::model::{PlanEntry, Role};
use super::{Entries, Evidence};

/// A one-hot encoding the guard left out of an edit or answer: the column has
/// more levels than the plan's categorical threshold (sqrt of the row count)
/// and was not named in `confirmed`.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOneHot {
    pub column: String,
    /// Distinct values the column would encode.
    pub levels: usize,
    /// The plan's threshold: sqrt(n_rows).
    pub threshold: usize,
    /// Net feature columns the encoding would add (levels, less the one the column
    /// already contributes when it is a feature now).
    pub features_added: usize,
    /// `question_id` when an answer was held; None for a direct edit.
    pub question_id: Option<String>,
}

fn is_active_one_hot(entry: &PlanEntry) -> bool {
    entry.role == Role::Feature && entry.encode.as_deref() == Some("one_hot")
}

/// Columns whose encoding this step newly set to one-hot above the threshold.
fn new_big_one_hots(
    before: &Entries,
    after: &Entries,
    evidence: &Evidence,
    threshold: f64,
) -> Vec<HeldOneHot> {
    after
        .iter()
        .filter(|(c, e)| is_active_one_hot(e) && !before.get(*c).is_some_and(is_active_one_hot))
        .filter_map(|(c, _)| {
            let levels = evidence.get(c.as_str())?.n_unique;
            (levels as f64 > threshold).then(|| HeldOneHot {
                column: c.clone(),
                levels,
                threshold: threshold as usize,
                features_added: levels
                    - usize::from(before.get(c).is_some_and(|b| b.role == Role::Feature)),
                question_id: None,
            })
        })
        .collect()
}

/// Record, in the entry's reason, that a one-hot above the threshold was confirmed.
fn note_confirmed(entries: &mut Entries, confirmed: &[HeldOneHot]) {
    for h in confirmed {
        if let Some(e) = entries.get_mut(&h.column) {
            e.reason = format!(
                "one_hot confirmed above threshold: {} levels exceed sqrt(n_rows) = {}; {}",
                h.levels, h.threshold, e.reason
            );
        }
    }
}

/// Partition any newly added above-threshold one-hot encodings into confirmed
/// (recorded in the entry's reason) and unconfirmed (returned to be held).
pub(super) fn confirm_or_hold(
    old_entries: &Entries,
    entries: &mut Entries,
    evidence: &Evidence,
    threshold: f64,
    confirmed: &[String],
) -> Vec<HeldOneHot> {
    let big = new_big_one_hots(old_entries, entries, evidence, threshold);
    let (ok, unconfirmed): (Vec<_>, Vec<_>) =
        big.into_iter().partition(|h| confirmed.contains(&h.column));
    note_confirmed(entries, &ok);
    unconfirmed
}

pub(super) fn tagged_with_question(unconfirmed: Vec<HeldOneHot>, ans: &Value) -> Vec<HeldOneHot> {
    let qid = ans
        .get("question_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    unconfirmed
        .into_iter()
        .map(|mut h| {
            h.question_id.clone_from(&qid);
            h
        })
        .collect()
}
