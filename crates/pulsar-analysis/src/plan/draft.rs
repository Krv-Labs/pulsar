//! `draft_plan`: measure a table and build a revision-1 plan
//! (`preprocessing_plan/draft.py` and the ordering half of `stakes.py`).
//!
//! With `stakes` on, each question gets a counterfactual k-NN overlap score
//! (see [`super::stakes`]) and the questions are ordered by how much they can
//! move the map; with it off they are ordered by rows affected alone.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use indexmap::IndexMap;
use pulsar_data::Table;
use rand::Rng;
use serde_json::{json, Map};

use super::defaults::{populate_default_entries, validate_role_designations, DraftContext};
use super::model::{PendingQuestion, PlanEditError, PlanQuestion, QuestionTopic};
use super::plan::PreprocessingPlan;
use super::proxies::dry_run_processed_space;
use super::questions::build_pending_questions;
use super::stakes::{compute_stakes, StakesOptions};
use crate::evidence::{measure_preprocessing_evidence, EvidenceError, EvidenceOptions};
use crate::pyround::round_to;

#[derive(Debug, Clone)]
pub struct DraftOptions {
    /// Columns that are outcomes to explain later; held out of the map.
    pub target_columns: Vec<String>,
    pub entity_column: String,
    pub time_column: String,
    /// Score each question's options by how much they change the map.
    pub stakes: bool,
    /// Rows used per stakes measurement (larger tables are sampled).
    pub stakes_sample_rows: usize,
    pub stakes_k: usize,
    /// Most questions to score; the rest are listed but unscored.
    pub stakes_max_questions: usize,
    pub seed: u64,
    /// Give row-index columns the identifier role (see
    /// [`DraftContext::row_index_ids`]); off reproduces Python's plans.
    pub row_index_ids: bool,
}

impl Default for DraftOptions {
    /// Python's defaults (stakes on, 2000 rows, k = 10, 16 questions),
    /// except that row-index columns are drafted as identifiers.
    fn default() -> Self {
        Self {
            target_columns: Vec::new(),
            entity_column: String::new(),
            time_column: String::new(),
            stakes: true,
            stakes_sample_rows: 2000,
            stakes_k: 10,
            stakes_max_questions: 16,
            seed: 0,
            row_index_ids: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DraftError {
    #[error(transparent)]
    Plan(#[from] PlanEditError),
    #[error(transparent)]
    Evidence(#[from] EvidenceError),
}

impl DraftError {
    pub fn code(&self) -> &str {
        match self {
            Self::Plan(e) => &e.code,
            Self::Evidence(e) => e.code(),
        }
    }
}

/// Framing questions (about the whole table, not one decision) go last.
fn framing_rank(topic: QuestionTopic) -> u8 {
    match topic {
        QuestionTopic::CodeColumns => 1,
        QuestionTopic::OutcomeColumns => 2,
        _ => 0,
    }
}

/// Most affected rows first within a rank (stable, so ties keep builder order).
fn order_pending(pending: &mut [PendingQuestion]) {
    pending.sort_by_key(|(q, _)| (framing_rank(q.topic), std::cmp::Reverse(q.rows_affected)));
}

/// Final order (scored questions first, lowest neighbours-kept first, then by
/// rows affected) and the stable `q1..qN` ids, linked back to their columns.
fn finalize(plan: &mut PreprocessingPlan, mut pending: Vec<PendingQuestion>) {
    pending.sort_by(|(a, _), (b, _)| {
        let key = |q: &PlanQuestion| {
            let kept = q.stakes.as_ref().and_then(|s| s["neighbors_kept"].as_f64());
            (
                framing_rank(q.topic),
                kept.is_none(),
                kept.unwrap_or(1.0),
                std::cmp::Reverse(q.rows_affected),
            )
        };
        let (ka, kb) = (key(a), key(b));
        (ka.0, ka.1)
            .cmp(&(kb.0, kb.1))
            .then(ka.2.total_cmp(&kb.2))
            .then(ka.3.cmp(&kb.3))
    });
    let mut questions: IndexMap<String, PlanQuestion> = IndexMap::new();
    for (idx, (mut q, linked)) in pending.into_iter().enumerate() {
        let qid = format!("q{}", idx + 1);
        q.question_id = qid.clone();
        for col in linked {
            if let Some(entry) = plan.entries.get_mut(&col) {
                entry.question_id = Some(qid.clone());
            }
        }
        questions.insert(qid, q);
    }
    plan.questions = questions;
}

/// `pp_` plus twelve random hex characters: the store's handle format
/// (Python used ten; the extra two keep ids unique across a long-lived store).
fn new_plan_id() -> String {
    let n: u64 = rand::thread_rng().gen();
    format!("pp_{:012x}", n & 0xFFFF_FFFF_FFFF)
}

/// UTC now as `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`.
fn utc_now_iso() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let (secs, micros) = (now.as_secs() as i64, now.subsec_micros());
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{micros:06}+00:00",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Measure `table` and construct a revision-1 plan.
pub fn draft_plan(
    table: &Table,
    dataset_id: &str,
    opts: &DraftOptions,
) -> Result<PreprocessingPlan, DraftError> {
    let (targets, entity, time) = validate_role_designations(
        &table.names,
        &opts.target_columns,
        &opts.entity_column,
        &opts.time_column,
    )?;
    let evidence = measure_preprocessing_evidence(table, &EvidenceOptions::default())?;

    let (entries, missing_groups, mut pending) = {
        let mut ctx = DraftContext::new(table, &evidence, &targets, &entity, &time);
        ctx.row_index_ids = opts.row_index_ids;
        populate_default_entries(&mut ctx);
        let pending = build_pending_questions(&mut ctx);
        (ctx.entries, ctx.missing_groups, pending)
    };
    order_pending(&mut pending);
    // Provisional ids while stakes are computed; final `q1..qN` ids follow.
    for (i, (q, _)) in pending.iter_mut().enumerate() {
        q.question_id = format!("_prov_{}", i + 1);
    }
    let eligible: Vec<usize> = pending
        .iter()
        .enumerate()
        .filter(|(_, (q, _))| {
            !matches!(
                q.topic,
                QuestionTopic::OutcomeColumns | QuestionTopic::CodeColumns
            )
        })
        .map(|(i, _)| i)
        .collect();

    let mut cost = Map::new();
    cost.insert("rows".into(), json!(evidence.n_rows));
    cost.insert("columns".into(), json!(evidence.n_columns));
    cost.insert(
        "evidence_ms".into(),
        json!(round_to(evidence.cost.elapsed_ms, 2)),
    );
    cost.insert("stakes_ms".into(), json!(0.0));
    cost.insert("stakes_rows_sampled".into(), json!(0));
    cost.insert("stakes_questions_computed".into(), json!(0));
    cost.insert("stakes_questions_skipped".into(), json!(eligible.len()));
    cost.insert("stakes_questions_failed".into(), json!(0));

    let mut plan = PreprocessingPlan {
        plan_id: new_plan_id(),
        dataset_id: dataset_id.to_string(),
        revision: 1,
        entries,
        questions: pending
            .iter()
            .map(|(q, _)| (q.question_id.clone(), q.clone()))
            .collect(),
        missing_groups,
        evidence,
        created_at: utc_now_iso(),
        cost,
        target_proxies: vec![],
        n_imputations: 1,
    };
    if opts.stakes && !eligible.is_empty() && opts.stakes_max_questions > 0 {
        let budget: Vec<usize> = eligible
            .iter()
            .copied()
            .take(opts.stakes_max_questions)
            .collect();
        let ids: Vec<String> = budget
            .iter()
            .map(|&i| pending[i].0.question_id.clone())
            .collect();
        let started = Instant::now();
        let results = compute_stakes(
            table,
            &mut plan,
            Some(&ids),
            &StakesOptions {
                sample_rows: opts.stakes_sample_rows,
                k: opts.stakes_k,
                seed: opts.seed,
            },
        );
        let stakes_ms = round_to(started.elapsed().as_secs_f64() * 1000.0, 2);
        let computed = ids
            .iter()
            .filter(|id| results.get(*id).is_some_and(Option::is_some))
            .count();
        let sampled_rows = results
            .values()
            .flatten()
            .next()
            .and_then(|s| s["rows_sampled"].as_u64())
            .unwrap_or(0);
        for (qid, stakes) in results {
            if let Some((q, _)) = pending.iter_mut().find(|(q, _)| q.question_id == qid) {
                q.stakes = stakes;
            }
        }
        plan.cost.insert("stakes_ms".into(), json!(stakes_ms));
        plan.cost
            .insert("stakes_rows_sampled".into(), json!(sampled_rows));
        plan.cost
            .insert("stakes_questions_computed".into(), json!(computed));
        plan.cost.insert(
            "stakes_questions_skipped".into(),
            json!(eligible.len() - ids.len()),
        );
        plan.cost.insert(
            "stakes_questions_failed".into(),
            json!(ids.len() - computed),
        );
    }
    finalize(&mut plan, pending);
    dry_run_processed_space(table, &mut plan);
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_is_rfc3339_utc_with_microseconds() {
        let t = utc_now_iso();
        assert_eq!(t.len(), 32, "{t}");
        assert!(t.ends_with("+00:00") && &t[10..11] == "T", "{t}");
        assert!(t.starts_with("20"), "{t}");
    }

    #[test]
    fn plan_ids_are_pp_plus_twelve_hex_digits() {
        let id = new_plan_id();
        assert!(id.starts_with("pp_") && id.len() == 15, "{id}");
        assert!(id[3..].chars().all(|c| c.is_ascii_hexdigit()), "{id}");
        assert_ne!(new_plan_id(), new_plan_id());
    }
}
