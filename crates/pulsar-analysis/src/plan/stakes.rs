//! Counterfactual k-NN neighbourhood-overlap stakes for plan questions
//! (`compute_stakes` in `preprocessing_plan/stakes.py`).
//!
//! For each question, every non-default option is applied to the plan, the
//! table is re-preprocessed, and the share of each row's k nearest neighbours
//! that survives is measured. The lowest share over the options is the
//! question's stake: how much that decision can move the map.
//!
//! Differences from the Python reference, both only on large or tie-heavy
//! data: when the table has more rows than `sample_rows`, the row sample is
//! drawn with a different generator (numpy's PCG64 is not reproduced), and
//! exact distance ties between rows are broken by row index where scikit-learn
//! breaks them by tree order. Both change which neighbours are counted, so
//! `neighbors_kept` can differ in its last digits on such data.

use indexmap::IndexMap;
use ndarray::Array2;
use pulsar_data::{preprocess, Table};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::{json, Value};

use super::edits::apply_answers_and_edits;
use super::model::{DecidedBy, QuestionTopic};
use super::plan::PreprocessingPlan;
use super::proxies::plan_config;
use crate::evidence::knn::{mean_jaccard, neighbor_lists};
use crate::pyround::round_to;

#[derive(Debug, Clone)]
pub struct StakesOptions {
    /// Rows used per measurement; larger tables are sampled down.
    pub sample_rows: usize,
    pub k: usize,
    /// Seed of the row sample.
    pub seed: u64,
}

impl Default for StakesOptions {
    fn default() -> Self {
        Self {
            sample_rows: 2000,
            k: 10,
            seed: 0,
        }
    }
}

fn note(plan: &mut PreprocessingPlan, text: String) {
    let notes = plan.cost.entry("stakes_notes").or_insert_with(|| json!([]));
    if let Some(list) = notes.as_array_mut() {
        list.push(json!(text));
    }
}

/// The plan applied to `table`: an all-numeric feature matrix.
fn plan_to_matrix(table: &Table, plan: &PreprocessingPlan) -> Result<Array2<f64>, String> {
    let config = plan_config(plan, "stakes_eval")?;
    preprocess(table, &config.preprocessing, None, None)
        .map(|p| p.matrix)
        .map_err(|e| e.to_string())
}

/// The base representation's neighbours, computed once for every option.
struct Base {
    cols: usize,
    neighbors: Option<Vec<Vec<usize>>>,
    k: usize,
}

impl Base {
    fn new(matrix: &Array2<f64>, k: usize) -> Self {
        let finite = matrix.iter().all(|v| v.is_finite());
        Self {
            cols: matrix.ncols(),
            // Zero-padding columns to match widths changes nothing after
            // z-scoring, so the base needs no per-option padding.
            neighbors: (matrix.ncols() > 0 && finite).then(|| neighbor_lists(matrix.view(), k)),
            k,
        }
    }

    /// Overlap with `other`, padding widths like the reference (an empty side
    /// is 0.0 overlap, two empty sides 1.0).
    fn overlap(&self, other: &Array2<f64>) -> Result<f64, String> {
        match (self.cols, other.ncols()) {
            (0, 0) => return Ok(1.0),
            (0, _) | (_, 0) => return Ok(0.0),
            _ => {}
        }
        let (Some(base), true) = (&self.neighbors, other.iter().all(|v| v.is_finite())) else {
            return Err("X_a and X_b must contain only finite values.".into());
        };
        Ok(mean_jaccard(
            base,
            &neighbor_lists(other.view(), self.k),
            self.k,
        ))
    }
}

fn evaluate_question(
    qid: &str,
    plan: &mut PreprocessingPlan,
    sample: &Table,
    base: &Base,
    n_sampled: usize,
) -> Option<Value> {
    let question = plan.questions.get(qid)?.clone();
    let alternatives: Vec<Vec<Value>> = question
        .options
        .iter()
        .filter(|o| Some(&o.option_id) != question.default_option_id.as_ref())
        .map(|o| o.edits.iter().map(|e| Value::Object(e.clone())).collect())
        .collect();
    if alternatives.is_empty() {
        return None;
    }

    let mut min_overlap: Option<f64> = None;
    let mut best_delta: i64 = 0;
    for edits in alternatives {
        let outcome: Result<(f64, i64), String> = if edits.is_empty() {
            Ok((1.0, 0))
        } else {
            apply_answers_and_edits(plan, &[], &edits, DecidedBy::Agent)
                .map_err(|e| e.message)
                .and_then(|alt| plan_to_matrix(sample, &alt))
                .and_then(|x_opt| {
                    Ok((
                        base.overlap(&x_opt)?,
                        x_opt.ncols() as i64 - base.cols as i64,
                    ))
                })
        };
        match outcome {
            Ok((overlap, delta)) => {
                let better = min_overlap.is_none_or(|m| {
                    overlap < m || (overlap == m && delta.abs() > best_delta.abs())
                });
                if better {
                    min_overlap = Some(overlap);
                    best_delta = delta;
                }
            }
            Err(e) => note(plan, format!("Stakes option failed for {qid}: {e}")),
        }
    }
    let min_overlap = min_overlap?;
    Some(json!({
        "neighbors_kept": round_to(min_overlap, 4),
        "feature_count_delta": best_delta,
        "rows_sampled": n_sampled,
        "k": base.k,
    }))
}

/// Stakes for `question_ids` (default: every open question), keyed by id.
/// `None` means not computed: no alternatives, an outcome question, or the
/// reason is recorded in `plan.cost["stakes_notes"]`.
pub fn compute_stakes(
    table: &Table,
    plan: &mut PreprocessingPlan,
    question_ids: Option<&[String]>,
    opts: &StakesOptions,
) -> IndexMap<String, Option<Value>> {
    let targets: Vec<String> = match question_ids {
        Some(ids) => ids.to_vec(),
        None => plan
            .questions
            .iter()
            .filter(|(_, q)| q.status == super::model::QuestionStatus::Open)
            .map(|(id, _)| id.clone())
            .collect(),
    };
    let mut results: IndexMap<String, Option<Value>> =
        targets.iter().map(|q| (q.clone(), None)).collect();
    if targets.is_empty() {
        return results;
    }

    let n_total = table.n_rows();
    if n_total < 2 {
        note(
            plan,
            "Dataset has fewer than 2 rows; stakes not computed.".into(),
        );
        return results;
    }
    let sample = if opts.sample_rows > 0 && n_total > opts.sample_rows {
        let mut rng = StdRng::seed_from_u64(opts.seed);
        let mut idx = rand::seq::index::sample(&mut rng, n_total, opts.sample_rows).into_vec();
        idx.sort_unstable();
        table.take_rows(&idx)
    } else {
        table.clone()
    };
    let n_sampled = sample.n_rows();
    let k_eff = opts.k.min(n_sampled - 1);
    if k_eff < 1 {
        note(
            plan,
            "Effective k < 1 on row sample; stakes not computed.".into(),
        );
        return results;
    }

    let base = match plan_to_matrix(&sample, plan) {
        Ok(x) => Base::new(&x, k_eff),
        Err(e) => {
            note(
                plan,
                format!("Baseline dry-run failed during stakes computation: {e}"),
            );
            return results;
        }
    };
    for qid in &targets {
        let skip = plan
            .questions
            .get(qid)
            .is_none_or(|q| q.topic == QuestionTopic::OutcomeColumns);
        if !skip {
            results.insert(
                qid.clone(),
                evaluate_question(qid, plan, &sample, &base, n_sampled),
            );
        }
    }
    results
}
