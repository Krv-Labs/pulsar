//! One-hot above the categorical threshold (sqrt of the row count) is held
//! until the agent confirms it; nothing is refused and nothing is swapped.

use pulsar_analysis::plan::{
    apply_answers_and_edits_guarded, draft_plan, DecidedBy, DraftOptions, Role,
};
use pulsar_data::{Column, Table};
use serde_json::json;

fn plan() -> pulsar_analysis::plan::PreprocessingPlan {
    let n = 100;
    let code: Vec<Option<String>> = (0..n).map(|i| Some(format!("c{}", i % 40))).collect();
    let few: Vec<Option<String>> = (0..n).map(|i| Some(format!("g{}", i % 3))).collect();
    let x: Vec<f64> = (0..n).map(|i| f64::from(i as u32) * 0.37 % 5.0).collect();
    let table = Table::new(
        vec!["code".into(), "few".into(), "x".into()],
        vec![Column::Text(code), Column::Text(few), Column::Float(x)],
    )
    .unwrap();
    draft_plan(&table, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap()
}

#[test]
fn unconfirmed_high_cardinality_edit_is_held_and_confirmed_applies() {
    let plan = plan();
    assert_eq!(plan.entries["code"].role, Role::Drop, "default drops it");
    let edit = json!({"column": "code", "role": "feature", "encode": "one_hot"});

    let (held_plan, held) = apply_answers_and_edits_guarded(
        &plan,
        &[],
        std::slice::from_ref(&edit),
        DecidedBy::Agent,
        &[],
    )
    .unwrap();
    assert_eq!(held.len(), 1);
    assert_eq!((held[0].column.as_str(), held[0].levels), ("code", 40));
    assert_eq!(held[0].threshold, 10);
    assert_eq!(held[0].features_added, 40);
    assert_eq!(
        held_plan.entries["code"], plan.entries["code"],
        "column unchanged"
    );

    let (applied, held) = apply_answers_and_edits_guarded(
        &plan,
        &[],
        &[edit],
        DecidedBy::Agent,
        &["code".to_string()],
    )
    .unwrap();
    assert!(held.is_empty());
    let e = &applied.entries["code"];
    assert_eq!(e.role, Role::Feature);
    assert_eq!(e.encode.as_deref(), Some("one_hot"));
    assert_eq!(e.decided_by, DecidedBy::Agent);
    assert!(
        e.reason.contains("confirmed above threshold"),
        "{}",
        e.reason
    );
}

#[test]
fn low_cardinality_one_hot_is_not_held() {
    let plan = plan();
    let edit = json!({"column": "few", "role": "feature", "encode": "one_hot"});
    let (_, held) =
        apply_answers_and_edits_guarded(&plan, &[], &[edit], DecidedBy::Agent, &[]).unwrap();
    assert!(held.is_empty());
}

/// 100 rows: `v` holds 95 distinct numbers and five "n/a" entries (96 levels
/// against a threshold of 10), which draws a `mixed_values` question.
fn mixed_plan() -> pulsar_analysis::plan::PreprocessingPlan {
    let v: Vec<Option<String>> = (0..100)
        .map(|i| {
            Some(if i < 95 {
                format!("{}.5", i)
            } else {
                "n/a".to_string()
            })
        })
        .collect();
    let x: Vec<f64> = (0..100).map(|i| f64::from(i as u32) * 0.37 % 5.0).collect();
    let table = Table::new(
        vec!["v".into(), "x".into()],
        vec![Column::Text(v), Column::Float(x)],
    )
    .unwrap();
    draft_plan(&table, "ds_bbbbbbbbbbbb", &DraftOptions::default()).unwrap()
}

#[test]
fn a_mixed_values_answer_that_one_hots_a_wide_column_is_held_whole() {
    use pulsar_analysis::plan::QuestionStatus;
    let plan = mixed_plan();
    let q = plan
        .questions
        .values()
        .find(|q| q.topic == pulsar_analysis::plan::QuestionTopic::MixedValues)
        .expect("a mixed_values question");
    let answer = json!({"question_id": q.question_id, "option_id": "as_categories"});

    let (held_plan, held) = apply_answers_and_edits_guarded(
        &plan,
        std::slice::from_ref(&answer),
        &[],
        DecidedBy::Agent,
        &[],
    )
    .unwrap();
    assert_eq!(held.len(), 1, "{held:?}");
    assert_eq!(held[0].column, "v");
    assert_eq!(held[0].question_id.as_deref(), Some(q.question_id.as_str()));
    assert!(held[0].levels > held[0].threshold);
    assert_eq!(
        held_plan.entries["v"], plan.entries["v"],
        "column unchanged"
    );
    assert_eq!(
        held_plan.questions[&q.question_id].status,
        QuestionStatus::Open
    );

    let (applied, held) = apply_answers_and_edits_guarded(
        &plan,
        &[answer],
        &[],
        DecidedBy::Agent,
        &["v".to_string()],
    )
    .unwrap();
    assert!(held.is_empty());
    let e = &applied.entries["v"];
    assert_eq!(e.encode.as_deref(), Some("one_hot"));
    assert!(
        e.reason.contains("confirmed above threshold"),
        "{}",
        e.reason
    );
    assert_eq!(
        applied.questions[&q.question_id].status,
        QuestionStatus::Answered
    );
}

#[test]
fn a_held_answer_leaves_independent_answers_in_the_same_call_applied() {
    use pulsar_analysis::plan::{QuestionStatus, QuestionTopic};
    let plan = mixed_plan();
    let topic_id = |t| {
        plan.questions
            .values()
            .find(|q| q.topic == t)
            .unwrap_or_else(|| panic!("no {t:?} question"))
            .question_id
            .clone()
    };
    let (mixed, outcome) = (
        topic_id(QuestionTopic::MixedValues),
        topic_id(QuestionTopic::OutcomeColumns),
    );
    let answers = [
        json!({"question_id": mixed, "option_id": "as_categories"}),
        json!({"question_id": outcome, "option_id": "none_of_them"}),
    ];
    let (next, held) =
        apply_answers_and_edits_guarded(&plan, &answers, &[], DecidedBy::Agent, &[]).unwrap();
    assert_eq!(held.len(), 1, "{held:?}");
    assert_eq!(
        next.entries["v"], plan.entries["v"],
        "held column unchanged"
    );
    assert_eq!(next.questions[&mixed].status, QuestionStatus::Open);
    assert_eq!(next.questions[&outcome].status, QuestionStatus::Answered);
    assert_eq!(next.revision, plan.revision + 1);
}

#[test]
fn two_unconfirmed_edits_cannot_reach_an_active_one_hot_without_confirmation() {
    let plan = plan();
    let (staged, held) = apply_answers_and_edits_guarded(
        &plan,
        &[],
        &[json!({"column": "code", "role": "drop", "encode": "one_hot"})],
        DecidedBy::Agent,
        &[],
    )
    .unwrap();
    assert!(held.is_empty(), "a dropped column is not an active one-hot");

    let flip = json!({"column": "code", "role": "feature"});
    let (still, held) = apply_answers_and_edits_guarded(
        &staged,
        &[],
        std::slice::from_ref(&flip),
        DecidedBy::Agent,
        &[],
    )
    .unwrap();
    assert_eq!(held.len(), 1, "{held:?}");
    assert_eq!(still.entries["code"].role, Role::Drop);

    let (applied, held) = apply_answers_and_edits_guarded(
        &staged,
        &[],
        &[flip],
        DecidedBy::Agent,
        &["code".to_string()],
    )
    .unwrap();
    assert!(held.is_empty());
    let e = &applied.entries["code"];
    assert_eq!(e.role, Role::Feature);
    assert!(
        e.reason.contains("confirmed above threshold"),
        "{}",
        e.reason
    );
}
