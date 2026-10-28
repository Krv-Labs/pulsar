//! A plan must survive being stored and reloaded: serialize to JSON text,
//! parse it back, and get an equal plan (column and question order, floats,
//! stakes, evidence and all). The store keeps plans as JSON, so any loss here
//! is a plan silently changing between two tool calls.

use pulsar_analysis::plan::{
    apply_answers_and_edits, draft_plan, DecidedBy, DraftOptions, PreprocessingPlan,
};
use pulsar_data::read_csv_path;
use pulsar_parity::golden_dir;
use serde_json::{json, Value};

fn through_text(plan: &PreprocessingPlan) -> PreprocessingPlan {
    let text = serde_json::to_string(&plan.to_value()).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    PreprocessingPlan::from_value(value).unwrap()
}

fn scenarios() -> Vec<(String, DraftOptions)> {
    let fx: Value = serde_json::from_str(
        &std::fs::read_to_string(golden_dir().join("plan/draft.json")).unwrap(),
    )
    .unwrap();
    fx.as_object()
        .unwrap()
        .values()
        .map(|s| {
            let args = &s["args"];
            let opts = DraftOptions {
                target_columns: serde_json::from_value(args["targets"].clone()).unwrap(),
                entity_column: args["entity"].as_str().unwrap().into(),
                time_column: args["time"].as_str().unwrap().into(),
                ..DraftOptions::default()
            };
            (s["csv"].as_str().unwrap().to_string(), opts)
        })
        .collect()
}

#[test]
fn every_drafted_plan_round_trips_through_json_text() {
    let mut checked = 0;
    for (csv, opts) in scenarios() {
        let table = read_csv_path(golden_dir().join(&csv)).unwrap();
        let plan = draft_plan(&table, "ds_aaaaaaaaaaaa", &opts).unwrap();
        assert_eq!(
            through_text(&plan),
            plan,
            "{csv}: drafted plan changed across a store round trip"
        );
        checked += 1;
    }
    assert!(checked >= 12, "only {checked} scenarios");
}

#[test]
fn an_edited_plan_with_candidates_and_answers_round_trips() {
    let table = read_csv_path(golden_dir().join("plan/wide.csv")).unwrap();
    let plan = draft_plan(&table, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
    let edits = [
        json!({"column": "visits1", "impute": [{"method": "fill_mean", "indicator": true}, {"method": "fill_median"}]}),
    ];
    let edited = apply_answers_and_edits(&plan, &[], &edits, DecidedBy::User).unwrap();
    assert!(edited.entries["visits1"].impute_candidates.is_some());
    assert_eq!(through_text(&edited), edited);

    // An answered question keeps its answer and status.
    let q = plan
        .questions
        .iter()
        .find(|(_, q)| q.topic == pulsar_analysis::plan::QuestionTopic::DateColumns)
        .unwrap();
    let answer = json!({"question_id": q.0, "option_id": "has_date"});
    let answered = apply_answers_and_edits(&plan, &[answer], &[], DecidedBy::Agent).unwrap();
    assert_eq!(through_text(&answered), answered);
    assert!(answered.questions[q.0].answer.is_some());
}

#[test]
fn the_reloaded_plan_behaves_like_the_original() {
    let table = read_csv_path(golden_dir().join("plan/wide.csv")).unwrap();
    let plan = draft_plan(&table, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
    let reloaded = through_text(&plan);
    let edits = [json!({"column": "visits1", "impute_method": "fill_mean"})];
    let (a, b) = (
        apply_answers_and_edits(&plan, &[], &edits, DecidedBy::Agent).unwrap(),
        apply_answers_and_edits(&reloaded, &[], &edits, DecidedBy::Agent).unwrap(),
    );
    assert_eq!(a, b);
}

#[test]
fn a_damaged_stored_plan_is_an_error_naming_the_problem() {
    let table = read_csv_path(golden_dir().join("plan/single.csv")).unwrap();
    let mut value = draft_plan(&table, "ds_aaaaaaaaaaaa", &DraftOptions::default())
        .unwrap()
        .to_value();
    value["entries"]["p"]["role"] = json!("boss");
    let err = PreprocessingPlan::from_value(value).unwrap_err();
    assert!(err.contains("boss") || err.contains("variant"), "{err}");
    assert!(PreprocessingPlan::from_value(json!({"plan_id": "pp_x"}))
        .unwrap_err()
        .contains("missing field"));
}
