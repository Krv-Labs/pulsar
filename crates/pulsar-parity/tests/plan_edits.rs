//! Parity for the plan edit engine: `apply_answers_and_edits` and
//! `diff_plan_entries` must reproduce what
//! `generate_plan_edits.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from Python.
//!
//! Every call is replayed in sequence from the drafted plan. A success is
//! compared whole (revision, every entry, every question, the diff); a failure
//! by its code, its exact message (agents read these) and its details. A
//! failed call leaves the plan unchanged, as in Python.

use std::collections::HashMap;

use pulsar_analysis::plan::{
    apply_answers_and_edits, diff_plan_entries, draft_plan, DecidedBy, DraftOptions,
    PreprocessingPlan,
};
use pulsar_data::read_csv_path;
use pulsar_parity::golden_dir;
use serde_json::{json, Value};

fn fixture() -> Vec<Value> {
    let path = golden_dir().join("plan/edits.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn keys(v: &Value) -> Vec<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

fn source(s: &str) -> DecidedBy {
    match s {
        "agent" => DecidedBy::Agent,
        "user" => DecidedBy::User,
        "default" => DecidedBy::Default,
        other => panic!("unexpected source {other:?} in the fixture"),
    }
}

fn draft(case: &Value) -> PreprocessingPlan {
    let table = read_csv_path(golden_dir().join(case["csv"].as_str().unwrap())).unwrap();
    let d = &case["draft"];
    let opts = DraftOptions {
        target_columns: serde_json::from_value(d["targets"].clone()).unwrap(),
        entity_column: d["entity"].as_str().unwrap().into(),
        time_column: d["time"].as_str().unwrap().into(),
        // These fixtures were frozen with stakes off (see plan_stakes.rs for on).
        stakes: false,
        // Python kept row-index columns as features.
        row_index_ids: false,
        ..DraftOptions::default()
    };
    let mut plan = draft_plan(&table, "ds_fixture", &opts).unwrap();
    plan.plan_id = "pp_fixture".into();
    plan.created_at = "2026-01-01T00:00:00+00:00".into();
    plan
}

fn compare_ok(label: &str, plan: &PreprocessingPlan, diff: &[Value], want: &Value) {
    assert_eq!(
        plan.revision,
        want["revision"].as_u64().unwrap() as u32,
        "{label}: revision"
    );
    assert_eq!(
        plan.n_imputations,
        want["n_imputations"].as_u64().unwrap() as u32,
        "{label}: n_imputations"
    );
    let entries = serde_json::to_value(&plan.entries).unwrap();
    assert_eq!(
        keys(&entries),
        keys(&want["entries"]),
        "{label}: column order"
    );
    for (col, we) in want["entries"].as_object().unwrap() {
        assert_eq!(&entries[col], we, "{label}: entry {col}");
    }
    let questions = serde_json::to_value(&plan.questions).unwrap();
    assert_eq!(
        keys(&questions),
        keys(&want["questions"]),
        "{label}: question order"
    );
    for (qid, wq) in want["questions"].as_object().unwrap() {
        assert_eq!(&questions[qid], wq, "{label}: question {qid}");
    }
    assert_eq!(Value::Array(diff.to_vec()), want["diff"], "{label}: diff");
}

fn replay(case: &Value, base: &PreprocessingPlan) {
    let name = case["name"].as_str().unwrap();
    let mut plan = base.clone();
    for (i, call) in case["calls"].as_array().unwrap().iter().enumerate() {
        let label = format!("{name} call {i}");
        let to_vec = |k: &str| call[k].as_array().unwrap().clone();
        let result = apply_answers_and_edits(
            &plan,
            &to_vec("answers"),
            &to_vec("edits"),
            source(call["source"].as_str().unwrap()),
        );
        let want = &call["result"];
        match (result, want["ok"].as_bool().unwrap()) {
            (Ok(new), true) => {
                compare_ok(&label, &new, &diff_plan_entries(&plan, &new), want);
                plan = new;
            }
            (Err(e), false) => {
                assert_eq!(e.code, want["code"].as_str().unwrap(), "{label}: code");
                let mut got_msg = e.message.clone();
                let mut want_msg = want["message"].as_str().unwrap().to_string();
                let mut got_details = Value::Object(e.details.clone());
                let mut want_details = want["details"].clone();
                // Python collects the columns an option edited in a `set`, so
                // which one the incompatible-options error names depends on
                // hash randomization. Rust names the first in edit order. Both
                // must name a column some option edits; the rest is exact.
                const MERGE: &str = "Selected options for column '";
                if want_msg.starts_with(MERGE) {
                    let named = |m: &str| m[MERGE.len()..].split('\'').next().unwrap().to_string();
                    let (got_col, want_col) = (named(&got_msg), named(&want_msg));
                    let edited: Vec<&str> = plan
                        .questions
                        .values()
                        .flat_map(|q| q.options.iter().flat_map(|o| o.edits.iter()))
                        .filter_map(|ed| ed.get("column").and_then(Value::as_str))
                        .collect();
                    assert!(
                        edited.contains(&got_col.as_str()) && edited.contains(&want_col.as_str()),
                        "{label}: {got_col} / {want_col}"
                    );
                    got_msg = got_msg.replace(&got_col, "<col>");
                    want_msg = want_msg.replace(&want_col, "<col>");
                    got_details["column"] = json!("<col>");
                    want_details["column"] = json!("<col>");
                }
                assert_eq!(got_msg, want_msg, "{label}: message");
                assert_eq!(got_details, want_details, "{label}: details");
            }
            (Ok(_), false) => panic!(
                "{label}: Python raised {:?}, Rust succeeded",
                want["message"]
            ),
            (Err(e), true) => panic!("{label}: Python succeeded, Rust raised {:?}", e.message),
        }
    }
}

#[test]
fn every_edit_and_answer_case_matches_python() {
    let cases = fixture();
    let mut drafts: HashMap<String, PreprocessingPlan> = HashMap::new();
    for case in &cases {
        let key = format!("{}|{}", case["csv"], case["draft"]);
        let base = drafts.entry(key).or_insert_with(|| draft(case));
        replay(case, base);
    }
}

#[test]
fn a_failed_call_never_modifies_the_plan() {
    let cases = fixture();
    let case = cases
        .iter()
        .find(|c| c["name"] == "a_failing_edit_discards_the_whole_call")
        .unwrap();
    let base = draft(case);
    let before = serde_json::to_value(&base.entries).unwrap();
    let edits: Vec<Value> = serde_json::from_value(case["calls"][0]["edits"].clone()).unwrap();
    assert!(apply_answers_and_edits(&base, &[], &edits, DecidedBy::Agent).is_err());
    assert_eq!(serde_json::to_value(&base.entries).unwrap(), before);
    assert_eq!(base.revision, 1);
}

/// Python crashes with an unhashable-type `TypeError` on a list role; the
/// port reports the invalid role instead of panicking.
#[test]
fn a_non_string_role_is_an_invalid_role_not_a_crash() {
    let case = fixture()
        .into_iter()
        .find(|c| c["name"] == "role_invalid")
        .unwrap();
    let base = draft(&case);
    let err = apply_answers_and_edits(
        &base,
        &[],
        &[json!({"column": "age", "role": ["a"]})],
        DecidedBy::Agent,
    )
    .unwrap_err();
    assert_eq!(err.code, "INVALID_ARGUMENT");
    assert!(
        err.message
            .starts_with("Invalid role \"['a']\" for column 'age'"),
        "{}",
        err.message
    );
}

#[test]
fn the_cases_cover_every_validation_branch_and_both_outcomes() {
    // Guards against a vacuous comparison if the fixture drifts.
    let cases = fixture();
    let messages: Vec<String> = cases
        .iter()
        .flat_map(|c| c["calls"].as_array().unwrap().iter())
        .filter(|k| k["result"]["ok"] == json!(false))
        .map(|k| k["result"]["message"].as_str().unwrap().to_string())
        .collect();
    let has = |needle: &str| messages.iter().any(|m| m.contains(needle));
    for needle in [
        "Invalid role",
        "Invalid encode method",
        "Invalid impute_method",
        "Invalid impute method",
        "requires 'impute_value'",
        "cannot be a boolean",
        "only valid with impute_method='fill_value'",
        "must be numeric",
        "cannot be used on text",
        "cannot be used on numbers",
        "has no blanks; indicator",
        "has no blanks; impute",
        "Invalid indicator",
        "group must be an evidence missingness group",
        "Unknown column",
        "Unsupported edit field",
        "Each edit must be a mapping",
        "Unknown question_id",
        "Unknown option_id",
        "must not be empty",
        "no longer applies",
        "incompatible roles",
        "Nothing to change",
        "Invalid source",
        "must be an integer",
        "must be a boolean",
        "cannot contain null entries",
        "cannot contain 'none'",
        "must be a mapping",
        "Each answer must be a mapping",
    ] {
        assert!(has(needle), "no fixture case raises: {needle}");
    }
    let ok = cases
        .iter()
        .flat_map(|c| c["calls"].as_array().unwrap())
        .filter(|k| k["result"]["ok"] == json!(true))
        .count();
    assert!(ok > 80, "only {ok} successful calls");
}
