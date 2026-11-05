//! Parity for counterfactual stakes: `draft_plan(stakes = true)` must
//! reproduce what `generate_plan_stakes.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze from Python: which questions are scored, each `neighbors_kept`,
//! the resulting question order and `q1..qN` ids, the entries' question links
//! and the cost counters.
//!
//! Not compared: timings, and the wording of the exception text inside
//! `stakes_notes` (only how many notes there are and what each is about).

use pulsar_analysis::plan::{draft_plan, DraftOptions};
use pulsar_data::read_csv_path;
use pulsar_parity::golden_dir;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = golden_dir().join("plan/stakes.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn keys(v: &Value) -> Vec<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

fn options(want: &Value) -> DraftOptions {
    let args = &want["args"];
    let o = &want["options"];
    let mut opts = DraftOptions {
        target_columns: serde_json::from_value(args["targets"].clone()).unwrap(),
        entity_column: args["entity"].as_str().unwrap().into(),
        time_column: args["time"].as_str().unwrap().into(),
        ..DraftOptions::default()
    };
    if let Some(b) = o["stakes"].as_bool() {
        opts.stakes = b;
    }
    if let Some(k) = o["stakes_k"].as_u64() {
        opts.stakes_k = k as usize;
    }
    if let Some(n) = o["stakes_max_questions"].as_u64() {
        opts.stakes_max_questions = n as usize;
    }
    opts
}

/// What a note is about, without the exception wording after the first `": "`.
fn note_topic(note: &str) -> String {
    note.split(": ").next().unwrap_or(note).to_string()
}

fn compare(name: &str) {
    let fx = fixture();
    let want = &fx[name];
    let table = read_csv_path(golden_dir().join(want["csv"].as_str().unwrap())).unwrap();
    let plan =
        draft_plan(&table, "ds_fixture", &options(want)).unwrap_or_else(|e| panic!("{name}: {e}"));

    let questions = serde_json::to_value(&plan.questions).unwrap();
    assert_eq!(
        keys(&questions),
        keys(&want["questions"]),
        "{name}: question ids/order"
    );
    for (qid, wq) in want["questions"].as_object().unwrap() {
        assert_eq!(
            &questions[qid]["stakes"], &wq["stakes"],
            "{name}: {qid} stakes ({})",
            wq["topic"]
        );
        assert_eq!(&questions[qid], wq, "{name}: question {qid}");
    }
    let entries = serde_json::to_value(&plan.entries).unwrap();
    for (col, we) in want["entries"].as_object().unwrap() {
        assert_eq!(&entries[col], we, "{name}: entry {col}");
    }

    let mut cost = plan.cost.clone();
    for k in ["evidence_ms", "stakes_ms"] {
        cost.remove(k);
    }
    let notes = |v: Option<&Value>| -> Vec<String> {
        v.and_then(Value::as_array)
            .map(|a| a.iter().map(|n| note_topic(n.as_str().unwrap())).collect())
            .unwrap_or_default()
    };
    assert_eq!(
        notes(cost.get("stakes_notes")),
        notes(want["cost"].get("stakes_notes")),
        "{name}: stakes notes"
    );
    cost.remove("stakes_notes");
    let mut want_cost = want["cost"].as_object().unwrap().clone();
    want_cost.remove("stakes_notes");
    assert_eq!(
        Value::Object(cost),
        Value::Object(want_cost),
        "{name}: cost counters"
    );
}

macro_rules! scenarios {
    ($($test:ident => $name:literal),* $(,)?) => {
        $(#[test] fn $test() { compare($name); })*
    };
}

scenarios! {
    plain_stakes_match_python => "plain",
    roles_stakes_match_python => "roles",
    wide_stakes_match_python => "wide",
    wide_with_targets_stakes_match_python => "wide_targets",
    single_stakes_match_python => "single",
    sums_stakes_match_python => "sums",
    penguins_stakes_match_python => "penguins",
    unicode_stakes_match_python => "unicode",
    proxy_event_stakes_match_python => "proxy_event",
    proxy_nev_stakes_match_python => "proxy_nev",
    wide_k3_stakes_match_python => "wide_k3",
    wide_k1_stakes_match_python => "wide_k1",
    wide_budget_2_stakes_match_python => "wide_budget_2",
    wide_budget_0_stakes_match_python => "wide_budget_0",
    wide_stakes_off_matches_python => "wide_stakes_off",
    single_k5_stakes_match_python => "single_k5",
    sums_k20_stakes_match_python => "sums_k20",
    mixed_one_mostly_stakes_match_python => "mixed_one_mostly",
    mixed_one_minor_stakes_match_python => "mixed_one_minor",
    mixed_many_single_stakes_match_python => "mixed_many_single",
    mixed_many_minor_stakes_match_python => "mixed_many_minor",
    a_failing_option_is_noted_and_the_rest_scored => "mixed_cat_commas",
}

#[test]
fn the_fixtures_exercise_scoring_failures_and_budgets() {
    // Guards against a vacuous comparison if the fixtures drift.
    let fx = fixture();
    let all = fx.as_object().unwrap();
    let scored = |s: &Value| {
        s["questions"]
            .as_object()
            .unwrap()
            .values()
            .filter(|q| !q["stakes"].is_null())
            .count()
    };
    assert!(
        all.values().filter(|s| scored(s) > 0).count() >= 6,
        "scored questions in several scenarios"
    );
    assert!(
        all.values()
            .any(|s| s["cost"]["stakes_notes"].as_array().is_some_and(|n| n
                .iter()
                .any(|x| x.as_str().unwrap().starts_with("Baseline dry-run failed")))),
        "baseline failure"
    );
    assert!(
        all.values()
            .any(|s| s["cost"]["stakes_notes"].as_array().is_some_and(|n| n
                .iter()
                .any(|x| x.as_str().unwrap().starts_with("Stakes option failed")))),
        "a failing option"
    );
    assert!(
        all.values()
            .any(|s| s["cost"]["stakes_questions_skipped"].as_u64().unwrap() > 0),
        "a question budget"
    );
    assert!(
        all.values()
            .any(|s| s["cost"]["stakes_questions_failed"].as_u64().unwrap() > 0),
        "unscored questions"
    );
    assert_eq!(
        fx["wide_stakes_off"]["questions"]["q1"]["stakes"],
        json!(null)
    );
}
