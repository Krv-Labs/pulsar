//! Parity for the plan question builders: `build_pending_questions` must
//! reproduce what `generate_plan_questions.py` (Python generator, see `tests/fixtures/golden/README.md`)
//! froze from the Python `preprocessing_plan.questions`. Question text,
//! options and edits are strings and ids, so everything is compared exactly,
//! including the entries the builders mutate and `blank_group_cols`.

use pulsar_analysis::plan::{
    build_pending_questions, populate_default_entries, validate_role_designations, DraftContext,
};
use pulsar_analysis::{measure_preprocessing_evidence, EvidenceOptions};
use pulsar_data::read_csv_path;
use pulsar_parity::golden_dir;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = golden_dir().join("plan/questions.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    serde_json::from_value(v.clone()).unwrap()
}

fn run(scenario: &Value) -> Value {
    let table = read_csv_path(golden_dir().join(scenario["csv"].as_str().unwrap())).unwrap();
    let evidence = measure_preprocessing_evidence(&table, &EvidenceOptions::default()).unwrap();
    let args = &scenario["args"];
    let (targets, entity, time) = validate_role_designations(
        &table.names,
        &strings(&args["targets"]),
        args["entity"].as_str().unwrap(),
        args["time"].as_str().unwrap(),
    )
    .unwrap();
    let mut ctx = DraftContext::new(&table, &evidence, &targets, &entity, &time);
    populate_default_entries(&mut ctx);
    let pending = build_pending_questions(&mut ctx);
    let mut blank: Vec<&String> = ctx.blank_group_cols.iter().collect();
    blank.sort();
    json!({
        "questions": pending
            .iter()
            .map(|(q, cols)| json!({"question": q, "stake_cols": cols}))
            .collect::<Vec<_>>(),
        "entries_after": ctx.entries,
        "blank_group_cols": blank,
    })
}

fn compare(name: &str) {
    let fx = fixture();
    let want = &fx[name];
    let got = run(want);

    let (gq, wq) = (
        got["questions"].as_array().unwrap(),
        want["questions"].as_array().unwrap(),
    );
    let topics = |qs: &[Value]| -> Vec<String> {
        qs.iter()
            .map(|q| q["question"]["topic"].as_str().unwrap().to_string())
            .collect()
    };
    assert_eq!(topics(gq), topics(wq), "{name}: question topics/order");
    for (i, (g, w)) in gq.iter().zip(wq).enumerate() {
        let topic = w["question"]["topic"].as_str().unwrap();
        assert_eq!(
            g["stake_cols"], w["stake_cols"],
            "{name}[{i}] {topic}: stake_cols"
        );
        for key in [
            "text",
            "applies_to",
            "rows_affected",
            "default_option_id",
            "status",
            "stakes",
            "answer",
        ] {
            assert_eq!(
                g["question"][key], w["question"][key],
                "{name}[{i}] {topic}: {key}"
            );
        }
        let (go, wo) = (
            g["question"]["options"].as_array().unwrap(),
            w["question"]["options"].as_array().unwrap(),
        );
        assert_eq!(go.len(), wo.len(), "{name}[{i}] {topic}: option count");
        for (j, (a, b)) in go.iter().zip(wo).enumerate() {
            assert_eq!(a, b, "{name}[{i}] {topic}: option {j} ({})", b["option_id"]);
        }
    }
    assert_eq!(
        got["blank_group_cols"], want["blank_group_cols"],
        "{name}: blank_group_cols"
    );
    let cols = |v: &Value| -> Vec<String> {
        v["entries_after"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    };
    assert_eq!(cols(&got), cols(want), "{name}: column order");
    for (col, entry) in want["entries_after"].as_object().unwrap() {
        assert_eq!(
            &got["entries_after"][col], entry,
            "{name}: entry {col} after the builders ran"
        );
    }
}

#[test]
fn plain_questions_match_python() {
    compare("plain");
}

#[test]
fn designated_roles_questions_match_python() {
    compare("roles");
}

#[test]
fn wide_table_questions_match_python() {
    compare("wide");
}

#[test]
fn wide_table_with_targets_matches_python() {
    compare("wide_targets");
}

#[test]
fn singular_wording_branches_match_python() {
    compare("single");
}

#[test]
fn each_mixed_values_wording_branch_matches_python() {
    for name in [
        "sums",
        "mixed_one_mostly",
        "mixed_one_minor",
        "mixed_many_single",
        "mixed_many_minor",
    ] {
        compare(name);
    }
}

#[test]
fn every_builder_and_wording_branch_is_exercised() {
    // Guards against a vacuous comparison if the fixtures drift.
    let fx = fixture();
    let mut topics = std::collections::BTreeSet::new();
    let mut texts = Vec::new();
    for sc in fx.as_object().unwrap().values() {
        for q in sc["questions"].as_array().unwrap() {
            topics.insert(q["question"]["topic"].as_str().unwrap().to_string());
            texts.push(q["question"]["text"].as_str().unwrap().to_string());
        }
    }
    for topic in [
        "blank_meaning",
        "code_columns",
        "conditional_column",
        "date_columns",
        "duplicate_signal",
        "mixed_values",
        "mostly_blank",
        "outcome_columns",
    ] {
        assert!(
            topics.contains(topic),
            "no {topic} question in the fixtures"
        );
    }
    let any = |needle: &str| texts.iter().any(|t| t.contains(needle));
    // The six wording branches of `mixed_values`: one or several columns x
    // a single shared number / mostly numbers / not mostly numbers.
    assert!(any("holds the number 5"), "one column, single number");
    assert!(any("mostly holds numbers"), "one column, mostly numbers");
    assert!(
        any("` holds numbers and the value"),
        "one column, not mostly numbers"
    );
    assert!(
        any("columns hold the number"),
        "several columns, single number"
    );
    assert!(
        any("columns mostly hold numbers"),
        "several columns, mostly numbers"
    );
    assert!(
        any("columns hold numbers and the value"),
        "several columns, not mostly numbers"
    );
    assert!(any("1 total in this table"), "single sum");
    assert!(any("totals in this table are exact sums"), "several sums");
    assert!(any("has identical values to"), "single duplicate");
    assert!(any("columns have identical values"), "several duplicates");
    assert!(any("including `"), "mostly-blank with 3+ others");
    assert!(any("blank exactly where"), "blank-meaning partner sentence");
}
