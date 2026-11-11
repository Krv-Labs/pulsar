//! Parity for `draft_plan` (without stakes), plan rendering and the
//! processed-space dry run: the Rust port must reproduce what
//! `generate_plan_draft.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from Python.
//!
//! Exact: question ids/order/text, entries, cost counters, target proxies,
//! the `preprocessing:` dict, the fingerprint (which hashes Python's
//! canonical JSON), marker legend, counts, processed-space sizes and whether
//! the plan is valid. Compared parsed, not byte for byte: the YAML text
//! (PyYAML and serde_yaml_ng quote differently). Not compared: timing, ids
//! and the wording of dry-run errors (only the column they name).

use pulsar_analysis::plan::{
    build_marker_legend, draft_plan, dry_run_processed_space, plan_fingerprint,
    plan_to_preprocessing_dict, plan_to_preprocessing_yaml, summarize_plan_counts, DraftOptions,
    ImputeSpec,
};
use pulsar_data::read_csv_path;
use pulsar_parity::golden_dir;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = golden_dir().join("plan/draft.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    serde_json::from_value(v.clone()).unwrap()
}

fn keys(v: &Value) -> Vec<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

fn compare(name: &str) {
    let fx = fixture();
    let want = &fx[name];
    let table = read_csv_path(golden_dir().join(want["csv"].as_str().unwrap())).unwrap();
    let args = &want["args"];
    let opts = DraftOptions {
        target_columns: strings(&args["targets"]),
        entity_column: args["entity"].as_str().unwrap().into(),
        time_column: args["time"].as_str().unwrap().into(),
        // These fixtures were frozen with stakes off (see plan_stakes.rs for on).
        stakes: false,
        ..DraftOptions::default()
    };
    let mut plan =
        draft_plan(&table, "ds_fixture", &opts).unwrap_or_else(|e| panic!("{name}: {e}"));

    // Pin what is random or time-dependent, as the generator did.
    plan.plan_id = "pp_fixture".into();
    plan.created_at = "2026-01-01T00:00:00+00:00".into();

    // The same mutation the generator applied before rendering.
    let mutations = &want["mutations"];
    if let Some(cands) = mutations["impute_candidates"].as_object() {
        for (col, specs) in cands {
            let specs: Vec<ImputeSpec> = serde_json::from_value(specs.clone()).unwrap();
            plan.entries.get_mut(col).unwrap().impute_candidates = Some(specs);
        }
    }
    if let Some(methods) = mutations["impute_method"].as_object() {
        for (col, m) in methods {
            plan.entries.get_mut(col).unwrap().impute =
                Some(ImputeSpec::method(m.as_str().unwrap()));
        }
    }
    if let Some(n) = mutations["n_imputations"].as_u64() {
        plan.n_imputations = n as u32;
    }

    // The dry run also refreshes the target proxies, so run it before comparing them.
    let table_for_dry = table.clone();
    let (space, validity) = dry_run_processed_space(&table_for_dry, &mut plan);

    // --- the drafted plan ---
    let questions = serde_json::to_value(&plan.questions).unwrap();
    assert_eq!(
        keys(&questions),
        keys(&want["questions"]),
        "{name}: question ids/order"
    );
    for (qid, wq) in want["questions"].as_object().unwrap() {
        assert_eq!(&questions[qid], wq, "{name}: question {qid}");
    }
    let entries = serde_json::to_value(&plan.entries).unwrap();
    assert_eq!(
        keys(&entries),
        keys(&want["entries"]),
        "{name}: column order"
    );
    for (col, we) in want["entries"].as_object().unwrap() {
        assert_eq!(&entries[col], we, "{name}: entry {col}");
    }
    let mut cost = plan.cost.clone();
    cost.remove("evidence_ms");
    assert_eq!(Value::Object(cost), want["cost"], "{name}: cost counters");
    assert_eq!(
        serde_json::to_value(&plan.target_proxies).unwrap(),
        want["target_proxies"],
        "{name}: target proxies"
    );

    // --- rendering ---
    let dict = Value::Object(plan_to_preprocessing_dict(&plan));
    assert_eq!(
        dict, want["preprocessing_dict"],
        "{name}: preprocessing dict"
    );
    assert_eq!(
        serde_json::to_string(&keys(&dict)).unwrap(),
        serde_json::to_string(&keys(&want["preprocessing_dict"])).unwrap(),
        "{name}: preprocessing section order"
    );
    assert_eq!(
        plan_fingerprint(&plan),
        want["fingerprint"].as_str().unwrap(),
        "{name}: fingerprint"
    );

    let (got_yaml, want_yaml) = (
        plan_to_preprocessing_yaml(&plan, true),
        want["yaml"].as_str().unwrap(),
    );
    assert_eq!(
        got_yaml.lines().next(),
        want_yaml.lines().next(),
        "{name}: yaml header"
    );
    let parse = |s: &str| serde_yaml_ng::from_str::<Value>(s).unwrap();
    assert_eq!(parse(&got_yaml), parse(want_yaml), "{name}: yaml content");

    assert_eq!(
        Value::Array(build_marker_legend(&plan, 5)),
        want["marker_legend"],
        "{name}: marker legend"
    );
    assert_eq!(
        summarize_plan_counts(&plan),
        want["counts"],
        "{name}: counts"
    );

    // --- dry run ---
    assert_eq!(
        validity.valid,
        want["valid"].as_bool().unwrap(),
        "{name}: valid ({:?})",
        validity.error
    );
    assert_eq!(
        space
            .map(|s| serde_json::to_value(s).unwrap())
            .unwrap_or(Value::Null),
        want["processed_space"],
        "{name}: processed space"
    );
    if let Some(py_error) = want["error"].as_str() {
        let rust_error = validity.error.unwrap();
        // Wording differs; the offending column must be the same.
        if let Some(col) = py_error
            .split("Column '")
            .nth(1)
            .and_then(|r| r.split('\'').next())
        {
            assert!(
                rust_error.contains(col),
                "{name}: rust error should name {col}: {rust_error}"
            );
        }
    }
}

macro_rules! scenarios {
    ($($test:ident => $name:literal),* $(,)?) => {
        $(#[test] fn $test() { compare($name); })*
    };
}

scenarios! {
    plain_draft_matches_python => "plain",
    designated_roles_draft_matches_python => "roles",
    wide_draft_matches_python => "wide",
    wide_with_targets_draft_matches_python => "wide_targets",
    single_draft_matches_python => "single",
    sums_draft_matches_python => "sums",
    penguins_draft_matches_python => "penguins",
    non_ascii_names_draft_matches_python => "unicode",
    target_proxies_for_a_boolean_target_match_python => "proxy_event",
    target_proxies_for_a_numeric_target_match_python => "proxy_nev",
    multi_imputation_render_matches_python => "multi_imputation",
    an_invalid_plan_is_reported_not_raised => "invalid",
    mixed_one_mostly_draft_matches_python => "mixed_one_mostly",
    mixed_one_minor_draft_matches_python => "mixed_one_minor",
    mixed_many_single_draft_matches_python => "mixed_many_single",
    mixed_many_minor_draft_matches_python => "mixed_many_minor",
}

#[test]
fn the_fixtures_exercise_the_interesting_paths() {
    // Guards against a vacuous comparison if the fixtures drift.
    let fx = fixture();
    let all = fx.as_object().unwrap();
    let any = |f: &dyn Fn(&Value) -> bool| all.values().any(f);
    assert!(any(&|s| s["valid"] == json!(true)) && any(&|s| s["valid"] == json!(false)));
    assert!(
        any(&|s| s["target_proxies"]
            .as_array()
            .is_some_and(|p| p.iter().any(|x| x["share_with_target"] == json!(1.0)))),
        "a proxy with share 1.0"
    );
    assert!(
        any(&|s| s["target_proxies"].as_array().is_some_and(|p| p
            .iter()
            .any(|x| x["share_with_target"].as_f64().is_some_and(|v| v < 1.0)))),
        "a partial-overlap proxy"
    );
    assert!(
        any(&|s| s["preprocessing_dict"]["n_imputations"] == json!(3)),
        "multi-imputation"
    );
    assert!(
        any(&|s| s["preprocessing_dict"]["missing_indicator_groups"].is_object()),
        "shared indicator groups"
    );
    assert!(
        // Python's YAML escapes non-ASCII, so look at the parsed config dict.
        any(&|s| {
            let dict = s["preprocessing_dict"].to_string();
            dict.contains("naïve") && dict.contains("日本")
        }),
        "non-ASCII names"
    );
    assert!(
        any(&|s| s["marker_legend"].as_array().is_some_and(|l| !l.is_empty())),
        "marker legend"
    );
}
