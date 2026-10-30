//! Parity for the draft/edit tool payloads and their Markdown: the Rust
//! builders must reproduce what
//! `generate_plan_payload.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from Python,
//! on the path the tools take (draft with stakes, dry run, payload; or apply
//! answers and edits, diff, dry run, payload).
//!
//! Compared exactly: every field of every payload, in summary, tiny-limit
//! (caps and omitted counts) and full variants, and the Markdown line for
//! line. Masked, because PyYAML and serde_yaml_ng render the same plan
//! differently: the YAML text and `preprocessing_yaml_chars`. Also masked: the
//! timing fields of `cost`, the wording of exception text in `stakes_notes`
//! (and of a dry-run validation error).

use pulsar_analysis::plan::{
    apply_answers_and_edits, build_draft_plan_payload, build_edit_plan_payload, diff_plan_entries,
    draft_plan, draft_plan_payload_to_markdown, dry_run_processed_space,
    edit_plan_payload_to_markdown, DecidedBy, Detail, DraftOptions, Limits, PreprocessingPlan,
};
use pulsar_data::read_csv_path;
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::{json, Value};

/// The evidence tolerance (see `evidence.rs`).
const RTOL: f64 = 1e-9;

fn fixture() -> Value {
    let path = golden_dir().join("plan/payloads.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn drafted(case: &Value, question_limit: usize) -> (pulsar_data::Table, PreprocessingPlan) {
    let table = read_csv_path(golden_dir().join(case["csv"].as_str().unwrap())).unwrap();
    let args = &case["args"];
    let list = |k: &str| -> Vec<String> {
        args.get(k)
            .map_or(vec![], |v| serde_json::from_value(v.clone()).unwrap())
    };
    let text = |k: &str| {
        args.get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let opts = DraftOptions {
        target_columns: list("targets"),
        entity_column: text("entity"),
        time_column: text("time"),
        stakes: true,
        stakes_max_questions: 2 * question_limit,
        // Python kept row-index columns as features.
        row_index_ids: false,
        ..DraftOptions::default()
    };
    let mut plan = draft_plan(&table, "ds_fixture", &opts).unwrap();
    plan.plan_id = "pp_fixture".into();
    plan.created_at = "2026-01-01T00:00:00+00:00".into();
    (table, plan)
}

fn limits(v: &Value) -> Limits {
    let d = Limits::default();
    let get = |k: &str, default: usize| {
        v.get(k)
            .and_then(Value::as_u64)
            .map_or(default, |n| n as usize)
    };
    Limits {
        question_limit: get("question_limit", d.question_limit),
        evidence_limit: get("evidence_limit", d.evidence_limit),
        change_limit: get("change_limit", d.change_limit),
    }
}

/// What a note is about, without the exception wording after the first `": "`.
fn mask_payload(payload: &mut Value) {
    let obj = payload.as_object_mut().unwrap();
    obj.remove("preprocessing_yaml_chars");
    obj.remove("preprocessing_yaml");
    if let Some(cost) = obj.get_mut("cost").and_then(Value::as_object_mut) {
        cost.remove("evidence_ms");
        cost.remove("stakes_ms");
        if let Some(notes) = cost.get_mut("stakes_notes").and_then(Value::as_array_mut) {
            for n in notes.iter_mut() {
                *n = json!(n.as_str().unwrap().split(": ").next().unwrap());
            }
        }
    }
    if let Some(v) = obj.get_mut("validation").and_then(Value::as_object_mut) {
        if v["error"].is_string() {
            v.insert("error".into(), json!("<error>"));
        }
    }
}

/// Mask the lines that depend on how YAML is rendered or on error wording.
fn mask_markdown(md: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_yaml = false;
    for line in md.lines() {
        if line == "```yaml" {
            in_yaml = true;
            out.push(line.to_string());
            out.push("<yaml>".into());
        } else if in_yaml {
            if line == "```" {
                in_yaml = false;
                out.push(line.to_string());
            }
        } else if line.starts_with("- Preprocessing YAML: ") && line.contains(" chars") {
            out.push("- Preprocessing YAML: <n> chars".into());
        } else if line.starts_with("- Validation error: ") {
            out.push("- Validation error: <error>".into());
        } else {
            out.push(line.to_string());
        }
    }
    out
}

fn compare_payload(label: &str, got: &Value, want: &Value, got_md: &str, want_md: &str) {
    let (mut got, mut want) = (got.clone(), want.clone());
    mask_payload(&mut got);
    mask_payload(&mut want);
    let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(keys(&got), keys(&want), "{label}: payload keys");
    // Floats within the evidence tolerance (plain vs pairwise summation moves
    // correlations in the last digits); strings, integers and structure exact.
    for (k, w) in want.as_object().unwrap() {
        assert_json_close(&format!("{label}: payload[{k}]"), &got[k], w, RTOL);
    }
    assert_eq!(
        mask_markdown(got_md),
        mask_markdown(want_md),
        "{label}: markdown"
    );
}

#[test]
fn every_draft_payload_and_markdown_matches_python() {
    let fx = fixture();
    let mut checked = 0;
    for (name, case) in fx["draft"].as_object().unwrap() {
        for (variant, v) in case["variants"].as_object().unwrap() {
            let l = limits(&v["limits"]);
            let (table, mut plan) = drafted(case, l.question_limit);
            let (space, _) = dry_run_processed_space(&table, &mut plan);
            let detail = if v["detail"] == "full" {
                Detail::Full
            } else {
                Detail::Summary
            };
            let payload = build_draft_plan_payload(&plan, space.as_ref(), detail, l);
            let md = draft_plan_payload_to_markdown(&payload);
            compare_payload(
                &format!("draft {name}/{variant}"),
                &payload,
                &v["payload"],
                &md,
                v["markdown"].as_str().unwrap(),
            );
            checked += 1;
        }
    }
    assert!(checked >= 30, "only {checked} draft payloads");
}

#[test]
fn every_edit_payload_and_markdown_matches_python() {
    let fx = fixture();
    let mut checked = 0;
    for (name, case) in fx["edit"].as_object().unwrap() {
        for (variant, v) in case["variants"].as_object().unwrap() {
            let (table, plan) = drafted(case, 8);
            let answers: Vec<Value> = serde_json::from_value(case["answers"].clone()).unwrap();
            let edits: Vec<Value> = serde_json::from_value(case["edits"].clone()).unwrap();
            let mut updated =
                apply_answers_and_edits(&plan, &answers, &edits, DecidedBy::Agent).unwrap();
            let changes = diff_plan_entries(&plan, &updated);
            let (space, validation) = dry_run_processed_space(&table, &mut updated);
            let payload = build_edit_plan_payload(
                &updated,
                &changes,
                space.as_ref(),
                &validation,
                limits(&v["limits"]),
            );
            let md = edit_plan_payload_to_markdown(&payload);
            compare_payload(
                &format!("edit {name}/{variant}"),
                &payload,
                &v["payload"],
                &md,
                v["markdown"].as_str().unwrap(),
            );
            checked += 1;
        }
    }
    assert!(checked >= 14, "only {checked} edit payloads");
}

#[test]
fn the_fixtures_exercise_caps_warnings_groups_and_both_next_tool_sets() {
    // Guards against a vacuous comparison if the fixtures drift.
    let fx = fixture();
    let drafts = fx["draft"].as_object().unwrap().values();
    let edits = fx["edit"].as_object().unwrap().values();
    let draft_payloads: Vec<&Value> = drafts
        .flat_map(|c| {
            c["variants"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| &v["payload"])
        })
        .collect();
    let edit_payloads: Vec<&Value> = edits
        .flat_map(|c| {
            c["variants"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| &v["payload"])
        })
        .collect();

    assert!(
        draft_payloads
            .iter()
            .any(|p| p["questions_omitted"].as_u64().unwrap() > 0),
        "an omitted-questions cap"
    );
    assert!(
        draft_payloads.iter().any(
            |p| p["evidence"]["relations_omitted"].as_u64().unwrap_or(0) > 0
                || p["evidence"]["most_missing_columns_omitted"]
                    .as_u64()
                    .unwrap_or(0)
                    > 0
        ),
        "an omitted-evidence cap"
    );
    assert!(
        draft_payloads
            .iter()
            .any(|p| !p["warnings"].as_array().unwrap().is_empty()),
        "a target-proxy warning"
    );
    assert!(
        draft_payloads
            .iter()
            .any(|p| p["processed_space"].is_null()),
        "an unavailable dry run"
    );
    assert!(
        draft_payloads.iter().any(|p| p.get("entries").is_some()),
        "a full payload"
    );
    assert!(
        edit_payloads
            .iter()
            .any(|p| p["change_summary"]
                .as_array()
                .unwrap()
                .iter()
                .any(|g| g["count"].as_u64().unwrap() > 1)),
        "grouped identical changes"
    );
    assert!(
        edit_payloads
            .iter()
            .any(|p| p["changes_omitted"].as_u64().unwrap() > 0),
        "an omitted-changes cap"
    );
    assert!(
        edit_payloads
            .iter()
            .any(|p| p["counts"]["confirmed_without_change"].as_u64().unwrap() > 0),
        "confirmed without change"
    );
    assert!(
        edit_payloads
            .iter()
            .any(|p| p["validation"]["valid"] == json!(false)),
        "an invalid plan"
    );
    let next: Vec<String> = edit_payloads
        .iter()
        .map(|p| p["next_tools"].to_string())
        .collect();
    assert!(
        next.iter()
            .any(|n| n.contains("validate_preprocessing_config"))
            && next
                .iter()
                .any(|n| !n.contains("validate_preprocessing_config")),
        "both next-tool sets"
    );
}
