//! `merge_small_groups` end to end: a cohort with a far blob of 9 rows from
//! two patients (a cluster of enough rows but under 8 patients) is merged
//! into its nearest group, and the dossier is then built on the merged
//! assignment without clustering again.

mod common;

use std::path::{Path, PathBuf};

use common::{call_json as call, ok_json as ok, release_of, serve, serve_phirewall, TempRoot};
use pulsar_store::{ClusterAssignment, ClusterAssignmentId};
use rmcp::service::{RoleClient, RunningService};
use rmcp::ClientHandler;
use serde_json::{json, Value};

const SUMMARY: &str = "io.phirewall/summary";

/// A Phirewall token for patient `i` (`phirewall-core/src/token.rs`).
fn token(i: usize) -> String {
    const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    const CHECK: &[u8; 37] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ*~$=U";
    let mut data = *b"D7000000";
    let mut n = i;
    for slot in data[2..].iter_mut().rev() {
        *slot = ALPHABET[n % 32];
        n /= 32;
    }
    let check = data.iter().fold(0u32, |acc, c| {
        (acc * 32 + ALPHABET.iter().position(|a| a == c).unwrap() as u32) % 37
    });
    let d = std::str::from_utf8(&data).unwrap();
    format!("{}-{}-{}", &d[..2], &d[2..], CHECK[check as usize] as char)
}

/// Two blobs of 20 patients x 3 rows, and a far blob of 9 rows from two
/// patients; `site` has two common levels and one of a single patient.
fn cohort(dir: &Path) -> PathBuf {
    let mut csv = String::from("pid,x0,x1,x2,x3,site\n");
    let jitter = |i: usize, j: usize| ((i * 7 + j * 13) % 11) as f64 / 110.0;
    for i in 0..129 {
        let (patient, center, site) = match i {
            0..=59 => (i / 3, 0.0, if i == 0 { "C" } else { "A" }),
            60..=119 => (i / 3, 5.0, "B"),
            _ => (40 + i % 2, -6.0, "B"),
        };
        let x: Vec<String> = (0..4)
            .map(|j| {
                let sign = if center < 0.0 && j % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                format!("{:.4}", sign * center + jitter(i, j))
            })
            .collect();
        csv.push_str(&format!("{},{},{site}\n", token(patient), x.join(",")));
    }
    let path = dir.join("cohort.csv");
    std::fs::write(&path, csv).unwrap();
    path
}

const CONFIG: &str = "run:\n  name: disclosure\npreprocessing:\n  drop_columns:\n  - pid\n  - site\nsweep:\n  projection:\n    method: jl\n    dimensions:\n      values:\n      - 2\n      - 3\n    seed:\n      values:\n      - 42\n  ball_mapper:\n    epsilon:\n      values:\n      - 0.8\n      - 1.2\ncosmic_graph:\n  construction: exact\n  construction_threshold: auto\n";

fn summary_validator() -> jsonschema::Validator {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/phirewall-summary-v1.json");
    let schema: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

async fn sweep<H: ClientHandler>(client: &RunningService<RoleClient, H>, csv: &Path, config: &str) {
    let ds = ok(
        client,
        "ingest_dataset",
        json!({"path": csv.display().to_string()}),
    )
    .await;
    ok(
        client,
        "run_topological_sweep",
        json!({"config_yaml": config, "dataset_id": ds["dataset_id"], "response_format": "json"}),
    )
    .await;
}

fn cluster_ids(dossier: &Value) -> Vec<i64> {
    let mut ids: Vec<i64> = dossier["clusters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["cluster_id"].as_i64().unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn merging_a_small_cluster_then_the_dossier_on_the_merged_assignment() {
    let root = TempRoot::new("merge");
    let csv = cohort(&root.0);
    let (state, client) = serve_phirewall(&root.0).await;
    sweep(&client, &csv, CONFIG).await;
    let dossier = ok(
        &client,
        "generate_cluster_dossier",
        json!({"response_format": "json", "method": "components", "interpretation_edge_weight_threshold": 0.0}),
    )
    .await;
    let before = dossier["cluster_count"].as_u64().unwrap();
    let original = dossier["cluster_assignment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(before >= 3, "the far blob is its own cluster: {before}");
    let original_labels = state
        .store
        .cluster_labels(&ClusterAssignmentId::parse(&original).unwrap())
        .unwrap();
    // Rows 120..129 are the far blob: 9 rows from two patients.
    let far = original_labels[120];
    assert_eq!(
        original_labels[120..].iter().filter(|&&l| l == far).count(),
        9,
        "the far blob is one cluster"
    );

    let merged = ok(&client, "merge_small_groups", json!({})).await;
    let merges = merged["merges"].as_array().unwrap();
    assert!(!merges.is_empty(), "{merged}");
    for m in merges {
        assert!(["edge_weight", "isolated"].contains(&m["linkage"].as_str().unwrap()));
        if m["from"].is_array() {
            // A collapsed run of isolated merges.
            assert_eq!(m["linkage"], "isolated");
            assert_eq!(
                m["count"].as_u64(),
                m["from"].as_array().map(|f| f.len() as u64)
            );
        } else {
            assert!(m["from"].is_i64() && m["into"].is_i64() && m["group"].is_i64());
            let share = m["weight_share"].as_f64().unwrap();
            assert!((0.0..=1.0).contains(&share));
        }
    }
    assert!(
        merges.iter().any(|m| m["from"].as_i64() == Some(far)
            || m["from"]
                .as_array()
                .is_some_and(|f| f.iter().any(|v| v.as_i64() == Some(far)))),
        "the 9-row, 2-patient blob was not merged: {merged}"
    );
    assert_eq!(merged["merged_from"], original);
    let new_id = merged["cluster_assignment_id"].as_str().unwrap().to_owned();
    assert_ne!(new_id, original);
    let n = merged["n_clusters"].as_u64().unwrap();
    assert!(n < before);
    // Sizes are disclosed row counts: multiples of 5, none under the floor.
    let sizes = merged["cluster_sizes"].as_array().unwrap();
    assert_eq!(sizes.len() as u64, n);
    for s in sizes {
        assert!(
            s["size"].as_u64().is_some_and(|v| v >= 10 && v % 5 == 0),
            "{s}"
        );
    }

    // The summary carries the unchanged grouping plus the optional fields.
    let summary = &merged[SUMMARY];
    let errors: Vec<String> = summary_validator()
        .iter_errors(summary)
        .map(|e| format!("{e} at {}", e.instance_path()))
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    let grouping = &summary["grouping"];
    assert_eq!(grouping["merged_from"], original);
    let summary_merges = grouping["merges"].as_array().unwrap();
    let total_merged: usize = merges
        .iter()
        .map(|m| m["count"].as_u64().unwrap_or(1) as usize)
        .sum();
    assert_eq!(summary_merges.len(), total_merged, "{grouping}");
    for m in summary_merges {
        assert!(m["from"].is_i64() && m["into"].is_i64() && m["group"].is_i64());
        assert!(["edge_weight", "isolated"].contains(&m["linkage"].as_str().unwrap()));
        let share = m["weight_share"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&share));
    }
    assert_eq!(grouping["group_count"].as_u64().unwrap(), n);
    let rel = release_of(&merged, "merge_small_groups");
    assert_eq!(rel["groups"].as_array().unwrap().len() as u64, n);
    assert_eq!(rel["labels"].as_array().unwrap().len() as u64, n);
    assert_eq!(
        state
            .session()
            .clusters
            .as_ref()
            .unwrap()
            .cluster_assignment_id
            .to_string(),
        new_id,
        "the merged labels are the session's clustering"
    );

    // The dossier on the stored labels: no re-clustering, the same id.
    let again = ok(
        &client,
        "generate_cluster_dossier",
        json!({"method": "assignment", "cluster_assignment_id": new_id, "response_format": "json"}),
    )
    .await;
    assert_eq!(again["cluster_assignment_id"], new_id);
    assert_eq!(again["cluster_count"].as_u64().unwrap(), n);
    assert_eq!(cluster_ids(&again).len() as u64, n);
    assert!(again["method_applied"]
        .as_str()
        .unwrap()
        .starts_with("merged:"));

    // Nothing left under 8 patients: merging again changes nothing.
    let noop = ok(&client, "merge_small_groups", json!({})).await;
    assert!(noop.get("merges").is_none(), "{noop}");
    assert!(noop.get("merged_from").is_none(), "{noop}");
    assert_eq!(noop["cluster_assignment_id"], new_id);
    // No merge happened, so the summary claims no merge provenance.
    let grouping = &noop[SUMMARY]["grouping"];
    assert!(grouping.get("merged_from").is_none(), "{grouping}");
    assert!(grouping.get("merges").is_none(), "{grouping}");

    // A no-op over a stored assignment whose ids are not first-row ordered:
    // the sizes must use its own ids, not renumber them to 0...
    let record = state
        .store
        .get_cluster_assignment(&ClusterAssignmentId::parse(&new_id).unwrap())
        .unwrap();
    let shifted: Vec<i64> = state
        .store
        .cluster_labels(&record.cluster_assignment_id)
        .unwrap()
        .into_iter()
        .map(|l| l + 1)
        .collect();
    let shifted_id = state
        .store
        .save_cluster_assignment(ClusterAssignment {
            run_id: record.run_id.clone(),
            dataset_id: record.dataset_id.clone(),
            method: "shifted".into(),
            interpretation_edge_weight_threshold: record.interpretation_edge_weight_threshold,
            threshold_source: record.threshold_source.clone(),
            construction_threshold: record.construction_threshold,
            labels: &shifted,
        })
        .unwrap()
        .cluster_assignment_id;
    let shifted_noop = ok(
        &client,
        "merge_small_groups",
        json!({"cluster_assignment_id": shifted_id.to_string()}),
    )
    .await;
    assert!(shifted_noop.get("merges").is_none(), "{shifted_noop}");
    assert!(shifted_noop.get("merged_from").is_none(), "{shifted_noop}");
    assert_eq!(
        shifted_noop["cluster_assignment_id"],
        shifted_id.to_string()
    );
    let mut reported: Vec<i64> = shifted_noop["cluster_sizes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["cluster_id"].as_i64().unwrap())
        .collect();
    reported.sort_unstable();
    let mut want = shifted;
    want.sort_unstable();
    want.dedup();
    assert_eq!(
        reported, want,
        "sizes must use the returned assignment's own ids: {shifted_noop}"
    );
    assert!(
        !reported.contains(&0),
        "the no-op renumbered: {shifted_noop}"
    );
}

#[tokio::test]
async fn the_floor_and_the_missing_token_column_are_explained() {
    let root = TempRoot::new("refuse");
    let csv = cohort(&root.0);
    let (_state, client) = serve(&root.0).await;
    sweep(&client, &csv, CONFIG).await;
    ok(
        &client,
        "generate_cluster_dossier",
        json!({"response_format": "json"}),
    )
    .await;
    let low = call(&client, "merge_small_groups", json!({"min_patients": 7})).await;
    assert_eq!(low.is_error, Some(true));
    let text = serde_json::to_string(&low.structured_content).unwrap();
    assert!(text.contains(">= 8"), "{text}");

    // Higher values are honoured: every group (20, 20 and 2 patients) is
    // under 25, so the merge chains into one group of the 42 patients.
    let high_res = call(&client, "merge_small_groups", json!({"min_patients": 25})).await;
    assert_eq!(high_res.is_error, Some(false));
    let high_text = high_res.content[0].as_text().unwrap().text.as_str();
    let high = high_res.structured_content.as_ref().unwrap();
    assert_eq!(high["min_patients"], 25);
    assert_eq!(high["n_clusters"], 1);
    assert_eq!(high["cluster_sizes"].as_array().unwrap().len(), 1);
    let top_merges = high["merges"].as_array().unwrap();
    assert_eq!(
        top_merges.len(),
        1,
        "consecutive isolated merges collapse: {top_merges:?}"
    );
    let real_merges = top_merges[0]["count"].as_u64().unwrap() as usize;
    assert!(real_merges >= 2);
    assert_eq!(
        high[SUMMARY]["grouping"]["merges"]
            .as_array()
            .unwrap()
            .len(),
        real_merges
    );
    assert!(
        high_text.contains(&format!("{real_merges} cluster(s) merged into 1 groups")),
        "markdown must count real merges ({real_merges}), not collapsed entries: {high_text}"
    );

    // method='assignment' with an id that is not in the store is refused.
    let bogus = call(
        &client,
        "generate_cluster_dossier",
        json!({"method": "assignment", "cluster_assignment_id": "ca_000000000000", "response_format": "json"}),
    )
    .await;
    assert_eq!(bogus.is_error, Some(true));
    let text = serde_json::to_string(&bogus.structured_content).unwrap();
    assert!(text.contains("CLUSTER_ASSIGNMENT_ID_UNKNOWN"), "{text}");
    assert!(text.contains("ca_000000000000"), "{text}");

    // The same data without its token column: patients cannot be counted.
    let plain = root.0.join("plain.csv");
    let rows: Vec<String> = std::fs::read_to_string(&csv)
        .unwrap()
        .lines()
        .map(|l| l.split_once(',').unwrap().1.to_owned())
        .collect();
    std::fs::write(&plain, rows.join("\n")).unwrap();
    let config = CONFIG.replace("  - pid\n", "");
    let plain_root = TempRoot::new("plain");
    let (_state, client) = serve(&plain_root.0).await;
    sweep(&client, &plain, &config).await;
    ok(
        &client,
        "generate_cluster_dossier",
        json!({"response_format": "json"}),
    )
    .await;
    let r = call(&client, "merge_small_groups", json!({})).await;
    assert_eq!(r.is_error, Some(true));
    let text = serde_json::to_string(&r.structured_content).unwrap();
    assert!(text.contains("NO_PATIENT_IDENTIFIER"), "{text}");
    assert!(text.contains("token column"), "{text}");
}

/// The tool's description (what the agent reads once), looked up by name.
async fn description<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    name: &str,
) -> String {
    let tools = client.list_all_tools().await.unwrap();
    tools
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is registered"))
        .description
        .as_deref()
        .unwrap()
        .to_owned()
}

/// Constant guidance lives in the tool description, not in every answer.
#[tokio::test]
async fn constant_guidance_is_in_the_description_not_the_answer() {
    let root = TempRoot::new("constant-guidance");
    let csv = cohort(&root.0);
    let (_state, client) = serve_phirewall(&root.0).await;
    let ds = ok(
        &client,
        "ingest_dataset",
        json!({"path": csv.display().to_string()}),
    )
    .await;
    let dataset_id = ds["dataset_id"].as_str().unwrap();

    let characterize = ok(
        &client,
        "characterize_dataset",
        json!({"dataset_id": dataset_id}),
    )
    .await;
    assert!(
        characterize.get("agent_guidance").is_none(),
        "{characterize}"
    );
    assert!(description(&client, "characterize_dataset")
        .await
        .contains("probe_columns"));

    let draft = ok(
        &client,
        "draft_preprocessing_plan",
        json!({"dataset_id": dataset_id, "response_format": "json"}),
    )
    .await;
    assert!(draft.get("how_to_use").is_none(), "{draft}");
    assert!(draft["weak_hints"].get("description").is_none(), "{draft}");
    let text = description(&client, "draft_preprocessing_plan").await;
    assert!(text.contains("TARGET_PROXY_COLUMNS") && text.contains("one_hot"));
    let full = ok(
        &client,
        "draft_preprocessing_plan",
        json!({"dataset_id": dataset_id, "detail": "full", "response_format": "json"}),
    )
    .await;
    assert!(full["how_to_use"].is_string(), "full keeps it");

    for detail in ["full", "summary"] {
        let created = ok(
            &client,
            "create_config",
            json!({"dataset_id": dataset_id, "detail": detail}),
        )
        .await;
        for gone in ["config_yaml_note", "agent_guidance", "parameter_reasoning"] {
            assert!(created.get(gone).is_none(), "{gone}: {created}");
        }
        for gone in [
            "cost_note",
            "agent_guidance",
            "compatibility_note",
            "parameter_reasoning",
        ] {
            assert!(
                created["sweep_strategy"].get(gone).is_none(),
                "{gone}: {created}"
            );
        }
        let note = created["calibration_note"].as_str().unwrap_or_default();
        assert!(!note.contains("Geometry calibrated under"), "{note}");
        assert_eq!(created.get("config_yaml").is_some(), detail == "full");
    }
    let text = description(&client, "create_config").await;
    assert!(
        text.contains("linear scan") && text.contains("[p5, p95]") && text.contains("config_yaml")
    );

    sweep(&client, &csv, CONFIG).await;

    let diagnosis = ok(&client, "diagnose_cosmic_graph", json!({})).await;
    assert!(
        diagnosis["weight_distribution"].get("basis").is_none(),
        "{diagnosis}"
    );
    assert!(description(&client, "diagnose_cosmic_graph")
        .await
        .contains("zeros include absent edges"));

    let curve = ok(&client, "get_threshold_stability_curve", json!({})).await;
    for gone in ["structural_breakpoints_filter", "full_detail_available"] {
        assert!(curve.get(gone).is_none(), "{gone}: {curve}");
    }
    assert!(description(&client, "get_threshold_stability_curve")
        .await
        .contains("small_component_absorption"));

    for detail in ["summary", "full"] {
        let dossier = ok(
            &client,
            "generate_cluster_dossier",
            json!({"method": "components", "interpretation_edge_weight_threshold": 0.0, "detail": detail, "response_format": "json"}),
        )
        .await;
        assert!(dossier.get("detail_hint").is_none(), "{dossier}");
    }
    assert!(description(&client, "generate_cluster_dossier")
        .await
        .contains("lossless arrays"));

    let merged = ok(&client, "merge_small_groups", json!({})).await;
    assert!(merged.get("detail_hint").is_none(), "{merged}");
    assert!(description(&client, "merge_small_groups")
        .await
        .contains("method='assignment'"));
}
