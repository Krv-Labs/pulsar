//! The response budget end to end: a dossier on a shattered graph fits the
//! budget and still publishes and reports its assignment; its `next` call
//! pages through every omitted cluster; the read tools take an explicit
//! (older) assignment; the middleware trims instead of failing; errors stay
//! short.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pulsar_mcp::budget::{render_budget, visible_chars, MAX_ERROR_CHARS};
use pulsar_mcp::{PulsarServer, ServerOptions, ServerState};
use pulsar_store::Store;
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, Tool};
use rmcp::service::{RoleClient, RunningService};
use rmcp::ServiceExt;
use serde_json::{json, Value};

const HARD: usize = 45_000;

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "pulsar-budget-{tag}-{}-{nanos:x}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

type Client = RunningService<RoleClient, ()>;

async fn connect(root: &Path, hard: Option<usize>) -> Client {
    let store = Store::open_at(root).unwrap();
    let options = ServerOptions {
        max_response_bytes: hard,
        transport: "in-memory".into(),
    };
    let state = Arc::new(ServerState::new(store, options).unwrap());
    let (a, b) = tokio::io::duplex(1 << 20);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);
    tokio::spawn(async move {
        let service = PulsarServer::new(state).serve((ar, aw)).await.unwrap();
        let _ = service.waiting().await;
    });
    ().serve((br, bw)).await.unwrap()
}

async fn call(client: &Client, name: &str, args: Value) -> CallToolResult {
    let mut params = CallToolRequestParams::new(name.to_owned());
    params.arguments = args.as_object().cloned();
    client.call_tool(params).await.unwrap()
}

fn text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect()
}

/// A successful call within the budget whose payload validates against
/// the tool's advertised output schema.
async fn ok(client: &Client, tools: &[Tool], name: &str, args: Value) -> Value {
    checked(tools, name, &call(client, name, args).await)
}

fn checked(tools: &[Tool], name: &str, result: &CallToolResult) -> Value {
    assert_eq!(result.is_error, Some(false), "{name}: {}", text(result));
    let size = visible_chars(result);
    assert!(size <= HARD, "{name}: {size} chars over the hard limit");
    let payload = result.structured_content.clone().unwrap();
    let schema = tools
        .iter()
        .find(|t| t.name == name)
        .and_then(|t| t.output_schema.clone())
        .unwrap();
    let validator = jsonschema::validator_for(&Value::Object((*schema).clone())).unwrap();
    let errors: Vec<String> = validator
        .iter_errors(&payload)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:?}");
    payload
}

async fn err(client: &Client, name: &str, args: Value) -> Value {
    let result = call(client, name, args).await;
    assert_eq!(result.is_error, Some(true), "{name} should fail");
    assert!(
        visible_chars(&result) < MAX_ERROR_CHARS,
        "{name}: error is {} chars",
        visible_chars(&result)
    );
    result.structured_content.unwrap()
}

/// 300 rows in 60 small, well separated groups over 24 numeric columns and
/// two categorical ones: a strict cut-off shatters it into many clusters.
fn cohort(dir: &Path) -> PathBuf {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut noise = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut header: Vec<String> = (0..24).map(|j| format!("lab_{j:02}")).collect();
    header.extend(["site".into(), "sex".into()]);
    let mut csv = header.join(",") + "\n";
    for i in 0..300 {
        let group = i % 60;
        let mut row: Vec<String> = (0..24)
            .map(|j| format!("{:.5}", ((group * (j + 3)) % 11) as f64 + noise() * 0.6))
            .collect();
        row.push(format!("site_{}", group % 7));
        row.push(["F", "M"][i % 2].into());
        csv += &(row.join(",") + "\n");
    }
    let path = dir.join("cohort.csv");
    std::fs::write(&path, csv).unwrap();
    path
}

async fn swept(client: &Client, tools: &[Tool], root: &Path) {
    let csv = cohort(root);
    let ingested = ok(
        client,
        tools,
        "ingest_dataset",
        json!({"path": csv.display().to_string()}),
    )
    .await;
    let dataset_id = ingested["dataset_id"].clone();
    ok(
        client,
        tools,
        "create_config",
        json!({"dataset_id": dataset_id}),
    )
    .await;
    ok(
        client,
        tools,
        "run_topological_sweep",
        json!({"dataset_id": dataset_id, "response_format": "json"}),
    )
    .await;
}

fn ids_of(clusters: &Value) -> Vec<i64> {
    clusters
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["cluster_id"].as_i64().unwrap())
        .collect()
}

/// Follow `next_call` from `first` to the end; returns every cluster id
/// shown along the way, in order.
async fn page_through(client: &Client, tools: &[Tool], first: &Value) -> Vec<i64> {
    let mut shown = vec![];
    let mut next = first["truncation"]["next_call"].clone();
    let mut pages = 0;
    while !next.is_null() {
        assert_eq!(next["tool"], "get_cluster_detail");
        let page = ok(
            client,
            tools,
            "get_cluster_detail",
            next["arguments"].clone(),
        )
        .await;
        assert_eq!(page["live"], true);
        let ids = ids_of(&page["clusters"]);
        assert!(!ids.is_empty(), "a page always moves: {page}");
        assert_eq!(page["truncation"]["shown"], ids.len());
        shown.extend(ids);
        next = page["truncation"]["next_call"].clone();
        pages += 1;
        assert!(pages < 500, "paging does not terminate");
    }
    shown
}

#[tokio::test(flavor = "multi_thread")]
async fn a_shattered_dossier_fits_publishes_and_pages_to_completion() {
    let root = TempRoot::new("dossier");
    let client = connect(&root.0, Some(HARD)).await;
    let tools = client.list_all_tools().await.unwrap();
    swept(&client, &tools, &root.0).await;

    // The incident: a strict cut-off with every section requested.
    let strict = json!({
        "method": "components",
        "interpretation_edge_weight_threshold": 0.95,
        "detail": "standard",
        "max_clusters": 1000,
        "response_format": "json",
    });
    let result = call(&client, "generate_cluster_dossier", strict).await;
    let dossier = checked(&tools, "generate_cluster_dossier", &result);
    let size = visible_chars(&result);
    assert!(size <= render_budget(Some(HARD)), "dossier is {size} chars");
    // Sections are shortened before any is dropped; the largest stay.
    assert!(
        dossier["clusters"].as_array().unwrap().len() >= 2,
        "{}",
        dossier["truncation"]
    );
    assert!(dossier["clusters"][0]["sections_trimmed"].is_object());
    let total = dossier["cluster_count"].as_u64().unwrap() as usize;
    assert!(total > 100, "the cut-off shatters the graph: {total}");
    let assignment = dossier["cluster_assignment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let t = &dossier["truncation"];
    assert_eq!(t["truncated"], true, "{t}");
    assert_eq!(t["total"], total);
    assert_eq!(t["budget_chars"], render_budget(Some(HARD)));
    assert!(t["next"].as_str().unwrap().contains(&assignment), "{t}");
    assert_eq!(
        t["next_call"]["arguments"]["cluster_assignment_id"],
        json!(assignment)
    );
    // Every cluster is indexed, even when the sections stop early.
    assert_eq!(dossier["cluster_index"]["total"], total);

    // The labels are live: the read tools see this assignment.
    let profile = ok(
        &client,
        &tools,
        "get_cluster_profile",
        json!({"cluster_id": ids_of(&dossier["cluster_index"]["clusters"])[0]}),
    )
    .await;
    assert_eq!(profile["cluster_assignment_id"], json!(assignment));

    // The sections shown plus every page cover each cluster exactly once.
    let mut seen = ids_of(&dossier["clusters"]);
    assert_eq!(seen.len(), t["shown"].as_u64().unwrap() as usize);
    seen.extend(page_through(&client, &tools, &dossier).await);
    let unique: BTreeSet<i64> = seen.iter().copied().collect();
    assert_eq!(unique.len(), seen.len(), "a cluster was shown twice");
    assert_eq!(unique.len(), total, "paging missed clusters");

    // detail='full' in Markdown: the same budget, the truncation mirrored
    // in one line of text.
    let md = call(
        &client,
        "generate_cluster_dossier",
        json!({"method": "components", "interpretation_edge_weight_threshold": 0.95, "detail": "full"}),
    )
    .await;
    let full = checked(&tools, "generate_cluster_dossier", &md);
    assert_eq!(full["truncation"]["truncated"], true);
    assert!(
        !full["clusters"].as_array().unwrap().is_empty(),
        "{}",
        full["truncation"]
    );
    assert!(visible_chars(&md) <= render_budget(Some(HARD)));
    let line = format!("Next: {}", full["truncation"]["next"].as_str().unwrap());
    assert!(text(&md).contains(&line), "{}", &text(&md)[..400]);
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn read_tools_take_an_older_assignment_and_say_which_they_used() {
    let root = TempRoot::new("durable");
    let client = connect(&root.0, Some(HARD)).await;
    let tools = client.list_all_tools().await.unwrap();
    swept(&client, &tools, &root.0).await;

    let strict = ok(
        &client,
        &tools,
        "generate_cluster_dossier",
        json!({"method": "components", "interpretation_edge_weight_threshold": 0.95}),
    )
    .await;
    let old = strict["cluster_assignment_id"].as_str().unwrap().to_owned();
    let old_ids = ids_of(&strict["cluster_index"]["clusters"]);
    let coarse = ok(
        &client,
        &tools,
        "generate_cluster_dossier",
        json!({"method": "components", "interpretation_edge_weight_threshold": 0.0}),
    )
    .await;
    let live = coarse["cluster_assignment_id"].as_str().unwrap().to_owned();
    assert_ne!(old, live);
    let live_total = coarse["cluster_count"].as_u64().unwrap();
    assert!(live_total < strict["cluster_count"].as_u64().unwrap());

    // A cluster only the older assignment has, read through its id.
    let target = *old_ids.last().unwrap();
    let profile = ok(
        &client,
        &tools,
        "get_cluster_profile",
        json!({"cluster_id": target, "cluster_assignment_id": old}),
    )
    .await;
    assert_eq!(profile["cluster_assignment_id"], json!(old));
    assert_eq!(profile["cluster"]["cluster_id"], target);
    let matrix = ok(
        &client,
        &tools,
        "get_cluster_signal_matrix",
        json!({"cluster_assignment_id": old, "max_clusters": 1000}),
    )
    .await;
    assert_eq!(matrix["cluster_assignment_id"], json!(old));
    let signal = ok(
        &client,
        &tools,
        "get_feature_signal",
        json!({"cluster_assignment_id": old, "feature_names": ["lab_01", "site"], "detail": "full", "max_clusters": 1000}),
    )
    .await;
    assert_eq!(signal["cluster_assignment_id"], json!(old));
    // Without the id the live labels answer.
    let current = ok(&client, &tools, "get_cluster_signal_matrix", json!({})).await;
    assert_eq!(current["cluster_assignment_id"], json!(live));

    // The older assignment pages too, and reading it leaves the live labels.
    let page = ok(
        &client,
        &tools,
        "get_cluster_detail",
        json!({"cluster_assignment_id": old, "cluster_ids": [target], "sections": ["central_rows", "feature_preview"]}),
    )
    .await;
    assert_eq!(page["live"], false);
    assert_eq!(ids_of(&page["clusters"]), vec![target]);
    assert!(page["clusters"][0]["central_rows"].is_array());
    assert!(page["clusters"][0]["feature_preview"].is_array());
    assert!(page["clusters"][0].get("numeric_features").is_none());
    let after = ok(
        &client,
        &tools,
        "get_cluster_profile",
        json!({"cluster_id": 0}),
    )
    .await;
    assert_eq!(after["cluster_assignment_id"], json!(live));

    // An unknown cluster names the assignment read, the live one, and how
    // to pick another, in a short error.
    let e = err(
        &client,
        "get_cluster_profile",
        json!({"cluster_id": 999_999}),
    )
    .await;
    assert_eq!(e["error_code"], "CLUSTER_ID_UNKNOWN");
    assert!(e["reason"].as_str().unwrap().contains(&live), "{e}");
    assert!(e["agent_action"]
        .as_str()
        .unwrap()
        .contains("cluster_assignment_id"));
    let e = err(
        &client,
        "get_cluster_detail",
        json!({"cluster_assignment_id": old, "cluster_ids": [999_999]}),
    )
    .await;
    assert_eq!(e["error_code"], "CLUSTER_ID_UNKNOWN");
    assert!(e["reason"].as_str().unwrap().contains(&old), "{e}");
    let e = err(
        &client,
        "get_cluster_detail",
        json!({"cluster_assignment_id": "ca_000000000000"}),
    )
    .await;
    assert!(e["agent_action"].as_str().unwrap().contains(&live), "{e}");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn an_oversized_result_is_trimmed_with_a_notice_not_failed() {
    let root = TempRoot::new("degrade");
    let client = connect(&root.0, Some(6_000)).await;
    let tools = client.list_all_tools().await.unwrap();
    let guide = ok(&client, &tools, "get_workflow_guide", json!({})).await;
    assert_eq!(guide["truncation"]["truncated"], true);
    assert!(guide["guide"].as_str().unwrap().ends_with('…'));
    let result = call(&client, "get_workflow_guide", json!({})).await;
    assert!(visible_chars(&result) <= render_budget(Some(6_000)));
    assert!(text(&result).starts_with("[truncated:"));
    client.cancel().await.unwrap();
}

/// The sweep carries a compact diagnose_cosmic_graph summary (headline facts
/// plus the call for the full view) and says plainly when the active config
/// was not saved, all within the response budget.
#[tokio::test(flavor = "multi_thread")]
async fn sweep_carries_a_compact_diagnose_summary_and_names_the_unsaved_config() {
    let root = TempRoot::new("sweep-summary");
    let client = connect(&root.0, Some(HARD)).await;
    let tools = client.list_all_tools().await.unwrap();
    swept(&client, &tools, &root.0).await;

    let result = call(
        &client,
        "run_topological_sweep",
        json!({"response_format": "markdown"}),
    )
    .await;
    let sweep = checked(&tools, "run_topological_sweep", &result);
    let size = visible_chars(&result);
    eprintln!(
        "sweep summary with diagnose summary: {size} chars (render budget {})",
        render_budget(Some(HARD))
    );
    assert!(size <= render_budget(Some(HARD)), "sweep is {size} chars");

    let summary = &sweep["diagnose_summary"];
    assert!(summary.is_object(), "{summary}");
    assert_eq!(summary["full_view"], "diagnose_cosmic_graph");
    assert!(summary["component_count"].as_u64().unwrap() > 0);
    assert!(!summary["top_component_sizes"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(summary["observed_patterns"].as_array().is_some());
    assert!(summary["risk_factors"].as_array().is_some());
    assert_eq!(sweep["next_required_check"], "diagnose_cosmic_graph");

    let advisory = sweep["config_advisory"].as_str().unwrap();
    assert!(advisory.contains("was not saved"), "{advisory}");

    let body = text(&result);
    assert!(
        body.contains("## Diagnose Summary"),
        "{}",
        &body[..body.len().min(600)]
    );
    assert!(body.contains("diagnose_cosmic_graph(run_id="), "{body}");
    assert!(body.contains("was not saved"), "{body}");
    client.cancel().await.unwrap();
}

/// Two well-separated blobs over many numeric columns: `compare_clusters`
/// compares O(features) rows.
fn wide_two_blob(dir: &Path) -> PathBuf {
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut noise = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let header: Vec<String> = (0..120).map(|j| format!("f{j:03}")).collect();
    let mut csv = header.join(",") + "\n";
    for i in 0..48 {
        let group = if i < 24 { 0.0 } else { 10.0 };
        let row: Vec<String> = (0..120)
            .map(|_| format!("{:.4}", group + noise() * 2.0))
            .collect();
        csv += &(row.join(",") + "\n");
    }
    let path = dir.join("wide.csv");
    std::fs::write(&path, csv).unwrap();
    path
}

async fn swept_wide(client: &Client, tools: &[rmcp::model::Tool], root: &Path) -> Value {
    let csv = wide_two_blob(root);
    let ingested = ok(
        client,
        tools,
        "ingest_dataset",
        json!({"path": csv.display().to_string()}),
    )
    .await;
    let dataset_id = ingested["dataset_id"].clone();
    // A minimal config keeps the sweep summary small, so a mid-size limit
    // binds the 120-feature comparison but not the sweep or the dossier.
    ok(
        client,
        tools,
        "run_topological_sweep",
        json!({"dataset_id": dataset_id, "config_yaml": WIDE_YAML, "response_format": "json"}),
    )
    .await;
    let dossier = ok(
        client,
        tools,
        "generate_cluster_dossier",
        json!({"response_format": "json"}),
    )
    .await;
    dossier
}

const WIDE_YAML: &str = r#"
version: 1
run:
  name: wide
sweep:
  projection:
    method: jl
    dimensions: [2]
    seed: [42]
  ball_mapper:
    epsilon: [0.5, 1.0]
"#;

/// `compare_clusters` fits itself to the budget by halving `max_features`,
/// the way the dossier fits itself, instead of failing over it.
#[tokio::test(flavor = "multi_thread")]
async fn compare_clusters_fits_itself_to_the_budget() {
    // A mid-size hard limit binds the 120-feature comparison but not the
    // sweep or the dossier, so the tool's own knob has to move.
    let root = TempRoot::new("compare-fit");
    let client = connect(&root.0, Some(18_000)).await;
    let tools = client.list_all_tools().await.unwrap();
    let dossier = swept_wide(&client, &tools, &root.0).await;
    // The index lists every cluster; the sections may stop early.
    let ids = ids_of(&dossier["cluster_index"]["clusters"]);
    assert!(ids.len() >= 2, "two blobs cluster apart: {dossier}");
    let (a, b) = (ids[0], ids[1]);

    let result = call(
        &client,
        "compare_clusters",
        json!({"cluster_a": a, "cluster_b": b, "max_features": 1000}),
    )
    .await;
    let payload = checked(&tools, "compare_clusters", &result);
    let size = visible_chars(&result);
    eprintln!(
        "compare_clusters capped: {size} chars (render budget {})",
        render_budget(Some(18_000))
    );
    assert!(
        size <= render_budget(Some(18_000)),
        "compare is {size} chars"
    );
    // The tool's own knob moved, not the middleware's trimmer.
    assert_eq!(
        payload["truncation"]["reduced"]["parameter"],
        "max_features"
    );
    assert!(
        payload["truncation"]["reduced"]["applied"]
            .as_u64()
            .unwrap()
            < payload["truncation"]["reduced"]["asked"].as_u64().unwrap()
    );
    assert_eq!(payload["features_compared"], 120);
    assert!(payload["features_omitted"].as_u64().unwrap() > 0);
    assert!(
        text(&result).contains("raise max_features"),
        "{}",
        &text(&result)[..200.min(text(&result).len())]
    );
    client.cancel().await.unwrap();

    // At the default budget all 120 features fit whole, with no truncation.
    let root = TempRoot::new("compare-whole");
    let client = connect(&root.0, Some(HARD)).await;
    let tools = client.list_all_tools().await.unwrap();
    let dossier = swept_wide(&client, &tools, &root.0).await;
    let ids = ids_of(&dossier["cluster_index"]["clusters"]);
    let result = call(
        &client,
        "compare_clusters",
        json!({"cluster_a": ids[0], "cluster_b": ids[1], "max_features": 1000}),
    )
    .await;
    let payload = checked(&tools, "compare_clusters", &result);
    let size = visible_chars(&result);
    eprintln!(
        "compare_clusters whole: {size} chars (render budget {})",
        render_budget(Some(HARD))
    );
    assert!(size <= render_budget(Some(HARD)));
    assert!(
        payload.get("truncation").is_none(),
        "{}",
        payload["truncation"]
    );
    assert_eq!(payload["features_omitted"], 0);
    client.cancel().await.unwrap();
}

/// A long-format panel: `n` entities x 3 visits over three drifting features.
fn panel_csv(dir: &Path, n: usize) -> PathBuf {
    let mut csv = String::from("patient,visit,a,b,c\n");
    for e in 0..n {
        for t in 1..=3 {
            let g = (e % 3) as f64;
            csv.push_str(&format!(
                "p{e:04},{t},{:.4},{:.4},{:.4}\n",
                g * 2.0 + t as f64 * 0.3,
                -g + t as f64 * 0.1,
                g * 0.5 + t as f64 * 0.05,
            ));
        }
    }
    let path = dir.join("panel.csv");
    std::fs::write(&path, csv).unwrap();
    path
}

const PANEL_YAML: &str = "run:\n  name: panel\nsweep:\n  projection:\n    method: jl\n    dimensions:\n      values: [2, 3]\n    seed:\n      values: [42]\n  ball_mapper:\n    epsilon:\n      values: [0.5, 1.0]\ncosmic_graph:\n  construction: exact\n";

/// `classify_trajectories(detail='full')` is O(entities): the per-entity map
/// is capped to the budget with a truncation note, while the class summary
/// and the preview always stay.
#[tokio::test(flavor = "multi_thread")]
async fn classify_full_is_capped_to_the_budget() {
    let root = TempRoot::new("classify-cap");
    let client = connect(&root.0, Some(HARD)).await;
    let tools = client.list_all_tools().await.unwrap();
    let csv = panel_csv(&root.0, 400);
    let ingested = ok(
        &client,
        &tools,
        "ingest_dataset",
        json!({"path": csv.display().to_string()}),
    )
    .await;
    ok(
        &client,
        &tools,
        "build_longitudinal_graph",
        json!({"dataset_id": ingested["dataset_id"], "entity_column": "patient", "time_column": "visit", "config_yaml": PANEL_YAML}),
    )
    .await;

    let result = call(
        &client,
        "classify_trajectories",
        json!({"detail": "full", "response_format": "json"}),
    )
    .await;
    let payload = checked(&tools, "classify_trajectories", &result);
    let size = visible_chars(&result);
    eprintln!(
        "classify full on 400 entities: {size} chars, omitted={} (render budget {})",
        payload["entities_omitted"],
        render_budget(Some(HARD))
    );
    assert!(
        size <= render_budget(Some(HARD)),
        "classify is {size} chars"
    );
    assert!(payload["entities_omitted"].as_u64().unwrap() > 0);
    assert_eq!(payload["truncation"]["truncated"], true);
    assert!(payload["classes_summary"].is_object());
    assert_eq!(
        payload["classification_preview"].as_object().unwrap().len(),
        25
    );

    // The summary shape still fits whole, with no truncation.
    let summary = ok(
        &client,
        &tools,
        "classify_trajectories",
        json!({"response_format": "json"}),
    )
    .await;
    assert!(summary.get("truncation").is_none());
    assert!(summary.get("classification").is_none());
    client.cancel().await.unwrap();
}

/// Budget audit of the tools the response-budget work had not probed yet:
/// the exports land in files (only a small receipt reaches the agent) and
/// the longitudinal tools stay bounded on the fixtures.
#[tokio::test(flavor = "multi_thread")]
async fn unprobed_tools_fit_the_budget_on_fixtures() {
    let root = TempRoot::new("audit");
    let client = connect(&root.0, Some(HARD)).await;
    let tools = client.list_all_tools().await.unwrap();
    swept(&client, &tools, &root.0).await;
    let dossier = ok(
        &client,
        &tools,
        "generate_cluster_dossier",
        json!({"response_format": "json"}),
    )
    .await;
    let assignment = dossier["cluster_assignment_id"].clone();

    // Every cluster id, paging through the detail pages.
    let mut ids = vec![];
    let mut next = Value::Null;
    loop {
        let args = if next.is_null() {
            json!({"cluster_assignment_id": assignment})
        } else {
            next["arguments"].clone()
        };
        let result = call(&client, "get_cluster_detail", args).await;
        let page = checked(&tools, "get_cluster_detail", &result);
        ids.extend(ids_of(&page["clusters"]));
        next = page["truncation"]["next_call"].clone();
        if next.is_null() {
            break;
        }
    }
    assert!(!ids.is_empty());
    let names: serde_json::Map<String, Value> = ids
        .iter()
        .map(|id| (id.to_string(), Value::String(format!("Cohort {id}"))))
        .collect();

    let out = root.0.join("labeled.csv");
    let result = call(
        &client,
        "export_labeled_data",
        json!({"cluster_assignment_id": assignment, "cluster_names": names, "output_path": out.display().to_string()}),
    )
    .await;
    checked(&tools, "export_labeled_data", &result);
    eprintln!("export_labeled_data: {} chars", visible_chars(&result));
    assert!(visible_chars(&result) <= render_budget(Some(HARD)));

    let result = call(
        &client,
        "export_dataset_bundle",
        json!({"output_dir": root.0.display().to_string(), "slug": "audit_bundle"}),
    )
    .await;
    checked(&tools, "export_dataset_bundle", &result);
    eprintln!("export_dataset_bundle: {} chars", visible_chars(&result));
    assert!(visible_chars(&result) <= render_budget(Some(HARD)));

    // The report lands next to the dataset file (in the temp root) when a
    // dataset_id is passed; without one it lands in the process cwd.
    let runtime = ok(&client, &tools, "get_runtime_context", json!({})).await;
    let result = call(
        &client,
        "export_html_report",
        json!({"dataset_id": runtime["latest_dataset_id"]}),
    )
    .await;
    checked(&tools, "export_html_report", &result);
    eprintln!("export_html_report: {} chars", visible_chars(&result));
    assert!(visible_chars(&result) <= render_budget(Some(HARD)));

    let csv = panel_csv(&root.0, 30);
    let ingested = ok(
        &client,
        &tools,
        "ingest_dataset",
        json!({"path": csv.display().to_string()}),
    )
    .await;
    let ds = ingested["dataset_id"].clone();
    let result = call(
        &client,
        "sample_longitudinal_entities",
        json!({"dataset_id": ds, "entity_column": "patient", "max_entities": 12}),
    )
    .await;
    checked(&tools, "sample_longitudinal_entities", &result);
    assert!(visible_chars(&result) <= render_budget(Some(HARD)));
    let result = call(
        &client,
        "build_longitudinal_graph",
        json!({"dataset_id": ds, "entity_column": "patient", "time_column": "visit", "config_yaml": PANEL_YAML, "representation": "both", "detail": "full"}),
    )
    .await;
    checked(&tools, "build_longitudinal_graph", &result);
    eprintln!(
        "build_longitudinal_graph full: {} chars",
        visible_chars(&result)
    );
    assert!(visible_chars(&result) <= render_budget(Some(HARD)));
    for (name, args) in [
        ("diagnose_longitudinal_graph", json!({"detail": "full"})),
        (
            "get_trajectory_archetypes",
            json!({"max_archetypes": 100, "max_entities_per_archetype": 100}),
        ),
        (
            "get_cross_time_neighbors",
            json!({"observation_id": 0, "max_neighbors": 1000}),
        ),
        ("classify_trajectories", json!({})),
    ] {
        let result = call(&client, name, args).await;
        checked(&tools, name, &result);
        eprintln!("{name}: {} chars", visible_chars(&result));
        assert!(
            visible_chars(&result) <= render_budget(Some(HARD)),
            "{name}"
        );
    }
    client.cancel().await.unwrap();
}
