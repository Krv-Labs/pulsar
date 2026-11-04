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
