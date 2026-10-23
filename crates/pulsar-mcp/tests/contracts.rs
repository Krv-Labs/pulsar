//! Contract tests through an in-process client (`test_mcp_contracts.py`):
//! every tool advertises its contract on the wire, `tools/list` is
//! deterministic and within budget, the canonical loop runs end to end with
//! every payload validating against its output schema, errors are
//! envelopes, async jobs and the tasks extension both deliver the sweep.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pulsar_mcp::{PulsarServer, ServerOptions, ServerState, CATALOG_BUDGET_CHARS};
use pulsar_store::Store;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ClientCapabilities, ClientConfig,
    GetTaskParams, Implementation, JsonObject, TaskStatus, Tool,
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::{ClientHandler, ServiceExt};
use serde_json::{json, Map, Value};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(tag: &str) -> Self {
        let nonce: u64 = rand_nonce();
        let path =
            std::env::temp_dir().join(format!("pulsar-mcp-{tag}-{}-{nonce:x}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn rand_nonce() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    t ^ (std::process::id() as u64).rotate_left(32)
}

fn golden(case: &str, file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/golden")
        .join(case)
        .join(file)
}

/// Serve a fresh server over an in-memory duplex and connect `handler`.
async fn connect<H: ClientHandler>(
    root: &Path,
    options: ServerOptions,
    handler: H,
) -> RunningService<RoleClient, H> {
    let store = Store::open_at(root).unwrap();
    let state = Arc::new(ServerState::new(store, options).unwrap());
    let (a, b) = tokio::io::duplex(1 << 20);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);
    tokio::spawn(async move {
        let service = PulsarServer::new(state)
            .serve((ar, aw))
            .await
            .expect("server starts");
        let _ = service.waiting().await;
    });
    handler.serve((br, bw)).await.expect("client connects")
}

fn test_options() -> ServerOptions {
    ServerOptions {
        max_response_bytes: Some(2_000_000),
        transport: "in-memory".into(),
    }
}

fn args(pairs: &[(&str, Value)]) -> JsonObject {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

async fn call<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    name: &str,
    a: JsonObject,
) -> CallToolResult {
    let mut params = CallToolRequestParams::new(name.to_owned());
    params.arguments = Some(a);
    client.call_tool(params).await.expect("call completes")
}

async fn ok<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    tools: &[Tool],
    name: &str,
    a: JsonObject,
) -> Value {
    let result = call(client, name, a).await;
    assert_eq!(result.is_error, Some(false), "{name}: {:?}", result.content);
    let payload = result
        .structured_content
        .clone()
        .expect("structured content");
    assert!(
        payload.is_object(),
        "{name}: structured content must be an object"
    );
    let schema = tools
        .iter()
        .find(|t| t.name == name)
        .and_then(|t| t.output_schema.clone())
        .expect("output schema");
    validate(&Value::Object((*schema).clone()), &payload, name);
    assert!(
        !result.content.is_empty(),
        "{name}: a text view is required"
    );
    payload
}

async fn err<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    name: &str,
    a: JsonObject,
) -> Value {
    let result = call(client, name, a).await;
    assert_eq!(result.is_error, Some(true), "{name} should fail");
    let payload = result.structured_content.expect("error envelope");
    assert_eq!(payload["status"], "error");
    assert_eq!(payload["tool"], name);
    assert!(payload["agent_action"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
    payload
}

/// A small JSON Schema validator covering the subset the catalog uses.
fn validate(schema: &Value, value: &Value, path: &str) {
    if let Some(any_of) = schema["anyOf"].as_array() {
        let ok = any_of
            .iter()
            .any(|s| std::panic::catch_unwind(|| validate(s, value, path)).is_ok());
        assert!(ok, "{path}: {value} matches none of {any_of:?}");
        return;
    }
    if let Some(options) = schema["enum"].as_array() {
        assert!(
            options.contains(value),
            "{path}: {value} not in {options:?}"
        );
    }
    match schema["type"].as_str() {
        Some("string") => assert!(value.is_string(), "{path}: expected string, got {value}"),
        Some("integer") => assert!(
            value.is_i64() || value.is_u64(),
            "{path}: expected integer, got {value}"
        ),
        Some("number") => assert!(value.is_number(), "{path}: expected number, got {value}"),
        Some("boolean") => assert!(value.is_boolean(), "{path}: expected boolean, got {value}"),
        Some("null") => assert!(value.is_null(), "{path}: expected null, got {value}"),
        Some("array") => {
            let items = value
                .as_array()
                .unwrap_or_else(|| panic!("{path}: expected array, got {value}"));
            if let Some(item_schema) = schema.get("items") {
                for (i, item) in items.iter().enumerate() {
                    validate(item_schema, item, &format!("{path}[{i}]"));
                }
            }
        }
        Some("object") => {
            let obj = value
                .as_object()
                .unwrap_or_else(|| panic!("{path}: expected object, got {value}"));
            let props = schema["properties"]
                .as_object()
                .cloned()
                .unwrap_or_default();
            if let Some(required) = schema["required"].as_array() {
                for key in required.iter().filter_map(Value::as_str) {
                    assert!(obj.contains_key(key), "{path}: missing required key {key}");
                }
            }
            let additional = schema["additionalProperties"].as_bool().unwrap_or(true);
            for (key, v) in obj {
                match props.get(key) {
                    Some(s) => validate(s, v, &format!("{path}.{key}")),
                    None => assert!(additional, "{path}: undeclared key {key}"),
                }
            }
        }
        _ => {}
    }
}

fn blobs_yaml() -> String {
    std::fs::read_to_string(golden("blobs", "config.yaml")).unwrap()
}

#[tokio::test]
async fn every_tool_declares_its_contract_on_the_wire() {
    let root = TempRoot::new("catalog");
    let client = connect(root.path(), test_options(), ()).await;
    let first = client.list_all_tools().await.unwrap();
    let second = client.list_all_tools().await.unwrap();
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        serde_json::to_value(&second).unwrap(),
        "tools/list must be deterministic"
    );
    assert!(first.len() >= 12);
    let names: BTreeSet<&str> = first.iter().map(|t| t.name.as_ref()).collect();
    for expected in [
        "ingest_dataset",
        "validate_config",
        "run_topological_sweep",
        "get_job",
        "wait_job",
        "diagnose_cosmic_graph",
        "get_threshold_stability_curve",
        "get_topological_skeleton",
        "get_sweep_history",
        "compare_sweeps",
        "get_workflow_guide",
        "get_runtime_context",
    ] {
        assert!(names.contains(expected), "missing tool {expected}");
    }
    let mut total = 0;
    for tool in &first {
        let description = tool.description.as_deref().unwrap_or("");
        assert!(
            description.len() > 40,
            "{}: description too short",
            tool.name
        );
        let input = Value::Object((*tool.input_schema).clone());
        assert_eq!(input["type"], "object", "{}: input schema", tool.name);
        for (prop, schema) in input["properties"].as_object().cloned().unwrap_or_default() {
            assert!(
                schema["description"].as_str().is_some_and(|d| d.len() > 10),
                "{}.{prop}: every parameter needs a description",
                tool.name
            );
        }
        assert!(tool.output_schema.is_some(), "{}: output schema", tool.name);
        let ann = tool
            .annotations
            .as_ref()
            .unwrap_or_else(|| panic!("{}: annotations", tool.name));
        assert!(
            ann.read_only_hint.is_some(),
            "{}: read_only_hint",
            tool.name
        );
        assert_eq!(
            ann.open_world_hint,
            Some(false),
            "{}: open_world_hint",
            tool.name
        );
        assert!(ann.title.is_some(), "{}: title", tool.name);
        total += description.len() + serde_json::to_string(&input).unwrap().len();
    }
    eprintln!(
        "MCP catalog budget: {total} chars (~{} tokens), limit {CATALOG_BUDGET_CHARS}",
        total / 4
    );
    assert!(total < CATALOG_BUDGET_CHARS);
    let info = client.peer_info().expect("server info");
    assert!(info.capabilities.tools.is_some());
    assert!(
        info.capabilities.supports_tasks(),
        "tasks extension advertised"
    );
    assert!(info
        .instructions
        .as_deref()
        .unwrap_or("")
        .contains("get_workflow_guide"));
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn canonical_loop_with_schema_validation_and_error_envelopes() {
    let root = TempRoot::new("loop");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();

    let guide = ok(&client, &tools, "get_workflow_guide", args(&[])).await;
    assert!(guide["guide"].as_str().unwrap().contains("wait_job"));

    let ingested = ok(
        &client,
        &tools,
        "ingest_dataset",
        args(&[(
            "path",
            json!(golden("blobs", "input.csv").display().to_string()),
        )]),
    )
    .await;
    let dataset_id = ingested["dataset_id"].as_str().unwrap().to_owned();
    assert!(dataset_id.starts_with("ds_"));

    let validated = ok(
        &client,
        &tools,
        "validate_config",
        args(&[
            ("config_yaml", json!(blobs_yaml())),
            ("dataset_id", json!(dataset_id)),
        ]),
    )
    .await;
    assert_eq!(validated["status"], "ok");
    assert!(validated["normalized_config_yaml"]
        .as_str()
        .unwrap()
        .contains("ball_mapper"));

    let sweep = ok(
        &client,
        &tools,
        "run_topological_sweep",
        args(&[
            ("config_yaml", json!(blobs_yaml())),
            ("dataset_id", json!(dataset_id)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let run_id = sweep["run_id"].as_str().unwrap().to_owned();
    assert_eq!(sweep["status"], "ok");
    assert_eq!(sweep["detail"], "summary");
    assert_eq!(sweep["next_required_check"], "diagnose_cosmic_graph");
    assert!(sweep["key_metrics"]["n_nodes"].as_u64().unwrap() == 150);
    assert!(sweep["threshold_stability_summary"].is_object());
    assert_eq!(sweep["projection_cache_status"]["status"], "miss");

    let full = ok(
        &client,
        &tools,
        "run_topological_sweep",
        args(&[
            ("detail", json!("full")),
            ("response_format", json!("markdown")),
        ]),
    )
    .await;
    assert_eq!(full["detail"], "full");
    assert!(full["metrics"]["component_sizes"].is_array());
    assert_eq!(
        full["projection_cache_status"]["status"], "hit",
        "same config reuses embeddings"
    );
    assert!(
        full["diff"].as_array().unwrap().is_empty(),
        "identical config → no diff"
    );

    let diag = ok(
        &client,
        &tools,
        "diagnose_cosmic_graph",
        args(&[("run_id", json!(run_id))]),
    )
    .await;
    assert_eq!(diag["source"], "live");
    assert_eq!(diag["graph_surface"]["kind"], "constructed_cosmic_graph");
    assert!(diag["minhash_profile"].is_object());
    let diag_md = ok(
        &client,
        &tools,
        "diagnose_cosmic_graph",
        args(&[
            ("detail", json!("full")),
            ("response_format", json!("markdown")),
        ]),
    )
    .await;
    assert!(diag_md["component_morphology"]["component_sizes"].is_array());

    let curve = ok(
        &client,
        &tools,
        "get_threshold_stability_curve",
        args(&[("run_id", json!(run_id))]),
    )
    .await;
    assert_eq!(curve["detail"], "summary");
    assert!(curve["threshold_candidates"].as_array().unwrap().len() <= 3);
    let curve_full = ok(
        &client,
        &tools,
        "get_threshold_stability_curve",
        args(&[
            ("detail", json!("full")),
            ("threshold_candidate_policy", json!("detail_seeking")),
        ]),
    )
    .await;
    assert_eq!(curve_full["threshold_candidate_policy"], "detail_seeking");
    assert_eq!(
        curve_full["thresholds"].as_array().unwrap().len(),
        curve_full["component_counts"].as_array().unwrap().len()
    );

    for detail in ["summary", "nodes", "edges", "full_nodes", "full"] {
        let skel = ok(
            &client,
            &tools,
            "get_topological_skeleton",
            args(&[
                ("detail", json!(detail)),
                ("max_edges", json!(5)),
                ("max_nodes", json!(7)),
            ]),
        )
        .await;
        assert_eq!(skel["graph"]["detail"], detail);
        if detail == "edges" {
            assert_eq!(skel["graph"]["edges_returned"], 5);
        }
        if detail == "full" {
            assert!(skel["config_yaml"].is_string());
        }
    }

    for detail in ["table", "summary", "full"] {
        let history = ok(
            &client,
            &tools,
            "get_sweep_history",
            args(&[
                ("detail", json!(detail)),
                ("response_format", json!("markdown")),
            ]),
        )
        .await;
        assert_eq!(history["n_runs"], 2);
    }
    let compare = ok(
        &client,
        &tools,
        "compare_sweeps",
        args(&[("run_a", json!(run_id)), ("run_b", json!(full["run_id"]))]),
    )
    .await;
    assert_eq!(compare["run_a"]["metrics"]["n_nodes"], 150);

    let job = ok(
        &client,
        &tools,
        "get_job",
        args(&[("run_id", json!(run_id))]),
    )
    .await;
    assert_eq!(job["status"], "complete");
    assert!(job["sweep"]["run_id"].as_str().is_some());

    let runtime = ok(&client, &tools, "get_runtime_context", args(&[])).await;
    assert_eq!(runtime["latest_run_id"], full["run_id"]);
    assert_eq!(runtime["model_cache"]["size"], 2);

    // Error envelopes.
    let e = err(
        &client,
        "diagnose_cosmic_graph",
        args(&[("run_id", json!("run_0123456789ab"))]),
    )
    .await;
    assert_eq!(e["error_code"], "RUN_ID_UNKNOWN");
    assert_eq!(e["details"]["run_id"], "run_0123456789ab");
    let e = err(
        &client,
        "diagnose_cosmic_graph",
        args(&[("run_id", json!("nope"))]),
    )
    .await;
    assert_eq!(e["error_code"], "INVALID_HANDLE");
    let e = err(
        &client,
        "run_topological_sweep",
        args(&[("component_limit", json!(0))]),
    )
    .await;
    assert_eq!(e["error_code"], "INVALID_ARGUMENT");
    let e = err(
        &client,
        "get_topological_skeleton",
        args(&[("detail", json!("bogus"))]),
    )
    .await;
    assert_eq!(e["error_code"], "INVALID_ARGUMENT");
    let e = err(
        &client,
        "ingest_dataset",
        args(&[("path", json!("/home/claude/data.csv"))]),
    )
    .await;
    assert_eq!(e["error_code"], "HOST_PATH_NOT_VISIBLE");
    let e = err(
        &client,
        "validate_config",
        args(&[("config_yaml", json!("run: [not a mapping"))]),
    )
    .await;
    assert_eq!(e["error_code"], "CONFIG_YAML_INVALID");
    let e = err(
        &client,
        "compare_sweeps",
        args(&[("run_a", json!(run_id)), ("run_b", json!(""))]),
    )
    .await;
    assert_eq!(e["error_code"], "INVALID_ARGUMENT");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn async_sweeps_are_polled_through_the_job_model_and_survive_a_new_client() {
    let root = TempRoot::new("async");
    let dataset_path = golden("blobs", "input.csv").display().to_string();
    let run_id;
    {
        let client = connect(root.path(), test_options(), ()).await;
        let tools = client.list_all_tools().await.unwrap();
        let ingested = ok(
            &client,
            &tools,
            "ingest_dataset",
            args(&[("path", json!(dataset_path))]),
        )
        .await;
        let submitted = ok(
            &client,
            &tools,
            "run_topological_sweep",
            args(&[
                ("config_yaml", json!(blobs_yaml())),
                ("dataset_id", ingested["dataset_id"].clone()),
                ("execution", json!("async")),
            ]),
        )
        .await;
        assert_eq!(submitted["status"], "running");
        assert_eq!(submitted["poll_with"], "wait_job");
        run_id = submitted["run_id"].as_str().unwrap().to_owned();
        let waited = ok(
            &client,
            &tools,
            "wait_job",
            args(&[("run_id", json!(run_id)), ("timeout_s", json!(120.0))]),
        )
        .await;
        assert_eq!(waited["status"], "complete");
        assert_eq!(waited["response_available"], true);
        assert_eq!(waited["sweep"]["run_id"], run_id);
        assert_eq!(waited["job"]["state"], "complete");
        assert_eq!(waited["job"]["progress"], 1.0);
        client.cancel().await.unwrap();
    }
    // A new server process over the same store: the run and its response are still there.
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let diag = ok(
        &client,
        &tools,
        "diagnose_cosmic_graph",
        args(&[("run_id", json!(run_id))]),
    )
    .await;
    assert_eq!(diag["source"], "persisted");
    assert_eq!(diag["scale"]["n_nodes"], 150);
    let job = ok(
        &client,
        &tools,
        "get_job",
        args(&[("run_id", json!(run_id))]),
    )
    .await;
    assert_eq!(job["response_available"], true);
    let curve = ok(
        &client,
        &tools,
        "get_threshold_stability_curve",
        args(&[("run_id", json!(run_id))]),
    )
    .await;
    assert_eq!(curve["source"], "persisted");
    let e = err(
        &client,
        "wait_job",
        args(&[("run_id", json!(run_id)), ("timeout_s", json!(0.0))]),
    )
    .await;
    assert_eq!(e["error_code"], "INVALID_ARGUMENT");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn oversized_responses_become_a_parseable_error_envelope() {
    let root = TempRoot::new("budget");
    let client = connect(
        root.path(),
        ServerOptions {
            max_response_bytes: Some(200),
            transport: "in-memory".into(),
        },
        (),
    )
    .await;
    let e = err(&client, "get_workflow_guide", args(&[])).await;
    assert_eq!(e["error_code"], "RESPONSE_TOO_LARGE");
    assert_eq!(e["details"]["limit_bytes"], 200);
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn task_capable_clients_get_a_task_handle_for_the_sweep() {
    let root = TempRoot::new("tasks");
    let config = ClientConfig::new(
        ClientCapabilities::builder().enable_tasks().build(),
        Implementation::new("pulsar-contract-test", "0"),
    );
    let client = connect(root.path(), test_options(), config).await;
    let tools = client.list_all_tools().await.unwrap();
    let ingested = ok(
        &client,
        &tools,
        "ingest_dataset",
        args(&[(
            "path",
            json!(golden("blobs", "input.csv").display().to_string()),
        )]),
    )
    .await;
    let mut params = CallToolRequestParams::new("run_topological_sweep");
    params.arguments = Some(args(&[
        ("config_yaml", json!(blobs_yaml())),
        ("dataset_id", ingested["dataset_id"].clone()),
        ("response_format", json!("json")),
    ]));
    let response = client.call_tool_once(params).await.unwrap();
    let CallToolResponse::Task(created) = response else {
        panic!("expected a task handle, got {response:?}");
    };
    let task_id = created.task.task_id.clone();
    let mut result: Option<Map<String, Value>> = None;
    for _ in 0..600 {
        let got = client
            .get_task(GetTaskParams::new(task_id.clone()))
            .await
            .unwrap();
        if got.task.status().is_terminal() {
            assert_eq!(
                got.task.status(),
                TaskStatus::Completed,
                "{:?}",
                got.task.payload
            );
            if let rmcp::model::TaskPayload::Completed { result: r } = got.task.payload {
                result = Some(r);
            }
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let result = result.expect("task completed");
    let payload = &result["structuredContent"];
    assert_eq!(payload["status"], "ok");
    assert!(payload["run_id"].as_str().unwrap().starts_with("run_"));
    client.cancel().await.unwrap();
}
