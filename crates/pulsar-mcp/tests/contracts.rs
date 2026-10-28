//! Contract tests through an in-process client (`test_mcp_contracts.py`):
//! every tool advertises its contract on the wire, `tools/list` is
//! deterministic and within budget, the canonical loop runs end to end with
//! every payload validating against its output schema, errors are
//! envelopes, async jobs and the tasks extension both deliver the sweep.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pulsar_mcp::{PulsarServer, ServerOptions, ServerState, CATALOG_BUDGET_CHARS};
use pulsar_store::{ClusterAssignmentId, ModelCache, Store};
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
    serve_state(Arc::new(ServerState::new(store, options).unwrap()), handler).await
}

/// Like `connect`, over a state the test built (e.g. a custom cache size).
async fn serve_state<H: ClientHandler>(
    state: Arc<ServerState>,
    handler: H,
) -> RunningService<RoleClient, H> {
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
        "generate_cluster_dossier",
        "get_cluster_profile",
        "get_feature_signal",
        "get_cluster_signal_matrix",
        "compare_clusters",
        "build_longitudinal_graph",
        "diagnose_longitudinal_graph",
        "get_trajectory_archetypes",
        "get_cross_time_neighbors",
        "classify_trajectories",
        "sample_longitudinal_entities",
        "export_labeled_data",
        "export_dataset_bundle",
        "create_graph_artifact",
        "export_html_report",
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
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(curve_full["threshold_candidate_policy"], "detail_seeking");
    assert_eq!(
        curve_full["thresholds"].as_array().unwrap().len(),
        curve_full["component_counts"].as_array().unwrap().len()
    );
    assert_eq!(
        curve_full["threshold_profiles"].as_array().unwrap().len(),
        curve_full["curve_point_count"].as_u64().unwrap() as usize
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

#[tokio::test]
async fn projection_cache_serves_any_live_run_not_only_the_latest() {
    let root = TempRoot::new("cache-aba");
    let client = connect(root.path(), test_options(), ()).await;
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
    let dataset_id = ingested["dataset_id"].clone();

    let config_a = blobs_yaml();
    let config_b = config_a.replace("      - 7\n", "      - 13\n");
    assert_ne!(config_a, config_b, "variant must change the seed grid");

    let mut statuses = Vec::new();
    for yaml in [&config_a, &config_b, &config_a] {
        let sweep = ok(
            &client,
            &tools,
            "run_topological_sweep",
            args(&[
                ("config_yaml", json!(yaml)),
                ("dataset_id", dataset_id.clone()),
                ("response_format", json!("json")),
            ]),
        )
        .await;
        statuses.push((
            sweep["projection_cache_status"]["status"]
                .as_str()
                .unwrap()
                .to_owned(),
            sweep["projection_cache_status"]["reason"]
                .as_str()
                .unwrap()
                .to_owned(),
        ));
    }
    assert_eq!(statuses[0], ("miss".into(), "no_cached_embeddings".into()));
    assert_eq!(statuses[1], ("miss".into(), "fingerprint_mismatch".into()));
    assert_eq!(
        statuses[2],
        ("hit".into(), "fingerprint_match".into()),
        "A → B → A must reuse A's embeddings"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn projection_cache_miss_keeps_lru_order() {
    let root = TempRoot::new("cache-lru");
    let mut state = ServerState::new(Store::open_at(root.path()).unwrap(), test_options()).unwrap();
    // Capacity normally comes from `PULSAR_MCP_MODEL_CACHE_SIZE`; set it here
    // instead so parallel tests do not race on the environment.
    state.cache = ModelCache::new(2);
    let state = Arc::new(state);
    let client = serve_state(state.clone(), ()).await;
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
    let dataset_id = ingested["dataset_id"].clone();

    let config_a = blobs_yaml();
    let config_b = config_a.replace("      - 7\n", "      - 13\n");
    let config_c = config_a.replace("      - 7\n", "      - 21\n");
    assert!(config_a != config_b && config_b != config_c && config_a != config_c);

    let mut sweeps = Vec::new();
    for yaml in [&config_a, &config_b, &config_c, &config_a] {
        sweeps.push(
            ok(
                &client,
                &tools,
                "run_topological_sweep",
                args(&[
                    ("config_yaml", json!(yaml)),
                    ("dataset_id", dataset_id.clone()),
                    ("response_format", json!("json")),
                ]),
            )
            .await,
        );
    }
    let reason = |s: &Value| s["projection_cache_status"]["reason"].clone();
    let run_id = |s: &Value| s["run_id"].as_str().unwrap().to_owned();
    // A was evicted by C, so the second A misses; the scan must not reorder
    // [B, C], so A's put evicts B (least recent) and C survives.
    assert_eq!(reason(&sweeps[3]), "fingerprint_mismatch");
    let live: Vec<String> = state
        .cache
        .run_ids()
        .iter()
        .map(|id| id.to_string())
        .collect();
    assert_eq!(live, vec![run_id(&sweeps[2]), run_id(&sweeps[3])]);

    let c_again = ok(
        &client,
        &tools,
        "run_topological_sweep",
        args(&[
            ("config_yaml", json!(config_c)),
            ("dataset_id", dataset_id.clone()),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(
        reason(&c_again),
        "fingerprint_match",
        "C must still be live"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn multi_imputation_config_validates_and_sweeps() {
    let root = TempRoot::new("multi-imputation");
    let csv_path = root.path().join("missing.csv");
    std::fs::write(
        &csv_path,
        "a,b,c\n1.0,2.0,3.0\n,2.5,3.5\n1.5,,4.0\n2.0,3.0,\n2.5,3.5,4.5\n3.0,4.0,5.0\n",
    )
    .unwrap();

    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let ingested = ok(
        &client,
        &tools,
        "ingest_dataset",
        args(&[("path", json!(csv_path.display().to_string()))]),
    )
    .await;
    let dataset_id = ingested["dataset_id"].clone();

    let yaml = r#"
version: 1
run:
  name: multi_imp
preprocessing:
  n_imputations: 3
  impute:
    a:
      method: [fill_mean, fill_median]
    b:
      - {method: fill_mean, indicator: false}
      - {method: fill_value, value: 0.0, indicator: true}
    c:
      method: sample_normal
      seed: 7
sweep:
  projection:
    method: jl
    dimensions: [2]
    seed: [42]
  ball_mapper:
    epsilon: [0.5, 1.0]
"#;

    let validated = ok(
        &client,
        &tools,
        "validate_config",
        args(&[
            ("config_yaml", json!(yaml)),
            ("dataset_id", dataset_id.clone()),
        ]),
    )
    .await;
    assert_eq!(validated["status"], "ok");
    let norm = validated["normalized_config_yaml"].as_str().unwrap();
    assert!(norm.contains("n_imputations: 3"), "{norm}");

    let sweep = ok(
        &client,
        &tools,
        "run_topological_sweep",
        args(&[
            ("config_yaml", json!(yaml)),
            ("dataset_id", dataset_id),
            ("detail", json!("summary")),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(sweep["status"], "ok");
    // 3 imputation variants * 1 dim * 1 seed * 2 epsilons = 6 ball maps.
    assert_eq!(sweep["key_metrics"]["n_ball_maps"], 6);
    client.cancel().await.unwrap();
}

// ---- preprocessing-plan tools -------------------------------------------------

fn plan_csv() -> String {
    golden("plan", "wide.csv").display().to_string()
}

async fn ingest_wide<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    tools: &[Tool],
) -> String {
    let ingested = ok(
        client,
        tools,
        "ingest_dataset",
        args(&[("path", json!(plan_csv()))]),
    )
    .await;
    ingested["dataset_id"].as_str().unwrap().to_owned()
}

fn question_of<'a>(payload: &'a Value, topic: &str) -> &'a Value {
    payload["questions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|q| q["topic"] == topic)
        .unwrap_or_else(|| panic!("no {topic} question in {:?}", payload["questions"]))
}

fn code_of(e: &Value) -> String {
    e["error_code"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn the_plan_loop_persists_checks_revisions_and_survives_a_restart() {
    let root = TempRoot::new("plan-loop");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let dataset_id = ingest_wide(&client, &tools).await;

    // Draft: labeled defaults, open questions with measured stakes, summary-first.
    let draft = ok(
        &client,
        &tools,
        "draft_preprocessing_plan",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let plan_id = draft["plan_id"].as_str().unwrap().to_owned();
    assert!(
        plan_id.starts_with("pp_") && plan_id.len() == 15,
        "{plan_id}"
    );
    assert_eq!(
        (draft["revision"].clone(), draft["detail"].clone()),
        (json!(1), json!("summary"))
    );
    assert!(draft["processed_space"]["feature_count"].as_u64().unwrap() > 0);
    assert!(
        draft["questions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|q| q["stakes"]["neighbors_kept"].is_number()),
        "stakes were computed"
    );
    assert!(
        draft.get("entries").is_none() && draft.get("preprocessing_yaml").is_none(),
        "summary omits the full object"
    );
    // Next steps only name tools this server registers.
    let registered: BTreeSet<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    for next in draft["next_tools"].as_array().unwrap() {
        assert!(
            registered.contains(next.as_str().unwrap()),
            "{next} is advertised but not registered"
        );
    }

    // Edit: answer a question; the new revision is stored and validated.
    let date_q = question_of(&draft, "date_columns")["question_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let edited = ok(
        &client,
        &tools,
        "edit_preprocessing_plan",
        args(&[
            ("plan_id", json!(plan_id)),
            ("base_revision", json!(1)),
            (
                "answers",
                json!([{"question_id": date_q, "option_id": "has_date"}]),
            ),
            ("source", json!("user")),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(edited["revision"], 2);
    assert_eq!(edited["validation"]["valid"], true);
    assert!(!edited["changes"].as_array().unwrap().is_empty());
    assert_ne!(edited["plan_fingerprint"], draft["plan_fingerprint"]);

    // A stale base_revision is a conflict that says what to do.
    let conflict = err(
        &client,
        "edit_preprocessing_plan",
        args(&[
            ("plan_id", json!(plan_id)),
            ("base_revision", json!(1)),
            ("edits", json!([{"column": "visits1", "role": "drop"}])),
        ]),
    )
    .await;
    assert_eq!(conflict["error_code"], "SESSION_REVISION_CONFLICT");
    assert_eq!(
        (
            conflict["details"]["current_revision"].clone(),
            conflict["details"]["base_revision"].clone()
        ),
        (json!(2), json!(1))
    );
    assert!(conflict["agent_action"]
        .as_str()
        .unwrap()
        .contains("base_revision=2"));
    client.cancel().await.unwrap();

    // Restart: a new server over the same store still has the plan.
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let again = ok(
        &client,
        &tools,
        "edit_preprocessing_plan",
        args(&[
            ("plan_id", json!(plan_id)),
            (
                "edits",
                json!([{"column": "visits1", "impute_method": "fill_mean"}]),
            ),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(
        again["revision"], 3,
        "base_revision omitted edits the current revision"
    );
    assert!(again["changes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["column"] == "visits1"));
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn a_target_edit_answers_the_outcome_question() {
    let root = TempRoot::new("plan-target");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let dataset_id = ingest_wide(&client, &tools).await;
    let draft = ok(
        &client,
        &tools,
        "draft_preprocessing_plan",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("stakes", json!(false)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let plan_id = draft["plan_id"].as_str().unwrap();
    assert_eq!(
        question_of(&draft, "outcome_columns")["topic"],
        "outcome_columns"
    );

    let edited = ok(
        &client,
        &tools,
        "edit_preprocessing_plan",
        args(&[
            ("plan_id", json!(plan_id)),
            ("edits", json!([{"column": "grade", "role": "target"}])),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let open_topics: Vec<&str> = edited["open_questions_preview"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| q["topic"].as_str().unwrap())
        .collect();
    assert!(
        !open_topics.contains(&"outcome_columns"),
        "naming a target answers the outcome question: {open_topics:?}"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn draft_detail_full_and_markdown_views() {
    let root = TempRoot::new("plan-full");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let dataset_id = ingest_wide(&client, &tools).await;

    let full = ok(
        &client,
        &tools,
        "draft_preprocessing_plan",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("detail", json!("full")),
            ("stakes", json!(false)),
        ]),
    )
    .await;
    assert!(full["entries"].as_object().unwrap().len() >= 10);
    assert!(full["preprocessing_yaml"]
        .as_str()
        .unwrap()
        .contains("preprocessing:"));
    assert_eq!(
        full["cost"]["stakes_questions_computed"], 0,
        "stakes=false skips scoring"
    );

    let result = call(
        &client,
        "draft_preprocessing_plan",
        args(&[("dataset_id", json!(dataset_id)), ("stakes", json!(false))]),
    )
    .await;
    let text = match &result.content[0] {
        rmcp::model::ContentBlock::Text(t) => t.text.clone(),
        other => panic!("expected text, got {other:?}"),
    };
    assert!(
        text.starts_with("# Preprocessing Plan `pp_"),
        "{}",
        &text[..60.min(text.len())]
    );
    assert!(text.contains("## Open Questions") && text.contains("## Evidence Headline"));
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn draft_and_edit_errors_tell_the_agent_what_to_do() {
    let root = TempRoot::new("plan-errors");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let dataset_id = ingest_wide(&client, &tools).await;
    let draft = ok(
        &client,
        &tools,
        "draft_preprocessing_plan",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("stakes", json!(false)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let plan_id = draft["plan_id"].as_str().unwrap().to_owned();

    // draft_preprocessing_plan
    assert_eq!(
        code_of(&err(&client, "draft_preprocessing_plan", args(&[])).await),
        "MISSING_INPUT"
    );
    assert_eq!(
        code_of(
            &err(
                &client,
                "draft_preprocessing_plan",
                args(&[("dataset_id", json!("ds_000000000000"))])
            )
            .await
        ),
        "DATASET_ID_UNKNOWN"
    );
    let bad_limit = err(
        &client,
        "draft_preprocessing_plan",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("question_limit", json!(0)),
        ]),
    )
    .await;
    assert_eq!(bad_limit["error_code"], "INVALID_ARGUMENT");
    assert!(bad_limit["reason"]
        .as_str()
        .unwrap()
        .contains("question_limit must be >= 1, got '0'"));
    let bad_target = err(
        &client,
        "draft_preprocessing_plan",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("target_columns", json!(["nope"])),
        ]),
    )
    .await;
    assert_eq!(bad_target["error_code"], "INVALID_ARGUMENT");
    assert_eq!(
        bad_target["details"]["unknown_target_columns"],
        json!(["nope"])
    );

    // edit_preprocessing_plan
    assert_eq!(
        code_of(&err(&client, "edit_preprocessing_plan", args(&[])).await),
        "MISSING_INPUT"
    );
    for (id, want) in [
        ("pp_000000000000", "PLAN_ID_UNKNOWN"),
        ("nonsense", "INVALID_HANDLE"),
    ] {
        let e = err(
            &client,
            "edit_preprocessing_plan",
            args(&[("plan_id", json!(id)), ("edits", json!([{}]))]),
        )
        .await;
        assert_eq!(code_of(&e), want, "{id}");
    }
    let nothing = err(
        &client,
        "edit_preprocessing_plan",
        args(&[("plan_id", json!(plan_id))]),
    )
    .await;
    assert_eq!(
        nothing["reason"],
        "Nothing to change: provide at least one answer or edit."
    );
    let unknown_col = err(
        &client,
        "edit_preprocessing_plan",
        args(&[
            ("plan_id", json!(plan_id)),
            ("edits", json!([{"column": "ghost", "role": "drop"}])),
        ]),
    )
    .await;
    assert_eq!(unknown_col["reason"], "Unknown column: 'ghost'.");
    assert_eq!(unknown_col["details"]["column"], "ghost");
    let no_blanks = err(
        &client,
        "edit_preprocessing_plan",
        args(&[
            ("plan_id", json!(plan_id)),
            (
                "edits",
                json!([{"column": "a", "impute_method": "fill_mean"}]),
            ),
        ]),
    )
    .await;
    assert!(no_blanks["agent_action"]
        .as_str()
        .unwrap()
        .contains("it has no blanks"));
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn validate_preprocessing_config_dry_runs_only_the_preprocessing() {
    let root = TempRoot::new("plan-validate");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let hint = |n: &str| {
        tools
            .iter()
            .find(|t| t.name == n)
            .and_then(|t| t.annotations.as_ref())
            .and_then(|a| a.read_only_hint)
    };
    assert_eq!(hint("validate_preprocessing_config"), Some(true));
    assert_eq!(hint("draft_preprocessing_plan"), Some(false));
    assert_eq!(hint("edit_preprocessing_plan"), Some(false));

    let yaml = |pre: &str| {
        format!(
            "run:\n  name: t\n{pre}\nsweep:\n  projection:\n    method: jl\n    dimensions: [2]\n    seed: [42]\n  ball_mapper:\n    epsilon: [0.5]\n"
        )
    };
    // No dataset yet: a clear, actionable error.
    let no_data = err(
        &client,
        "validate_preprocessing_config",
        args(&[(
            "config_yaml",
            json!(yaml("preprocessing:\n  drop_columns: [id]")),
        )]),
    )
    .await;
    assert_eq!(no_data["error_code"], "SESSION_DATA_MISSING");

    let dataset_id = ingest_wide(&client, &tools).await;
    let draft = ok(
        &client,
        &tools,
        "draft_preprocessing_plan",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("detail", json!("full")),
            ("stakes", json!(false)),
        ]),
    )
    .await;
    let plan_yaml = draft["preprocessing_yaml"].as_str().unwrap();

    // The plan's own preprocessing block validates, using the session's dataset.
    let valid = ok(
        &client,
        &tools,
        "validate_preprocessing_config",
        args(&[("config_yaml", json!(yaml(plan_yaml)))]),
    )
    .await;
    assert_eq!(
        (
            valid["status"].clone(),
            valid["valid"].clone(),
            valid["nan_remaining"].clone()
        ),
        (json!("ok"), json!(true), json!(0))
    );
    assert_eq!(valid["input_rows"], 60);
    assert_eq!(
        valid["output_feature_count"], draft["processed_space"]["feature_count"],
        "agrees with the draft's dry run"
    );
    assert_eq!(
        valid["missingness_flag_count"],
        draft["processed_space"]["missing_marker_count"]
    );
    assert!(valid["feature_names_preview"].as_array().unwrap().len() <= 9);

    // Errors: bad YAML syntax, an unknown key, a plan the data cannot satisfy, empty text.
    let with = |text: &str| {
        args(&[
            ("config_yaml", json!(text)),
            ("dataset_id", json!(dataset_id)),
        ])
    };
    let syntax = err(
        &client,
        "validate_preprocessing_config",
        with("run: [unclosed"),
    )
    .await;
    assert_eq!(syntax["error_code"], "CONFIG_YAML_INVALID");
    let unknown_key = err(
        &client,
        "validate_preprocessing_config",
        with(&yaml("preprocessing:\n  nonsense: 1")),
    )
    .await;
    assert_eq!(unknown_key["error_code"], "CONFIG_INVALID");
    assert_eq!(unknown_key["details"]["valid"], false);
    let unfit = err(
        &client,
        "validate_preprocessing_config",
        with(&yaml(
            "preprocessing:\n  impute:\n    last_seen: {method: fill_median}",
        )),
    )
    .await;
    assert_eq!(unfit["error_code"], "CONFIG_INVALID", "{unfit}");
    assert_eq!(
        code_of(&err(&client, "validate_preprocessing_config", with("")).await),
        "CONFIG_YAML_INVALID"
    );
    client.cancel().await.unwrap();
}

// ---- characterization tools ---------------------------------------------------

#[tokio::test]
async fn characterize_and_probe_read_raw_data_and_explain_failures() {
    let root = TempRoot::new("characterize");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    for name in ["characterize_dataset", "probe_columns"] {
        let t = tools.iter().find(|t| t.name == name).expect(name);
        assert_eq!(
            t.annotations.as_ref().and_then(|a| a.read_only_hint),
            Some(true),
            "{name}"
        );
    }

    let dataset_id = ingest_wide(&client, &tools).await;
    let by_id = ok(
        &client,
        &tools,
        "characterize_dataset",
        args(&[("dataset_id", json!(dataset_id))]),
    )
    .await;
    assert_eq!(by_id["status"], "ok");
    assert!(by_id["n_samples"].as_u64().unwrap() > 1);
    // Only registered tools are advertised as the next step.
    let next = by_id["recommended_next_tool"].as_str().unwrap();
    assert!(tools.iter().any(|t| t.name == next), "{next}");

    // A raw path gives the same measurements.
    let by_path = ok(
        &client,
        &tools,
        "characterize_dataset",
        args(&[("csv_path", json!(plan_csv()))]),
    )
    .await;
    assert_eq!(by_path["n_samples"], by_id["n_samples"]);
    assert_eq!(
        by_path["raw_numeric_geometry"],
        by_id["raw_numeric_geometry"]
    );

    let markdown = call(
        &client,
        "characterize_dataset",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("response_format", json!("markdown")),
        ]),
    )
    .await;
    let text = format!("{:?}", markdown.content);
    for named in ["create_config", "refine_config"] {
        // Never advertise a tool the server does not have.
        assert!(
            !text.contains(named) || tools.iter().any(|t| t.name == named),
            "{named} advertised but not registered"
        );
    }
    assert!(
        text.contains("create_config"),
        "the calibration step is named"
    );

    // Probe: known columns, an unknown one, and the 20-column limit.
    let names: Vec<String> = by_id["column_profile_preview"]
        .as_array()
        .unwrap()
        .iter()
        .take(2)
        .map(|c| c["name"].as_str().unwrap().to_owned())
        .collect();
    let mut wanted = names.clone();
    wanted.push("does_not_exist".into());
    let probe = ok(
        &client,
        &tools,
        "probe_columns",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("columns", json!(wanted)),
        ]),
    )
    .await;
    assert_eq!(probe["columns_requested"], 3);
    assert_eq!(probe["columns_returned"], 2);
    assert_eq!(probe["missing_columns"], json!(["does_not_exist"]));

    let too_many = err(
        &client,
        "probe_columns",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("columns", json!(vec!["a"; 21])),
        ]),
    )
    .await;
    assert_eq!(code_of(&too_many), "TOO_MANY_COLUMNS");
    assert_eq!(too_many["details"]["max"], 20);

    // Errors that unblock the agent.
    assert_eq!(
        code_of(&err(&client, "characterize_dataset", args(&[])).await),
        "DATASET_NOT_SPECIFIED"
    );
    let missing = err(
        &client,
        "characterize_dataset",
        args(&[("csv_path", json!("/no/such/dir/data.csv"))]),
    )
    .await;
    assert_eq!(code_of(&missing), "FILE_NOT_FOUND", "{missing}");
    // csv_path gets ingest_dataset's checks: only regular files are read.
    let mut not_regular = vec![root.path().display().to_string()];
    if cfg!(unix) {
        not_regular.push("/dev/null".into());
    }
    for path in not_regular {
        let refused = err(
            &client,
            "characterize_dataset",
            args(&[("csv_path", json!(path))]),
        )
        .await;
        assert_eq!(code_of(&refused), "FILE_NOT_FOUND", "{path}: {refused}");
    }
    assert!(by_path["dataset_id"].as_str().unwrap().starts_with("ds_"));
    let one_numeric = err(
        &client,
        "characterize_dataset",
        args(&[(
            "csv_path",
            json!(golden("plan", "characterize_one_numeric.csv")
                .display()
                .to_string()),
        )]),
    )
    .await;
    assert!(
        one_numeric["reason"]
            .as_str()
            .unwrap()
            .contains("at least 2 numeric columns"),
        "{one_numeric}"
    );
    let unknown = err(
        &client,
        "probe_columns",
        args(&[("dataset_id", json!("ds_nope")), ("columns", json!(["a"]))]),
    )
    .await;
    assert!(!code_of(&unknown).is_empty());
    client.cancel().await.unwrap();
}

// ---- create_config -------------------------------------------------------------

#[tokio::test]
async fn create_config_calibrates_a_baseline_that_validates_and_becomes_active() {
    let root = TempRoot::new("create-config");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let dataset_id = ingest_wide(&client, &tools).await;

    // Default plan: drafted, stored and reported with provenance.
    let created = ok(
        &client,
        &tools,
        "create_config",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("intent", json!("baseline")),
        ]),
    )
    .await;
    assert_eq!(created["calibration_space"], "processed");
    assert_eq!(created["preprocessing_plan"]["source"], "baseline_defaults");
    assert_eq!(created["preprocessing_plan"]["revision"], 1);
    let yaml = created["config_yaml"].as_str().unwrap();
    assert!(yaml.contains("name: baseline"), "{yaml}");
    let strategy = &created["sweep_strategy"];
    let (dims, seeds, steps) = (
        strategy["projection_dimensions"].as_array().unwrap().len(),
        strategy["projection_seeds"].as_array().unwrap().len(),
        strategy["epsilon_steps"].as_u64().unwrap() as usize,
    );
    assert_eq!(
        strategy["estimated_ball_maps"].as_u64().unwrap() as usize,
        strategy["imputation_variants"].as_u64().unwrap() as usize * dims * seeds * steps
    );
    let p = &created["knn_distance_percentiles"];
    assert!(p["p5"].as_f64().unwrap() <= p["p95"].as_f64().unwrap());
    // Only registered tools are advertised.
    let guidance = strategy["agent_guidance"].as_str().unwrap();
    for named in ["refine_config", "compare_sweeps"] {
        assert_eq!(
            guidance.contains(named),
            tools.iter().any(|t| t.name == named),
            "{named} in {guidance}"
        );
    }

    // It is the session's active config, and a valid one.
    let context = ok(&client, &tools, "get_runtime_context", args(&[])).await;
    assert_eq!(context["active_config_yaml"], created["config_yaml"]);
    assert_eq!(context["active_config_dataset_id"], json!(dataset_id));
    let valid = ok(
        &client,
        &tools,
        "validate_config",
        args(&[("config_yaml", json!(yaml))]),
    )
    .await;
    assert_eq!(valid["status"], "ok");

    // An explicit plan is used as is; a second default plan is not stored again.
    let plan_id = created["preprocessing_plan"]["plan_id"].as_str().unwrap();
    let explicit = ok(
        &client,
        &tools,
        "create_config",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("plan_id", json!(plan_id)),
        ]),
    )
    .await;
    assert_eq!(explicit["preprocessing_plan"]["source"], "explicit_plan");
    assert_eq!(explicit["preprocessing_plan"]["plan_id"], json!(plan_id));
    assert_eq!(
        explicit["preprocessing_plan"]["plan_fingerprint"],
        created["preprocessing_plan"]["plan_fingerprint"]
    );

    // Edit the plan; a call without plan_id warns that the edit is bypassed.
    let edited = ok(
        &client,
        &tools,
        "edit_preprocessing_plan",
        args(&[
            ("plan_id", json!(plan_id)),
            ("edits", json!([{"column": "visits1", "role": "drop"}])),
        ]),
    )
    .await;
    assert_eq!(edited["revision"], 2);
    let bypassed = ok(
        &client,
        &tools,
        "create_config",
        args(&[("dataset_id", json!(dataset_id))]),
    )
    .await;
    let warnings = bypassed["warnings"].as_array().expect("warnings");
    let not_used = warnings
        .iter()
        .find(|w| w["code"] == "PREPROCESSING_PLAN_NOT_USED")
        .unwrap_or_else(|| panic!("no bypass warning in {warnings:?}"));
    assert!(not_used["message"]
        .as_str()
        .unwrap()
        .contains(&format!("Pass plan_id={plan_id}")));
    assert_ne!(bypassed["preprocessing_plan"]["plan_id"], json!(plan_id));

    // Passing the edited plan uses it: its revision and fingerprint differ.
    let used = ok(
        &client,
        &tools,
        "create_config",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("plan_id", json!(plan_id)),
        ]),
    )
    .await;
    assert_eq!(used["preprocessing_plan"]["revision"], 2);
    assert_ne!(
        used["preprocessing_plan"]["plan_fingerprint"],
        created["preprocessing_plan"]["plan_fingerprint"]
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn create_config_errors_tell_the_agent_what_to_do() {
    let root = TempRoot::new("create-config-errors");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let dataset_id = ingest_wide(&client, &tools).await;

    assert_eq!(
        code_of(&err(&client, "create_config", args(&[])).await),
        "MISSING_INPUT"
    );
    assert!(!code_of(
        &err(
            &client,
            "create_config",
            args(&[("dataset_id", json!("ds_nope"))])
        )
        .await
    )
    .is_empty());
    let unknown_plan = err(
        &client,
        "create_config",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("plan_id", json!("pp_doesnotexist")),
        ]),
    )
    .await;
    assert!(!code_of(&unknown_plan).is_empty());

    // A plan of another dataset is refused.
    let other = ok(
        &client,
        &tools,
        "ingest_dataset",
        args(&[(
            "path",
            json!(golden("blobs", "input.csv").display().to_string()),
        )]),
    )
    .await;
    let other_id = other["dataset_id"].as_str().unwrap();
    let plan = ok(
        &client,
        &tools,
        "draft_preprocessing_plan",
        args(&[("dataset_id", json!(other_id)), ("stakes", json!(false))]),
    )
    .await;
    let mismatch = err(
        &client,
        "create_config",
        args(&[
            ("dataset_id", json!(dataset_id)),
            ("plan_id", plan["plan_id"].clone()),
        ]),
    )
    .await;
    assert_eq!(code_of(&mismatch), "INVALID_ARGUMENT");
    assert_eq!(mismatch["details"]["plan_dataset_id"], json!(other_id));
    client.cancel().await.unwrap();
}

// ---- refine_config -------------------------------------------------------------

fn text_of(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn refine_config_edits_the_active_config_and_reports_a_diff() {
    let root = TempRoot::new("refine-config");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();

    // Nothing to refine yet.
    assert_eq!(
        code_of(&err(&client, "refine_config", args(&[])).await),
        "ACTIVE_CONFIG_MISSING"
    );

    let dataset_id = ingest_wide(&client, &tools).await;
    let created = ok(
        &client,
        &tools,
        "create_config",
        args(&[("dataset_id", json!(dataset_id))]),
    )
    .await;
    let before = created["config_yaml"].as_str().unwrap().to_owned();

    // A nested override changes one leaf; the active config follows.
    let refined = ok(
        &client,
        &tools,
        "refine_config",
        args(&[(
            "overrides",
            json!({"sweep": {"ball_mapper": {"epsilon": {"range": {"max": 9.0}}}}}),
        )]),
    )
    .await;
    assert_eq!(
        refined["applied_overrides"],
        json!(["sweep.ball_mapper.epsilon.range.max"])
    );
    assert_eq!(refined["diff"][0]["new"], 9.0);
    assert_eq!(refined["dataset_id"], json!(dataset_id));
    assert_ne!(refined["config_yaml"].as_str().unwrap(), before);
    let context = ok(&client, &tools, "get_runtime_context", args(&[])).await;
    assert_eq!(context["active_config_yaml"], refined["config_yaml"]);

    // Short keys and deletion; the result still validates.
    let again = ok(
        &client,
        &tools,
        "refine_config",
        args(&[(
            "overrides",
            json!({"projection_dimensions": [2, 3], "remove_keys": ["output.n_reps"]}),
        )]),
    )
    .await;
    assert!(again["applied_overrides"]
        .as_array()
        .unwrap()
        .contains(&json!("-output.n_reps")));
    assert!(again["diff"]
        .as_array()
        .unwrap()
        .iter()
        .any(|d| d["removed"] == true));
    let valid = ok(
        &client,
        &tools,
        "validate_config",
        args(&[("config_yaml", again["config_yaml"].clone())]),
    )
    .await;
    assert_eq!(valid["status"], "ok");

    // Text views.
    let markdown = call(
        &client,
        "refine_config",
        args(&[
            ("overrides", json!({"n_reps": 2})),
            ("response_format", json!("markdown")),
        ]),
    )
    .await;
    let md = text_of(&markdown);
    assert!(md.starts_with("# Refined Config"), "{md}");
    assert!(md.contains("| `output.n_reps` |"), "{md}");
    let yaml_view = call(
        &client,
        "refine_config",
        args(&[
            ("overrides", json!({"n_reps": 3})),
            ("response_format", json!("yaml")),
        ]),
    )
    .await;
    assert!(text_of(&yaml_view).starts_with("version:"), "{yaml_view:?}");

    // The sparsify caution.
    let sparse = ok(
        &client,
        &tools,
        "refine_config",
        args(&[("overrides", json!({"cosmic_graph": {"sparsify": true}}))]),
    )
    .await;
    assert_eq!(sparse["warnings"][0]["path"], "cosmic_graph.sparsify");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn refine_config_rejects_bad_overrides_without_changing_anything() {
    let root = TempRoot::new("refine-config-errors");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let dataset_id = ingest_wide(&client, &tools).await;
    let created = ok(
        &client,
        &tools,
        "create_config",
        args(&[("dataset_id", json!(dataset_id))]),
    )
    .await;

    let unknown = err(
        &client,
        "refine_config",
        args(&[("overrides", json!({"bogus": 1}))]),
    )
    .await;
    assert_eq!(code_of(&unknown), "UNKNOWN_OVERRIDE_KEY");
    assert!(unknown["reason"].as_str().unwrap().contains("['bogus']"));

    let wrong_shape = err(
        &client,
        "refine_config",
        args(&[("overrides", json!({"projection_dimensions": 5}))]),
    )
    .await;
    assert_eq!(code_of(&wrong_shape), "CONFIG_REFINE_FAILED");

    let invalid = err(
        &client,
        "refine_config",
        args(&[("overrides", json!({"projection_method": "bogus"}))]),
    )
    .await;
    assert_eq!(code_of(&invalid), "CONFIG_REFINE_FAILED");
    assert!(invalid["details"]["issues"]
        .as_array()
        .is_some_and(|i| !i.is_empty()));

    let not_yaml = err(
        &client,
        "refine_config",
        args(&[("config_yaml", json!("- just\n- a list"))]),
    )
    .await;
    assert_eq!(code_of(&not_yaml), "CONFIG_REFINE_FAILED");

    // The active config survived every rejection.
    let context = ok(&client, &tools, "get_runtime_context", args(&[])).await;
    assert_eq!(context["active_config_yaml"], created["config_yaml"]);

    // An explicit config_yaml is refined without touching the active config.
    let explicit = ok(
        &client,
        &tools,
        "refine_config",
        args(&[
            ("config_yaml", created["config_yaml"].clone()),
            ("overrides", json!({"n_reps": 7})),
        ]),
    )
    .await;
    assert!(explicit.get("dataset_id").is_none());
    let context = ok(&client, &tools, "get_runtime_context", args(&[])).await;
    assert_eq!(context["active_config_yaml"], created["config_yaml"]);
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn the_cluster_dossier_persists_and_publishes_its_assignment() {
    let root = TempRoot::new("dossier");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    const TOOL: &str = "generate_cluster_dossier";

    let none = err(&client, TOOL, args(&[])).await;
    assert_eq!(none["error_code"], "MODEL_NOT_LOADED");

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
    let sweep_args = args(&[
        ("config_yaml", json!(blobs_yaml())),
        ("dataset_id", json!(dataset_id)),
        ("response_format", json!("json")),
    ]);
    let sweep = ok(&client, &tools, "run_topological_sweep", sweep_args.clone()).await;
    let run_id = sweep["run_id"].as_str().unwrap().to_owned();

    let summary = ok(
        &client,
        &tools,
        TOOL,
        args(&[("response_format", json!("json"))]),
    )
    .await;
    assert_eq!(summary["run_id"], json!(run_id));
    assert_eq!(summary["clusters_run_id"], json!(run_id));
    assert_eq!(summary["threshold_source"], "inherited_construction");
    assert_eq!(summary["threshold_inherited"], true);
    assert_eq!(
        summary["interpretation_edge_weight_threshold"],
        summary["construction_threshold"]
    );
    assert!(!summary["clusters"].as_array().unwrap().is_empty());
    assert!(summary["graph_metrics"]["component_sizes_summary"].is_object());
    assert!(summary["graph_metrics"].get("component_sizes").is_none());

    // The assignment is persisted with its provenance.
    let ca = summary["cluster_assignment_id"].as_str().unwrap();
    let store = Store::open_at(root.path()).unwrap();
    let id = ClusterAssignmentId::parse(ca).unwrap();
    let record = store.get_cluster_assignment(&id).unwrap();
    assert_eq!(record.label_count, 150);
    assert_eq!(
        record.run_id.as_ref().map(|r| r.to_string()),
        Some(run_id.clone())
    );
    assert_eq!(
        record.dataset_id.as_ref().map(|d| d.to_string()),
        Some(dataset_id.clone())
    );
    assert_eq!(store.cluster_labels(&id).unwrap().len(), 150);
    assert_eq!(summary["cluster_assignment"]["label_count"], 150);

    let standard = call(
        &client,
        TOOL,
        args(&[("detail", json!("standard")), ("max_clusters", json!(2))]),
    )
    .await;
    assert_eq!(standard.is_error, Some(false), "{:?}", standard.content);
    let text = text_of(&standard);
    assert!(text.contains("## Two Levers"), "{text}");
    assert!(text.contains("## Cluster Assignment Artifact"), "{text}");
    let standard = standard.structured_content.unwrap();
    assert_eq!(standard["surface_labels_ref"], "get_workflow_guide");
    assert_ne!(
        standard["cluster_assignment_id"],
        json!(ca),
        "every dossier persists anew"
    );

    let full = ok(
        &client,
        &tools,
        TOOL,
        args(&[
            ("detail", json!("full")),
            ("method", json!("components")),
            ("interpretation_edge_weight_threshold", json!(0.5)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(full["threshold_source"], "explicit");
    assert!(full["surface_labels"]["clusters"].is_string());

    let bad = err(&client, TOOL, args(&[("max_clusters", json!(0))])).await;
    assert_eq!(bad["error_code"], "INVALID_ARGUMENT");
    let bad = err(
        &client,
        TOOL,
        args(&[("interpretation_edge_weight_threshold", json!(1.5))]),
    )
    .await;
    assert_eq!(bad["error_code"], "INVALID_ARGUMENT");
    assert_eq!(bad["details"]["interpretation_edge_weight_threshold"], 1.5);
    let cut = err(
        &client,
        TOOL,
        args(&[("method", json!("spectral")), ("max_k", json!(1))]),
    )
    .await;
    assert_eq!(cut["error_code"], "NO_STABLE_SPECTRAL_CUT");
    assert_eq!(cut["details"]["k_min"], 2);
    assert!(cut["details"].get("candidate_scores").is_none());

    // A new sweep clears the published labels; the old run can no longer
    // receive a dossier.
    let second = ok(&client, &tools, "run_topological_sweep", sweep_args).await;
    assert_ne!(second["run_id"], json!(run_id));
    let stale = err(&client, TOOL, args(&[("run_id", json!(run_id))])).await;
    assert_eq!(stale["error_code"], "SESSION_REVISION_CONFLICT");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn the_cluster_tools_read_the_published_labels() {
    let root = TempRoot::new("cluster-tools");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let profile_args = || args(&[("cluster_id", json!(0))]);

    let none = err(&client, "get_cluster_profile", profile_args()).await;
    assert_eq!(none["error_code"], "MODEL_NOT_LOADED");

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
    let sweep_args = args(&[
        ("config_yaml", json!(blobs_yaml())),
        ("dataset_id", ingested["dataset_id"].clone()),
        ("response_format", json!("json")),
    ]);
    let sweep = ok(&client, &tools, "run_topological_sweep", sweep_args.clone()).await;
    let run_id = sweep["run_id"].as_str().unwrap().to_owned();

    for (tool, a) in [
        ("get_cluster_profile", profile_args()),
        (
            "get_feature_signal",
            args(&[("feature_names", json!(["x0"]))]),
        ),
        ("get_cluster_signal_matrix", args(&[])),
        (
            "compare_clusters",
            args(&[("cluster_a", json!(0)), ("cluster_b", json!(1))]),
        ),
    ] {
        let e = err(&client, tool, a).await;
        assert_eq!(e["error_code"], "CLUSTERS_NOT_BUILT", "{tool}");
    }

    let dossier = ok(
        &client,
        &tools,
        "generate_cluster_dossier",
        args(&[
            ("method", json!("components")),
            ("interpretation_edge_weight_threshold", json!(0.5)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let ids: Vec<i64> = dossier["clusters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["cluster_id"].as_i64().unwrap())
        .collect();
    assert!(ids.len() >= 2, "blobs separate at 0.5: {ids:?}");
    let (a, b) = (ids[0], ids[1]);

    let profile = call(
        &client,
        "get_cluster_profile",
        args(&[("cluster_id", json!(a))]),
    )
    .await;
    assert!(text_of(&profile).contains("interpretation slice"));
    let profile = ok(
        &client,
        &tools,
        "get_cluster_profile",
        args(&[
            ("cluster_id", json!(a)),
            ("detail", json!("full")),
            ("max_features", json!(2)),
            ("run_id", json!(run_id)),
        ]),
    )
    .await;
    assert_eq!(profile["clusters_run_id"], json!(run_id));
    assert_eq!(profile["cluster"]["cluster_id"], a);
    assert_eq!(profile["detail"], "full");
    assert!(profile["surface_labels"].is_object());
    assert!(profile["cluster_result"]["cluster_sizes"].is_array());
    let unknown = err(
        &client,
        "get_cluster_profile",
        args(&[("cluster_id", json!(9999))]),
    )
    .await;
    assert_eq!(unknown["error_code"], "CLUSTER_ID_UNKNOWN");
    let bad = err(
        &client,
        "get_cluster_profile",
        args(&[("cluster_id", json!(a)), ("max_features", json!(0))]),
    )
    .await;
    assert_eq!(bad["error_code"], "INVALID_ARGUMENT");
    let missing = err(&client, "get_cluster_profile", args(&[])).await;
    assert_eq!(missing["error_code"], "INVALID_ARGUMENT");

    let signal = ok(
        &client,
        &tools,
        "get_feature_signal",
        args(&[
            ("feature_names", json!(["x0", "x1"])),
            ("cluster_ids", json!([a])),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(signal["cluster_selection"]["ranked_by"], "signal");
    let rows = &signal["signals"]["clusters"];
    assert_eq!(rows.as_array().map(Vec::len), Some(1), "{signal}");
    assert_eq!(rows[0]["cluster_id"], a);
    assert_eq!(signal["surface_labels_ref"], "get_workflow_guide");

    let matrix = call(&client, "get_cluster_signal_matrix", args(&[])).await;
    let text = text_of(&matrix);
    assert!(text.contains("## Cluster Selection"), "{text}");
    assert!(text.contains("## Topological Signal Matrix"), "{text}");
    let matrix = ok(
        &client,
        &tools,
        "get_cluster_signal_matrix",
        args(&[
            ("max_clusters", json!(1)),
            ("return_markdown", json!(false)),
        ]),
    )
    .await;
    assert_eq!(matrix["signal_matrix"]["clusters_returned"], 1);
    assert_eq!(
        matrix["signal_matrix"]["clusters_omitted"],
        json!(ids.len() - 1)
    );

    let compared = ok(
        &client,
        &tools,
        "compare_clusters",
        args(&[
            ("cluster_a", json!(a)),
            ("cluster_b", json!(b)),
            ("max_features", json!(1)),
        ]),
    )
    .await;
    assert_eq!(compared["results"].as_array().unwrap().len(), 1);
    assert_eq!(
        compared["features_omitted"].as_u64().unwrap() + 1,
        compared["features_compared"].as_u64().unwrap()
    );
    let empty = err(
        &client,
        "compare_clusters",
        args(&[("cluster_a", json!(a)), ("cluster_b", json!(9999))]),
    )
    .await;
    assert_eq!(empty["error_code"], "COMPARISON_EMPTY");

    let stale = err(
        &client,
        "get_feature_signal",
        args(&[
            ("feature_names", json!(["x0"])),
            (
                "run_id",
                json!(sweep["run_id"]
                    .as_str()
                    .unwrap()
                    .replace(|c: char| c.is_ascii_hexdigit(), "0")),
            ),
        ]),
    )
    .await;
    assert_eq!(stale["error_code"], "CLUSTERS_STALE");
    assert_eq!(stale["details"]["clusters_run_id"], json!(run_id));

    // The persisted assignment exports as a labeled CSV.
    let ca = dossier["cluster_assignment_id"].clone();
    let all_names: serde_json::Map<String, Value> = ids
        .iter()
        .map(|i| (i.to_string(), json!(format!("Cohort {i}"))))
        .collect();
    let out = root.path().join("out/labeled.csv");
    let exported = ok(
        &client,
        &tools,
        "export_labeled_data",
        args(&[
            ("cluster_assignment_id", ca.clone()),
            ("cluster_names", Value::Object(all_names.clone())),
            ("output_path", json!(out.display().to_string())),
            ("run_id", json!(run_id)),
        ]),
    )
    .await;
    assert_eq!(exported["row_count"], 150);
    let csv = std::fs::read_to_string(&out).unwrap();
    assert!(csv
        .lines()
        .next()
        .unwrap()
        .ends_with(",topological_cluster_id,topological_cluster_name"));
    assert_eq!(csv.lines().count(), 151);
    let partial = err(
        &client,
        "export_labeled_data",
        args(&[
            ("cluster_assignment_id", ca.clone()),
            ("cluster_names", json!({"0": "Only one"})),
            ("output_path", json!(out.display().to_string())),
        ]),
    )
    .await;
    assert_eq!(partial["error_code"], "CLUSTER_NAMES_INVALID");
    let unknown = err(
        &client,
        "export_labeled_data",
        args(&[
            ("cluster_assignment_id", json!("ca_000000000000")),
            ("cluster_names", Value::Object(all_names.clone())),
            ("output_path", json!(out.display().to_string())),
        ]),
    )
    .await;
    assert_eq!(unknown["error_code"], "CLUSTER_ASSIGNMENT_ID_UNKNOWN");

    // The session's clusters export as a Parquet bundle.
    let bundle_dir = root.path().join("bundles");
    let bundle = ok(
        &client,
        &tools,
        "export_dataset_bundle",
        args(&[
            ("output_dir", json!(bundle_dir.display().to_string())),
            ("slug", json!("blobs")),
            ("cluster_names", json!({ids[0].to_string(): "First"})),
            ("include_clean", json!(true)),
        ]),
    )
    .await;
    assert_eq!(bundle["manifest"]["counts"]["nodes"], 150);
    assert_eq!(bundle["manifest"]["counts"]["groups"], json!(ids.len()));
    for rel in [
        "tabular/raw.parquet",
        "tabular/clean.parquet",
        "graph/nodes.parquet",
        "graph/edges.parquet",
        "graph/cosmic.parquet",
        "graph/groups.parquet",
        "export_manifest.json",
    ] {
        assert!(bundle_dir.join("blobs").join(rel).exists(), "{rel}");
    }
    // A re-export without the clean table removes it and keeps foreign files.
    std::fs::write(bundle_dir.join("blobs/notes.txt"), "mine").unwrap();
    ok(
        &client,
        &tools,
        "export_dataset_bundle",
        args(&[
            ("output_dir", json!(bundle_dir.display().to_string())),
            ("slug", json!("blobs")),
            ("cluster_assignment_id", ca.clone()),
            ("layout", json!("spectral")),
            ("write_manifest", json!(false)),
        ]),
    )
    .await;
    assert!(!bundle_dir.join("blobs/tabular/clean.parquet").exists());
    assert!(!bundle_dir.join("blobs/export_manifest.json").exists());
    assert!(bundle_dir.join("blobs/notes.txt").exists());
    let bad = err(
        &client,
        "export_dataset_bundle",
        args(&[
            ("output_dir", json!(bundle_dir.display().to_string())),
            ("slug", json!("x")),
            ("edges_threshold", json!(1.5)),
        ]),
    )
    .await;
    assert_eq!(bad["error_code"], "INVALID_EDGES_THRESHOLD");
    let bad = err(
        &client,
        "export_dataset_bundle",
        args(&[
            ("output_dir", json!(bundle_dir.display().to_string())),
            ("slug", json!("x")),
            ("cluster_descriptions", json!({"99": "nope"})),
        ]),
    )
    .await;
    assert_eq!(bad["error_code"], "CLUSTER_NAMES_INVALID");

    // A new sweep clears the labels.
    let second = ok(&client, &tools, "run_topological_sweep", sweep_args).await;
    let missing = err(
        &client,
        "export_dataset_bundle",
        args(&[
            ("output_dir", json!(bundle_dir.display().to_string())),
            ("slug", json!("x")),
        ]),
    )
    .await;
    assert_eq!(missing["error_code"], "CLUSTERS_MISSING");
    let stale = err(
        &client,
        "export_dataset_bundle",
        args(&[
            ("output_dir", json!(bundle_dir.display().to_string())),
            ("slug", json!("x")),
            ("cluster_assignment_id", ca.clone()),
        ]),
    )
    .await;
    assert_eq!(
        stale["error_code"],
        "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH"
    );
    let mismatch = err(
        &client,
        "export_labeled_data",
        args(&[
            ("cluster_assignment_id", ca),
            ("cluster_names", Value::Object(all_names)),
            ("output_path", json!(out.display().to_string())),
            ("run_id", second["run_id"].clone()),
        ]),
    )
    .await;
    assert_eq!(
        mismatch["error_code"],
        "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH"
    );
    let cleared = err(&client, "get_cluster_signal_matrix", args(&[])).await;
    assert_eq!(cleared["error_code"], "CLUSTERS_NOT_BUILT");
    client.cancel().await.unwrap();
}

/// A long-format panel: `n` entities x 3 visits, three drifting features, and
/// entity e7 missing visit 2 (ragged).
fn write_panel(dir: &std::path::Path, n: usize) -> PathBuf {
    let mut csv = String::from("patient,visit,a,b,c\n");
    for e in 0..n {
        for t in 1..=3 {
            if e == 7 && t == 2 {
                continue;
            }
            let g = (e % 3) as f64;
            let x = |k: f64| g * 2.0 + (t as f64) * 0.3 * k + ((e * 7 + t * 3) % 5) as f64 * 0.05;
            csv.push_str(&format!(
                "p{e:02},{t},{:.4},{:.4},{:.4}\n",
                x(1.0),
                x(-1.0),
                x(0.5)
            ));
        }
    }
    let path = dir.join("panel.csv");
    std::fs::write(&path, csv).unwrap();
    path
}

const PANEL_YAML: &str = "run:\n  name: panel\nsweep:\n  projection:\n    method: jl\n    dimensions:\n      values: [2, 3]\n    seed:\n      values: [42]\n  ball_mapper:\n    epsilon:\n      values: [0.5, 1.0]\ncosmic_graph:\n  construction: exact\n";

#[tokio::test]
async fn the_longitudinal_tools_build_measure_and_explain_failures() {
    let root = TempRoot::new("longitudinal");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    let csv = write_panel(root.path(), 30);

    let none = err(&client, "diagnose_longitudinal_graph", args(&[])).await;
    assert_eq!(none["error_code"], "LONGITUDINAL_ID_UNKNOWN");

    let ingested = ok(
        &client,
        &tools,
        "ingest_dataset",
        args(&[("path", json!(csv.display().to_string()))]),
    )
    .await;
    let ds = ingested["dataset_id"].clone();

    // Sampling: validation, then a deterministic Parquet sample.
    let bad = err(
        &client,
        "sample_longitudinal_entities",
        args(&[
            ("dataset_id", ds.clone()),
            ("entity_column", json!("patient")),
            ("min_time_points", json!(2)),
        ]),
    )
    .await;
    assert_eq!(bad["error_code"], "INVALID_SAMPLE_SIZE");
    let unknown_col = err(
        &client,
        "sample_longitudinal_entities",
        args(&[("dataset_id", ds.clone()), ("entity_column", json!("who"))]),
    )
    .await;
    assert_eq!(unknown_col["error_code"], "ENTITY_COLUMN_UNKNOWN");
    let sample_args = args(&[
        ("dataset_id", ds.clone()),
        ("entity_column", json!("patient")),
        ("max_entities", json!(12)),
        ("time_column", json!("visit")),
        ("min_time_points", json!(3)),
    ]);
    let sampled = ok(
        &client,
        &tools,
        "sample_longitudinal_entities",
        sample_args.clone(),
    )
    .await;
    assert_eq!(sampled["eligible_entities"], 29, "p07 misses a visit");
    assert_eq!(sampled["sampled_entities"], 12);
    assert_eq!(sampled["sampled_rows"], 36);
    let again = ok(&client, &tools, "sample_longitudinal_entities", sample_args).await;
    assert_eq!(
        again["dataset_id"], sampled["dataset_id"],
        "the sample is deterministic"
    );

    // Build preconditions.
    let build = |extra: &[(&str, Value)]| {
        let mut a = vec![
            ("dataset_id", ds.clone()),
            ("entity_column", json!("patient")),
            ("time_column", json!("visit")),
            ("config_yaml", json!(PANEL_YAML)),
        ];
        a.extend(extra.iter().cloned());
        args(&a)
    };
    let e = err(
        &client,
        "build_longitudinal_graph",
        args(&[("dataset_id", ds.clone())]),
    )
    .await;
    assert_eq!(e["error_code"], "PANEL_KEYS_MISSING");
    let e = err(
        &client,
        "build_longitudinal_graph",
        args(&[
            ("dataset_id", ds.clone()),
            ("entity_column", json!("patient")),
            ("time_column", json!("visit")),
        ]),
    )
    .await;
    assert_eq!(e["error_code"], "CONFIG_REQUIRED");
    let e = err(
        &client,
        "build_longitudinal_graph",
        build(&[("dataset_id", json!(""))]),
    )
    .await;
    assert_eq!(e["error_code"], "DATASET_ID_REQUIRED");
    let e = err(
        &client,
        "build_longitudinal_graph",
        build(&[("similarity_threshold", json!(-0.1))]),
    )
    .await;
    assert_eq!(e["error_code"], "INVALID_ARGUMENT");
    let e = err(
        &client,
        "build_longitudinal_graph",
        build(&[
            ("representation", json!("temporal")),
            ("on_missing", json!("allow_ragged")),
        ]),
    )
    .await;
    assert_eq!(e["error_code"], "RAGGED_PANEL_NOT_SUPPORTED");

    let built = ok(
        &client,
        &tools,
        "build_longitudinal_graph",
        build(&[("representation", json!("both")), ("detail", json!("full"))]),
    )
    .await;
    let lid = built["longitudinal_id"].as_str().unwrap().to_owned();
    assert!(lid.starts_with("lg_"), "{lid}");
    assert_eq!(built["panel"]["n_entities_kept"], 29);
    assert_eq!(
        built["panel"]["dropped_entities"]["preview"],
        json!(["p07"])
    );
    assert_eq!(built["config_adaptation"]["panel_feature_count"], 3);
    assert_eq!(
        built["graph_surface"]["temporal"]["tensor_shape"],
        json!([29, 29, 3])
    );
    assert!(
        built["graph_surface"]["trajectory"]["n_edges"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(built["times"], json!([1, 2, 3]));
    let md = call(
        &client,
        "build_longitudinal_graph",
        build(&[("response_format", json!("markdown"))]),
    )
    .await;
    assert!(
        text_of(&md).contains("# Longitudinal Panel `lg_"),
        "{}",
        text_of(&md)
    );

    // Measurements on the "both" panel.
    let diag = ok(
        &client,
        &tools,
        "diagnose_longitudinal_graph",
        args(&[("longitudinal_id", json!(lid)), ("detail", json!("full"))]),
    )
    .await;
    assert_eq!(
        diag["surfaces"]["temporal"]["aggregations"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
    assert!(diag["surfaces"]["trajectory"]["component_morphology"].is_object());
    let diag_md = call(
        &client,
        "diagnose_longitudinal_graph",
        args(&[
            ("longitudinal_id", json!(lid)),
            ("response_format", json!("markdown")),
        ]),
    )
    .await;
    assert!(text_of(&diag_md).contains("### Threshold ladder"));
    assert!(text_of(&diag_md).contains("| persistence | [0,1] |"));

    let arch = ok(
        &client,
        &tools,
        "get_trajectory_archetypes",
        args(&[
            ("longitudinal_id", json!(lid)),
            ("response_format", json!("json")),
            ("max_archetypes", json!(2)),
        ]),
    )
    .await;
    assert_eq!(arch["n_entities"], 29);
    assert!(arch["archetypes_returned"].as_u64().unwrap() <= 2);
    let e = err(
        &client,
        "get_trajectory_archetypes",
        args(&[("threshold", json!(-1.0))]),
    )
    .await;
    assert_eq!(e["error_code"], "INVALID_ARGUMENT");

    let by_label = ok(
        &client,
        &tools,
        "get_cross_time_neighbors",
        args(&[
            ("longitudinal_id", json!(lid)),
            ("entity_id", json!("p01")),
            ("t", json!(2)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(by_label["source"]["entity_id"], "p01");
    assert_eq!(by_label["source"]["time_label"], 2);
    assert_eq!(by_label["direction"], "any");
    let by_text_label = ok(
        &client,
        &tools,
        "get_cross_time_neighbors",
        args(&[
            ("entity_id", json!("p01")),
            ("t", json!("2")),
            ("direction", json!("forward")),
        ]),
    )
    .await;
    assert_eq!(
        by_text_label["source"], by_label["source"],
        "str(t) also matches"
    );
    let obs = by_label["source"]["obs_id"].clone();
    let by_id = ok(
        &client,
        &tools,
        "get_cross_time_neighbors",
        args(&[("observation_id", obs)]),
    )
    .await;
    assert_eq!(by_id["source"], by_label["source"]);
    let e = err(
        &client,
        "get_cross_time_neighbors",
        args(&[("entity_id", json!("p01"))]),
    )
    .await;
    assert_eq!(e["error_code"], "OBSERVATION_NOT_SPECIFIED");
    let e = err(
        &client,
        "get_cross_time_neighbors",
        args(&[("entity_id", json!("p01")), ("t", json!(9))]),
    )
    .await;
    assert_eq!(e["error_code"], "OBSERVATION_NOT_FOUND");
    let e = err(
        &client,
        "get_cross_time_neighbors",
        args(&[("observation_id", json!(10_000))]),
    )
    .await;
    assert_eq!(e["error_code"], "OBSERVATION_NOT_FOUND");

    let classes = ok(
        &client,
        &tools,
        "classify_trajectories",
        args(&[
            ("method", json!("levenshtein")),
            ("detail", json!("full")),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(classes["entities_returned"], 29);
    assert_eq!(classes["classification"].as_object().unwrap().len(), 29);
    let summary = call(&client, "classify_trajectories", args(&[])).await;
    let text = text_of(&summary);
    assert!(text.contains("## Cohort Summary"), "{text}");
    let summary = summary.structured_content.unwrap();
    assert_eq!(summary["entities_omitted"], 4);
    assert!(summary.get("classification").is_none());

    // A temporal-only panel has no trajectory surface.
    let temporal = ok(
        &client,
        &tools,
        "build_longitudinal_graph",
        build(&[("representation", json!("temporal"))]),
    )
    .await;
    let e = err(
        &client,
        "get_trajectory_archetypes",
        args(&[("longitudinal_id", temporal["longitudinal_id"].clone())]),
    )
    .await;
    assert_eq!(e["error_code"], "REPRESENTATION_NOT_BUILT");
    let e = err(
        &client,
        "diagnose_longitudinal_graph",
        args(&[
            ("longitudinal_id", temporal["longitudinal_id"].clone()),
            ("representation", json!("trajectory")),
        ]),
    )
    .await;
    assert_eq!(e["error_code"], "REPRESENTATION_NOT_BUILT");

    // The session keeps the four newest panels; older handles name what to rebuild.
    for _ in 0..3 {
        ok(&client, &tools, "build_longitudinal_graph", build(&[])).await;
    }
    let evicted = err(
        &client,
        "diagnose_longitudinal_graph",
        args(&[("longitudinal_id", json!(lid))]),
    )
    .await;
    assert_eq!(evicted["error_code"], "LONGITUDINAL_NOT_LOADED");
    assert_eq!(evicted["details"]["representation"], "both");
    assert_eq!(evicted["details"]["panel"]["entity_column"], "patient");
    let e = err(
        &client,
        "diagnose_longitudinal_graph",
        args(&[("longitudinal_id", json!("lg_000000000000"))]),
    )
    .await;
    assert_eq!(e["error_code"], "LONGITUDINAL_ID_UNKNOWN");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn graph_artifacts_are_estimated_built_and_measured() {
    let root = TempRoot::new("artifact");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    const TOOL: &str = "create_graph_artifact";

    let none = err(&client, TOOL, args(&[])).await;
    assert_eq!(none["error_code"], "MODEL_NOT_LOADED");
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
    let sweep_args = args(&[
        ("config_yaml", json!(blobs_yaml())),
        ("dataset_id", ingested["dataset_id"].clone()),
        ("response_format", json!("json")),
    ]);
    let sweep = ok(&client, &tools, "run_topological_sweep", sweep_args.clone()).await;
    let run_id = sweep["run_id"].clone();

    let estimate = ok(&client, &tools, TOOL, args(&[])).await;
    assert_eq!(estimate["estimate_only"], true);
    assert_eq!(estimate["input"]["n_nodes"], 150);
    assert!(estimate["estimated_cost"]["sketch_dim"].as_u64().unwrap() > 0);
    assert_eq!(estimate["build_call"]["args"]["estimate_only"], false);
    let e = err(
        &client,
        TOOL,
        args(&[("kind", json!("structural_backbone"))]),
    )
    .await;
    assert_eq!(e["error_code"], "GRAPH_ARTIFACT_KIND_UNSUPPORTED");
    let e = err(&client, TOOL, args(&[("epsilon", json!(0.0))])).await;
    assert_eq!(e["error_code"], "INVALID_ARGUMENT");

    let built = ok(
        &client,
        &tools,
        TOOL,
        args(&[("estimate_only", json!(false)), ("sketch_dim", json!(16))]),
    )
    .await;
    let artifact_id = built["artifact_id"].as_str().unwrap().to_owned();
    assert!(artifact_id.starts_with("graph_"), "{artifact_id}");
    assert_eq!(built["staleness"]["stale"], false);
    assert!(built["artifact_edge_count"].as_u64().unwrap() > 0);

    let diag = ok(
        &client,
        &tools,
        "diagnose_cosmic_graph",
        args(&[
            ("surface", json!("artifact")),
            ("artifact_id", json!(artifact_id)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(diag["source"], "artifact");
    assert_eq!(
        diag["graph_surface"]["threshold_role"],
        "artifact_construction"
    );
    assert_eq!(diag["scale"]["n_edges"], built["artifact_edge_count"]);
    let skeleton = ok(
        &client,
        &tools,
        "get_topological_skeleton",
        args(&[
            ("surface", json!("artifact")),
            ("artifact_id", json!(artifact_id)),
            ("detail", json!("edges")),
            ("max_edges", json!(5)),
        ]),
    )
    .await;
    assert_eq!(skeleton["artifact_kind"], "spectral_sparsifier");
    let e = err(
        &client,
        "get_topological_skeleton",
        args(&[("surface", json!("artifact"))]),
    )
    .await;
    assert_eq!(e["error_code"], "ARTIFACT_ID_REQUIRED");
    let e = err(
        &client,
        "diagnose_cosmic_graph",
        args(&[
            ("surface", json!("artifact")),
            ("artifact_id", json!("graph_000000000000")),
        ]),
    )
    .await;
    assert_eq!(e["error_code"], "ARTIFACT_ID_UNKNOWN");

    // A new sweep makes the artifact stale but keeps it readable; building
    // for the superseded run is refused, estimating it is not.
    ok(&client, &tools, "run_topological_sweep", sweep_args).await;
    let stale = ok(
        &client,
        &tools,
        "diagnose_cosmic_graph",
        args(&[
            ("surface", json!("artifact")),
            ("artifact_id", json!(artifact_id)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    assert_eq!(stale["artifact_staleness"]["stale"], true);
    ok(&client, &tools, TOOL, args(&[("run_id", run_id.clone())])).await;
    let e = err(
        &client,
        TOOL,
        args(&[("run_id", run_id), ("estimate_only", json!(false))]),
    )
    .await;
    assert_eq!(e["error_code"], "SESSION_REVISION_CONFLICT");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn the_html_report_is_written_beside_the_dataset() {
    let root = TempRoot::new("report");
    let client = connect(root.path(), test_options(), ()).await;
    let tools = client.list_all_tools().await.unwrap();
    const TOOL: &str = "export_html_report";
    let csv = root.path().join("blobs.csv");
    std::fs::copy(golden("blobs", "input.csv"), &csv).unwrap();
    let ingested = ok(
        &client,
        &tools,
        "ingest_dataset",
        args(&[("path", json!(csv.display().to_string()))]),
    )
    .await;
    let ds = ingested["dataset_id"].clone();
    ok(
        &client,
        &tools,
        "run_topological_sweep",
        args(&[
            ("config_yaml", json!(blobs_yaml())),
            ("dataset_id", ds.clone()),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let missing = err(&client, TOOL, args(&[("dataset_id", ds.clone())])).await;
    assert_eq!(missing["error_code"], "CLUSTERS_MISSING");
    let dossier = ok(
        &client,
        &tools,
        "generate_cluster_dossier",
        args(&[
            ("method", json!("components")),
            ("interpretation_edge_weight_threshold", json!(0.5)),
            ("response_format", json!("json")),
        ]),
    )
    .await;
    let first = dossier["clusters"][0]["cluster_id"].as_i64().unwrap();

    let unnamed = ok(&client, &tools, TOOL, args(&[("dataset_id", ds.clone())])).await;
    assert!(unnamed["warning"]
        .as_str()
        .unwrap()
        .contains("cluster_names was not provided"));
    let named = ok(
        &client,
        &tools,
        TOOL,
        args(&[
            ("dataset_id", ds.clone()),
            (
                "cluster_names",
                json!({first.to_string(): "Alpha & <Omega>"}),
            ),
        ]),
    )
    .await;
    assert!(named.get("warning").is_none());
    let path = std::path::PathBuf::from(named["report_path"].as_str().unwrap());
    assert_eq!(
        path,
        root.path()
            .join("blobs_report.html")
            .canonicalize()
            .unwrap_or(root.path().join("blobs_report.html"))
    );
    assert!(named["report_url"].as_str().unwrap().starts_with("file://"));
    let html = std::fs::read_to_string(&path).unwrap();
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("Alpha &amp; &lt;Omega&gt;"));
    assert!(html.contains("<svg id='cosmicGraph'"));
    assert!(html.contains("id='paramsYaml'"));

    let bad = err(
        &client,
        TOOL,
        args(&[("cluster_names", json!({"999": "nope"}))]),
    )
    .await;
    assert_eq!(bad["error_code"], "CLUSTER_NAMES_INVALID");
    let unknown = err(
        &client,
        TOOL,
        args(&[("dataset_id", json!("ds_000000000000"))]),
    )
    .await;
    assert_eq!(unknown["error_code"], "DATASET_ID_UNKNOWN");
    client.cancel().await.unwrap();
}
