//! Shared test harness for MCP integration tests (`connect`, `serve`, `ok`,
//! and Phirewall `io.phirewall/release` v1 contract checks).

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pulsar_mcp::{PulsarServer, ServerOptions, ServerState};
use pulsar_store::Store;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientConfig, Implementation,
    JsonObject, Tool,
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::{ClientHandler, ServiceExt};
use serde_json::{json, Value};

pub const RELEASE: &str = "io.phirewall/release";

pub fn phirewall_client_config() -> ClientConfig {
    let mut exp = BTreeMap::new();
    let mut v1 = JsonObject::new();
    v1.insert("version".into(), json!(1));
    exp.insert(RELEASE.into(), v1);
    let mut caps = ClientCapabilities::default();
    caps.experimental = Some(exp);
    ClientConfig::new(caps, Implementation::new("phirewall-test", "0"))
}

pub struct TempRoot(pub PathBuf);

impl TempRoot {
    pub fn new(tag: &str) -> Self {
        let nonce: u64 = {
            use std::time::{SystemTime, UNIX_EPOCH};
            let t = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            t ^ (std::process::id() as u64).rotate_left(32)
        };
        let path =
            std::env::temp_dir().join(format!("pulsar-mcp-{tag}-{}-{nonce:x}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn test_options() -> ServerOptions {
    ServerOptions {
        max_response_bytes: Some(2_000_000),
        transport: "in-memory".into(),
    }
}

/// Serve `state` over an in-memory duplex and connect `handler`.
pub async fn serve_state<H: ClientHandler>(
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

/// Serve a fresh server at `root` with `options` and connect `handler`.
pub async fn connect_with<H: ClientHandler>(
    root: &Path,
    options: ServerOptions,
    handler: H,
) -> RunningService<RoleClient, H> {
    let store = Store::open_at(root).unwrap();
    serve_state(Arc::new(ServerState::new(store, options).unwrap()), handler).await
}

/// Serve a fresh server at `root` and return both `state` and default client.
pub async fn serve(root: &Path) -> (Arc<ServerState>, RunningService<RoleClient, ()>) {
    let store = Store::open_at(root).unwrap();
    let state = Arc::new(ServerState::new(store, test_options()).unwrap());
    let client = serve_state(state.clone(), ()).await;
    (state, client)
}

/// Serve a fresh server at `root` and return both `state` and a Phirewall-capable client.
pub async fn serve_phirewall(
    root: &Path,
) -> (Arc<ServerState>, RunningService<RoleClient, ClientConfig>) {
    let store = Store::open_at(root).unwrap();
    let state = Arc::new(ServerState::new(store, test_options()).unwrap());
    let client = serve_state(state.clone(), phirewall_client_config()).await;
    (state, client)
}

/// Serve a fresh server at `root` and return a connected default client.
pub async fn connect(root: &Path) -> RunningService<RoleClient, ()> {
    connect_with(root, test_options(), ()).await
}

pub fn args(pairs: &[(&str, Value)]) -> JsonObject {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

pub async fn call<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    name: &str,
    a: JsonObject,
) -> CallToolResult {
    let mut params = CallToolRequestParams::new(name.to_owned());
    params.arguments = Some(a);
    client.call_tool(params).await.expect("call completes")
}

pub async fn call_json<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    name: &str,
    a: Value,
) -> CallToolResult {
    call(client, name, serde_json::from_value(a).unwrap()).await
}

pub fn validate_schema(schema: &Value, instance: &Value, context: &str) {
    let validator =
        jsonschema::validator_for(schema).unwrap_or_else(|e| panic!("{context} schema: {e}"));
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|e| format!("{e} at {}", e.instance_path()))
        .collect();
    assert!(errors.is_empty(), "{context}: {errors:#?}\n{instance}");
}

pub fn release_validator() -> &'static jsonschema::Validator {
    static VALIDATOR: std::sync::OnceLock<jsonschema::Validator> = std::sync::OnceLock::new();
    VALIDATOR.get_or_init(|| {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/phirewall-release-v1.json");
        let schema: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        jsonschema::validator_for(&schema).expect("the vendored release schema compiles")
    })
}

/// Validate `payload["io.phirewall/release"]` against `phirewall-release-v1.json`
/// and the semantic rules of `release-v1.md`.
pub fn release_of<'a>(payload: &'a Value, name: &str) -> &'a Value {
    let rel = payload
        .get(RELEASE)
        .unwrap_or_else(|| panic!("{name}: no {RELEASE}"));
    let errors: Vec<String> = release_validator()
        .iter_errors(rel)
        .map(|e| format!("{} at {}", e, e.instance_path()))
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:#?}\n{rel}");

    let mut group_ids = BTreeSet::new();
    for g in rel["groups"].as_array().unwrap() {
        let id = g["id"].as_str().unwrap();
        assert!(
            group_ids.insert(id.to_owned()),
            "{name}: duplicate group {id}"
        );
        if g["rows"]["enc"] == "ranges" {
            let mut prev_end = 0u64;
            for (i, r) in g["rows"]["v"].as_array().unwrap().iter().enumerate() {
                let start = r[0].as_u64().unwrap();
                let end = r[1].as_u64().unwrap();
                assert!(
                    start < end,
                    "{name}: group {id} range {i} empty [{start}, {end})"
                );
                if i > 0 {
                    assert!(
                        prev_end <= start,
                        "{name}: group {id} range {i} overlaps [{prev_end}, {start})"
                    );
                }
                prev_end = end;
            }
        }
    }

    let mut seen_paths = BTreeSet::new();
    for l in rel["labels"].as_array().unwrap() {
        let path = l["path"].as_str().unwrap();
        assert!(
            !path.starts_with("/io.phirewall~1"),
            "{name}: label path inside phirewall block: {path}"
        );
        assert!(
            seen_paths.insert(path.to_owned()),
            "{name}: duplicate label path {path}"
        );
        let gid = l["group"].as_str().unwrap();
        assert!(
            group_ids.contains(gid),
            "{name}: unknown group {gid} at {path}"
        );
        let target = payload
            .pointer(path)
            .unwrap_or_else(|| panic!("{name}: pointer {path} does not resolve"));
        match l["kind"].as_str().unwrap() {
            "count" => assert!(target.is_number(), "{name}: {path} not a number: {target}"),
            "share" => {
                let of = l["of"].as_str().unwrap();
                assert!(
                    of == "all" || group_ids.contains(of),
                    "{name}: invalid share.of {of} at {path}"
                );
                assert!(target.is_number(), "{name}: {path} not a number: {target}");
            }
            "statistic" => {
                let vs = l["vs"].as_str().unwrap();
                assert!(
                    vs == "rest" || group_ids.contains(vs),
                    "{name}: invalid statistic.vs {vs} at {path}"
                );
                assert!(target.is_number(), "{name}: {path} not a number: {target}");
            }
            "label" => assert!(target.is_string(), "{name}: {path} not a string: {target}"),
            "records" => assert!(target.is_array(), "{name}: {path} not an array: {target}"),
            other => panic!("{name}: unexpected kind {other}"),
        }
    }
    rel
}

pub async fn ok_with_tools<H: ClientHandler>(
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
    validate_schema(&Value::Object((*schema).clone()), &payload, name);
    if payload.get(RELEASE).is_some() {
        release_of(&payload, name);
    }
    assert!(
        !result.content.is_empty(),
        "{name}: a text view is required"
    );
    payload
}

pub async fn ok_json<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    name: &str,
    a: Value,
) -> Value {
    let tools = client.list_all_tools().await.unwrap();
    ok_with_tools(client, &tools, name, serde_json::from_value(a).unwrap()).await
}

pub async fn ok_pairs<H: ClientHandler>(
    client: &RunningService<RoleClient, H>,
    name: &str,
    pairs: &[(&str, Value)],
) -> Value {
    let tools = client.list_all_tools().await.unwrap();
    ok_with_tools(client, &tools, name, args(pairs)).await
}
