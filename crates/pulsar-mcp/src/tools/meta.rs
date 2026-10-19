//! `get_workflow_guide` and `get_runtime_context`.

use rmcp::model::CallToolResult;
use serde_json::json;

use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::guide::WORKFLOW_PROMPT;
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const GUIDE: &str = "get_workflow_guide";
pub const RUNTIME: &str = "get_runtime_context";

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct NoParams {}

pub fn guide_spec() -> ToolSpec {
    ToolSpec {
        name: GUIDE,
        title: "Workflow guide",
        description: "Return the recommended end-to-end Pulsar workflow as markdown. Call once at the start of a session for the procedure (ingest, calibrate, run, diagnose, interpret). Structured content is {\"guide\": <markdown>}; the text view is the same markdown.",
        input_schema: input_schema::<NoParams>(),
        output_schema: s::as_object(s::object(&[("guide", s::string())], &["guide"], false)),
        read_only: true,
        idempotent: true,
    }
}

pub async fn guide() -> Result<CallToolResult, ToolError> {
    Ok(ok_result(
        json!({"guide": WORKFLOW_PROMPT}),
        Some(WORKFLOW_PROMPT.to_owned()),
    ))
}

const MODEL_CONTINUITY: &str = "Runs are persisted in the store the moment they complete, so run_id handles work on every transport and across restarts. The in-memory model cache only makes reads of recent runs faster.";

pub fn runtime_spec() -> ToolSpec {
    ToolSpec {
        name: RUNTIME,
        title: "Runtime context",
        description: "Report the server's runtime context: paths, session handles, transport, model cache. Returns cwd, cache_dir, session_id, latest handles, active config, transport, cached runs (model_cache) and path guidance. Call before file operations or when a handle seems stale.",
        input_schema: input_schema::<NoParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("cwd", s::string()),
                ("cache_dir", s::string()),
                ("temp_dir", s::string()),
                ("transport", s::string()),
                ("transport_assumption", s::string()),
                ("model_continuity", s::string()),
                ("session_id", s::string()),
                ("dataset_handle_persistence", s::string()),
                ("run_handle_persistence", s::string()),
                ("model_state_persistence", s::string()),
                ("session_model_loaded", s::boolean()),
                ("session_data_loaded", s::boolean()),
                ("latest_dataset_id", s::nullable(s::string())),
                ("latest_run_id", s::nullable(s::string())),
                ("active_config_yaml", s::nullable(s::string())),
                ("active_config_dataset_id", s::nullable(s::string())),
                ("workflow_guide_available_via", s::string()),
                ("path_guidance", s::array(s::string())),
                (
                    "model_cache",
                    s::object(
                        &[
                            ("size", s::integer()),
                            ("capacity", s::integer()),
                            ("run_ids", s::array(s::string())),
                            ("memory_bytes", s::integer()),
                            (
                                "entries",
                                s::array(s::object(
                                    &[("run_id", s::string()), ("memory_bytes", s::integer())],
                                    &["run_id", "memory_bytes"],
                                    false,
                                )),
                            ),
                            ("model_bytes_estimated", s::boolean()),
                        ],
                        &["size", "capacity", "run_ids"],
                        false,
                    ),
                ),
            ],
            &["cwd", "cache_dir", "transport", "session_id", "latest_dataset_id", "latest_run_id", "model_cache"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn runtime(state: &ServerState) -> Result<CallToolResult, ToolError> {
    let session = state.session_snapshot();
    let model_loaded = session
        .latest_run_id
        .as_ref()
        .is_some_and(|id| state.cache.contains(id));
    let payload = json!({
        "cwd": std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default(),
        "cache_dir": state.store.root().display().to_string(),
        "temp_dir": std::env::temp_dir().display().to_string(),
        "transport": state.options.transport,
        "transport_assumption": state.options.transport,
        "model_continuity": MODEL_CONTINUITY,
        "session_id": "default",
        "dataset_handle_persistence": "on-disk store under cache_dir/datasets",
        "run_handle_persistence": "on-disk store under cache_dir/runs (record, graph and affinity sidecars)",
        "model_state_persistence": "in-memory cache of recent runs; persisted runs reload from disk",
        "session_model_loaded": model_loaded,
        "session_data_loaded": model_loaded,
        "latest_dataset_id": session.latest_dataset_id,
        "latest_run_id": session.latest_run_id,
        "active_config_yaml": session.active_config_yaml,
        "active_config_dataset_id": session.active_config_dataset_id,
        "workflow_guide_available_via": "get_workflow_guide",
        "path_guidance": [
            "Use ingest_dataset(path) for host-visible absolute paths.",
            "SANDBOX ISOLATION: If your file is in a sandbox (e.g. /home/claude), do not use bash/cp as a bridge by default: ask the user for a host-visible absolute path. Only if the user confirms the server host can see `cache_dir` (shown above), copy the file there and call `ingest_dataset(path)` on the destination.",
            "config_path must be server-visible; config_yaml should be raw YAML (no Markdown).",
        ],
        "model_cache": state.cache_stats(),
    });
    Ok(ok_result(payload, None))
}
