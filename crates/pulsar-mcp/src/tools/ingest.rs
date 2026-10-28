//! `ingest_dataset`: register a host-visible file and hand out its handle.

use rmcp::model::CallToolResult;
use serde_json::json;

use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const NAME: &str = "ingest_dataset";

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct Params {
    /// Absolute path to a CSV or Parquet file on the machine running this server (host-visible; sandbox-only paths fail).
    pub path: String,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: NAME,
        title: "Ingest Dataset",
        description: "Register a host-visible CSV/Parquet file and return its `dataset_id`. Returns the dataset record (`dataset_id`, resolved path, size). Pass `dataset_id` to validate_config and run_topological_sweep. Start here; no other tool reads raw paths.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("dataset_id", s::string()),
                ("path", s::string()),
                ("name", s::string()),
                ("size_bytes", s::integer()),
                ("mtime_ns", s::integer()),
                ("ingested_at", s::number()),
                ("source", s::string()),
            ],
            &["dataset_id", "path", "name", "size_bytes", "mtime_ns", "ingested_at", "source"],
            false,
        )),
        read_only: false,
        idempotent: true,
    }
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    if p.path.trim().is_empty() {
        return Err(ToolError::invalid_argument(
            NAME,
            "path is required.",
            "Pass the absolute, host-visible path of a CSV or Parquet file.",
        ));
    }
    let store = state.store.clone();
    let path = p.path.clone();
    let record = tokio::task::spawn_blocking(move || store.register_dataset(&path))
        .await
        .map_err(|e| ToolError::worker_stopped(NAME, "INGEST_FAILED", &e))?;
    let record = match record {
        Ok(r) => r,
        Err(pulsar_store::StoreError::DatasetNotFound { .. }) => {
            return Err(ToolError::path_access(
                NAME,
                &p.path,
                "Ask the user for a host-visible absolute dataset path, then call ingest_dataset again.",
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads. Ask the user for a host-visible absolute path, or call get_runtime_context to find `cache_dir`; only if the user confirms the server host can see it, copy the file there and retry ingest_dataset(path) with the new path.",
            ));
        }
        Err(e) => return Err(ToolError::from_store(NAME, e)),
    };
    state.session().latest_dataset_id = Some(record.dataset_id.clone());
    Ok(ok_result(
        json!({
            "dataset_id": record.dataset_id,
            "path": record.path,
            "name": record.name,
            "size_bytes": record.size_bytes,
            "mtime_ns": record.mtime_ns,
            "ingested_at": record.ingested_at,
            "source": record.source,
        }),
        None,
    ))
}
