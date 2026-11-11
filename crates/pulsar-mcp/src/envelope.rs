//! Tool results: a text view plus structured content on success, and the
//! canonical error envelope (`error_code`, `tool`, `agent_action`, handle
//! context) with the MCP error flag set on failure (`pulsar/mcp/errors.py`).

use pulsar_analysis::AnalysisError;
use pulsar_store::{HandleKind, StoreError};
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{json, Map, Value};

const SANDBOX_PATH_PREFIXES: [&str; 5] = [
    "/home/claude",
    "/mnt/",
    "/workspace/",
    "/tmp/claude",
    "/System/Volumes/Data/home/claude",
];

/// A tool failure the agent can act on.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolError {
    pub tool: &'static str,
    pub error_code: String,
    pub reason: String,
    pub agent_action: String,
    pub details: Map<String, Value>,
    pub metrics: Map<String, Value>,
}

impl ToolError {
    pub fn new(
        tool: &'static str,
        error_code: &str,
        reason: impl Into<String>,
        agent_action: impl Into<String>,
    ) -> Self {
        Self {
            tool,
            error_code: error_code.to_owned(),
            reason: reason.into(),
            agent_action: agent_action.into(),
            details: Map::new(),
            metrics: Map::new(),
        }
    }

    pub fn invalid_argument(
        tool: &'static str,
        reason: impl Into<String>,
        agent_action: impl Into<String>,
    ) -> Self {
        Self::new(tool, "INVALID_ARGUMENT", reason, agent_action)
    }

    /// A blocking worker that panicked or was cancelled. No input reaches
    /// this path by design, so the envelope states a defect and keeps the
    /// join error as the evidence; `code` is the calling tool's own code.
    pub fn worker_stopped(tool: &'static str, code: &str, join: &tokio::task::JoinError) -> Self {
        Self::new(
            tool,
            code,
            format!("the worker stopped unexpectedly: {join}"),
            "An internal computation stopped unexpectedly: this is a defect and the message is the evidence. Report it with the message.",
        )
    }

    pub fn detail(mut self, key: &str, value: Value) -> Self {
        self.details.insert(key.to_owned(), value);
        self
    }

    /// The envelope as structured content.
    pub fn envelope(&self) -> Value {
        json!({
            "status": "error",
            "tool": self.tool,
            "reason": self.reason,
            "error_code": self.error_code,
            "agent_action": self.agent_action,
            "details": self.details,
            "metrics": self.metrics,
        })
    }

    pub fn into_result(self) -> CallToolResult {
        let envelope = self.envelope();
        let text = serde_json::to_string_pretty(&envelope).unwrap_or_default();
        let mut result = CallToolResult::error(vec![ContentBlock::text(text)]);
        result.structured_content = Some(envelope);
        result
    }

    /// Unknown or unusable handle, with the handle in `details`.
    pub fn unknown_handle(tool: &'static str, kind: HandleKind, value: &str) -> Self {
        let action = match kind {
            HandleKind::Run => {
                "Check the run_id against get_sweep_history. If the run exists but did not complete, \
                 poll get_job; otherwise re-run run_topological_sweep. Omit run_id to use the latest run."
            }
            _ => "Create or retrieve a valid handle before retrying this tool.",
        };
        Self::new(
            tool,
            kind.unknown_code(),
            format!("Unknown {kind} '{value}'."),
            action,
        )
        .detail(kind.field(), json!(value))
    }

    /// Path-visibility failure: sandbox paths get the sandbox guidance.
    pub fn path_access(
        tool: &'static str,
        path: &str,
        missing_action: &str,
        sandbox_action: &str,
    ) -> Self {
        let matched = SANDBOX_PATH_PREFIXES.iter().find(|p| path.starts_with(*p));
        let context = json!({
            "attempted_path": path,
            "looks_like_sandbox_path": matched.is_some(),
            "matched_prefix": matched,
        });
        if matched.is_some() {
            Self::new(
                tool,
                "HOST_PATH_NOT_VISIBLE",
                "Path is not visible to the MCP server host filesystem.",
                sandbox_action,
            )
            .detail("path_context", context)
        } else {
            Self::new(
                tool,
                "FILE_NOT_FOUND",
                "Path does not exist on the MCP host filesystem.",
                missing_action,
            )
            .detail("path_context", context)
        }
    }

    pub fn model_not_loaded(tool: &'static str) -> Self {
        Self::new(
            tool,
            "MODEL_NOT_LOADED",
            "No run is available: nothing has been fitted in this session.",
            "Call run_topological_sweep first, or pass the run_id of a persisted run.",
        )
    }

    /// Map a store failure onto the envelope, keeping its code and action.
    pub fn from_store(tool: &'static str, error: StoreError) -> Self {
        let mut out = Self::new(tool, error.code(), error.to_string(), error.agent_action());
        match &error {
            StoreError::UnknownHandle { kind, id } => {
                out.details.insert(kind.field().to_owned(), json!(id));
            }
            StoreError::RunNotComplete {
                run_id,
                state,
                error,
            } => {
                out.details.insert("run_id".into(), json!(run_id));
                out.details.insert("job_state".into(), json!(state));
                out.details.insert("job_error".into(), json!(error));
            }
            StoreError::DatasetChanged { dataset_id, path } => {
                out.details.insert("dataset_id".into(), json!(dataset_id));
                out.details.insert("path".into(), json!(path));
            }
            StoreError::InvalidHandle { kind, value } => {
                out.details.insert(kind.field().to_owned(), json!(value));
            }
            _ => {}
        }
        out
    }

    pub fn from_analysis(tool: &'static str, error: AnalysisError) -> Self {
        Self::new(
            tool,
            error.code(),
            error.to_string(),
            "Check the arguments against the tool description, then retry.",
        )
    }
}

/// A successful result: `text` is the human view (defaults to the pretty
/// JSON of `payload`); `payload` is the structured content.
pub fn ok_result(payload: Value, text: Option<String>) -> CallToolResult {
    let text = text.unwrap_or_else(|| {
        let pretty = serde_json::to_string_pretty(&payload).unwrap_or_default();
        if pretty.len() > 20_000 {
            serde_json::to_string(&payload).unwrap_or(pretty)
        } else {
            pretty
        }
    });
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(payload);
    result
}

/// Bytes a result costs the client: text blocks plus the structured JSON.
pub fn result_size_bytes(result: &CallToolResult) -> usize {
    let mut size = 0;
    for block in &result.content {
        if let ContentBlock::Text(t) = block {
            size += t.text.len();
        }
    }
    if let Some(s) = &result.structured_content {
        size += serde_json::to_vec(s).map(|v| v.len()).unwrap_or(0);
    }
    size
}

/// The `RESPONSE_TOO_LARGE` backstop (`ResponseBudgetMiddleware`).
pub fn too_large(tool: &str, size: usize, limit: usize) -> CallToolResult {
    let tool: &'static str = Box::leak(tool.to_owned().into_boxed_str());
    ToolError::new(
        tool,
        "RESPONSE_TOO_LARGE",
        format!("Response is {size} bytes, over the {limit}-byte limit."),
        "Request detail='summary', lower max_* / preview limits, or use a narrower tool.",
    )
    .detail("size_bytes", json!(size))
    .detail("limit_bytes", json!(limit))
    .into_result()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_carries_code_action_and_details() {
        let err =
            ToolError::unknown_handle("diagnose_cosmic_graph", HandleKind::Run, "run_0123456789ab");
        let result = err.clone().into_result();
        assert_eq!(result.is_error, Some(true));
        let s = result.structured_content.unwrap();
        assert_eq!(s["error_code"], "RUN_ID_UNKNOWN");
        assert_eq!(s["tool"], "diagnose_cosmic_graph");
        assert_eq!(s["details"]["run_id"], "run_0123456789ab");
        assert!(s["agent_action"]
            .as_str()
            .unwrap()
            .contains("get_sweep_history"));

        let sandbox = ToolError::path_access("ingest_dataset", "/home/claude/x.csv", "m", "s");
        assert_eq!(sandbox.error_code, "HOST_PATH_NOT_VISIBLE");
        let missing = ToolError::path_access("ingest_dataset", "/data/x.csv", "m", "s");
        assert_eq!(missing.error_code, "FILE_NOT_FOUND");

        let ok = ok_result(json!({"a": 1}), None);
        assert_eq!(ok.is_error, Some(false));
        assert!(result_size_bytes(&ok) > 0);
        let big = too_large("x", 10, 5);
        assert_eq!(
            big.structured_content.unwrap()["error_code"],
            "RESPONSE_TOO_LARGE"
        );
    }
}
