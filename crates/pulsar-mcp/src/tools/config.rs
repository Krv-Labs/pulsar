//! `validate_config`: the params.yaml contract, backed by `pulsar-config`.

use pulsar_config::PulsarConfig;
use pulsar_store::DatasetId;
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const NAME: &str = "validate_config";

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct Params {
    /// Full Pulsar config as raw YAML text (no Markdown fences).
    pub config_yaml: String,
    /// Optional dataset handle from ingest_dataset; run.data is resolved against it and it becomes the session's current dataset.
    pub dataset_id: String,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: NAME,
        title: "Validate Config",
        description: "Validate a full Pulsar config and normalize it into canonical YAML. Returns issues, warnings and normalized_config_yaml on success; a structured error with an error_code and agent_action when the config is invalid. Use before run_topological_sweep when you hand-edited the YAML.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("resolved_dataset_path", s::nullable(s::string())),
                ("issues", s::array(s::open_object())),
                ("warnings", s::array(s::open_object())),
                ("normalized_config_yaml", s::string()),
            ],
            &["status", "resolved_dataset_path", "issues"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

/// Strip Markdown fences an agent may have wrapped the YAML in.
pub fn strip_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest
            .strip_prefix("yaml")
            .or_else(|| rest.strip_prefix("yml"))
            .unwrap_or(rest);
        let rest = rest.strip_suffix("```").unwrap_or(rest);
        return rest.trim();
    }
    trimmed
}

/// Parse YAML into a resolved config, as a tool error on failure.
pub fn parse_config(tool: &'static str, yaml: &str) -> Result<PulsarConfig, ToolError> {
    let text = strip_fences(yaml);
    if text.is_empty() {
        return Err(ToolError::new(
            tool,
            "CONFIG_YAML_INVALID",
            "config_yaml is empty.",
            "Pass the full Pulsar config as raw YAML (see `pulsar schema config` for the format).",
        ));
    }
    pulsar_config::from_yaml_str(text).map_err(|errors| {
        let issues: Vec<Value> = errors
            .iter()
            .map(|e| serde_json::to_value(e).unwrap_or_else(|_| json!(e.to_string())))
            .collect();
        ToolError::new(
            tool,
            "CONFIG_YAML_INVALID",
            "Config validation failed.",
            "Fix the listed issues (each names its YAML path), then validate_config again before running the sweep.",
        )
        .detail("issues", Value::Array(issues))
    })
}

/// Resolve an optional `dataset_id` to the checked dataset path.
pub fn resolve_dataset(
    state: &ServerState,
    tool: &'static str,
    dataset_id: &str,
) -> Result<Option<(DatasetId, String)>, ToolError> {
    if dataset_id.is_empty() {
        return Ok(None);
    }
    let id = DatasetId::parse(dataset_id).map_err(|e| ToolError::from_store(tool, e))?;
    let record = state
        .store
        .check_dataset(&id)
        .map_err(|e| ToolError::from_store(tool, e))?;
    Ok(Some((id, record.path.display().to_string())))
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    let mut cfg = parse_config(NAME, &p.config_yaml)?;
    let dataset = resolve_dataset(state, NAME, &p.dataset_id)?;
    if let Some((_, path)) = &dataset {
        cfg.data = path.clone();
    }
    let mut issues = Vec::new();
    let resolved_dataset_path = if cfg.data.is_empty() {
        issues.push(json!({
            "path": "run.data",
            "message": "Dataset path is missing",
            "expected": "A readable CSV or Parquet path via run.data or dataset_id",
            "suggestion": "Pass dataset_id or set run.data in the YAML",
        }));
        None
    } else if !std::path::Path::new(&cfg.data).exists() {
        issues.push(json!({
            "path": "run.data",
            "message": format!("Dataset path does not exist: {}", cfg.data),
            "expected": "A host-visible CSV or Parquet path",
            "suggestion": "Use ingest_dataset(path) on a server-visible path, then pass dataset_id",
        }));
        Some(cfg.data.clone())
    } else {
        Some(cfg.data.clone())
    };
    if !issues.is_empty() {
        let code = if cfg.data.is_empty() {
            "DATASET_PATH_MISSING"
        } else {
            "FILE_NOT_FOUND"
        };
        return Err(ToolError::new(
            NAME,
            code,
            "Config validation failed.",
            "Use ingest_dataset on a host-visible path, then validate_config with dataset_id or set run.data to a readable host path.",
        )
        .detail("resolved_dataset_path", json!(resolved_dataset_path))
        .detail("issues", Value::Array(issues)));
    }
    let normalized = pulsar_config::to_yaml(&cfg);
    let mut warnings = Vec::new();
    if cfg.cosmic_graph.sparsify {
        warnings.push(json!({
            "path": "cosmic_graph.sparsify",
            "message": "Spectral sparsification is enabled; density and edge counts will not be comparable to non-sparsified runs.",
        }));
    }
    {
        let mut session = state.session();
        session.active_config_yaml = Some(normalized.clone());
        if let Some((id, _)) = &dataset {
            session.active_config_dataset_id = Some(id.clone());
            session.latest_dataset_id = Some(id.clone());
        }
    }
    Ok(ok_result(
        json!({
            "status": "ok",
            "resolved_dataset_path": resolved_dataset_path,
            "issues": [],
            "warnings": warnings,
            "normalized_config_yaml": normalized,
        }),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_are_stripped() {
        assert_eq!(
            strip_fences("```yaml\nrun:\n  name: x\n```"),
            "run:\n  name: x"
        );
        assert_eq!(strip_fences("  run: 1  "), "run: 1");
    }
}
