//! `refine_config`: constrained overrides of a sweep config
//! (`refine_config` in `pulsar/mcp/tools/config.py`).
//!
//! Applies the override engine to the given YAML, or to the session's active
//! config (storing the result back), and returns the normalized YAML with a
//! diff. Use between sweeps to shift or concentrate the grid.

use pulsar_analysis::config_overrides::{check_overrides, parse_yaml_mapping, OverrideError};
use pulsar_analysis::plan::py_json_dumps;
use rmcp::model::CallToolResult;
use schemars::JsonSchema;
use serde_json::{json, Map, Value};

use super::config::strip_fences;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const NAME: &str = "refine_config";

const SPARSIFY_MESSAGE: &str = "Spectral sparsification is enabled (cosmic_graph.sparsify: true). It is SLOW on large datasets \u{2014} a Spielman-Srivastava sparsifier that solves a preconditioned-CG system per JL sketch row \u{2014} and it runs after the already-sparse cosmic graph is built, so it is pure additional construction cost with no downstream speedup.";
const SPARSIFY_SUGGESTION: &str = "Set cosmic_graph.sparsify: false unless you specifically need a spectrum/effective-resistance-preserving graph for downstream spectral analysis on a small graph. It is not a structural-analysis or performance tool.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RefineFormat {
    #[default]
    Json,
    Markdown,
    Yaml,
}

#[derive(serde::Deserialize, JsonSchema, Default)]
#[serde(default)]
pub struct Params {
    /// Full Pulsar config YAML to refine. Leave empty to refine the session's active config (set by create_config) and store the result back.
    pub config_yaml: String,
    /// Nested dict merged into the config, e.g. {'sweep': {'ball_mapper': {'epsilon': {'range': {'min': 0.2, 'max': 1.0, 'steps': 6}}}}}. Short keys (projection_dimensions, projection_seeds, epsilon_values, construction_threshold, n_reps, n_imputations, drop_columns, run_name) set a whole parameter. Delete keys with {'remove_keys': ['dotted.path']}.
    pub overrides: Option<Map<String, Value>>,
    /// Text view: 'json', 'markdown' (diff plus YAML) or 'yaml' (bare YAML). structured_content is identical.
    pub response_format: RefineFormat,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: NAME,
        title: "Refine Config",
        description: "Apply constrained overrides to a Pulsar config and return normalized YAML plus a diff. With no config_yaml it edits the session's active config in place. Deletion via overrides={\"remove_keys\": [...]} is idempotent and shows as removed:true in the diff. Use between sweeps to concentrate or shift the grid.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("applied_overrides", s::array(s::string())),
                ("diff", s::array(s::open_object())),
                ("config_yaml", s::string()),
                ("warnings", s::array(s::open_object())),
                ("dataset_id", s::nullable(s::string())),
            ],
            &["status", "applied_overrides", "diff", "config_yaml"],
            false,
        )),
        read_only: false,
        idempotent: false,
    }
}

fn failed(code: &str, reason: impl Into<String>, action: &str) -> ToolError {
    ToolError::new(NAME, code, reason, action)
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    let overrides = p.overrides.unwrap_or_default();
    let use_active = p.config_yaml.trim().is_empty();
    let source = if use_active {
        state.session_snapshot().active_config_yaml.ok_or_else(|| {
            failed(
                "ACTIVE_CONFIG_MISSING",
                "No config_yaml provided and no active config in session.",
                "Pass config_yaml or call create_config(dataset_id) first.",
            )
        })?
    } else {
        p.config_yaml.clone()
    };

    let checked = check_overrides(&overrides).map_err(override_error)?;
    let mut raw = parse_yaml_mapping(strip_fences(&source)).map_err(|reason| {
        failed(
            "CONFIG_REFINE_FAILED",
            reason,
            "Check config_yaml is valid YAML and overrides is a nested dict, then retry.",
        )
    })?;
    let applied = checked.apply(&mut raw).map_err(override_error)?;
    let config = pulsar_config::from_value(Value::Object(raw)).map_err(|errors| {
        let issues: Vec<Value> = errors
            .iter()
            .map(|e| serde_json::to_value(e).unwrap_or_else(|_| json!(e.to_string())))
            .collect();
        failed(
            "CONFIG_REFINE_FAILED",
            "The refined config is not valid; nothing was changed.",
            "Fix the listed issues (each names its YAML path) or adjust the overrides, then retry.",
        )
        .detail("issues", Value::Array(issues))
    })?;
    let config_yaml = pulsar_config::to_yaml(&config);

    let mut payload = json!({
        "status": "ok",
        "applied_overrides": applied.applied,
        "diff": applied.diff,
        "config_yaml": config_yaml,
    });
    if config.cosmic_graph.sparsify {
        payload["warnings"] = json!([{
            "path": "cosmic_graph.sparsify",
            "message": SPARSIFY_MESSAGE,
            "suggestion": SPARSIFY_SUGGESTION,
        }]);
    }
    if use_active {
        let mut session = state.session();
        session.active_config_yaml = Some(config_yaml.clone());
        payload["dataset_id"] = json!(session.active_config_dataset_id);
    }
    let text = match p.response_format {
        RefineFormat::Json => None,
        RefineFormat::Yaml => Some(config_yaml),
        RefineFormat::Markdown => Some(to_markdown(&payload)),
    };
    Ok(ok_result(payload, text))
}

fn override_error(error: OverrideError) -> ToolError {
    match error {
        OverrideError::UnknownKey(reason) => failed(
            "UNKNOWN_OVERRIDE_KEY",
            reason,
            "Use only valid override keys. See error message for valid key list.",
        ),
        OverrideError::Invalid(reason) => failed(
            "CONFIG_REFINE_FAILED",
            reason,
            "Check config_yaml is valid YAML and overrides is a nested dict, then retry.",
        ),
    }
}

/// A value in a diff cell: strings as text, anything else as JSON, cut at 120 characters.
fn cell(value: &Value) -> String {
    const LIMIT: usize = 120;
    if value.is_null() {
        return "`null`".into();
    }
    let text = match value {
        Value::String(s) => s.clone(),
        other => py_json_dumps(other),
    };
    let text = if text.chars().count() > LIMIT {
        format!("{}...", text.chars().take(LIMIT - 3).collect::<String>())
    } else {
        text
    };
    format!("`{text}`")
}

fn to_markdown(payload: &Value) -> String {
    let mut lines = vec!["# Refined Config".to_owned(), String::new()];
    lines.push(format!(
        "- Status: {}",
        payload["status"].as_str().unwrap_or("unknown")
    ));
    if let Some(id) = payload["dataset_id"].as_str().filter(|s| !s.is_empty()) {
        lines.push(format!("- Dataset ID: `{id}`"));
    }
    let applied = payload["applied_overrides"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    lines.push(format!("- Applied overrides: {}", applied.len()));
    if !applied.is_empty() {
        let paths: Vec<String> = applied
            .iter()
            .map(|p| format!("`{}`", p.as_str().unwrap_or("")))
            .collect();
        lines.push(format!("- Paths: {}", paths.join(", ")));
    }
    if let Some(warnings) = payload["warnings"].as_array().filter(|w| !w.is_empty()) {
        lines.extend([String::new(), "## Warnings".into(), String::new()]);
        for w in warnings {
            let mut line = format!(
                "- `{}`: {}",
                w["path"].as_str().unwrap_or("$"),
                w["message"].as_str().unwrap_or("")
            );
            if let Some(suggestion) = w["suggestion"].as_str().filter(|s| !s.is_empty()) {
                line.push_str(&format!(" Suggestion: {suggestion}"));
            }
            lines.push(line);
        }
    }
    lines.extend([String::new(), "## Diff".into(), String::new()]);
    match payload["diff"].as_array().filter(|d| !d.is_empty()) {
        Some(diff) => {
            lines.extend([
                "| Path | Old | New |".to_owned(),
                "|---|---|---|".to_owned(),
            ]);
            for row in diff {
                let new = if row["removed"] == true {
                    "`<removed>`".to_owned()
                } else {
                    cell(&row["new"])
                };
                lines.push(format!(
                    "| `{}` | {} | {} |",
                    row["path"].as_str().unwrap_or(""),
                    cell(&row["old"]),
                    new
                ));
            }
        }
        None => lines.push("- No effective changes.".into()),
    }
    lines.extend([
        String::new(),
        "## Config YAML".into(),
        String::new(),
        "```yaml".into(),
        payload["config_yaml"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_owned(),
        "```".into(),
    ]);
    lines.join("\n").trim().to_owned()
}
