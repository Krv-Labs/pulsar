//! `characterize_dataset` and `probe_columns`: the first look at a dataset
//! (`pulsar/mcp/tools/meta.py` and `reporting.py`).
//!
//! Both read the raw data, not a fitted model, and return bounded measurement
//! packets with no domain interpretation. `characterize_dataset` is the map
//! (counts, geometry, a ranked preview of columns worth a look);
//! `probe_columns` is the magnifying glass for named columns.

use std::collections::HashSet;

use pulsar_analysis::characterization::{
    characterization_payload_to_markdown, characterize_dataset, compact_characterization_payload,
    probe_columns_payload, probe_columns_payload_to_markdown, profile_column_details,
    CharacterizationError, ColumnProfile,
};
use pulsar_data::{DataError, Table};
use rmcp::model::CallToolResult;
use schemars::JsonSchema;
use serde_json::{json, Value};

use super::config::resolve_dataset;
use super::params::Format;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const CHARACTERIZE: &str = "characterize_dataset";
pub const PROBE: &str = "probe_columns";
/// Columns one `probe_columns` call may inspect.
const MAX_PROBE_COLUMNS: usize = 20;
/// Row cap and seed of the geometry sample (as in v0.5).
const SUBSAMPLE: usize = 1000;
const SEED: u64 = 42;

pub(super) fn registered(name: &str) -> bool {
    super::catalog().iter().any(|t| t.name == name)
}

pub(super) async fn load(tool: &'static str, path: &str) -> Result<Table, ToolError> {
    let owned = std::path::PathBuf::from(path);
    tokio::task::spawn_blocking(move || pulsar_data::read_table(&owned))
        .await
        .map_err(|e| ToolError::new(tool, "DATASET_LOAD_FAILED", e.to_string(), "Retry the call."))?
        .map_err(|e| match &e {
            DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => ToolError::path_access(
                tool,
                path,
                "Ask the user for a host-visible absolute dataset path, or use ingest_dataset first and pass dataset_id.",
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads or bash/cp as a bridge. Ask the user for a host-visible absolute path, or call get_runtime_context; only if the user confirms the server host can see `cache_dir`, copy the file there, ingest_dataset(path), and retry this tool.",
            ),
            _ => ToolError::new(
                tool,
                e.code(),
                e.to_string(),
                "Check that the dataset is readable tabular data (CSV or Parquet), then retry or re-ingest it.",
            )
            .detail("path", json!(path)),
        })
}

/// Until `create_config` and `refine_config` are registered, point at the
/// tools that exist: a server must not advertise a tool it does not have.
fn point_at_registered_tools(payload: &mut Value, markdown: &mut String) {
    if !registered("create_config") {
        payload["recommended_next_tool"] = json!("draft_preprocessing_plan");
        payload["agent_guidance"] = json!(
            "Use draft_preprocessing_plan(dataset_id) to draft a preprocessing plan. Use probe_columns(dataset_id, columns) only for specific columns listed in column_profile_preview or named by the user."
        );
        *markdown = markdown.replace(
            "- `create_config` for processed-space calibration.",
            "- `draft_preprocessing_plan` to draft a preprocessing plan.",
        );
    }
    if !registered("refine_config") {
        *markdown = markdown.replace(
            "- Set via `refine_config(..., cosmic_graph.minhash_d=<value>)`.",
            "- Set `cosmic_graph.minhash_d` in the config YAML.",
        );
    }
}

// ---- characterize_dataset ------------------------------------------------------

#[derive(serde::Deserialize, JsonSchema, Default)]
#[serde(default)]
pub struct CharacterizeParams {
    /// Host-visible absolute path to a CSV/Parquet file. Ignored when dataset_id is given; use either this or dataset_id.
    pub csv_path: String,
    /// Dataset handle returned by ingest_dataset. Preferred over csv_path; takes precedence when both are given.
    pub dataset_id: String,
    /// Text view: 'markdown' or 'json'. structured_content is identical.
    pub response_format: Format,
}

pub fn characterize_spec() -> ToolSpec {
    ToolSpec {
        name: CHARACTERIZE,
        title: "Characterize dataset",
        description: "Summarize dataset geometry and schema: row/feature counts, variance curve, k-NN scale and a ranked column preview. Returns a bounded measurement packet with no domain interpretation. On massive datasets it adds a minhash_advisory suggesting cosmic_graph.minhash_d. Use it to see the data's scale and structure before drafting a preprocessing plan.",
        input_schema: input_schema::<CharacterizeParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("dataset_id", s::string()),
                ("n_samples", s::any()),
                ("n_features", s::any()),
                ("n_columns_total", s::any()),
                ("missingness_pct", s::any()),
                ("numeric_missingness_pct", s::any()),
                ("overall_missingness_pct", s::any()),
                ("raw_numeric_geometry", s::open_object()),
                ("schema_summary", s::open_object()),
                ("column_name_preview", s::open_object()),
                ("column_profile_preview", s::array(s::open_object())),
                ("omitted_column_profiles", s::integer()),
                ("recommended_next_tool", s::string()),
                ("agent_guidance", s::string()),
                ("minhash_advisory", s::open_object()),
            ],
            &["status", "n_samples", "raw_numeric_geometry", "schema_summary", "column_profile_preview", "recommended_next_tool"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

fn characterization_error(error: CharacterizationError) -> ToolError {
    let action = match &error {
        CharacterizationError::TooFewNumericColumns { .. } => {
            "Geometry needs at least two numeric columns. Inspect the schema with probe_columns, or fix the column types and re-ingest."
        }
        CharacterizationError::TooFewRows { .. } => "The dataset needs at least two rows to measure neighbour distances.",
        _ => "Check the dataset is readable tabular data, then retry or re-ingest it.",
    };
    ToolError::new(CHARACTERIZE, error.code(), error.to_string(), action)
}

pub async fn characterize(
    state: &ServerState,
    p: CharacterizeParams,
) -> Result<CallToolResult, ToolError> {
    let (dataset_id, path) = match resolve_dataset(state, CHARACTERIZE, &p.dataset_id)? {
        Some((id, path)) => (id, path),
        // Same checks as ingest_dataset (canonical path, regular file), so
        // special files such as /dev/stdin are refused and the caller gets
        // the dataset_id the next-step guidance asks for.
        None if !p.csv_path.is_empty() => match state.store.register_dataset(&p.csv_path) {
            Ok(record) => (record.dataset_id, record.path.display().to_string()),
            Err(pulsar_store::StoreError::DatasetNotFound { .. }) => {
                return Err(ToolError::path_access(
                    CHARACTERIZE,
                    &p.csv_path,
                    "Ask the user for a host-visible absolute path to a regular CSV/Parquet file, or use ingest_dataset first and pass dataset_id.",
                    "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads or bash/cp as a bridge. Ask the user for a host-visible absolute path, or call get_runtime_context; only if the user confirms the server host can see `cache_dir`, copy the file there, ingest_dataset(path), and retry this tool.",
                ));
            }
            Err(e) => return Err(ToolError::from_store(CHARACTERIZE, e)),
        },
        None => {
            return Err(ToolError::new(
                CHARACTERIZE,
                "DATASET_NOT_SPECIFIED",
                "Provide either csv_path or dataset_id.",
                "Call ingest_dataset(path) to get a dataset_id, then pass dataset_id.",
            ))
        }
    };
    state.session().latest_dataset_id = Some(dataset_id.clone());
    let table = load(CHARACTERIZE, &path).await?;
    let profile =
        tokio::task::spawn_blocking(move || characterize_dataset(&table, SUBSAMPLE, SEED))
            .await
            .map_err(|e| {
                ToolError::new(
                    CHARACTERIZE,
                    "CHARACTERIZATION_FAILED",
                    e.to_string(),
                    "Retry; report the error if it recurs.",
                )
            })?
            .map_err(characterization_error)?;

    let mut payload = compact_characterization_payload(&profile);
    payload["dataset_id"] = json!(dataset_id.to_string());
    let mut markdown = characterization_payload_to_markdown(&payload);
    point_at_registered_tools(&mut payload, &mut markdown);
    let text = match p.response_format {
        Format::Markdown => markdown,
        Format::Json => serde_json::to_string_pretty(&payload).unwrap_or_default(),
    };
    Ok(ok_result(payload, Some(text)))
}

// ---- probe_columns -------------------------------------------------------------

#[derive(serde::Deserialize, JsonSchema, Default)]
#[serde(default)]
pub struct ProbeParams {
    /// Dataset handle returned by ingest_dataset.
    pub dataset_id: String,
    /// Names of columns of the raw dataset to inspect (at most 20); unknown names are listed under missing_columns.
    pub columns: Vec<String>,
    /// Text view: 'markdown' or 'json'. structured_content is identical.
    pub response_format: Format,
}

pub fn probe_spec() -> ToolSpec {
    ToolSpec {
        name: PROBE,
        title: "Probe Columns",
        description: "Drill into specific columns: sample values, missingness, ranges and top values. Reads the raw dataset, not a fitted model. Use after characterize_dataset or draft_preprocessing_plan flags columns to review; values are truncated and at most 20 columns are allowed per call.",
        input_schema: input_schema::<ProbeParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("columns_requested", s::integer()),
                ("columns_returned", s::integer()),
                ("missing_columns", s::array(s::string())),
                ("column_profiles", s::array(s::open_object())),
            ],
            &["status", "columns_requested", "columns_returned", "missing_columns", "column_profiles"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn probe(state: &ServerState, p: ProbeParams) -> Result<CallToolResult, ToolError> {
    if p.columns.len() > MAX_PROBE_COLUMNS {
        return Err(ToolError::new(
            PROBE,
            "TOO_MANY_COLUMNS",
            "Too many columns requested. Max 20 at a time.",
            "Split the request into batches of at most 20 columns.",
        )
        .detail("requested", json!(p.columns.len()))
        .detail("max", json!(MAX_PROBE_COLUMNS)));
    }
    if p.dataset_id.is_empty() {
        return Err(ToolError::new(
            PROBE,
            "MISSING_INPUT",
            "dataset_id is required.",
            "Pass dataset_id from ingest_dataset.",
        ));
    }
    let (id, path) = resolve_dataset(state, PROBE, &p.dataset_id)?
        .expect("non-empty dataset_id resolves or errors");
    state.session().latest_dataset_id = Some(id);
    let table = load(PROBE, &path).await?;

    let columns = p.columns.clone();
    let payload = tokio::task::spawn_blocking(move || {
        let present: HashSet<&str> = table.names.iter().map(String::as_str).collect();
        let (mut profiles, mut missing): (Vec<ColumnProfile>, Vec<String>) = (vec![], vec![]);
        for c in &columns {
            match present
                .contains(c.as_str())
                .then(|| profile_column_details(&table, c, 10))
                .flatten()
            {
                Some(profile) => profiles.push(profile),
                None => missing.push(c.clone()),
            }
        }
        probe_columns_payload(&profiles, &columns, &missing)
    })
    .await
    .map_err(|e| {
        ToolError::new(
            PROBE,
            "PROBE_FAILED",
            e.to_string(),
            "Check the column names and retry with fewer columns.",
        )
    })?;

    let text = match p.response_format {
        Format::Markdown => probe_columns_payload_to_markdown(&payload),
        Format::Json => serde_json::to_string_pretty(&payload).unwrap_or_default(),
    };
    Ok(ok_result(payload, Some(text)))
}
