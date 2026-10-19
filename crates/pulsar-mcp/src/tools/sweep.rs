//! `run_topological_sweep`, `get_job`, `wait_job`, `get_sweep_history`,
//! `compare_sweeps`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pulsar_analysis::diagnostics::sparse_upper_percentile;
use pulsar_analysis::{
    agent_threshold_options, finalization_gate, GraphAnalysis, Policy, ThresholdGraph,
};
use pulsar_config::{PulsarConfig, Threshold};
use pulsar_pipeline::projection_fingerprint;
use pulsar_store::{summarize_history, JobState, RunId, RunRecord};
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use super::config::{parse_config, resolve_dataset};
use super::params::{Detail, Format};
use super::{CallContext, ToolSpec};
use crate::envelope::{ok_result, ToolError};
use crate::jobs::{run_sweep, submit, ProgressTarget, SweepRequest};
use crate::markdown;
use crate::schema::{self as s, input_schema};
use crate::state::{LiveRun, ServerState};

pub const SWEEP: &str = "run_topological_sweep";
pub const GET_JOB: &str = "get_job";
pub const WAIT_JOB: &str = "wait_job";
pub const HISTORY: &str = "get_sweep_history";
pub const COMPARE: &str = "compare_sweeps";

/// The persisted full sweep response (`runs/<id>/response.json`).
const RESPONSE_BLOB: &str = "response.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Execution {
    /// Wait for the fit and return the sweep result (a task handle on clients that support MCP tasks).
    #[default]
    Sync,
    /// Return `{status: "running", run_id}` at once; poll with wait_job / get_job.
    Async,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct SweepParams {
    /// Host-visible absolute path to a *.yaml/*.yml config file. Used only when config_yaml is empty.
    pub config_path: String,
    /// Full Pulsar config as raw YAML (preferred). If this and config_path are empty, the session's active config is used.
    pub config_yaml: String,
    /// Dataset handle from ingest_dataset; overrides run.data in the config. Empty uses the config's run.data.
    pub dataset_id: String,
    /// If true, also write the resolved config to <run name>_params.yaml next to the data for reproducibility.
    pub save_config: bool,
    /// 'summary' = compact decision packet (key metrics, bounded component preview); 'full' = the complete raw response including all metrics.
    pub detail: Detail,
    /// Text rendering only: 'markdown' or 'json'. structured_content is identical in both.
    pub response_format: Format,
    /// Summary detail only: include the normalized config YAML in the payload.
    pub include_config_yaml: bool,
    /// Summary detail only: how many component sizes to preview (largest first); the rest are counted as omitted. Must be >= 1.
    pub component_limit: usize,
    /// 'sync' (default) waits for the fit; 'async' returns the run_id immediately for wait_job / get_job polling.
    pub execution: Execution,
}

impl Default for SweepParams {
    fn default() -> Self {
        Self {
            config_path: String::new(),
            config_yaml: String::new(),
            dataset_id: String::new(),
            save_config: false,
            detail: Detail::Summary,
            response_format: Format::Markdown,
            include_config_yaml: false,
            component_limit: 20,
            execution: Execution::Sync,
        }
    }
}

pub fn sweep_spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: SWEEP,
        title: "Run Topological Sweep",
        description: "Run the Pulsar topological sweep (project, ball-map grid, cosmic graph) and persist the run. Returns run_id, key graph metrics, threshold and a finalization gate; follow with diagnose_cosmic_graph. Prefer inline config_yaml; otherwise the active session config is used. execution='async' returns the run_id immediately for wait_job polling.",
        input_schema: input_schema::<SweepParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("detail", s::string()),
                ("run_id", s::string()),
                ("dataset_id", s::nullable(s::string())),
                ("data_shape", s::array(s::integer())),
                ("construction_threshold", s::number()),
                ("graph_health", s::any()),
                ("analysis_status", s::string()),
                ("next_required_check", s::string()),
                ("finalization_gate", s::any()),
                ("constructed_graph_connected", s::boolean()),
                ("full_affinity_connected", s::boolean()),
                ("spectral_clustering_allowed", s::boolean()),
                ("is_connected", s::boolean()),
                ("singleton_fraction", s::number()),
                ("metrics", open.clone()),
                ("key_metrics", open.clone()),
                ("component_sizes_preview", s::array(s::any())),
                ("component_sizes_omitted", s::integer()),
                ("diff", s::array(s::any())),
                ("diff_summary", open.clone()),
                ("threshold_stability_summary", s::nullable(open.clone())),
                ("projection_method", s::string()),
                ("projection_cached", s::boolean()),
                ("projection_cache_status", s::any()),
                ("pca_cached", s::boolean()),
                ("pca_cache_status", s::any()),
                ("memory_usage_mb", s::number()),
                ("config_advisory", s::string()),
                ("config_yaml_included", s::boolean()),
                ("config_yaml_available_via", s::string()),
                ("config_yaml_normalized", s::string()),
                ("saved_config_path", s::string()),
                ("next_tools", s::array(s::string())),
                ("job", open),
                ("poll_with", s::string()),
            ],
            &["status", "run_id"],
            false,
        )),
        read_only: false,
        idempotent: false,
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct JobParams {
    /// The run_id returned by run_topological_sweep.
    pub run_id: String,
}

fn job_schema() -> Value {
    s::object(
        &[
            ("status", s::string()),
            ("run_id", s::string()),
            ("job", s::open_object()),
            ("run_complete", s::boolean()),
            ("response_available", s::boolean()),
            ("sweep", s::open_object()),
            ("poll_with", s::string()),
        ],
        &["status", "run_id", "job"],
        false,
    )
}

pub fn get_job_spec() -> ToolSpec {
    ToolSpec {
        name: GET_JOB,
        title: "Get Job",
        description: "Report the state of a sweep job (queued, running with progress, complete, failed) without waiting. Use after run_topological_sweep(execution='async'); when complete, the job's sweep payload is included.",
        input_schema: input_schema::<JobParams>(),
        output_schema: s::as_object(job_schema()),
        read_only: true,
        idempotent: true,
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct WaitParams {
    /// The run_id returned by run_topological_sweep.
    pub run_id: String,
    /// Longest time to wait, in seconds (0.1 to 600). On timeout the job's current progress is returned.
    pub timeout_s: f64,
}

impl Default for WaitParams {
    fn default() -> Self {
        Self {
            run_id: String::new(),
            timeout_s: 30.0,
        }
    }
}

pub fn wait_job_spec() -> ToolSpec {
    ToolSpec {
        name: WAIT_JOB,
        title: "Wait For Job",
        description: "Block until a sweep job completes or `timeout_s` elapses, then return its state. A completed job returns the same sweep summary run_topological_sweep would have; a failed job returns its error envelope; a timeout returns the current progress so you can call wait_job again.",
        input_schema: input_schema::<WaitParams>(),
        output_schema: s::as_object(job_schema()),
        read_only: true,
        idempotent: true,
    }
}

// ----------------------------------------------------------------- running

/// Resolve the request, queue the job, then either wait or return at once.
pub async fn run(
    state: Arc<ServerState>,
    p: SweepParams,
    ctx: CallContext,
) -> Result<CallToolResult, ToolError> {
    if p.component_limit < 1 {
        return Err(ToolError::invalid_argument(
            SWEEP,
            format!("component_limit must be >= 1, got {}", p.component_limit),
            "Pass component_limit >= 1.",
        ));
    }
    let prepared = prepare(&state, &p).await?;
    let job = submit(&state, &prepared.request)?;
    let run_id = job.id();
    match p.execution {
        Execution::Async => {
            let record = job.record();
            tokio::spawn(execute(state.clone(), prepared, job, ctx.progress));
            Ok(ok_result(
                json!({
                    "status": "running",
                    "run_id": run_id,
                    "job": record,
                    "poll_with": "wait_job",
                }),
                Some(format!(
                    "Sweep queued as `{run_id}`. Poll it with wait_job(run_id) or get_job(run_id)."
                )),
            ))
        }
        Execution::Sync => {
            let response = execute(state, prepared, job, ctx.progress).await?;
            Ok(render(
                response,
                p.detail,
                p.response_format,
                p.component_limit,
                p.include_config_yaml,
            ))
        }
    }
}

/// Everything `execute` needs besides the job.
pub struct Prepared {
    request: SweepRequest,
    normalized_yaml: String,
    previous: Option<RunRecord>,
    projection_method: &'static str,
    projection_cache_status: Value,
    save_config: bool,
}

async fn prepare(state: &ServerState, p: &SweepParams) -> Result<Prepared, ToolError> {
    let raw_yaml = if !p.config_yaml.trim().is_empty() {
        p.config_yaml.clone()
    } else if !p.config_path.trim().is_empty() {
        let path = p.config_path.clone();
        if !(path.ends_with(".yaml") || path.ends_with(".yml")) {
            return Err(ToolError::invalid_argument(
                SWEEP,
                format!("config_path must be a *.yaml or *.yml file, got {path:?}"),
                "Pass a YAML config path, or the YAML itself in config_yaml.",
            ));
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => {
                return Err(ToolError::path_access(
                    SWEEP,
                    &path,
                    "Pass a host-visible config path, or the YAML itself in config_yaml.",
                    "The config path is inside a sandbox the server cannot see; pass the YAML inline in config_yaml.",
                ));
            }
        }
    } else {
        state.session_snapshot().active_config_yaml.ok_or_else(|| {
            ToolError::new(
                SWEEP,
                "CONFIG_MISSING",
                "No config_yaml or config_path was given and the session has no active config.",
                "Pass the sweep config as raw YAML in config_yaml (validate_config checks it first).",
            )
        })?
    };
    let mut cfg = parse_config(SWEEP, &raw_yaml)?;

    let (dataset_id, data_path) = match resolve_dataset(state, SWEEP, &p.dataset_id)? {
        Some((id, path)) => {
            cfg.data = path.clone();
            (Some(id), PathBuf::from(path))
        }
        None if cfg.data.is_empty() => {
            return Err(ToolError::new(
                SWEEP,
                "DATASET_PATH_MISSING",
                "The config has no run.data and no dataset_id was given.",
                "Call ingest_dataset(path) and pass its dataset_id, or set run.data in the config.",
            ));
        }
        None => {
            let path = PathBuf::from(&cfg.data);
            match state.store.register_dataset(&path) {
                Ok(record) => (Some(record.dataset_id), record.path),
                Err(pulsar_store::StoreError::DatasetNotFound { .. }) => {
                    return Err(ToolError::path_access(
                        SWEEP,
                        &cfg.data,
                        "Use ingest_dataset on a host-visible path and pass dataset_id.",
                        "run.data points into a sandbox the server cannot see; ingest a host-visible copy and pass dataset_id.",
                    ));
                }
                Err(e) => return Err(ToolError::from_store(SWEEP, e)),
            }
        }
    };

    let read_path = data_path.clone();
    let table = tokio::task::spawn_blocking(move || pulsar_data::read_table(&read_path))
        .await
        .map_err(|e| ToolError::new(SWEEP, "INGEST_FAILED", e.to_string(), "Retry the sweep."))?
        .map_err(|e| {
            ToolError::new(
                SWEEP,
                e.code(),
                e.to_string(),
                "Check that the dataset is readable tabular data (CSV or Parquet), then retry.",
            )
            .detail("path", json!(data_path.display().to_string()))
        })?;

    let normalized_yaml = pulsar_config::to_yaml(&cfg);
    let schema: Vec<(String, &'static str)> = table.schema();
    let fingerprint = projection_fingerprint(&cfg, table.n_rows(), Some(&schema));
    let projection_method = match cfg.projection.method {
        pulsar_config::ProjectionMethod::Jl => "jl",
        pulsar_config::ProjectionMethod::Pca => "pca",
    };

    let session = state.session_snapshot();
    let previous = session
        .latest_run_id
        .as_ref()
        .and_then(|id| state.store.get_run(id).ok());
    let cached = session
        .latest_run_id
        .as_ref()
        .and_then(|id| state.cache.get(id));
    let (precomputed, projection_cache_status) = match cached {
        Some(live)
            if live.projection_fingerprint == fingerprint && !live.fitted.embeddings.is_empty() =>
        {
            (
                Some(live.fitted.embeddings.clone()),
                cache_status(projection_method, "hit", "fingerprint_match"),
            )
        }
        Some(_) => (
            None,
            cache_status(projection_method, "miss", "fingerprint_mismatch"),
        ),
        None => (
            None,
            cache_status(projection_method, "miss", "no_cached_embeddings"),
        ),
    };

    {
        let mut s = state.session();
        s.active_config_yaml = Some(normalized_yaml.clone());
        s.active_config_dataset_id = dataset_id.clone();
    }

    Ok(Prepared {
        request: SweepRequest {
            cfg,
            table,
            dataset_id,
            data_path: Some(data_path),
            precomputed,
            projection_fingerprint: fingerprint,
        },
        normalized_yaml,
        previous,
        projection_method,
        projection_cache_status,
        save_config: p.save_config,
    })
}

fn cache_status(method: &str, status: &str, reason: &str) -> Value {
    json!({
        "scope": "session",
        "artifact": "projection_embeddings",
        "method": method,
        "status": status,
        "reason": reason,
    })
}

/// Fit, persist the run, build and persist the full response.
async fn execute(
    state: Arc<ServerState>,
    prepared: Prepared,
    job: pulsar_store::JobHandle,
    progress: Option<ProgressTarget>,
) -> Result<Value, ToolError> {
    let Prepared {
        request,
        normalized_yaml,
        previous,
        projection_method,
        projection_cache_status,
        save_config,
    } = prepared;
    let projection_cached = projection_cache_status["status"] == "hit";
    let cfg_for_save = save_config.then(|| request.cfg.clone());
    let live = run_sweep(state.clone(), request, job, progress).await?;

    let saved_path = match cfg_for_save {
        Some(cfg) => Some(save_config_file(&cfg, live.data_path.as_deref())?),
        None => None,
    };
    let record = state
        .store
        .get_run(&live.run_id)
        .map_err(|e| ToolError::from_store(SWEEP, e))?;
    let sweep_count = state
        .store
        .list_runs(record.dataset_id.as_ref())
        .map(|l| l.runs.len())
        .unwrap_or(1);
    let response = build_response(
        &live,
        &record,
        previous.as_ref(),
        &normalized_yaml,
        projection_method,
        projection_cached,
        projection_cache_status,
        saved_path.as_deref(),
        sweep_count,
    );
    state
        .store
        .write_run_json(&live.run_id, RESPONSE_BLOB, &response)
        .map_err(|e| ToolError::from_store(SWEEP, e))?;
    Ok(response)
}

fn save_config_file(cfg: &PulsarConfig, data_path: Option<&Path>) -> Result<String, ToolError> {
    let name = if cfg.run_name.is_empty() {
        "pulsar"
    } else {
        cfg.run_name.as_str()
    };
    let dir = data_path
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let path = dir.join(format!("{name}_params.yaml"));
    std::fs::write(&path, pulsar_config::to_yaml(cfg)).map_err(|e| {
        ToolError::new(
            SWEEP,
            "CONFIG_SAVE_FAILED",
            format!("could not write {}: {e}", path.display()),
            "The run completed; retry with save_config=false or a writable data directory.",
        )
    })?;
    Ok(path.display().to_string())
}

/// `_graph_health_summary`: `(label, is_connected)`.
fn graph_health(metrics: &pulsar_analysis::GraphMetrics) -> (&'static str, bool) {
    let is_connected = metrics.component_count <= 1;
    let label = if metrics.density > 0.8 {
        "hairball"
    } else if metrics.singleton_fraction > 0.25
        || metrics.component_count > 3.max((metrics.n_nodes as f64 * 0.1) as usize)
    {
        "fragmented"
    } else {
        "connected"
    };
    (label, is_connected)
}

/// `[min, max]` of the resolved epsilon grid, as the Python history showed it.
pub fn epsilon_label(cfg: &PulsarConfig) -> String {
    let eps = &cfg.ball_mapper.epsilons;
    match (
        eps.iter().copied().reduce(f64::min),
        eps.iter().copied().reduce(f64::max),
    ) {
        (Some(lo), Some(hi)) => format!("[{lo:.3}, {hi:.3}]"),
        _ => "n/a".into(),
    }
}

pub fn threshold_label(cfg: &PulsarConfig) -> Value {
    match cfg.cosmic_graph.construction_threshold {
        Threshold::Auto => json!("auto"),
        Threshold::Value(v) => json!(v),
    }
}

fn config_of(record: &RunRecord) -> Option<PulsarConfig> {
    pulsar_config::from_yaml_str(&record.config_yaml).ok()
}

/// `_compute_sweep_diff`.
fn sweep_diff(
    previous: Option<&RunRecord>,
    current: &RunRecord,
    current_cfg: &PulsarConfig,
) -> Vec<Value> {
    let Some(prev) = previous else {
        return Vec::new();
    };
    let mut diff = Vec::new();
    let p_dims = &prev.config.projection_dimensions;
    let c_dims = &current.config.projection_dimensions;
    if p_dims != c_dims {
        diff.push(json!({"field": "projection_dimensions", "previous": p_dims, "current": c_dims}));
    }
    let p_eps = config_of(prev)
        .map(|c| epsilon_label(&c))
        .unwrap_or_else(|| "n/a".into());
    let c_eps = epsilon_label(current_cfg);
    if p_eps != c_eps {
        diff.push(json!({"field": "epsilon", "previous": p_eps, "current": c_eps}));
    }
    if let (Some(p), Some(c)) = (&prev.dataset_id, &current.dataset_id) {
        if p != c {
            diff.push(json!({"field": "dataset_id", "previous": p, "current": c}));
        }
    }
    let (pm, cm) = (&prev.metrics, &current.metrics);
    if pm.n_edges != cm.n_edges {
        diff.push(json!({"field": "n_edges", "previous": pm.n_edges, "current": cm.n_edges}));
    }
    if pm.component_count != cm.component_count {
        diff.push(json!({"field": "component_count", "previous": pm.component_count, "current": cm.component_count}));
    }
    if pm.giant_fraction != cm.giant_fraction {
        diff.push(json!({"field": "giant_fraction", "previous": pm.giant_fraction, "current": cm.giant_fraction}));
    }
    diff
}

#[allow(clippy::too_many_arguments)]
fn build_response(
    live: &LiveRun,
    record: &RunRecord,
    previous: Option<&RunRecord>,
    normalized_yaml: &str,
    projection_method: &str,
    projection_cached: bool,
    projection_cache_status: Value,
    saved_path: Option<&str>,
    sweep_count: usize,
) -> Value {
    let fitted = &live.fitted;
    let affinity = fitted.weighted_edges(Some(0.0));
    let minhash_d = match fitted.config.cosmic_graph.construction {
        pulsar_config::Construction::Minhash => Some(fitted.config.cosmic_graph.minhash_d as usize),
        pulsar_config::Construction::Exact => None,
    };
    let metrics = GraphAnalysis::new(&fitted.graph)
        .map(|a| {
            a.metrics(
                fitted.construction_threshold,
                fitted.ball_maps.len(),
                Some(&affinity),
                minhash_d,
            )
        })
        .ok();
    let metrics_value = metrics
        .as_ref()
        .and_then(|m| serde_json::to_value(m).ok())
        .unwrap_or(json!({}));
    let (health, is_connected) = metrics
        .as_ref()
        .map(graph_health)
        .unwrap_or(("empty", true));
    let threshold_graph = ThresholdGraph::from_edges(fitted.n, &affinity);
    let full_affinity_connected = fitted.n > 0 && threshold_graph.component_count_at(0.0) <= 1;

    let gate = match (&metrics, &fitted.stability) {
        (Some(m), Some(stability)) => {
            let plateaus: Vec<_> = stability.plateaus.iter().take(10).cloned().collect();
            let options = agent_threshold_options(
                &threshold_graph,
                &plateaus,
                &stability.thresholds,
                &stability.component_counts,
                Policy::Balanced,
                7,
            );
            finalization_gate(
                m,
                sweep_count,
                &fitted.config.projection.dimensions,
                Some(&options.candidates),
            )
        }
        (Some(m), None) => {
            finalization_gate(m, sweep_count, &fitted.config.projection.dimensions, None)
        }
        (None, _) => {
            json!({"status": "ok", "reason": "No dominant-component resolution gate triggered."})
        }
    };

    let config_advisory = match saved_path {
        Some(path) => format!("Config changes persisted to host at: {path}"),
        None => "Config changes are in-memory (session-only). Set save_config=True to persist to params.yaml.".to_owned(),
    };
    let mut response = Map::new();
    response.insert("status".into(), json!("ok"));
    response.insert("run_id".into(), json!(record.run_id));
    response.insert("metrics".into(), metrics_value);
    response.insert("projection_method".into(), json!(projection_method));
    response.insert("projection_cached".into(), json!(projection_cached));
    response.insert(
        "projection_cache_status".into(),
        projection_cache_status.clone(),
    );
    response.insert("pca_cached".into(), json!(projection_cached));
    response.insert("pca_cache_status".into(), projection_cache_status);
    response.insert(
        "memory_usage_mb".into(),
        json!((live.memory_bytes() as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0),
    );
    response.insert(
        "diff".into(),
        Value::Array(sweep_diff(previous, record, &fitted.config)),
    );
    response.insert("config_advisory".into(), json!(config_advisory));
    response.insert("config_yaml_normalized".into(), json!(normalized_yaml));
    response.insert(
        "data_shape".into(),
        json!([live.table.n_rows(), live.table.names.len()]),
    );
    response.insert("is_connected".into(), json!(is_connected));
    response.insert("constructed_graph_connected".into(), json!(is_connected));
    response.insert(
        "full_affinity_connected".into(),
        json!(full_affinity_connected),
    );
    response.insert(
        "singleton_fraction".into(),
        json!(record.metrics.singleton_fraction),
    );
    response.insert(
        "spectral_clustering_allowed".into(),
        json!(full_affinity_connected),
    );
    response.insert("graph_health".into(), json!(health));
    response.insert("analysis_status".into(), json!("diagnostics_required"));
    response.insert("next_required_check".into(), json!("diagnose_cosmic_graph"));
    response.insert("finalization_gate".into(), gate);
    response.insert(
        "construction_threshold".into(),
        json!(fitted.construction_threshold),
    );
    if let Some(summary) = &record.threshold_stability_summary {
        response.insert("threshold_stability_summary".into(), json!(summary));
    }
    if let Some(path) = saved_path {
        response.insert("saved_config_path".into(), json!(path));
    }
    if let Some(ds) = &record.dataset_id {
        response.insert("dataset_id".into(), json!(ds));
    }
    Value::Object(response)
}

/// `build_sweep_summary_payload`.
pub fn summary_payload(
    response: &Value,
    component_limit: usize,
    include_config_yaml: bool,
) -> Value {
    let metrics = &response["metrics"];
    let sizes: Vec<Value> = metrics["component_sizes"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let preview: Vec<Value> = sizes.iter().take(component_limit).cloned().collect();
    let key_metrics: Map<String, Value> = [
        "n_nodes",
        "n_edges",
        "density",
        "component_count",
        "giant_fraction",
        "singleton_fraction",
        "n_ball_maps",
        "grid_adequacy_status",
    ]
    .iter()
    .filter_map(|k| metrics.get(*k).map(|v| ((*k).to_owned(), v.clone())))
    .collect();
    let diff: Vec<Value> = response["diff"].as_array().cloned().unwrap_or_default();
    let mut parameter_changes = Vec::new();
    let mut metric_changes = Map::new();
    for row in &diff {
        let field = row["field"].as_str().unwrap_or("");
        match field {
            "n_edges" | "component_count" | "giant_fraction" => {
                metric_changes.insert(
                    field.to_owned(),
                    json!({"previous": row["previous"], "current": row["current"]}),
                );
            }
            "projection_dimensions" | "pca_dims" => {
                parameter_changes.push(json!("sweep.projection.dimensions.values"))
            }
            "epsilon" => parameter_changes.push(json!("cosmic_graph.epsilon")),
            other => parameter_changes.push(json!(other)),
        }
    }
    let changes_returned = parameter_changes.len() + metric_changes.len();
    let mut payload = json!({
        "status": response["status"],
        "detail": "summary",
        "run_id": response["run_id"],
        "dataset_id": response.get("dataset_id").cloned().unwrap_or(Value::Null),
        "data_shape": response["data_shape"],
        "construction_threshold": response["construction_threshold"],
        "graph_health": response["graph_health"],
        "analysis_status": response["analysis_status"],
        "next_required_check": response["next_required_check"],
        "finalization_gate": response["finalization_gate"],
        "constructed_graph_connected": response["constructed_graph_connected"],
        "full_affinity_connected": response["full_affinity_connected"],
        "spectral_clustering_allowed": response["spectral_clustering_allowed"],
        "key_metrics": key_metrics,
        "component_sizes_preview": preview,
        "component_sizes_omitted": sizes.len().saturating_sub(component_limit.min(sizes.len())),
        "diff_summary": {
            "parameter_changes": parameter_changes,
            "metric_changes": metric_changes,
            "changes_returned": changes_returned,
        },
        "threshold_stability_summary": response.get("threshold_stability_summary").cloned().unwrap_or(Value::Null),
        "projection_method": response["projection_method"],
        "projection_cached": response["projection_cached"],
        "projection_cache_status": response["projection_cache_status"],
        "pca_cached": response["pca_cached"],
        "pca_cache_status": response["pca_cache_status"],
        "memory_usage_mb": response["memory_usage_mb"],
        "config_advisory": response["config_advisory"],
        "config_yaml_included": include_config_yaml,
        "config_yaml_available_via": "get_runtime_context",
        "next_tools": ["diagnose_cosmic_graph", "get_sweep_history", "get_threshold_stability_curve"],
    });
    let obj = payload.as_object_mut().expect("object");
    if include_config_yaml {
        obj.insert(
            "config_yaml_normalized".into(),
            response["config_yaml_normalized"].clone(),
        );
    }
    if let Some(path) = response.get("saved_config_path") {
        obj.insert("saved_config_path".into(), path.clone());
    }
    payload
}

fn render(
    response: Value,
    detail: Detail,
    format: Format,
    component_limit: usize,
    include_config_yaml: bool,
) -> CallToolResult {
    let summary = summary_payload(&response, component_limit, include_config_yaml);
    let text = match format {
        Format::Markdown => Some(markdown::sweep(&summary)),
        Format::Json => None,
    };
    match detail {
        Detail::Full => {
            let mut full = response;
            if let Some(obj) = full.as_object_mut() {
                obj.insert("detail".into(), json!("full"));
            }
            ok_result(full, text)
        }
        Detail::Summary => ok_result(summary, text),
    }
}

// --------------------------------------------------------------------- jobs

fn job_payload(state: &ServerState, run_id: &RunId) -> Result<(Value, JobState), ToolError> {
    let job = state
        .store
        .get_job(run_id)
        .map_err(|e| ToolError::from_store(GET_JOB, e))?;
    let response = if job.state == JobState::Complete {
        state
            .store
            .read_run_json(run_id, RESPONSE_BLOB)
            .ok()
            .flatten()
    } else {
        None
    };
    let mut payload = json!({
        "status": job.state.as_str(),
        "run_id": run_id,
        "job": job,
        "run_complete": job.state == JobState::Complete,
        "response_available": response.is_some(),
    });
    if let Some(r) = response {
        payload["sweep"] = summary_payload(&r, 20, false);
    }
    if !job.state.is_terminal() {
        payload["poll_with"] = json!("wait_job");
    }
    Ok((payload, job.state))
}

pub async fn get_job(state: &ServerState, p: JobParams) -> Result<CallToolResult, ToolError> {
    let run_id = RunId::parse(&p.run_id).map_err(|e| ToolError::from_store(GET_JOB, e))?;
    let (payload, _) = job_payload(state, &run_id)?;
    Ok(ok_result(payload, None))
}

pub async fn wait_job(state: &ServerState, p: WaitParams) -> Result<CallToolResult, ToolError> {
    if !(0.1..=600.0).contains(&p.timeout_s) {
        return Err(ToolError::invalid_argument(
            WAIT_JOB,
            format!("timeout_s must be between 0.1 and 600, got {}", p.timeout_s),
            "Pass timeout_s in [0.1, 600].",
        ));
    }
    let run_id = RunId::parse(&p.run_id).map_err(|e| ToolError::from_store(WAIT_JOB, e))?;
    let deadline = Instant::now() + Duration::from_secs_f64(p.timeout_s);
    loop {
        let (payload, job_state) = job_payload(state, &run_id)?;
        match job_state {
            JobState::Complete => {
                let text = payload
                    .get("sweep")
                    .map(markdown::sweep)
                    .unwrap_or_else(|| format!("Run `{run_id}` is complete."));
                return Ok(ok_result(payload, Some(text)));
            }
            JobState::Failed => {
                let error = payload["job"]["error"].clone();
                let code = error["code"].as_str().unwrap_or("JOB_FAILED").to_owned();
                let message = error["message"]
                    .as_str()
                    .unwrap_or("the sweep failed")
                    .to_owned();
                return Err(ToolError::new(
                    WAIT_JOB,
                    &code,
                    message,
                    "Fix the problem the error names (config or data), then submit a new sweep.",
                )
                .detail("run_id", json!(run_id))
                .detail("job", payload["job"].clone()));
            }
            JobState::Queued | JobState::Running => {
                if Instant::now() >= deadline {
                    let progress = payload["job"]["progress"].as_f64().unwrap_or(0.0);
                    let stage = payload["job"]["stage"].as_str().unwrap_or("").to_owned();
                    return Ok(ok_result(
                        payload,
                        Some(format!(
                            "Run `{run_id}` is still {}: {stage} ({:.0}%). Call wait_job again.",
                            job_state.as_str(),
                            progress * 100.0
                        )),
                    ));
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
    }
}

// ------------------------------------------------------------------ history

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HistoryDetail {
    #[default]
    Table,
    Summary,
    Full,
}

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct HistoryParams {
    /// 'table' = compact per-run rows; 'summary' = synthesized patterns across runs; 'full' = both.
    pub detail: HistoryDetail,
    /// Text rendering only: 'markdown' or 'json'. structured_content is identical in both.
    pub response_format: Format,
    /// Optional dataset_id to restrict the history to one dataset. Empty lists every persisted run.
    pub dataset_id: String,
}

pub fn history_spec() -> ToolSpec {
    ToolSpec {
        name: HISTORY,
        title: "Sweep History",
        description: "List the persisted sweeps with graph metrics per run. Use after several run_topological_sweep calls to see how epsilon and projection dimensions moved nodes, edges, components and giant fraction; detail='summary' synthesizes patterns across runs. Reads the store, so it covers every completed run.",
        input_schema: input_schema::<HistoryParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("detail", s::string()),
                ("n_runs", s::integer()),
                ("runs", s::array(s::open_object())),
                ("summary", s::open_object()),
            ],
            &["status", "detail", "n_runs"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn history(state: &ServerState, p: HistoryParams) -> Result<CallToolResult, ToolError> {
    let dataset = resolve_dataset(state, HISTORY, &p.dataset_id)?.map(|(id, _)| id);
    let listing = state
        .store
        .list_runs(dataset.as_ref())
        .map_err(|e| ToolError::from_store(HISTORY, e))?;
    let rows: Vec<Value> = listing
        .runs
        .iter()
        .enumerate()
        .map(|(i, r)| {
            json!({
                "run_index": i + 1,
                "run_id": r.run_id,
                "dataset_id": r.dataset_id,
                "projection_dimensions": r.config.projection_dimensions,
                "epsilon": config_of(r).map(|c| epsilon_label(&c)).unwrap_or_else(|| "n/a".into()),
                "nodes": r.metrics.n_nodes,
                "edges": r.metrics.n_edges,
                "components": r.metrics.component_count,
                "giant_fraction": r.metrics.giant_fraction,
            })
        })
        .collect();
    let summary = summarize_history(&listing.runs);
    let detail = match p.detail {
        HistoryDetail::Table => "table",
        HistoryDetail::Summary => "summary",
        HistoryDetail::Full => "full",
    };
    let mut payload = json!({"status": "ok", "detail": detail, "n_runs": rows.len()});
    if matches!(p.detail, HistoryDetail::Table | HistoryDetail::Full) {
        payload["runs"] = Value::Array(rows);
    }
    if matches!(p.detail, HistoryDetail::Summary | HistoryDetail::Full) {
        payload["summary"] = json!(summary);
    }
    let text = match p.response_format {
        Format::Markdown => {
            let mut view = payload.clone();
            if view.get("runs").is_none() {
                view["runs"] = json!(listing
                    .runs
                    .iter()
                    .enumerate()
                    .map(|(i, r)| json!({"run_index": i + 1, "run_id": r.run_id, "projection_dimensions": r.config.projection_dimensions, "epsilon": config_of(r).map(|c| epsilon_label(&c)).unwrap_or_else(|| "n/a".into()), "nodes": r.metrics.n_nodes, "edges": r.metrics.n_edges, "components": r.metrics.component_count, "giant_fraction": r.metrics.giant_fraction}))
                    .collect::<Vec<_>>());
            }
            Some(markdown::history(&view))
        }
        Format::Json => None,
    };
    Ok(ok_result(payload, text))
}

// ------------------------------------------------------------------ compare

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct CompareParams {
    /// First run_id, as returned by run_topological_sweep (persisted runs only).
    pub run_a: String,
    /// Second run_id to compare against run_a.
    pub run_b: String,
}

pub fn compare_spec() -> ToolSpec {
    let side = s::object(
        &[
            ("run_id", s::string()),
            ("dataset_id", s::nullable(s::string())),
            ("projection_dimensions", s::array(s::any())),
            ("epsilon", s::string()),
            ("threshold", s::any()),
            (
                "metrics",
                s::object(
                    &[
                        ("n_nodes", s::any()),
                        ("n_edges", s::any()),
                        ("component_count", s::any()),
                        ("giant_fraction", s::any()),
                        ("weight_p95", s::any()),
                    ],
                    &[],
                    true,
                ),
            ),
        ],
        &["run_id", "metrics"],
        false,
    );
    ToolSpec {
        name: COMPARE,
        title: "Compare Sweeps",
        description: "Compare two persisted sweeps side by side by config and graph metrics. Returns each run's projection dimensions, epsilon, construction threshold, nodes, edges, components, giant fraction and weight p95; the text view is a markdown table. Measurement only: it does not rank the runs.",
        input_schema: input_schema::<CompareParams>(),
        output_schema: s::as_object(s::object(
            &[("status", s::string()), ("run_a", side.clone()), ("run_b", side)],
            &["status", "run_a", "run_b"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

fn compare_side(state: &ServerState, raw: &str) -> Result<Value, ToolError> {
    let view = state.resolve_run(COMPARE, raw)?;
    let affinity = view
        .affinity()
        .map_err(|e| ToolError::from_store(COMPARE, e))?;
    let n = affinity.n;
    let weights: Vec<f64> = affinity.edges.iter().map(|e| e.2).collect();
    let weight_p95 = sparse_upper_percentile(&weights, n * n.saturating_sub(1) / 2, 95.0);
    let cfg = view.config();
    let record = &view.record;
    Ok(json!({
        "run_id": record.run_id,
        "dataset_id": record.dataset_id,
        "projection_dimensions": record.config.projection_dimensions,
        "epsilon": cfg.as_ref().map(epsilon_label).unwrap_or_else(|| "n/a".into()),
        "threshold": cfg.as_ref().map(threshold_label).unwrap_or(json!("auto")),
        "metrics": {
            "n_nodes": record.metrics.n_nodes,
            "n_edges": record.metrics.n_edges,
            "component_count": record.metrics.component_count,
            "giant_fraction": record.metrics.giant_fraction,
            "weight_p95": weight_p95,
        },
    }))
}

pub async fn compare(state: &ServerState, p: CompareParams) -> Result<CallToolResult, ToolError> {
    if p.run_a.is_empty() || p.run_b.is_empty() {
        return Err(ToolError::invalid_argument(
            COMPARE,
            "run_a and run_b are both required.",
            "Pass two run_ids from run_topological_sweep or get_sweep_history.",
        ));
    }
    let payload = json!({
        "status": "ok",
        "run_a": compare_side(state, &p.run_a)?,
        "run_b": compare_side(state, &p.run_b)?,
    });
    let text = markdown::compare(&payload);
    Ok(ok_result(payload, Some(text)))
}
