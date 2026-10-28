//! The longitudinal tools (`pulsar/mcp/tools/longitudinal.py` and
//! `sample_longitudinal_entities` in `tools/ingestion.py`): pivot an ingested
//! long-format table into a panel, build its representations, and measure
//! them. Payloads come from `pulsar_analysis::longitudinal`; this module owns
//! parameters, handles, errors and the text views.
//!
//! Known divergences from Python:
//! - Handles are `lg_<12 hex>` store handles (Python minted `lng_...`), and
//!   each build saves a store record. The matrices stay in the session, which
//!   keeps the last [`MAX_LONGITUDINAL_ARTIFACTS`]; a handle whose matrices
//!   are gone returns `LONGITUDINAL_NOT_LOADED` with what to rebuild.
//! - `sample_longitudinal_entities` always writes the sample as Parquet
//!   (Python kept CSV sources as CSV).
//! - The build reports no intermediate progress.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pulsar_analysis::longitudinal::{
    adapt_config_to_panel, archetype_payload, classify_trajectories_payload, cross_time_payload,
    estimate_costs, guard_representation, pivot_panel, prepare_panel_frame,
    select_longitudinal_entities, temporal_diagnosis, trajectory_diagnosis, PanelError,
};
use pulsar_analysis::payloads::bounded_list;
use pulsar_analysis::pyround::round_to;
use pulsar_core::TemporalCosmicGraph;
use pulsar_data::preprocess::py_float_repr;
use pulsar_data::Key;
use pulsar_pipeline::{CosmicTrajectory, EntityIds};
use pulsar_store::{DatasetSource, LongitudinalId, Representation};
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use super::characterize::load;
use super::config::{parse_config, resolve_dataset};
use super::params::Format;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::{LongitudinalArtifact, ServerState, MAX_LONGITUDINAL_ARTIFACTS};

pub const BUILD: &str = "build_longitudinal_graph";
pub const DIAGNOSE: &str = "diagnose_longitudinal_graph";
pub const ARCHETYPES: &str = "get_trajectory_archetypes";
pub const NEIGHBORS: &str = "get_cross_time_neighbors";
pub const CLASSIFY: &str = "classify_trajectories";
pub const SAMPLE: &str = "sample_longitudinal_entities";

const CLASSIFICATION_PREVIEW_LIMIT: usize = 25;

// ---- shared helpers ----------------------------------------------------------------------

fn panel_error(tool: &'static str, e: PanelError) -> ToolError {
    let mut out = ToolError::new(tool, e.error_code, e.reason, e.agent_action);
    out.details.extend(*e.details);
    out
}

fn invalid(tool: &'static str, reason: &str) -> ToolError {
    ToolError::invalid_argument(tool, reason, "Fix the argument named in reason and retry.")
}

/// The artifact for `longitudinal_id` (empty: the most recent one).
fn resolve(
    state: &ServerState,
    tool: &'static str,
    id: &str,
) -> Result<Arc<LongitudinalArtifact>, ToolError> {
    let session = state.session();
    if id.is_empty() {
        return session.longitudinal.last().cloned().ok_or_else(|| {
            ToolError::new(
                tool,
                "LONGITUDINAL_ID_UNKNOWN",
                "No longitudinal panel has been built in this session.",
                "Call build_longitudinal_graph first, or pass the longitudinal_id it returned.",
            )
            .detail("longitudinal_id", json!(id))
        });
    }
    if let Some(a) = session
        .longitudinal
        .iter()
        .find(|a| a.longitudinal_id.as_str() == id)
    {
        return Ok(a.clone());
    }
    drop(session);
    let record = LongitudinalId::parse(id)
        .ok()
        .and_then(|lid| state.store.get_longitudinal(&lid).ok());
    match record {
        Some(r) => Err(ToolError::new(
            tool,
            "LONGITUDINAL_NOT_LOADED",
            format!("Longitudinal panel '{id}' is recorded but its graphs are no longer in memory."),
            "Rebuild it with build_longitudinal_graph using the dataset_id, columns, representation and config_yaml in details.",
        )
        .detail("longitudinal_id", json!(id))
        .detail("dataset_id", json!(r.dataset_id))
        .detail("representation", json!(r.representation))
        .detail("panel", r.panel)
        .detail("config_yaml", json!(r.config_yaml))),
        None => Err(ToolError::new(
            tool,
            "LONGITUDINAL_ID_UNKNOWN",
            format!("Unknown longitudinal_id '{id}'."),
            "Create or retrieve a valid longitudinal_id before retrying this tool.",
        )
        .detail("longitudinal_id", json!(id))),
    }
}

fn require_trajectory<'a>(
    tool: &'static str,
    a: &'a LongitudinalArtifact,
) -> Result<&'a CosmicTrajectory, ToolError> {
    a.trajectory.as_ref().ok_or_else(|| {
        ToolError::new(
            tool,
            "REPRESENTATION_NOT_BUILT",
            "This panel was built without the trajectory representation.",
            "Rebuild with build_longitudinal_graph(representation='trajectory') or representation='both'.",
        )
        .detail("longitudinal_id", json!(a.longitudinal_id))
        .detail("representation", json!(a.representation))
    })
}

fn internal(tool: &'static str, e: impl std::fmt::Display) -> ToolError {
    ToolError::new(
        tool,
        "LONGITUDINAL_QUERY_FAILED",
        e.to_string(),
        "Call diagnose_longitudinal_graph to confirm the panel is intact, or rebuild it with build_longitudinal_graph.",
    )
}

/// Python `str()` of a JSON scalar.
fn py_text(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) if n.is_f64() => py_float_repr(n.as_f64().unwrap_or(f64::NAN)),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => py_list(other),
    }
}

/// Python `str(list)` of JSON scalars (strings quoted).
fn py_list(v: &Value) -> String {
    let items: Vec<String> = v
        .as_array()
        .into_iter()
        .flatten()
        .map(|x| match x {
            Value::String(s) => pulsar_analysis::plan::py_repr(s),
            other => py_text(other),
        })
        .collect();
    format!("[{}]", items.join(", "))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn next_tools_md(payload: &Value) -> Vec<String> {
    let mut lines = vec![String::new(), "## Next Tools".to_owned()];
    lines.extend(
        payload["next_tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| format!("- `{}`", t.as_str().unwrap_or_default())),
    );
    lines
}

fn ladder_md(sweep: &Value) -> Vec<String> {
    let mut lines = vec![
        "| Threshold | Obs clusters | Distinct trajectories | Largest group |".to_owned(),
        "|---|---|---|---|".to_owned(),
    ];
    for row in sweep["rows"].as_array().into_iter().flatten() {
        lines.push(format!(
            "| {} | {} | {} | {} |",
            py_text(&row["threshold"]),
            row["observation_clusters"],
            row["distinct_trajectories"],
            row["largest_trajectory_group"]
        ));
    }
    lines.push(String::new());
    lines.push(format!("_{}_", sweep["note"].as_str().unwrap_or_default()));
    lines
}

fn render(
    payload: Value,
    format: Format,
    markdown: impl FnOnce(&Value) -> String,
) -> CallToolResult {
    match format {
        Format::Json => ok_result(payload, None),
        Format::Markdown => {
            let text = markdown(&payload);
            ok_result(payload, Some(text))
        }
    }
}

fn bounded_schema() -> Value {
    s::object(
        &[
            ("total", s::integer()),
            ("preview", s::array(s::any())),
            ("omitted", s::integer()),
        ],
        &["total", "preview"],
        true,
    )
}

fn sweep_schema() -> Value {
    s::object(
        &[
            ("basis", s::string()),
            ("note", s::string()),
            (
                "rows",
                s::array(s::object(
                    &[
                        ("threshold", s::number()),
                        ("observation_clusters", s::integer()),
                        ("distinct_trajectories", s::integer()),
                        ("largest_trajectory_group", s::integer()),
                    ],
                    &[
                        "threshold",
                        "observation_clusters",
                        "distinct_trajectories",
                        "largest_trajectory_group",
                    ],
                    false,
                )),
            ),
        ],
        &["basis", "note", "rows"],
        false,
    )
}

fn report_properties() -> Vec<(&'static str, Value)> {
    vec![
        ("policy_applied", s::string()),
        ("n_entities_input", s::integer()),
        ("n_entities_kept", s::integer()),
        ("dropped_entities", bounded_schema()),
        ("n_times", s::integer()),
        ("n_features", s::integer()),
        ("cells_missing_fraction", s::number()),
        ("alignment", s::enumeration(&["complete", "ragged"])),
        ("observation_count", s::integer()),
    ]
}

const REPORT_REQUIRED: [&str; 6] = [
    "policy_applied",
    "n_entities_kept",
    "n_times",
    "n_features",
    "alignment",
    "observation_count",
];

fn weight_dist_schema() -> Value {
    s::object(
        &[
            ("count", s::integer()),
            ("p25", s::nullable(s::number())),
            ("p50", s::nullable(s::number())),
            ("p95", s::nullable(s::number())),
            ("max", s::nullable(s::number())),
        ],
        &[],
        true,
    )
}

// ---- build_longitudinal_graph ------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationParam {
    #[default]
    Trajectory,
    Temporal,
    Both,
}

impl RepresentationParam {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trajectory => "trajectory",
            Self::Temporal => "temporal",
            Self::Both => "both",
        }
    }

    fn store(self) -> Representation {
        match self {
            Self::Trajectory => Representation::Trajectory,
            Self::Temporal => Representation::Temporal,
            Self::Both => Representation::Both,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MissingParam {
    #[default]
    DropEntity,
    ForwardFill,
    AllowRagged,
}

impl MissingParam {
    fn as_str(self) -> &'static str {
        match self {
            Self::DropEntity => "drop_entity",
            Self::ForwardFill => "forward_fill",
            Self::AllowRagged => "allow_ragged",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    #[default]
    Summary,
    Full,
}

impl Detail {
    fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Full => "full",
        }
    }
}

fn json_format() -> Format {
    Format::Json
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct BuildParams {
    /// Required. From ingest_dataset or sample_longitudinal_entities: long-format panel, one row per (entity, time step), numeric features.
    #[serde(default)]
    pub dataset_id: String,
    /// Column identifying the tracked entity (e.g. an id). Required.
    #[serde(default)]
    pub entity_column: String,
    /// Column holding the discrete, orderable time step. Required; continuous timestamps must be binned before ingest.
    #[serde(default)]
    pub time_column: String,
    /// Numeric columns to embed. Omit to use every numeric column except the entity/time keys and any preprocessing.drop_columns.
    #[serde(default)]
    pub feature_columns: Option<Vec<String>>,
    /// 'trajectory' (default): sparse graph over (entity, time) observations; edges may cross time steps. 'temporal': dense (n, n, T) entity tensor, memory-guarded. 'both': build each.
    #[serde(default)]
    pub representation: RepresentationParam,
    /// Pulsar params YAML from create_config. Omit to reuse the session's active config. Excess projection dimensions are dropped (see config_adaptation).
    #[serde(default)]
    pub config_yaml: String,
    /// Entities missing at some time step: 'drop_entity' removes them, 'forward_fill' carries values forward, 'allow_ragged' keeps them (trajectory only).
    #[serde(default)]
    pub on_missing: MissingParam,
    /// Construction-time floor (>= 0) on the weight of trajectory edges (similarity between two (entity, time) observations); edges below it are never built. 0 keeps all. Interpretation thresholds on later tools filter on top of this.
    #[serde(default)]
    pub similarity_threshold: f64,
    /// 'summary' (default): panel shape, build cost and graph surface. 'full': also the normalized config_yaml and the list of time labels.
    #[serde(default)]
    pub detail: Detail,
    /// Text view: 'json' (default) or 'markdown' table. structured_content is identical.
    #[serde(default = "json_format")]
    pub response_format: Format,
}

pub fn build_spec() -> ToolSpec {
    let open = s::open_object();
    let mut panel_props = vec![
        ("entity_column", s::string()),
        ("time_column", s::string()),
        ("n_features", s::integer()),
        ("feature_columns_preview", s::array(s::string())),
    ];
    panel_props.extend(report_properties());
    ToolSpec {
        name: BUILD,
        title: "Build Longitudinal Graph",
        description: "Pivot a long-format dataset into a panel and build longitudinal graphs. Returns a longitudinal_id handle, panel shape and alignment report, build cost, and graph surface. Does not touch the static sweep/cluster state. Use before diagnose_longitudinal_graph, get_trajectory_archetypes, get_cross_time_neighbors or classify_trajectories.",
        input_schema: input_schema::<BuildParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("detail", s::enumeration(&["summary", "full"])),
                ("longitudinal_id", s::string()),
                ("dataset_id", s::nullable(s::string())),
                ("representation", s::enumeration(&["trajectory", "temporal", "both"])),
                ("panel", s::object(&panel_props, &["entity_column", "time_column", "n_features"], true)),
                (
                    "build_cost",
                    s::object(
                        &[
                            ("n_observations", s::integer()),
                            ("n_entities", s::integer()),
                            ("n_times", s::integer()),
                            ("temporal_tensor_mb", s::number()),
                            ("temporal_peak_mb", s::number()),
                            ("temporal_tensor_limit_mb", s::number()),
                            ("estimated_ball_maps_trajectory", s::integer()),
                            ("estimated_ball_maps_temporal", s::integer()),
                            ("projection_dimensions", s::array(s::integer())),
                            ("epsilons", s::array(s::number())),
                        ],
                        &["estimated_ball_maps_trajectory", "estimated_ball_maps_temporal"],
                        true,
                    ),
                ),
                ("config_adaptation", open.clone()),
                (
                    "graph_surface",
                    s::object(
                        &[
                            (
                                "trajectory",
                                s::object(
                                    &[
                                        ("n_observations", s::integer()),
                                        ("n_balls", s::integer()),
                                        ("n_edges", s::integer()),
                                        ("cross_time_edges", s::integer()),
                                        ("cross_time_fraction", s::number()),
                                    ],
                                    &[],
                                    true,
                                ),
                            ),
                            ("temporal", s::object(&[("tensor_shape", s::array(s::integer())), ("tensor_mb", s::number())], &[], true)),
                        ],
                        &[],
                        false,
                    ),
                ),
                ("next_tools", s::array(s::string())),
                ("config_yaml", s::string()),
                ("times", s::array(s::any())),
            ],
            &[
                "status",
                "detail",
                "longitudinal_id",
                "dataset_id",
                "representation",
                "panel",
                "build_cost",
                "config_adaptation",
                "graph_surface",
                "next_tools",
            ],
            false,
        )),
        read_only: false,
        idempotent: false,
    }
}

/// Set when the calling future is dropped (the client cancelled): the worker
/// stops at its next stage boundary.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

struct Built {
    trajectory: Option<CosmicTrajectory>,
    temporal: Option<TemporalCosmicGraph>,
    panel: pulsar_analysis::longitudinal::Panel,
    costs: Value,
    notes: Map<String, Value>,
}

fn build_worker(
    table: pulsar_data::Table,
    config: pulsar_config::PulsarConfig,
    p: &BuildParams,
    cancelled: &AtomicBool,
) -> Result<Built, ToolError> {
    let check = |stage: &str| {
        if cancelled.load(Ordering::Relaxed) {
            Err(ToolError::new(
                BUILD,
                "OPERATION_CANCELLED",
                format!("Longitudinal build cancelled during {stage}."),
                "Retry build_longitudinal_graph when ready.",
            ))
        } else {
            Ok(())
        }
    };
    let features = p.feature_columns.clone().unwrap_or_default();
    let frame = prepare_panel_frame(
        &table,
        &config.preprocessing,
        &p.entity_column,
        &p.time_column,
        &features,
    )
    .map_err(|e| panel_error(BUILD, e))?;
    check("panel preparation")?;
    let panel = pivot_panel(
        &frame,
        &p.entity_column,
        &p.time_column,
        &features,
        p.on_missing.as_str(),
    )
    .map_err(|e| panel_error(BUILD, e))?;
    let rep = p.representation.as_str();
    guard_representation(&panel, rep).map_err(|e| panel_error(BUILD, e))?;
    let (config, notes) =
        adapt_config_to_panel(&config, &panel).map_err(|e| panel_error(BUILD, e))?;
    let costs = estimate_costs(&panel, &config);
    check("representation build")?;
    let failed = |e: String| {
        ToolError::new(
            BUILD,
            "LONGITUDINAL_BUILD_FAILED",
            e,
            "Check entity_column, time_column and config_yaml against the dataset; try representation='trajectory' or a smaller panel.",
        )
    };
    let trajectory = if rep != "temporal" {
        Some(
            CosmicTrajectory::from_snapshots(
                &panel.snapshots,
                &config,
                &EntityIds::PerSnapshot(panel.snapshot_entity_ids.clone()),
                Some(&panel.times),
                p.similarity_threshold,
                true,
            )
            .map_err(|e| failed(e.to_string()))?,
        )
    } else {
        None
    };
    let temporal = if rep != "trajectory" {
        check("temporal representation")?;
        // Python's temporal build always used randomized PCA on the
        // projection grid's dimensions and seeds.
        Some(
            TemporalCosmicGraph::from_snapshots(
                &panel.snapshots,
                &config.projection.dimensions,
                &config.projection.seeds,
                &config.ball_mapper.epsilons,
                0.0,
            )
            .map_err(|e| failed(e.to_string()))?,
        )
    } else {
        None
    };
    check("artifact publish")?;
    Ok(Built {
        trajectory,
        temporal,
        panel,
        costs,
        notes,
    })
}

fn surface_summary(
    trajectory: Option<&CosmicTrajectory>,
    temporal: Option<&TemporalCosmicGraph>,
) -> Value {
    let mut out = Map::new();
    if let Some(ct) = trajectory {
        let total = ct.similarity.edges.len();
        let cross = ct.cross_time(0.0).edges.len();
        out.insert(
            "trajectory".into(),
            json!({
                "n_observations": ct.n_observations(),
                "n_balls": ct.balls.len(),
                "n_edges": total,
                "cross_time_edges": cross,
                "cross_time_fraction": if total > 0 { round_to(cross as f64 / total as f64, 4) } else { 0.0 },
            }),
        );
    }
    if let Some(tcg) = temporal {
        let (a, b, c) = tcg.shape();
        out.insert(
            "temporal".into(),
            json!({"tensor_shape": [a, b, c], "tensor_mb": round_to(tcg.nbytes() as f64 / (1024.0 * 1024.0), 2)}),
        );
    }
    Value::Object(out)
}

pub async fn build(state: &ServerState, p: BuildParams) -> Result<CallToolResult, ToolError> {
    if p.entity_column.is_empty() || p.time_column.is_empty() {
        return Err(ToolError::new(
            BUILD,
            "PANEL_KEYS_MISSING",
            "entity_column and time_column are required.",
            "Call probe_columns or characterize_dataset to identify the entity and time columns, then retry.",
        ));
    }
    if p.similarity_threshold < 0.0 {
        return Err(invalid(BUILD, "similarity_threshold must be >= 0."));
    }
    let yaml = if p.config_yaml.is_empty() {
        state
            .session()
            .active_config_yaml
            .clone()
            .unwrap_or_default()
    } else {
        p.config_yaml.clone()
    };
    if yaml.is_empty() {
        return Err(ToolError::new(
            BUILD,
            "CONFIG_REQUIRED",
            "No config available for the longitudinal build.",
            "Call create_config(dataset_id) first — it calibrates epsilon and the projection grid against this dataset — then pass config_yaml.",
        )
        .detail("dataset_id", json!(p.dataset_id)));
    }
    let Some((dataset_id, path)) = resolve_dataset(state, BUILD, &p.dataset_id)? else {
        return Err(ToolError::new(
            BUILD,
            "DATASET_ID_REQUIRED",
            "dataset_id is required for a longitudinal build.",
            "Call ingest_dataset first and pass the returned dataset_id.",
        ));
    };
    let mut config = parse_config(BUILD, &yaml)?;
    config.data = path.clone();
    let normalized = pulsar_config::to_yaml(&config);
    let table = load(BUILD, &path).await?;

    let cancelled = Arc::new(AtomicBool::new(false));
    let _guard = CancelOnDrop(cancelled.clone());
    let flag = cancelled.clone();
    let (built, p) = tokio::task::spawn_blocking(move || {
        let r = build_worker(table, config, &p, &flag);
        (r, p)
    })
    .await
    .map_err(|e| {
        ToolError::new(
            BUILD,
            "LONGITUDINAL_BUILD_FAILED",
            e.to_string(),
            "Retry build_longitudinal_graph.",
        )
    })?;
    let built = built?;

    let rep = p.representation.as_str();
    let panel_summary = json!({
        "entity_column": p.entity_column,
        "time_column": p.time_column,
        "n_entities": built.panel.n_entities(),
        "n_times": built.panel.n_times(),
        "n_features": built.panel.feature_columns.len(),
    });
    let record = state
        .store
        .save_longitudinal(
            Some(dataset_id.clone()),
            p.representation.store(),
            &normalized,
            panel_summary,
        )
        .map_err(|e| ToolError::from_store(BUILD, e))?;
    let graph_surface = surface_summary(built.trajectory.as_ref(), built.temporal.as_ref());
    let mut panel = Map::new();
    panel.insert("entity_column".into(), json!(p.entity_column));
    panel.insert("time_column".into(), json!(p.time_column));
    panel.insert(
        "n_features".into(),
        json!(built.panel.feature_columns.len()),
    );
    panel.insert(
        "feature_columns_preview".into(),
        json!(built
            .panel
            .feature_columns
            .iter()
            .take(10)
            .collect::<Vec<_>>()),
    );
    if let Value::Object(report) = &built.panel.report {
        panel.extend(report.clone());
    }
    let mut payload = json!({
        "status": "ok",
        "detail": p.detail.as_str(),
        "longitudinal_id": record.longitudinal_id,
        "dataset_id": dataset_id,
        "representation": rep,
        "panel": panel,
        "build_cost": built.costs,
        "config_adaptation": built.notes,
        "graph_surface": graph_surface,
        "next_tools": ["diagnose_longitudinal_graph", "get_trajectory_archetypes", "get_cross_time_neighbors"],
    });
    if p.detail == Detail::Full {
        payload["config_yaml"] = json!(normalized);
        payload["times"] = json!(built.panel.times);
    }
    let artifact = Arc::new(LongitudinalArtifact {
        longitudinal_id: record.longitudinal_id,
        dataset_id: Some(dataset_id),
        representation: rep,
        trajectory: built.trajectory,
        temporal: built.temporal,
        entity_column: p.entity_column.clone(),
        time_column: p.time_column.clone(),
        feature_columns: built.panel.feature_columns.clone(),
        times: built.panel.times.clone(),
        report: built.panel.report.clone(),
        config_yaml: normalized,
    });
    {
        let mut session = state.session();
        session.longitudinal.push(artifact);
        let excess = session
            .longitudinal
            .len()
            .saturating_sub(MAX_LONGITUDINAL_ARTIFACTS);
        session.longitudinal.drain(..excess);
    }
    Ok(render(payload, p.response_format, build_markdown))
}

fn build_markdown(payload: &Value) -> String {
    let panel = &payload["panel"];
    let cost = &payload["build_cost"];
    let mut lines = vec![
        format!(
            "# Longitudinal Panel `{}`",
            payload["longitudinal_id"].as_str().unwrap_or_default()
        ),
        String::new(),
        format!(
            "- Representation: **{}**",
            payload["representation"].as_str().unwrap_or_default()
        ),
        format!(
            "- Panel: {} entities x {} times = {} observations",
            panel["n_entities_kept"], panel["n_times"], panel["observation_count"]
        ),
        format!("- Features: {}", panel["n_features"]),
        format!(
            "- Alignment: {} (policy `{}`, {} entities dropped)",
            panel["alignment"].as_str().unwrap_or_default(),
            panel["policy_applied"].as_str().unwrap_or_default(),
            panel["dropped_entities"]["total"]
        ),
        format!(
            "- Missing cells: {:.2}%",
            num(&panel["cells_missing_fraction"]) * 100.0
        ),
        format!(
            "- Estimated ball maps: {} (trajectory) / {} (temporal)",
            cost["estimated_ball_maps_trajectory"], cost["estimated_ball_maps_temporal"]
        ),
    ];
    let surface = &payload["graph_surface"];
    if let Some(traj) = surface.get("trajectory") {
        lines.extend([
            String::new(),
            "## Trajectory surface".to_owned(),
            format!(
                "- {} observations, {} balls, {} edges",
                traj["n_observations"], traj["n_balls"], traj["n_edges"]
            ),
            format!(
                "- Cross-time edges: {} ({:.1}%)",
                traj["cross_time_edges"],
                num(&traj["cross_time_fraction"]) * 100.0
            ),
        ]);
    }
    if let Some(temp) = surface.get("temporal") {
        lines.extend([
            String::new(),
            "## Temporal surface".to_owned(),
            format!(
                "- Tensor {} ({} MB)",
                py_list(&temp["tensor_shape"]),
                py_text(&temp["tensor_mb"])
            ),
        ]);
    }
    lines.extend(next_tools_md(payload));
    lines.join("\n")
}

// ---- diagnose_longitudinal_graph ----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceParam {
    #[default]
    Auto,
    Trajectory,
    Temporal,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct DiagnoseParams {
    /// From build_longitudinal_graph. Omit for the most recent panel.
    #[serde(default)]
    pub longitudinal_id: String,
    /// Surface to measure: 'trajectory' (observation graph), 'temporal' (entity tensor aggregations), or 'auto' for every surface the panel was built with.
    #[serde(default)]
    pub representation: SurfaceParam,
    /// 'full' adds component morphology and build metadata.
    #[serde(default)]
    pub detail: Detail,
    /// Text view: 'json' (default) or 'markdown' table. structured_content is identical.
    #[serde(default = "json_format")]
    pub response_format: Format,
}

pub fn diagnose_spec() -> ToolSpec {
    let report_props = report_properties();
    let next = s::array(s::string());
    let trajectory = s::object(
        &[
            ("representation", s::enumeration(&["trajectory"])),
            ("node_identity", s::string()),
            (
                "panel",
                s::object(
                    &[
                        ("n_observations", s::integer()),
                        ("n_entities", s::integer()),
                        ("n_times", s::integer()),
                    ],
                    &["n_observations", "n_entities", "n_times"],
                    false,
                ),
            ),
            (
                "similarity_surface",
                s::object(
                    &[
                        ("surface", s::string()),
                        ("construction", s::any()),
                        ("similarity_threshold_applied", s::any()),
                        ("n_edges", s::integer()),
                        ("cross_time_edges", s::integer()),
                        ("cross_time_fraction", s::number()),
                        ("weight_distribution", weight_dist_schema()),
                    ],
                    &["n_edges", "cross_time_edges", "cross_time_fraction"],
                    true,
                ),
            ),
            (
                "cover",
                s::object(
                    &[
                        ("n_balls", s::integer()),
                        ("balls_by_eps", s::open_object()),
                        ("ball_sizes", s::open_object()),
                        ("memberships_per_observation", weight_dist_schema()),
                    ],
                    &["n_balls"],
                    true,
                ),
            ),
            ("threshold_sweep", sweep_schema()),
            ("next_tools", next.clone()),
            ("component_morphology", s::open_object()),
            ("meta", s::open_object()),
        ],
        &[
            "representation",
            "node_identity",
            "panel",
            "similarity_surface",
            "cover",
            "threshold_sweep",
            "next_tools",
        ],
        false,
    );
    let aggregation = s::object(
        &[
            ("aggregation", s::string()),
            ("value_range", s::string()),
            ("measures", s::string()),
            ("why", s::string()),
            ("best_for", s::array(s::string())),
            ("avoid_for", s::array(s::string())),
            ("distribution", weight_dist_schema()),
            ("self_scaled_cut", s::number()),
            ("edges_at_cut", s::integer()),
            ("components_at_cut", s::integer()),
            ("converging_pairs", s::integer()),
            ("diverging_pairs", s::integer()),
        ],
        &[
            "aggregation",
            "distribution",
            "self_scaled_cut",
            "edges_at_cut",
        ],
        true,
    );
    let temporal = s::object(
        &[
            ("representation", s::enumeration(&["temporal"])),
            ("node_identity", s::string()),
            (
                "panel",
                s::object(
                    &[("n_entities", s::integer()), ("n_times", s::integer())],
                    &["n_entities", "n_times"],
                    false,
                ),
            ),
            (
                "tensor",
                s::object(
                    &[
                        ("shape", s::array(s::integer())),
                        ("bytes", s::integer()),
                        ("mb", s::number()),
                    ],
                    &["shape", "mb"],
                    false,
                ),
            ),
            ("cut_basis", s::string()),
            ("aggregations", s::array(aggregation)),
            ("next_tools", next),
            ("detail", s::enumeration(&["full"])),
            ("threshold", s::any()),
        ],
        &[
            "representation",
            "node_identity",
            "panel",
            "tensor",
            "cut_basis",
            "aggregations",
            "next_tools",
        ],
        false,
    );
    ToolSpec {
        name: DIAGNOSE,
        title: "Diagnose Longitudinal Graph",
        description: "Measure a built longitudinal panel without prescribing an interpretation. Trajectory surface: edge/cover counts, weight distribution, cross-time edge fraction, threshold ladder. Temporal surface: the six tensor aggregations side by side, each cut at its own q90, with why/best_for/avoid_for.",
        input_schema: input_schema::<DiagnoseParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("detail", s::enumeration(&["summary", "full"])),
                ("longitudinal_id", s::string()),
                ("panel", s::object(&report_props, &REPORT_REQUIRED, true)),
                ("surfaces", s::object(&[("trajectory", trajectory), ("temporal", temporal)], &[], false)),
            ],
            &["status", "detail", "longitudinal_id", "panel", "surfaces"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn diagnose(state: &ServerState, p: DiagnoseParams) -> Result<CallToolResult, ToolError> {
    let artifact = resolve(state, DIAGNOSE, &p.longitudinal_id)?;
    let detail = p.detail.as_str();
    let want = p.representation;
    let a = artifact.clone();
    let surfaces = tokio::task::spawn_blocking(move || {
        let mut surfaces = Map::new();
        if want != SurfaceParam::Temporal {
            if let Some(ct) = &a.trajectory {
                surfaces.insert("trajectory".into(), trajectory_diagnosis(ct, detail));
            }
        }
        if want != SurfaceParam::Trajectory {
            if let Some(tcg) = &a.temporal {
                surfaces.insert("temporal".into(), temporal_diagnosis(tcg, detail));
            }
        }
        surfaces
    })
    .await
    .map_err(|e| internal(DIAGNOSE, e))?;
    if surfaces.is_empty() {
        let requested = match want {
            SurfaceParam::Auto => "auto",
            SurfaceParam::Trajectory => "trajectory",
            SurfaceParam::Temporal => "temporal",
        };
        return Err(ToolError::new(
            DIAGNOSE,
            "REPRESENTATION_NOT_BUILT",
            format!("Panel has no '{requested}' surface to diagnose."),
            "Rebuild with build_longitudinal_graph(representation=...) including the surface you want to measure.",
        )
        .detail("longitudinal_id", json!(artifact.longitudinal_id))
        .detail("built", json!(artifact.representation))
        .detail("requested", json!(requested)));
    }
    let payload = json!({
        "status": "ok",
        "detail": detail,
        "longitudinal_id": artifact.longitudinal_id,
        "panel": artifact.report,
        "surfaces": surfaces,
    });
    Ok(render(payload, p.response_format, diagnosis_markdown))
}

fn diagnosis_markdown(payload: &Value) -> String {
    let mut lines = vec![format!(
        "# Longitudinal Diagnosis `{}`",
        payload["longitudinal_id"].as_str().unwrap_or_default()
    )];
    if let Some(traj) = payload["surfaces"].get("trajectory") {
        let sim = &traj["similarity_surface"];
        let wd = &sim["weight_distribution"];
        lines.extend([
            String::new(),
            "## Trajectory surface (observation nodes)".to_owned(),
            format!(
                "- {} observations ({} entities x {} times)",
                traj["panel"]["n_observations"],
                traj["panel"]["n_entities"],
                traj["panel"]["n_times"]
            ),
            format!(
                "- {} edges; {} cross-time ({:.1}%)",
                sim["n_edges"],
                sim["cross_time_edges"],
                num(&sim["cross_time_fraction"]) * 100.0
            ),
            format!(
                "- Weights p25/p50/p95: {} / {} / {}",
                py_text(&wd["p25"]),
                py_text(&wd["p50"]),
                py_text(&wd["p95"])
            ),
            format!("- Cover: {} balls", traj["cover"]["n_balls"]),
            String::new(),
            "### Threshold ladder".to_owned(),
        ]);
        lines.extend(ladder_md(&traj["threshold_sweep"]));
    }
    if let Some(temporal) = payload["surfaces"].get("temporal") {
        lines.extend([
            String::new(),
            "## Temporal surface (entity nodes)".to_owned(),
            format!(
                "- Tensor {} ({} MB)",
                py_list(&temporal["tensor"]["shape"]),
                py_text(&temporal["tensor"]["mb"])
            ),
            format!("- _{}_", temporal["cut_basis"].as_str().unwrap_or_default()),
            String::new(),
            "| Aggregation | Range | Edges@cut | Components@cut | Best for | Why |".to_owned(),
            "|---|---|---|---|---|---|".to_owned(),
        ]);
        for agg in temporal["aggregations"].as_array().into_iter().flatten() {
            let best: Vec<&str> = agg["best_for"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            lines.push(format!(
                "| {} | {} | {} | {} | {} | {} |",
                agg["aggregation"].as_str().unwrap_or_default(),
                agg["value_range"].as_str().unwrap_or_default(),
                agg["edges_at_cut"],
                agg["components_at_cut"],
                best.join(", "),
                agg["why"].as_str().unwrap_or_default()
            ));
        }
    }
    lines.join("\n")
}

// ---- get_trajectory_archetypes ------------------------------------------------------------

fn ten() -> usize {
    10
}

fn five() -> usize {
    5
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ArchetypesParams {
    /// From build_longitudinal_graph. Omit for the most recent panel.
    #[serde(default)]
    pub longitudinal_id: String,
    /// Interpretation filter (>= 0) on CO_SIMILAR edge weights used to cluster observations; not the construction threshold. 0 uses every built edge.
    #[serde(default)]
    pub threshold: f64,
    /// Most populous archetypes to return (>= 1).
    #[serde(default = "ten")]
    pub max_archetypes: usize,
    /// Example entity ids listed per archetype (>= 0).
    #[serde(default = "five")]
    pub max_entities_per_archetype: usize,
    /// 'summary' and 'full' return the same bounded payload (the value is only echoed); widen with max_archetypes / max_entities_per_archetype.
    #[serde(default)]
    pub detail: Detail,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    #[serde(default)]
    pub response_format: Format,
}

pub fn archetypes_spec() -> ToolSpec {
    ToolSpec {
        name: ARCHETYPES,
        title: "Trajectory Archetypes",
        description: "Group entities by the cluster sequence they trace through time: an archetype is a set of entities with an identical per-time-step cluster sequence. Needs the trajectory representation. Cluster counts move steeply with threshold, so the threshold ladder is always included; read it before trusting one row.",
        input_schema: input_schema::<ArchetypesParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("detail", s::enumeration(&["summary", "full"])),
                ("longitudinal_id", s::string()),
                ("representation", s::enumeration(&["trajectory"])),
                ("interpretation_threshold", s::number()),
                ("threshold_role", s::string()),
                ("n_entities", s::integer()),
                ("n_times", s::integer()),
                ("distinct_trajectories", s::integer()),
                ("archetypes_returned", s::integer()),
                ("archetypes_omitted", s::integer()),
                (
                    "archetypes",
                    s::array(s::object(
                        &[
                            ("sequence", s::array(s::nullable(s::integer()))),
                            ("n_entities", s::integer()),
                            ("fraction", s::number()),
                            ("transitions", s::integer()),
                            ("entities", bounded_schema()),
                        ],
                        &["sequence", "n_entities", "fraction", "transitions", "entities"],
                        false,
                    )),
                ),
                ("threshold_sweep", sweep_schema()),
                ("next_tools", s::array(s::string())),
            ],
            &[
                "status",
                "detail",
                "longitudinal_id",
                "representation",
                "interpretation_threshold",
                "threshold_role",
                "n_entities",
                "n_times",
                "distinct_trajectories",
                "archetypes_returned",
                "archetypes_omitted",
                "archetypes",
                "threshold_sweep",
                "next_tools",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn archetypes(
    state: &ServerState,
    p: ArchetypesParams,
) -> Result<CallToolResult, ToolError> {
    if p.threshold < 0.0 {
        return Err(invalid(ARCHETYPES, "threshold must be >= 0."));
    }
    if p.max_archetypes < 1 {
        return Err(invalid(ARCHETYPES, "max_archetypes must be >= 1."));
    }
    let artifact = resolve(state, ARCHETYPES, &p.longitudinal_id)?;
    require_trajectory(ARCHETYPES, &artifact)?;
    let a = artifact.clone();
    let (threshold, max_a, max_e) = (p.threshold, p.max_archetypes, p.max_entities_per_archetype);
    let mut payload = tokio::task::spawn_blocking(move || {
        archetype_payload(
            a.trajectory.as_ref().expect("checked"),
            threshold,
            max_a,
            max_e,
        )
    })
    .await
    .map_err(|e| internal(ARCHETYPES, e))?;
    payload["status"] = json!("ok");
    payload["detail"] = json!(p.detail.as_str());
    payload["longitudinal_id"] = json!(artifact.longitudinal_id);
    Ok(render(payload, p.response_format, archetypes_markdown))
}

fn archetypes_markdown(payload: &Value) -> String {
    let mut lines = vec![
        format!(
            "# Trajectory Archetypes `{}`",
            payload["longitudinal_id"].as_str().unwrap_or_default()
        ),
        String::new(),
        format!(
            "- Interpretation threshold: {} ({})",
            py_text(&payload["interpretation_threshold"]),
            payload["threshold_role"].as_str().unwrap_or_default()
        ),
        format!(
            "- {} entities over {} time steps",
            payload["n_entities"], payload["n_times"]
        ),
        format!(
            "- {} distinct trajectories ({} not shown)",
            payload["distinct_trajectories"], payload["archetypes_omitted"]
        ),
        String::new(),
        "| Sequence | Entities | Share | Transitions | Example members |".to_owned(),
        "|---|---|---|---|---|".to_owned(),
    ];
    for arch in payload["archetypes"].as_array().into_iter().flatten() {
        let members: Vec<&str> = arch["entities"]["preview"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        lines.push(format!(
            "| {} | {} | {:.1}% | {} | {} |",
            py_list(&arch["sequence"]),
            arch["n_entities"],
            num(&arch["fraction"]) * 100.0,
            arch["transitions"],
            members.join(", ")
        ));
    }
    lines.extend([String::new(), "## Threshold ladder".to_owned()]);
    lines.extend(ladder_md(&payload["threshold_sweep"]));
    lines.extend(next_tools_md(payload));
    lines.join("\n")
}

// ---- get_cross_time_neighbors -------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DirectionParam {
    #[default]
    Any,
    Forward,
    Backward,
}

impl DirectionParam {
    fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Forward => "forward",
            Self::Backward => "backward",
        }
    }
}

fn twenty() -> usize {
    20
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct NeighborsParams {
    /// From build_longitudinal_graph. Omit for the most recent panel.
    #[serde(default)]
    pub longitudinal_id: String,
    /// Entity to look up (as text), used together with t. Ignored when observation_id is given.
    #[serde(default)]
    pub entity_id: String,
    /// Original label from the panel's time column for the source observation (not the positional index). Used with entity_id.
    #[serde(default)]
    pub t: Option<Value>,
    /// Positional observation id (an obs_id from an earlier response); alternative to entity_id plus t.
    #[serde(default)]
    pub observation_id: Option<i64>,
    /// Interpretation filter (>= 0) on edge weight; 0 keeps all.
    #[serde(default)]
    pub threshold: f64,
    /// Highest-weight neighbors to return (>= 1).
    #[serde(default = "twenty")]
    pub max_neighbors: usize,
    /// 'any' time step, 'forward' (later steps only) or 'backward' (earlier steps only).
    #[serde(default)]
    pub direction: DirectionParam,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    #[serde(default)]
    pub response_format: Format,
}

pub fn neighbors_spec() -> ToolSpec {
    ToolSpec {
        name: NEIGHBORS,
        title: "Cross-Time Neighbors",
        description: "Rank observations from other time steps that resemble one observation (\"who does this entity at this time look like at a different time?\"). Needs the trajectory representation. Identify the source by observation_id or by entity_id plus t; the response gives both time_label and the positional t index.",
        input_schema: input_schema::<NeighborsParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("longitudinal_id", s::string()),
                ("representation", s::enumeration(&["trajectory"])),
                ("surface", s::string()),
                ("interpretation_threshold", s::number()),
                ("direction", s::enumeration(&["any", "forward", "backward"])),
                (
                    "source",
                    s::object(
                        &[("obs_id", s::integer()), ("entity_id", s::string()), ("t", s::integer()), ("time_label", s::any())],
                        &["obs_id", "entity_id", "t"],
                        false,
                    ),
                ),
                ("neighbors_returned", s::integer()),
                ("neighbors_omitted", s::integer()),
                ("cross_time_degree", s::integer()),
                (
                    "neighbors",
                    s::array(s::object(
                        &[
                            ("obs_id", s::integer()),
                            ("entity_id", s::string()),
                            ("t", s::integer()),
                            ("time_label", s::any()),
                            ("delta_t", s::integer()),
                            ("weight", s::number()),
                        ],
                        &["obs_id", "entity_id", "t", "delta_t", "weight"],
                        false,
                    )),
                ),
                ("next_tools", s::array(s::string())),
            ],
            &[
                "status",
                "longitudinal_id",
                "representation",
                "surface",
                "interpretation_threshold",
                "direction",
                "source",
                "neighbors_returned",
                "neighbors_omitted",
                "cross_time_degree",
                "neighbors",
                "next_tools",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

/// Python `==` between a panel time label and a JSON argument.
fn key_equals(key: &Key, v: &Value) -> bool {
    match (key, v) {
        (Key::Text(a), Value::String(b)) => a == b,
        (Key::Int(a), Value::Number(n)) => n.as_f64() == Some(*a as f64),
        (Key::Float(a), Value::Number(n)) => n.as_f64() == Some(*a),
        (Key::Bool(a), Value::Bool(b)) => a == b,
        (Key::Bool(a), Value::Number(n)) => n.as_f64() == Some(f64::from(u8::from(*a))),
        (Key::Int(a), Value::Bool(b)) => *a == i64::from(*b),
        _ => false,
    }
}

pub async fn neighbors(
    state: &ServerState,
    p: NeighborsParams,
) -> Result<CallToolResult, ToolError> {
    if p.max_neighbors < 1 {
        return Err(invalid(NEIGHBORS, "max_neighbors must be >= 1."));
    }
    if p.threshold < 0.0 {
        return Err(invalid(NEIGHBORS, "threshold must be >= 0."));
    }
    let artifact = resolve(state, NEIGHBORS, &p.longitudinal_id)?;
    let ct = require_trajectory(NEIGHBORS, &artifact)?;
    let t = p.t.clone().filter(|v| !v.is_null());
    let obs_id: i64 = match p.observation_id {
        Some(id) => id,
        None => {
            let (Some(t), false) = (&t, p.entity_id.is_empty()) else {
                return Err(ToolError::new(
                    NEIGHBORS,
                    "OBSERVATION_NOT_SPECIFIED",
                    "Provide observation_id, or both entity_id and t.",
                    "Pass entity_id and t (from the panel's entity/time columns), or an observation_id from a previous response.",
                ));
            };
            let time_index = artifact
                .times
                .iter()
                .position(|k| key_equals(k, t))
                .or_else(|| {
                    artifact
                        .times
                        .iter()
                        .position(|k| k.to_string() == py_text(t))
                });
            let Some(time_index) = time_index else {
                return Err(ToolError::new(
                    NEIGHBORS,
                    "OBSERVATION_NOT_FOUND",
                    format!(
                        "No panel time label matches t={}.",
                        match t {
                            Value::String(s) => pulsar_analysis::plan::py_repr(s),
                            other => py_text(other),
                        }
                    ),
                    "Pass a time label from the panel's time_column.",
                )
                .detail("entity_id", json!(p.entity_id))
                .detail("t", t.clone())
                .detail("available_times", bounded_list(&artifact.times, 20)));
            };
            let matches: Vec<usize> = ct
                .obs
                .iter()
                .enumerate()
                .filter(|(_, o)| o.entity_id.to_string() == p.entity_id && o.t == time_index)
                .map(|(k, _)| k)
                .collect();
            if matches.len() != 1 {
                return Err(ToolError::new(
                    NEIGHBORS,
                    "OBSERVATION_NOT_FOUND",
                    format!("No unique observation for entity '{}' at t={}.", p.entity_id, py_text(t)),
                    "Check entity_id and the original time label against the panel; call diagnose_longitudinal_graph for the panel shape.",
                )
                .detail("entity_id", json!(p.entity_id))
                .detail("t", t.clone())
                .detail("matches", json!(matches.len())));
            }
            matches[0] as i64
        }
    };
    let n = ct.n_observations();
    if obs_id < 0 || obs_id as usize >= n {
        return Err(ToolError::new(
            NEIGHBORS,
            "OBSERVATION_NOT_FOUND",
            format!("observation_id {obs_id} is out of range."),
            format!("Valid observation ids are 0..{}.", n as i64 - 1),
        )
        .detail("observation_id", json!(obs_id)));
    }
    let a = artifact.clone();
    let (threshold, max_n, direction) = (p.threshold, p.max_neighbors, p.direction.as_str());
    let mut payload = tokio::task::spawn_blocking(move || {
        cross_time_payload(
            a.trajectory.as_ref().expect("checked"),
            obs_id as usize,
            threshold,
            max_n,
            direction,
        )
    })
    .await
    .map_err(|e| internal(NEIGHBORS, e))?;
    payload["status"] = json!("ok");
    payload["longitudinal_id"] = json!(artifact.longitudinal_id);
    Ok(render(payload, p.response_format, neighbors_markdown))
}

fn neighbors_markdown(payload: &Value) -> String {
    let source = &payload["source"];
    let mut lines = vec![
        format!(
            "# Cross-Time Neighbors of entity `{}` at time `{}` (trajectory index t={})",
            source["entity_id"].as_str().unwrap_or_default(),
            py_text(&source["time_label"]),
            source["t"]
        ),
        String::new(),
        format!(
            "- Surface: {}",
            payload["surface"].as_str().unwrap_or_default()
        ),
        format!(
            "- Interpretation threshold: {}, direction `{}`",
            py_text(&payload["interpretation_threshold"]),
            payload["direction"].as_str().unwrap_or_default()
        ),
        format!(
            "- Cross-time degree: {} ({} not shown)",
            payload["cross_time_degree"], payload["neighbors_omitted"]
        ),
        String::new(),
        "| Entity | Time label | t index | delta steps | Weight |".to_owned(),
        "|---|---|---|---|---|".to_owned(),
    ];
    for nb in payload["neighbors"].as_array().into_iter().flatten() {
        lines.push(format!(
            "| {} | {} | {} | {:+} | {:.4} |",
            nb["entity_id"].as_str().unwrap_or_default(),
            py_text(&nb["time_label"]),
            nb["t"],
            nb["delta_t"].as_i64().unwrap_or(0),
            num(&nb["weight"])
        ));
    }
    lines.extend(next_tools_md(payload));
    lines.join("\n")
}

// ---- classify_trajectories ----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClassifyMethod {
    #[default]
    Complexity,
    Transition,
    Sequence,
    Levenshtein,
    Dtw,
}

impl ClassifyMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::Complexity => "complexity",
            Self::Transition => "transition",
            Self::Sequence => "sequence",
            Self::Levenshtein => "levenshtein",
            Self::Dtw => "dtw",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Complexity => "Shannon Entropy & Transition Counts (Stable vs Volatile)",
            Self::Transition => "Markov Self-Retention Probabilities",
            Self::Sequence => "Structural Path Profile (Singleton vs Multistate)",
            Self::Levenshtein => "Levenshtein Edit Distance Complete-Linkage Clustering",
            Self::Dtw => "Dynamic Time Warping Complete-Linkage Clustering",
        }
    }
}

fn classify_threshold() -> f64 {
    0.15
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ClassifyParams {
    /// From build_longitudinal_graph. Omit for the most recent panel.
    #[serde(default)]
    pub longitudinal_id: String,
    /// 'complexity': entropy and transition count. 'transition': Markov self-retention. 'sequence': path shape. 'levenshtein'/'dtw': complete-linkage clustering of pairwise distance into 4 cohorts (O(n^2)).
    #[serde(default)]
    pub method: ClassifyMethod,
    /// Interpretation filter (>= 0) on CO_SIMILAR edges that defines the clusters whose per-entity sequences are classified.
    #[serde(default = "classify_threshold")]
    pub threshold: f64,
    /// 'summary' (default) returns the per-class summary plus a 25-entity preview. 'full' adds the complete per-entity `classification` map (O(entities)).
    #[serde(default)]
    pub detail: Detail,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    #[serde(default)]
    pub response_format: Format,
}

pub fn classify_spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: CLASSIFY,
        title: "Classify Trajectories",
        description: "Classify each entity's cluster path through time into trajectory classes. Returns a per-class summary plus a bounded entity preview (all entities with detail=\"full\"). Needs the trajectory representation. Use get_trajectory_archetypes to group by exact sequence; this tool gives rule- or distance-based classes.",
        input_schema: input_schema::<ClassifyParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("longitudinal_id", s::string()),
                ("representation", s::enumeration(&["trajectory"])),
                ("method", s::enumeration(&["complexity", "transition", "sequence", "levenshtein", "dtw"])),
                ("interpretation_threshold", s::number()),
                ("n_entities", s::integer()),
                ("classes_summary", open.clone()),
                ("detail", s::enumeration(&["summary", "full"])),
                ("classification_preview", open.clone()),
                ("entities_returned", s::integer()),
                ("entities_omitted", s::integer()),
                ("classification", open),
                ("next_tools", s::array(s::string())),
            ],
            &[
                "status",
                "longitudinal_id",
                "representation",
                "method",
                "interpretation_threshold",
                "n_entities",
                "classes_summary",
                "detail",
                "classification_preview",
                "entities_returned",
                "entities_omitted",
                "next_tools",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn classify(state: &ServerState, p: ClassifyParams) -> Result<CallToolResult, ToolError> {
    if p.threshold < 0.0 {
        return Err(invalid(CLASSIFY, "threshold must be >= 0."));
    }
    let artifact = resolve(state, CLASSIFY, &p.longitudinal_id)?;
    require_trajectory(CLASSIFY, &artifact)?;
    let a = artifact.clone();
    let (method, threshold) = (p.method, p.threshold);
    let mut payload = tokio::task::spawn_blocking(move || {
        classify_trajectories_payload(
            a.trajectory.as_ref().expect("checked"),
            method.as_str(),
            threshold,
        )
    })
    .await
    .map_err(|e| internal(CLASSIFY, e))?
    .map_err(|e| internal(CLASSIFY, e))?;
    let full = match payload
        .as_object_mut()
        .and_then(|m| m.shift_remove("classification"))
    {
        Some(Value::Object(m)) => m,
        _ => Map::new(),
    };
    let preview: Map<String, Value> = full
        .iter()
        .take(CLASSIFICATION_PREVIEW_LIMIT)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let is_full = p.detail == Detail::Full;
    payload["status"] = json!("ok");
    payload["longitudinal_id"] = json!(artifact.longitudinal_id);
    payload["detail"] = json!(p.detail.as_str());
    payload["entities_returned"] = json!(if is_full { full.len() } else { preview.len() });
    payload["entities_omitted"] = json!(if is_full {
        0
    } else {
        full.len() - preview.len()
    });
    payload["classification_preview"] = Value::Object(preview);
    if is_full {
        payload["classification"] = Value::Object(full);
    }
    Ok(render(payload, p.response_format, |v| {
        classification_markdown(v, method)
    }))
}

fn classification_markdown(payload: &Value, method: ClassifyMethod) -> String {
    let mut lines = vec![
        format!(
            "# Trajectory Classification `{}`",
            payload["longitudinal_id"].as_str().unwrap_or_default()
        ),
        String::new(),
        format!("- **Method**: `{}` ({})", method.as_str(), method.label()),
        format!(
            "- **Interpretation Threshold**: {}",
            py_text(&payload["interpretation_threshold"])
        ),
        format!("- **Entities Classified**: {}", payload["n_entities"]),
        String::new(),
        "## Cohort Summary".to_owned(),
        String::new(),
        "| Cohort Class | Entity Count | Population Share |".to_owned(),
        "|---|---|---|".to_owned(),
    ];
    let mut classes: Vec<(&String, &Value)> = payload["classes_summary"]
        .as_object()
        .into_iter()
        .flatten()
        .collect();
    classes.sort_by_key(|(_, info)| std::cmp::Reverse(info["count"].as_u64().unwrap_or(0)));
    for (class, info) in classes {
        lines.push(format!(
            "| {class} | {} | {:.1}% |",
            info["count"],
            num(&info["fraction"]) * 100.0
        ));
    }
    lines.extend([
        String::new(),
        "## Entity Classification Samples (first 15)".to_owned(),
        String::new(),
        "| Entity ID | Cohort Class | Visits | Sequence (Observed) |".to_owned(),
        "|---|---|---|---|".to_owned(),
    ]);
    let preview = payload["classification_preview"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    for (id, info) in preview.iter().take(15) {
        lines.push(format!(
            "| {id} | {} | {} | {} |",
            info["class"].as_str().unwrap_or_default(),
            info["length"],
            py_list(&info["sequence"])
        ));
    }
    let shown = preview.len().min(15) as u64;
    let n = payload["n_entities"].as_u64().unwrap_or(0);
    if n > shown {
        lines.push(format!(
            "| ... | and {} more entities (detail=\"full\" returns all) | | |",
            n - shown
        ));
    }
    lines.extend(next_tools_md(payload));
    lines.join("\n")
}

// ---- sample_longitudinal_entities ---------------------------------------------------------

fn two_thousand() -> i64 {
    2000
}

fn three() -> i64 {
    3
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SampleParams {
    /// Source dataset handle returned by ingest_dataset.
    pub dataset_id: String,
    /// Column that identifies the repeated entity (e.g. a patient or device id); all rows of a sampled entity are kept.
    pub entity_column: String,
    /// Maximum number of entities to keep (>= 1).
    #[serde(default = "two_thousand")]
    pub max_entities: i64,
    /// Minimum number of rows an entity needs to be eligible.
    #[serde(default = "three")]
    pub min_observations: i64,
    /// Optional discrete time column; required when min_time_points is set.
    #[serde(default)]
    pub time_column: Option<String>,
    /// Optional minimum number of distinct time_column values an entity needs to be eligible.
    #[serde(default)]
    pub min_time_points: Option<i64>,
}

pub fn sample_spec() -> ToolSpec {
    ToolSpec {
        name: SAMPLE,
        title: "Sample Longitudinal Entities",
        description: "Create a deterministic, row-preserving sample of longitudinal entities. Returns a NEW `dataset_id` (registered, Parquet) plus eligible/sampled counts; no rows or identifiers are returned. Use before build_longitudinal_graph on large panels. The source must already be ingested.",
        input_schema: input_schema::<SampleParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("dataset_id", s::string()),
                ("source_dataset_id", s::string()),
                ("entity_column", s::string()),
                ("max_entities", s::integer()),
                ("min_observations", s::integer()),
                ("time_column", s::nullable(s::string())),
                ("min_time_points", s::nullable(s::integer())),
                ("eligible_entities", s::integer()),
                ("sampled_entities", s::integer()),
                ("sampled_rows", s::integer()),
            ],
            &[
                "dataset_id",
                "source_dataset_id",
                "entity_column",
                "max_entities",
                "min_observations",
                "eligible_entities",
                "sampled_entities",
                "sampled_rows",
            ],
            false,
        )),
        read_only: false,
        idempotent: true,
    }
}

pub async fn sample(state: &ServerState, p: SampleParams) -> Result<CallToolResult, ToolError> {
    let bad_tp = p.min_time_points.is_some_and(|v| v < 1);
    if p.max_entities < 1
        || p.min_observations < 1
        || bad_tp
        || (p.min_time_points.is_some() && p.time_column.is_none())
    {
        return Err(ToolError::new(
            SAMPLE,
            "INVALID_SAMPLE_SIZE",
            "max_entities and min_observations must be positive, and min_time_points requires a positive value and time_column.",
            "Use positive values and provide time_column when setting min_time_points.",
        ));
    }
    let (source_id, path) = resolve_dataset(state, SAMPLE, &p.dataset_id)?.ok_or_else(|| {
        ToolError::new(
            SAMPLE,
            "DATASET_ID_UNKNOWN",
            "Unknown dataset_id ''.",
            "Create or retrieve a valid dataset_id before retrying this tool.",
        )
        .detail("dataset_id", json!(""))
    })?;
    let table = load(SAMPLE, &path).await?;
    if table.column(&p.entity_column).is_none() {
        return Err(ToolError::new(
            SAMPLE,
            "ENTITY_COLUMN_UNKNOWN",
            format!(
                "Entity column '{}' is not present in the dataset.",
                p.entity_column
            ),
            "Call characterize_dataset or probe_columns and pass an existing entity column.",
        )
        .detail("dataset_id", json!(source_id))
        .detail("entity_column", json!(p.entity_column)));
    }
    if let Some(tc) = &p.time_column {
        if table.column(tc).is_none() {
            return Err(ToolError::new(
                SAMPLE,
                "TIME_COLUMN_UNKNOWN",
                format!("Time column '{tc}' is not present in the dataset."),
                "Pass an existing discrete time column or omit time_column.",
            )
            .detail("dataset_id", json!(source_id))
            .detail("time_column", json!(tc)));
        }
    }
    let (selected, eligible, rows) = select_longitudinal_entities(
        &table,
        &p.entity_column,
        p.max_entities as usize,
        p.min_observations as usize,
        p.time_column.as_deref(),
        p.min_time_points.map(|v| v as usize),
    );
    if rows.is_empty() {
        return Err(ToolError::new(
            SAMPLE,
            "NO_ELIGIBLE_ENTITIES",
            "No entities meet the requested minimum observation and timepoint counts.",
            "Lower the minimum counts or use different entity and time columns.",
        )
        .detail("dataset_id", json!(source_id))
        .detail("entity_column", json!(p.entity_column))
        .detail("min_observations", json!(p.min_observations))
        .detail("time_column", json!(p.time_column))
        .detail("min_time_points", json!(p.min_time_points)));
    }
    let sampled = table.take_rows(&rows);
    let store = state.store.clone();
    let stem = std::path::Path::new(&path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "dataset".into());
    let record = tokio::task::spawn_blocking(move || {
        let staging = store.root().join(format!(
            "sample_{}.parquet",
            LongitudinalId::random().as_str()
        ));
        pulsar_data::write_parquet(&sampled, &staging).map_err(|e| e.to_string())?;
        let out = store
            .register_dataset_from_file(
                &format!("{stem}_sample.parquet"),
                &staging,
                DatasetSource::EntitySample,
            )
            .map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&staging);
        out
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r)
    .map_err(|e| {
        ToolError::new(
            SAMPLE,
            "SAMPLE_FAILED",
            e,
            "Check the dataset file and column names, then retry.",
        )
    })?;
    Ok(ok_result(
        json!({
            "dataset_id": record.dataset_id,
            "source_dataset_id": source_id,
            "entity_column": p.entity_column,
            "max_entities": p.max_entities,
            "min_observations": p.min_observations,
            "time_column": p.time_column,
            "min_time_points": p.min_time_points,
            "eligible_entities": eligible,
            "sampled_entities": selected.len(),
            "sampled_rows": rows.len(),
        }),
        None,
    ))
}
