//! `generate_cluster_dossier` (`pulsar/mcp/tools/clustering.py`): cluster
//! the latest run, build the evidence dossier, persist the assignment and
//! publish the labels into the session for the other cluster tools.
//!
//! Publication follows the Python stage/commit protocol: the O(rows) work
//! and the assignment write run on a blocking thread, and the assignment is
//! committed (two renames) only if the session's latest run is still the one
//! the labels were computed on. A refused or cancelled dossier drops its
//! [`StagedClusterAssignment`], so nothing is persisted.
//!
//! Known divergences from Python:
//! - Python also refused a publish when the session's data frame had been
//!   rebound under an unchanged run id. Here the table belongs to the live
//!   run, so the run check covers it.
//! - The evidence cache is keyed by comparing its inputs instead of a
//!   SHA-256 of their JSON; the fingerprint was never in a payload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pulsar_analysis::characterization::{
    group_summary, GroupSummaryThresholds, GROUP_SUMMARY_DEFAULTS,
};
use pulsar_analysis::cluster_payloads::{
    build_cluster_selection, build_evidence_payload, build_summary_evidence_payload,
    cluster_result_payload, cluster_selection_to_markdown, prepend_threshold_markdown,
    summary_evidence_payload_to_markdown, threshold_surface_payload, two_lever_markdown_note,
};
use pulsar_analysis::clustering::resolve::{
    resolve_clusters_with_threshold_graph, ResolveError, ResolveMethod, SpectralCutDiagnostics,
};
use pulsar_analysis::dossier::{
    build_dossier_with_central_rows, dossier_to_markdown, TopologicalDossier,
};
use pulsar_analysis::feature_evidence::{build_feature_evidence_index, FeatureEvidenceIndex};
use pulsar_analysis::payloads::{bounded_list, size_summary};
use pulsar_analysis::plan::row_index_like;
use pulsar_analysis::GraphAnalysis;
use pulsar_store::{
    ClusterAssignment, ClusterAssignmentId, ClusterAssignmentRecord, RunId,
    StagedClusterAssignment, Store,
};
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

use super::params::Format;
use super::{dossier_budget, ToolSpec};
use crate::envelope::ToolError;
use crate::phirewall;
use crate::schema::{self as s, input_schema};
use crate::state::{
    CachedResolution, ClusterState, EvidenceKey, LiveRun, RunGraphCache, RunView, ServerState,
};

pub const NAME: &str = "generate_cluster_dossier";

const THRESHOLD_RANGE_MESSAGE: &str = "Threshold must be finite and between 0.0 and 1.0.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MethodParam {
    #[default]
    Auto,
    Spectral,
    Components,
    Assignment,
}

impl MethodParam {
    /// The cut to run; `None` for a stored assignment, which is not re-cut.
    fn resolve(self) -> Option<ResolveMethod> {
        match self {
            Self::Auto => Some(ResolveMethod::Auto),
            Self::Spectral => Some(ResolveMethod::Spectral),
            Self::Components => Some(ResolveMethod::Components),
            Self::Assignment => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DossierDetail {
    #[default]
    Summary,
    Standard,
    Full,
}

impl DossierDetail {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Standard => "standard",
            Self::Full => "full",
        }
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct Params {
    /// 'components': components after dropping edges <= the interpretation threshold. 'spectral': weighted-affinity cut (may fail NO_STABLE_SPECTRAL_CUT). 'auto' (default): 'components' if the threshold > 0, else an H0 stability cut then components or spectral. 'assignment': evidence on cluster_assignment_id's stored labels (e.g. from merge_small_groups), no re-cut.
    pub method: MethodParam,
    /// Stored clustering to build on; required for method='assignment', ignored otherwise. Its run must be the session's latest.
    pub cluster_assignment_id: Option<String>,
    /// Largest k tried by the spectral search (2..max_k); ignored for 'components'. Default 15.
    pub max_k: usize,
    /// Edge-weight cutoff in [0, 1]; edges <= it are dropped before clustering. Omit to inherit the construction threshold (spectral: 0.0).
    pub interpretation_edge_weight_threshold: Option<f64>,
    /// 'summary' (default): headline evidence per cluster. 'standard': adds feature evidence (heavy). 'full': lossless audit payload (heaviest).
    pub detail: DossierDetail,
    /// Clusters shown, largest first (>= 1); clusters_omitted counts the rest. Default 8.
    pub max_clusters: usize,
    /// Feature hints shown per cluster. Default 5.
    pub feature_preview_limit: usize,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    pub response_format: Format,
    /// Column names hidden from the report; does not change the model or clustering.
    pub exclude_columns: Vec<String>,
    /// Run to cluster, from run_topological_sweep. Omit for the latest; must be the latest run or SESSION_REVISION_CONFLICT.
    pub run_id: String,
    /// Summary grouping: groups listed, largest first (>= 1). Default 6.
    pub summary_max_groups: Option<usize>,
    /// Summary grouping: features listed, strongest first (>= 1). Default 5.
    pub summary_max_features: Option<usize>,
    /// Summary grouping: smallest |sd| called a difference (finite, >= 0). Default 0.5.
    pub summary_min_sd: Option<f64>,
    /// Summary grouping: smallest group/level shown (>= 8, default; can only be raised).
    pub summary_min_rows: Option<usize>,
    /// Summary grouping: differences per group (>= 1). Default 3.
    pub summary_max_traits: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            method: MethodParam::Auto,
            cluster_assignment_id: None,
            max_k: 15,
            interpretation_edge_weight_threshold: None,
            detail: DossierDetail::Summary,
            max_clusters: 8,
            feature_preview_limit: 5,
            response_format: Format::Markdown,
            exclude_columns: Vec::new(),
            run_id: String::new(),
            summary_max_groups: None,
            summary_max_features: None,
            summary_min_sd: None,
            summary_min_rows: None,
            summary_max_traits: None,
        }
    }
}

pub fn spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: NAME,
        title: "Cluster Evidence Dossier",
        description: "Cluster the latest run's graph and build a statistical evidence dossier. Persists a cluster_assignment_id and publishes the labels into the session for cluster tools and exports; a new sweep clears them. Start with detail='summary'; escalate after narrowing focus. An oversized result is truncated, with truncation.next (get_cluster_detail) paging the rest, rather than failing. method='assignment' with cluster_assignment_id builds evidence on stored labels without re-clustering, e.g. the assignment merge_small_groups returns.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("run_id", s::string()),
                ("clusters_run_id", s::nullable(s::string())),
                ("construction_threshold", s::number()),
                ("interpretation_edge_weight_threshold", s::number()),
                ("interpretation_edge_weight_threshold_applied", s::number()),
                ("method_applied", s::string()),
                ("cluster_count", s::integer()),
                (
                    "profiled_features",
                    s::array(s::object(
                        &[
                            ("feature", s::integer()),
                            ("column", s::string()),
                            ("used_for_grouping", s::boolean()),
                        ],
                        &["feature", "column", "used_for_grouping"],
                        false,
                    )),
                ),
                ("threshold_inherited", s::boolean()),
                ("threshold_source", s::string()),
                ("threshold_surface", open.clone()),
                ("cluster_assignment", open.clone()),
                ("cluster_assignment_id", s::string()),
                ("cluster_selection", open.clone()),
                (phirewall::KEY, open.clone()),
                ("clusters", s::array(open.clone())),
                ("recommended_next_tools", s::array(s::string())),
                ("detail_hint", s::string()),
                ("surface_labels_ref", s::string()),
                ("surface_labels", open.clone()),
                ("cluster_index", open),
                ("truncation", s::truncation()),
            ],
            &[
                "status",
                "clusters_run_id",
                "interpretation_edge_weight_threshold",
                "threshold_surface",
                "cluster_assignment_id",
                "clusters",
                "recommended_next_tools",
            ],
            // summary/standard/full add differently shaped evidence blocks
            true,
        )),
        read_only: false,
        idempotent: false,
    }
}

pub(crate) fn surface_labels() -> Value {
    json!({
        "graph_metrics": "Computed on the persisted cosmic graph @ construction_threshold (the fitted graph). Components, density, and singletons describe this graph.",
        "clusters": "Computed on the interpretation slice @ interpretation_edge_weight_threshold of the [0,1]-normalized weighted adjacency (sparsified only when enabled). Cluster counts and membership reflect this slice, not the persisted graph.",
    })
}

fn assignment_summary(record: &ClusterAssignmentRecord) -> Value {
    json!({
        "cluster_assignment_id": record.cluster_assignment_id,
        "run_id": record.run_id,
        "dataset_id": record.dataset_id,
        "method": record.method,
        "interpretation_edge_weight_threshold": record.interpretation_edge_weight_threshold,
        "threshold_source": record.threshold_source,
        "construction_threshold": record.construction_threshold,
        "label_count": record.label_count,
        "cluster_ids": bounded_list(&record.cluster_ids, 20),
    })
}

fn assignment_markdown(record: &ClusterAssignmentRecord) -> String {
    let show = |v: Option<String>| v.unwrap_or_else(|| "None".into());
    [
        "## Cluster Assignment Artifact".to_owned(),
        String::new(),
        format!("- `cluster_assignment_id`: `{}`", record.cluster_assignment_id),
        format!("- `run_id`: `{}`", show(record.run_id.as_ref().map(|r| r.to_string()))),
        format!(
            "- `dataset_id`: `{}`",
            show(record.dataset_id.as_ref().map(|d| d.to_string()))
        ),
        "- Use this handle with `export_labeled_data` to export this exact clustering, including its method and interpretation threshold.".to_owned(),
        String::new(),
    ]
    .join("\n")
}

/// Columns the dossier never profiles: the requested exclusions, the
/// identifiers the run's preprocessing plan names, and row indexes.
pub(crate) fn excluded_columns(live: &LiveRun, requested: &[String]) -> Vec<String> {
    let mut out = requested.to_vec();
    let row_indexes = live
        .table
        .names
        .iter()
        .zip(&live.table.columns)
        .filter(|(name, column)| row_index_like(name, column))
        .map(|(name, _)| name);
    for name in live.identifier_columns.iter().chain(row_indexes) {
        if !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

/// The group_summary thresholds: the defaults with the requested overrides.
fn summary_thresholds(p: &Params) -> Result<GroupSummaryThresholds, ToolError> {
    let d = GROUP_SUMMARY_DEFAULTS;
    let list = |name: &str, value: Option<usize>, min: usize, default: usize| match value {
        Some(v) if v < min => Err(ToolError::invalid_argument(
            NAME,
            format!("{name} must be >= {min}, got '{v}'"),
            format!("Pass {name} >= {min}, or omit it for {default}."),
        )),
        v => Ok(v.unwrap_or(default)),
    };
    let min_sd = p.summary_min_sd.unwrap_or(d.min_sd);
    if !(min_sd.is_finite() && min_sd >= 0.0) {
        return Err(ToolError::invalid_argument(
            NAME,
            format!("summary_min_sd must be a finite number >= 0, got '{min_sd}'"),
            format!("Pass summary_min_sd >= 0, or omit it for {}.", d.min_sd),
        ));
    }
    let min_rows = p.summary_min_rows.unwrap_or(d.min_rows);
    if min_rows < d.min_rows {
        // A privacy floor: groups and levels this small are never shown.
        return Err(ToolError::invalid_argument(
            NAME,
            format!(
                "summary_min_rows must be >= {}, got '{min_rows}'",
                d.min_rows
            ),
            format!(
                "Pass summary_min_rows >= {} or omit it; the floor can only be raised.",
                d.min_rows
            ),
        ));
    }
    Ok(GroupSummaryThresholds {
        max_groups: list("summary_max_groups", p.summary_max_groups, 1, d.max_groups)?,
        max_features: list(
            "summary_max_features",
            p.summary_max_features,
            1,
            d.max_features,
        )?,
        min_sd,
        min_rows,
        max_traits: list("summary_max_traits", p.summary_max_traits, 1, d.max_traits)?,
    })
}

fn unit_threshold(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn failed(reason: impl Into<String>) -> ToolError {
    ToolError::new(
        NAME,
        "DOSSIER_FAILED",
        reason,
        "Check the reason, adjust method / interpretation threshold, and retry.",
    )
}

fn resolve_error(e: ResolveError, detail: DossierDetail) -> ToolError {
    let ResolveError::NoStableSpectralCut(diag) = e else {
        return failed(e.to_string());
    };
    let details = diag.details(detail == DossierDetail::Full);
    let show = |v: &Value| match v {
        Value::Null => "None".to_owned(),
        other => other.to_string(),
    };
    let reason = format!(
        "{} Spectral search ran (k={}-{}) but no evaluated cut exceeded the acceptance floor {}; best candidate k={}, silhouette={}.",
        SpectralCutDiagnostics::REASON,
        show(&details["k_min"]),
        show(&details["max_k"]),
        show(&details["accepted_silhouette_min"]),
        show(&details["best_k"]),
        show(&details["best_silhouette_score"]),
    );
    let mut out = ToolError::new(
        NAME,
        SpectralCutDiagnostics::ERROR_CODE,
        reason,
        "Try method='components' with an explicit interpretation_edge_weight_threshold (see get_threshold_stability_curve), or treat the graph as one continuous structure.",
    );
    if let Value::Object(map) = details {
        out.details.extend(map);
    }
    out
}

/// What the blocking worker computes besides the staged assignment.
struct Computed {
    labels: Vec<i64>,
    meta: Value,
    evidence: Arc<FeatureEvidenceIndex>,
    evidence_key: EvidenceKey,
    /// `_graph_metrics_payload`: component sizes as a bounded summary.
    graph_metrics: Value,
    dossier: Option<TopologicalDossier>,
    dossiers: HashMap<String, TopologicalDossier>,
    resolution: Option<CachedResolution>,
    group_summary: Value,
    summary_thresholds: GroupSummaryThresholds,
}

/// Inputs of the blocking worker, captured before it starts.
struct Job {
    live: Arc<LiveRun>,
    graph_cache: Arc<RunGraphCache>,
    store: Store,
    run_id: RunId,
    cached: Option<Arc<ClusterState>>,
    /// `None` when `given` supplies the labels.
    method: Option<ResolveMethod>,
    max_k: usize,
    interpretation_threshold: f64,
    threshold_source: String,
    construction_threshold: f64,
    n_ball_maps: usize,
    minhash_d: Option<usize>,
    exclude_columns: Vec<String>,
    /// Columns the preprocessing dropped: profiled but not used for grouping.
    dropped: Vec<String>,
    summary_thresholds: GroupSummaryThresholds,
    max_clusters_to_characterize: usize,
    detail: DossierDetail,
    /// Stored labels to build on instead of clustering (method='assignment').
    given: Option<ClusterAssignmentRecord>,
    given_labels: Vec<i64>,
}

/// Where the published assignment comes from.
enum Staging {
    /// A clustering computed now, written but not yet visible.
    New(StagedClusterAssignment),
    /// A stored assignment the dossier was built on.
    Existing(ClusterAssignmentRecord),
}

/// What the clustering step settled: labels plus the facts a payload reports.
struct Resolved {
    labels: Vec<i64>,
    method_used: String,
    n_clusters: usize,
    silhouette: Option<f64>,
    threshold_applied: f64,
    plateaus: Value,
    failure_reason: Option<String>,
    resolution: Option<CachedResolution>,
}

fn compute(job: Job) -> Result<(Computed, Staging), ToolError> {
    let fitted = &job.live.fitted;
    let table = &job.live.table;
    let affinity = &job.graph_cache.affinity;
    let result = match (&job.given, job.method) {
        (Some(record), _) => {
            let labels = job.given_labels.clone();
            let resolution = job
                .cached
                .as_ref()
                .filter(|c| c.run_id == job.run_id && c.labels == labels)
                .and_then(|c| c.resolution.clone());
            Resolved {
                n_clusters: record.cluster_ids.len(),
                labels,
                method_used: record.method.clone(),
                silhouette: None,
                threshold_applied: record.interpretation_edge_weight_threshold,
                plateaus: Value::Null,
                failure_reason: None,
                resolution,
            }
        }
        (None, method) => {
            let method = method.expect("a clustering method");
            if let Some((cached, res)) = job.cached.as_ref().and_then(|c| {
                let r = c.resolution.as_ref()?;
                (c.run_id == job.run_id
                    && r.method == method
                    && r.max_k == job.max_k
                    && r.interpretation_threshold == job.interpretation_threshold)
                    .then(|| (c, r.clone()))
            }) {
                Resolved {
                    labels: cached.labels.clone(),
                    method_used: res.method_used.clone(),
                    n_clusters: res.n_clusters,
                    silhouette: res.silhouette,
                    threshold_applied: res.threshold_applied,
                    plateaus: res.plateaus.clone(),
                    failure_reason: res.failure_reason.clone(),
                    resolution: Some(res),
                }
            } else {
                let r = resolve_clusters_with_threshold_graph(
                    affinity,
                    &job.graph_cache.threshold_graph,
                    fitted
                        .stability
                        .as_ref()
                        .or(job.graph_cache.stability.as_deref()),
                    method,
                    job.max_k,
                    job.interpretation_threshold,
                )
                .map_err(|e| resolve_error(e, job.detail))?;
                let labels: Vec<i64> = r.labels.iter().map(|&l| l as i64).collect();
                let method_used = r.method_used.to_owned();
                let n_clusters = r.n_clusters;
                let silhouette = r.silhouette_score;
                let threshold_applied = r.interpretation_edge_weight_threshold_applied;
                let plateaus = r
                    .stability_plateaus
                    .as_ref()
                    .map_or(Value::Null, |p| json!(p));
                let failure_reason = r.failure_reason.clone();
                let resolution = Some(CachedResolution {
                    method,
                    max_k: job.max_k,
                    interpretation_threshold: job.interpretation_threshold,
                    method_used: method_used.clone(),
                    n_clusters,
                    silhouette,
                    threshold_applied,
                    plateaus: plateaus.clone(),
                    failure_reason: failure_reason.clone(),
                });
                Resolved {
                    labels,
                    method_used,
                    n_clusters,
                    silhouette,
                    threshold_applied,
                    plateaus,
                    failure_reason,
                    resolution,
                }
            }
        }
    };
    let labels = result.labels.clone();
    let meta = cluster_result_payload(
        &labels,
        &result.method_used,
        result.n_clusters,
        result.silhouette,
        result.threshold_applied,
        &result.plateaus,
        result.failure_reason.as_deref(),
    );

    let mut sorted_exclude = job.exclude_columns.clone();
    sorted_exclude.sort();
    let evidence_key = EvidenceKey {
        method: result.method_used.clone(),
        interpretation_edge_weight_threshold: result.threshold_applied,
        exclude_columns: sorted_exclude,
        max_clusters_to_characterize: job.max_clusters_to_characterize,
    };
    // Evidence and dossiers depend only on (run_id, labels, exclude_columns, max_clusters).
    let evidence_hit = job.cached.as_ref().filter(|c| {
        c.run_id == job.run_id
            && c.labels == labels
            && c.evidence_key.exclude_columns == evidence_key.exclude_columns
            && c.evidence_key.max_clusters_to_characterize
                == evidence_key.max_clusters_to_characterize
    });
    let (evidence, mut dossiers) = match evidence_hit {
        Some(c) => {
            let map = c.dossiers.lock().unwrap_or_else(|e| e.into_inner()).clone();
            (c.evidence.clone(), map)
        }
        None => (
            Arc::new(
                build_feature_evidence_index(
                    table,
                    &labels,
                    &fitted.graph.edges,
                    &job.exclude_columns,
                    Some(job.max_clusters_to_characterize),
                )
                .map_err(|e| failed(e.to_string()))?,
            ),
            HashMap::new(),
        ),
    };

    // Python logs and degrades to an empty block when diagnosis fails.
    let metrics = job.graph_cache.metrics.clone().or_else(|| {
        GraphAnalysis::new(&fitted.graph).ok().map(|a| {
            a.metrics(
                fitted.construction_threshold,
                job.n_ball_maps,
                Some(&affinity.edges),
                job.minhash_d,
            )
        })
    });
    let full_metrics = metrics
        .as_ref()
        .and_then(|m| serde_json::to_value(m).ok())
        .unwrap_or_else(|| json!({}));
    let dossier = match job.detail {
        DossierDetail::Summary => None,
        detail => {
            let key = detail.as_str();
            if let Some(cached_dossier) = dossiers.get(key) {
                Some(cached_dossier.clone())
            } else {
                let existing_central: Option<HashMap<i64, Vec<Value>>> =
                    dossiers.values().next().map(|d| {
                        d.clusters
                            .iter()
                            .map(|c| (c.cluster_id, c.central_rows.clone()))
                            .collect()
                    });
                let built = build_dossier_with_central_rows(
                    &evidence,
                    table,
                    &labels,
                    &fitted.graph.edges,
                    key,
                    full_metrics.clone(),
                    existing_central.as_ref(),
                )
                .map_err(|e| failed(e.to_string()))?;
                dossiers.insert(key.to_owned(), built.clone());
                Some(built)
            }
        }
    };
    let mut graph_metrics = full_metrics;
    if let Value::Object(map) = &mut graph_metrics {
        map.remove("component_sizes");
        let sizes = metrics.map(|m| m.component_sizes).unwrap_or_default();
        map.insert("component_sizes_summary".into(), size_summary(&sizes, 20));
    }

    // The key holds the method, cut-off and hidden columns; the dropped
    // columns are the run's, so only the thresholds can still differ.
    let group_summary = match evidence_hit.filter(|c| c.evidence_key == evidence_key) {
        Some(c) if c.summary_thresholds == job.summary_thresholds => c.group_summary.clone(),
        _ => serde_json::to_value(group_summary(
            table,
            &labels,
            &job.exclude_columns,
            &job.dropped,
            // The summary's method is a code word: `merged:x` is `merged_x`.
            &result.method_used.replace(':', "_"),
            result.threshold_applied,
            &job.summary_thresholds,
        ))
        .map_err(|e| failed(e.to_string()))?,
    };

    let staged = match job.given {
        Some(record) => Staging::Existing(record),
        None => Staging::New(
            job.store
                .stage_cluster_assignment(ClusterAssignment {
                    run_id: Some(job.run_id.clone()),
                    dataset_id: job.live.dataset_id.clone(),
                    method: result.method_used.clone(),
                    interpretation_edge_weight_threshold: job.interpretation_threshold,
                    threshold_source: job.threshold_source.to_owned(),
                    construction_threshold: job.construction_threshold,
                    labels: &labels,
                })
                .map_err(|e| ToolError::from_store(NAME, e))?,
        ),
    };
    Ok((
        Computed {
            labels,
            meta,
            evidence,
            evidence_key,
            graph_metrics,
            dossier,
            dossiers,
            resolution: result.resolution,
            group_summary,
            summary_thresholds: job.summary_thresholds,
        },
        staged,
    ))
}

/// Commit `staged` and publish the labels, unless the session's latest run
/// moved on while they were computed. On refusal `staged` is dropped, which
/// discards it.
fn publish(
    state: &ServerState,
    expected_run_id: &RunId,
    computed: &Computed,
    staged: Staging,
) -> Result<ClusterAssignmentRecord, ToolError> {
    let mut session = state.session();
    if session.latest_run_id.as_ref() != Some(expected_run_id) {
        return Err(ToolError::new(
            NAME,
            "SESSION_REVISION_CONFLICT",
            "Active run changed while computing cluster dossier; refusing to publish stale clusters.",
            "Nothing was persisted or published. The live run changed while the dossier was computing; re-run generate_cluster_dossier against the current run.",
        ));
    }
    let record = match staged {
        Staging::New(staged) => staged
            .commit()
            .map_err(|e| ToolError::from_store(NAME, e))?,
        Staging::Existing(record) => record,
    };
    session.clusters = Some(Arc::new(ClusterState {
        run_id: expected_run_id.clone(),
        cluster_assignment_id: record.cluster_assignment_id.clone(),
        labels: computed.labels.clone(),
        meta: computed.meta.clone(),
        evidence: computed.evidence.clone(),
        evidence_key: computed.evidence_key.clone(),
        group_summary: computed.group_summary.clone(),
        summary_thresholds: computed.summary_thresholds,
        resolution: computed.resolution.clone(),
        dossiers: Mutex::new(computed.dossiers.clone()),
    }));
    Ok(record)
}

fn shown_cluster_ids(payload: &Value) -> Vec<i64> {
    payload["clusters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["cluster_id"].as_i64())
        .collect()
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    let format = p.response_format;
    let (payload, text) = run(state, p).await?;
    let text = match format {
        Format::Markdown => text,
        Format::Json => None,
    };
    // The labels are published: from here on the result must reach the
    // agent, so it is fitted to the budget instead of failing over it.
    Ok(dossier_budget::fit_published(state, payload, text))
}

/// Load a stored assignment, its labels, its `RunView` and its `LiveRun`,
/// checking that its run is the session's latest and that its label count
/// matches the run's rows.
pub(crate) fn resolve_assignment_run(
    state: &ServerState,
    tool: &'static str,
    id: &ClusterAssignmentId,
) -> Result<(ClusterAssignmentRecord, Vec<i64>, RunView, Arc<LiveRun>), ToolError> {
    let record = state
        .store
        .get_cluster_assignment(id)
        .map_err(|e| ToolError::from_store(tool, e))?;
    let labels = state
        .store
        .cluster_labels(id)
        .map_err(|e| ToolError::from_store(tool, e))?;
    let run_id = record.run_id.clone().ok_or_else(|| {
        ToolError::new(
            tool,
            "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
            format!("Assignment {id} was not made from a run."),
            "Use an assignment returned by generate_cluster_dossier.",
        )
    })?;
    if state.session().latest_run_id.as_ref() != Some(&run_id) {
        return Err(ToolError::new(
            tool,
            "SESSION_REVISION_CONFLICT",
            format!(
                "Assignment {id} belongs to run {run_id}, which is not the session's latest run."
            ),
            "Omit run_id or re-run generate_cluster_dossier on the current run.",
        )
        .detail("run_id", json!(run_id)));
    }
    let view = state.resolve_run(tool, run_id.as_ref())?;
    let live = view
        .live
        .clone()
        .ok_or_else(|| ToolError::model_not_loaded(tool).detail("run_id", json!(run_id)))?;
    if labels.len() != live.table.n_rows() {
        return Err(ToolError::new(
            tool,
            "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
            format!(
                "Assignment {id} has {} labels but run {run_id} has {} rows.",
                labels.len(),
                live.table.n_rows()
            ),
            "Use an assignment made from the current run's data.",
        ));
    }
    Ok((record, labels, view, live))
}

/// The dossier's payload and its markdown, before the response format picks.
pub(crate) async fn run(
    state: &ServerState,
    p: Params,
) -> Result<(Value, Option<String>), ToolError> {
    let (stored, view, live) = match p.method {
        MethodParam::Assignment => {
            let id = match p.cluster_assignment_id.as_deref() {
                Some(id) if !id.is_empty() => {
                    ClusterAssignmentId::parse(id).map_err(|e| ToolError::from_store(NAME, e))?
                }
                _ => {
                    return Err(ToolError::invalid_argument(
                        NAME,
                        "method='assignment' needs cluster_assignment_id",
                        "Pass the cluster_assignment_id of a stored clustering (generate_cluster_dossier and merge_small_groups return one), or use another method to cluster now.",
                    ))
                }
            };
            let (record, labels, view, live) = resolve_assignment_run(state, NAME, &id)?;
            if !p.run_id.is_empty() && p.run_id != view.run_id().as_ref() {
                return Err(ToolError::new(
                    NAME,
                    "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
                    format!(
                        "Assignment {id} belongs to run {}, not {}.",
                        view.run_id(),
                        p.run_id
                    ),
                    "Omit run_id (the assignment's own run is used) or pass an assignment made from that run.",
                ));
            }
            (Some((record, labels)), view, live)
        }
        _ => {
            let view = state.resolve_run(NAME, &p.run_id)?;
            let run_id = view.run_id().clone();
            if !p.run_id.is_empty() && state.session().latest_run_id.as_ref() != Some(&run_id) {
                return Err(ToolError::new(
                    NAME,
                    "SESSION_REVISION_CONFLICT",
                    "Refusing to publish a cluster dossier for a run that is not the session's latest.",
                    "Omit run_id or re-run the sweep; dossiers and artifacts can only be published for the session's latest run.",
                )
                .detail("run_id", json!(run_id)));
            }
            let live = view
                .live
                .clone()
                .ok_or_else(|| ToolError::model_not_loaded(NAME).detail("run_id", json!(run_id)))?;
            (None, view, live)
        }
    };
    let run_id = view.run_id().clone();
    if p.max_clusters < 1 {
        return Err(ToolError::invalid_argument(
            NAME,
            format!("max_clusters must be >= 1, got '{}'", p.max_clusters),
            "Pass max_clusters >= 1.",
        ));
    }
    if let Some(t) = p.interpretation_edge_weight_threshold {
        if !unit_threshold(t) {
            return Err(ToolError::invalid_argument(
                NAME,
                format!("interpretation_edge_weight_threshold: {THRESHOLD_RANGE_MESSAGE}"),
                "Pass interpretation_edge_weight_threshold as a finite number in [0.0, 1.0], or omit it to inherit the construction threshold.",
            )
            .detail("interpretation_edge_weight_threshold", json!(t)));
        }
    }
    let construction_threshold = live.fitted.construction_threshold;
    if !unit_threshold(construction_threshold) {
        return Err(failed(format!(
            "resolved_construction_threshold: {THRESHOLD_RANGE_MESSAGE}"
        )));
    }
    let (interpretation_threshold, threshold_source, threshold_inherited) = match &stored {
        // The stored assignment keeps the cut it was made with.
        Some((record, _)) => (
            record.interpretation_edge_weight_threshold,
            record.threshold_source.clone(),
            false,
        ),
        None => match (p.interpretation_edge_weight_threshold, p.method) {
            (None, MethodParam::Spectral) => {
                (0.0, "spectral_default_full_affinity".to_owned(), false)
            }
            (None, _) => (
                construction_threshold,
                "inherited_construction".to_owned(),
                true,
            ),
            (Some(t), _) => (t, "explicit".to_owned(), false),
        },
    };
    let surface = threshold_surface_payload(
        construction_threshold,
        interpretation_threshold,
        threshold_inherited,
        &threshold_source,
    );

    let summary_thresholds = summary_thresholds(&p)?;
    let exclude_columns = excluded_columns(&live, &p.exclude_columns);
    let dropped = live.fitted.config.preprocessing.drop_columns.clone();
    let graph_cache = state.graph_cache_for_live(&live, &view.record);
    let live_run = live.clone();
    let job = Job {
        live,
        graph_cache,
        store: state.store.clone(),
        run_id: run_id.clone(),
        cached: state.session().clusters.clone(),
        method: p.method.resolve(),
        max_k: p.max_k,
        interpretation_threshold,
        threshold_source: threshold_source.clone(),
        construction_threshold,
        n_ball_maps: view.record.n_ball_maps,
        minhash_d: view.minhash_d(),
        exclude_columns,
        dropped: dropped.clone(),
        summary_thresholds,
        // Only characterize the largest clusters so a shattered graph
        // cannot stall the O(clusters x features) evidence pass.
        max_clusters_to_characterize: (p.max_clusters * 2).max(50),
        detail: p.detail,
        given: stored.as_ref().map(|(r, _)| r.clone()),
        given_labels: stored.map(|(_, l)| l).unwrap_or_default(),
    };
    // A cancelled call drops this future; the worker's staged assignment is
    // then dropped with its result, which discards it.
    let (computed, staged) = tokio::task::spawn_blocking(move || compute(job))
        .await
        .map_err(|e| ToolError::worker_stopped(NAME, "DOSSIER_FAILED", &e))??;
    let record = publish(state, &run_id, &computed, staged)?;

    let header = two_lever_markdown_note(construction_threshold, interpretation_threshold)
        + &assignment_markdown(&record);
    let (mut payload, text) = match &computed.dossier {
        None => {
            // JSON callers get a tighter preview; the payload shape is identical.
            let preview = match p.response_format {
                Format::Json => p.feature_preview_limit.min(3),
                Format::Markdown => p.feature_preview_limit,
            };
            let mut payload = build_summary_evidence_payload(
                &computed.evidence,
                &computed.meta,
                &computed.graph_metrics,
                Some(p.max_clusters),
                preview,
                5,
            );
            payload["cluster_selection"] =
                build_cluster_selection(&computed.meta, &shown_cluster_ids(&payload), "size");
            payload["recommended_next_tools"] = json!([
                "get_cluster_profile",
                "get_feature_signal",
                "get_cluster_signal_matrix",
                "compare_clusters",
            ]);
            payload["detail_hint"] = json!("Use get_cluster_profile/get_feature_signal for evidence; detail='full' returns lossless arrays for audit.");
            let text = header
                + &cluster_selection_to_markdown(&payload["cluster_selection"])
                + &summary_evidence_payload_to_markdown(&payload);
            (payload, text)
        }
        Some(dossier) => {
            let full = p.detail == DossierDetail::Full;
            let mut payload = build_evidence_payload(
                dossier,
                &computed.meta,
                p.detail.as_str(),
                Some(p.max_clusters),
                p.feature_preview_limit,
            );
            if full {
                payload["surface_labels"] = surface_labels();
            } else {
                payload["surface_labels_ref"] = json!("get_workflow_guide");
            }
            payload["cluster_selection"] =
                build_cluster_selection(&computed.meta, &shown_cluster_ids(&payload), "size");
            payload["recommended_next_tools"] = json!([
                "get_feature_signal",
                "compare_clusters",
                "export_html_report"
            ]);
            payload["detail_hint"] = json!("Use targeted evidence tools for routine loops; detail='full' is the lossless audit payload.");
            let text = header
                + &dossier_to_markdown(
                    dossier,
                    (!full).then_some(p.max_clusters),
                    (!full).then_some(p.feature_preview_limit),
                );
            (payload, text)
        }
    };
    payload["run_id"] = json!(run_id);
    payload["clusters_run_id"] = json!(run_id);
    payload["construction_threshold"] = json!(construction_threshold);
    payload["interpretation_edge_weight_threshold"] = json!(interpretation_threshold);
    payload["interpretation_edge_weight_threshold_applied"] =
        computed.meta["interpretation_edge_weight_threshold_applied"].clone();
    payload["method_applied"] = computed.meta["method_used"].clone();
    payload["cluster_count"] = computed.meta["n_clusters"].clone();
    // A column the preprocessing dropped (a label kept for comparison, say)
    // is profiled but did not shape the groups. `feature` is the column's
    // position in the table, as in the summary's grouping.
    let names = &live_run.table.names;
    payload["profiled_features"] = computed
        .evidence
        .working_columns
        .iter()
        .filter_map(|c| {
            let feature = names.iter().position(|n| n == c)?;
            Some(
                json!({"feature": feature, "column": c, "used_for_grouping": !dropped.contains(c)}),
            )
        })
        .collect();
    payload["threshold_inherited"] = json!(threshold_inherited);
    payload["threshold_source"] = json!(threshold_source);
    payload["threshold_surface"] = surface.clone();
    payload["cluster_assignment"] = assignment_summary(&record);
    payload["cluster_assignment_id"] = json!(record.cluster_assignment_id);
    payload[phirewall::KEY] = phirewall::summary(
        Some(run_id.to_string()),
        live_run.dataset_id.as_ref().map(|d| d.to_string()),
        live_run.table.n_rows(),
        [("grouping", computed.group_summary.clone())],
    );
    Ok((payload, Some(prepend_threshold_markdown(&text, &surface))))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use pulsar_store::StoreOptions;

    use super::*;
    use crate::state::ServerOptions;

    struct TempRoot(PathBuf);

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn computed() -> Computed {
        let csv = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/golden/blobs/input.csv");
        let table = pulsar_data::read_table(&csv).unwrap();
        let labels = vec![0_i64; table.n_rows()];
        let evidence = build_feature_evidence_index(&table, &labels, &[], &[], None).unwrap();
        Computed {
            labels,
            meta: json!({}),
            evidence: Arc::new(evidence),
            evidence_key: EvidenceKey {
                method: "components".into(),
                interpretation_edge_weight_threshold: 0.0,
                exclude_columns: vec![],
                max_clusters_to_characterize: 50,
            },
            graph_metrics: json!({}),
            dossier: None,
            dossiers: HashMap::new(),
            resolution: None,
            group_summary: json!({}),
            summary_thresholds: GROUP_SUMMARY_DEFAULTS,
        }
    }

    fn assignment_files(root: &Path) -> usize {
        std::fs::read_dir(root.join("cluster_assignments"))
            .unwrap()
            .count()
    }

    #[test]
    fn a_publish_after_the_run_moved_on_persists_nothing() {
        let root = TempRoot(
            std::env::temp_dir().join(format!("pulsar-dossier-publish-{}", RunId::random())),
        );
        let store = Store::open_with(&root.0, StoreOptions::default()).unwrap();
        let state = ServerState::new(store, ServerOptions::default()).unwrap();
        let computed = computed();
        let stage = || {
            state
                .store
                .stage_cluster_assignment(ClusterAssignment {
                    run_id: None,
                    dataset_id: None,
                    method: "components".into(),
                    interpretation_edge_weight_threshold: 0.0,
                    threshold_source: "explicit".into(),
                    construction_threshold: 0.0,
                    labels: &computed.labels,
                })
                .unwrap()
        };
        let (computed_on, latest) = (RunId::random(), RunId::random());

        state.session().latest_run_id = Some(latest.clone());
        let err = publish(&state, &computed_on, &computed, Staging::New(stage())).unwrap_err();
        assert_eq!(err.error_code, "SESSION_REVISION_CONFLICT");
        assert!(state.session().clusters.is_none());
        assert_eq!(
            assignment_files(&root.0),
            0,
            "the staged files are discarded"
        );

        let record = publish(&state, &latest, &computed, Staging::New(stage())).unwrap();
        let published = state.session().clusters.clone().unwrap();
        assert_eq!(published.run_id, latest);
        assert_eq!(
            published.cluster_assignment_id,
            record.cluster_assignment_id
        );
        assert_eq!(assignment_files(&root.0), 2);
    }
}
