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

use std::sync::Arc;

use pulsar_analysis::cluster_payloads::{
    build_cluster_selection, build_evidence_payload, build_summary_evidence_payload,
    cluster_result_payload, cluster_selection_to_markdown, prepend_threshold_markdown,
    summary_evidence_payload_to_markdown, threshold_surface_payload, two_lever_markdown_note,
};
use pulsar_analysis::clustering::resolve::{
    resolve_clusters, ResolveError, ResolveMethod, SpectralCutDiagnostics,
};
use pulsar_analysis::dossier::{build_dossier, dossier_to_markdown, TopologicalDossier};
use pulsar_analysis::feature_evidence::{build_feature_evidence_index, FeatureEvidenceIndex};
use pulsar_analysis::payloads::{bounded_list, size_summary};
use pulsar_analysis::GraphAnalysis;
use pulsar_data::Column;
use pulsar_pipeline::Graph;
use pulsar_store::{
    ClusterAssignment, ClusterAssignmentRecord, RunId, StagedClusterAssignment, Store,
};
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

use super::params::Format;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::{ClusterState, EvidenceKey, LiveRun, ServerState};

pub const NAME: &str = "generate_cluster_dossier";

const THRESHOLD_RANGE_MESSAGE: &str = "Threshold must be finite and between 0.0 and 1.0.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MethodParam {
    #[default]
    Auto,
    Spectral,
    Components,
}

impl From<MethodParam> for ResolveMethod {
    fn from(m: MethodParam) -> Self {
        match m {
            MethodParam::Auto => ResolveMethod::Auto,
            MethodParam::Spectral => ResolveMethod::Spectral,
            MethodParam::Components => ResolveMethod::Components,
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
    /// 'components': connected components after dropping edges <= the interpretation threshold. 'spectral': spectral cut of the weighted affinity (may return NO_STABLE_SPECTRAL_CUT). 'auto' (default): 'components' when the resolved interpretation threshold is > 0; otherwise an H0 stability cut, then components or spectral.
    pub method: MethodParam,
    /// Largest k tried by the spectral search (2..max_k); ignored for method='components'. Default 15.
    pub max_k: usize,
    /// Edge-weight cutoff in [0, 1]; edges <= this are dropped before clustering. Omit to inherit the construction threshold (spectral: 0.0, the full affinity).
    pub interpretation_edge_weight_threshold: Option<f64>,
    /// 'summary' (default): cheap per-cluster headline evidence. 'standard': adds per-cluster feature evidence (heavy). 'full': lossless audit payload (heaviest).
    pub detail: DossierDetail,
    /// Show at most this many clusters, largest first (>= 1); clusters_omitted reports the rest. Default 8.
    pub max_clusters: usize,
    /// Feature hints shown per cluster. Default 5.
    pub feature_preview_limit: usize,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    pub response_format: Format,
    /// Column names to hide from the report. Display only: does not change the model or the clustering.
    pub exclude_columns: Vec<String>,
    /// Run to cluster, from run_topological_sweep. Omit for the latest run. Labels are published into the session, so it must be the latest run (else SESSION_REVISION_CONFLICT).
    pub run_id: String,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            method: MethodParam::Auto,
            max_k: 15,
            interpretation_edge_weight_threshold: None,
            detail: DossierDetail::Summary,
            max_clusters: 8,
            feature_preview_limit: 5,
            response_format: Format::Markdown,
            exclude_columns: Vec::new(),
            run_id: String::new(),
        }
    }
}

pub fn spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: NAME,
        title: "Cluster Evidence Dossier",
        description: "Cluster the latest run's graph and build a statistical evidence dossier. Persists a cluster_assignment_id and publishes the labels into the session for the cluster tools and exports; a new sweep clears them. Start with detail='summary'; escalate only after narrowing focus.",
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
                            ("column", s::string()),
                            ("used_for_grouping", s::boolean()),
                        ],
                        &["column", "used_for_grouping"],
                        false,
                    )),
                ),
                ("threshold_inherited", s::boolean()),
                ("threshold_source", s::string()),
                ("threshold_surface", open.clone()),
                ("cluster_assignment", open.clone()),
                ("cluster_assignment_id", s::string()),
                ("cluster_selection", open.clone()),
                ("clusters", s::array(open.clone())),
                ("recommended_next_tools", s::array(s::string())),
                ("detail_hint", s::string()),
                ("surface_labels_ref", s::string()),
                ("surface_labels", open),
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

/// Whether a column is a row index rather than a measurement: a row-id name
/// (`rowid`, `row_id`, `index`, `Unnamed: 0`, blank) or integers counting
/// the rows from 0 or 1.
fn row_index_like(name: &str, column: &Column) -> bool {
    let key: String = name
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase();
    if matches!(
        key.as_str(),
        "" | "rowid" | "rownum" | "rownumber" | "index" | "unnamed0"
    ) {
        return true;
    }
    let Column::Int(values) = column else {
        return false;
    };
    let counts_from = |start: i64| (start..).zip(values).all(|(i, &v)| i == v);
    values.len() > 1 && (counts_from(0) || counts_from(1))
}

/// Columns the dossier never profiles: the requested exclusions, the
/// identifiers the run's preprocessing plan names, and row indexes.
fn excluded_columns(live: &LiveRun, requested: &[String]) -> Vec<String> {
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
}

/// Inputs of the blocking worker, captured before it starts.
struct Job {
    live: Arc<LiveRun>,
    store: Store,
    run_id: RunId,
    cached: Option<Arc<ClusterState>>,
    method: ResolveMethod,
    max_k: usize,
    interpretation_threshold: f64,
    threshold_source: &'static str,
    construction_threshold: f64,
    n_ball_maps: usize,
    minhash_d: Option<usize>,
    exclude_columns: Vec<String>,
    max_clusters_to_characterize: usize,
    detail: DossierDetail,
}

fn compute(job: Job) -> Result<(Computed, StagedClusterAssignment), ToolError> {
    let fitted = &job.live.fitted;
    let table = &job.live.table;
    let affinity = Graph {
        n: fitted.n,
        edges: fitted.weighted_edges(Some(0.0)),
    };
    let result = resolve_clusters(
        &affinity,
        fitted.stability.as_ref(),
        job.method,
        job.max_k,
        job.interpretation_threshold,
    )
    .map_err(|e| resolve_error(e, job.detail))?;
    let labels: Vec<i64> = result.labels.iter().map(|&l| l as i64).collect();
    let plateaus = result
        .stability_plateaus
        .as_ref()
        .map_or(Value::Null, |p| json!(p));
    let meta = cluster_result_payload(
        &labels,
        result.method_used,
        result.n_clusters,
        result.silhouette_score,
        result.interpretation_edge_weight_threshold_applied,
        &plateaus,
        result.failure_reason.as_deref(),
    );

    let mut sorted_exclude = job.exclude_columns.clone();
    sorted_exclude.sort();
    let evidence_key = EvidenceKey {
        method: result.method_used.to_owned(),
        interpretation_edge_weight_threshold: result.interpretation_edge_weight_threshold_applied,
        exclude_columns: sorted_exclude,
        max_clusters_to_characterize: job.max_clusters_to_characterize,
    };
    let evidence = match &job.cached {
        Some(c)
            if c.run_id == job.run_id && c.labels == labels && c.evidence_key == evidence_key =>
        {
            c.evidence.clone()
        }
        _ => Arc::new(
            build_feature_evidence_index(
                table,
                &labels,
                &fitted.graph.edges,
                &job.exclude_columns,
                Some(job.max_clusters_to_characterize),
            )
            .map_err(|e| failed(e.to_string()))?,
        ),
    };

    // Python logs and degrades to an empty block when diagnosis fails.
    let metrics = GraphAnalysis::new(&fitted.graph).ok().map(|a| {
        a.metrics(
            fitted.construction_threshold,
            job.n_ball_maps,
            Some(&affinity.edges),
            job.minhash_d,
        )
    });
    let full_metrics = metrics
        .as_ref()
        .and_then(|m| serde_json::to_value(m).ok())
        .unwrap_or_else(|| json!({}));
    let dossier = match job.detail {
        DossierDetail::Summary => None,
        detail => Some(
            build_dossier(
                &evidence,
                table,
                &labels,
                &fitted.graph.edges,
                detail.as_str(),
                full_metrics.clone(),
            )
            .map_err(|e| failed(e.to_string()))?,
        ),
    };
    let mut graph_metrics = full_metrics;
    if let Value::Object(map) = &mut graph_metrics {
        map.remove("component_sizes");
        let sizes = metrics.map(|m| m.component_sizes).unwrap_or_default();
        map.insert("component_sizes_summary".into(), size_summary(&sizes, 20));
    }

    let staged = job
        .store
        .stage_cluster_assignment(ClusterAssignment {
            run_id: Some(job.run_id.clone()),
            dataset_id: job.live.dataset_id.clone(),
            method: result.method_used.to_owned(),
            interpretation_edge_weight_threshold: job.interpretation_threshold,
            threshold_source: job.threshold_source.to_owned(),
            construction_threshold: job.construction_threshold,
            labels: &labels,
        })
        .map_err(|e| ToolError::from_store(NAME, e))?;
    Ok((
        Computed {
            labels,
            meta,
            evidence,
            evidence_key,
            graph_metrics,
            dossier,
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
    staged: StagedClusterAssignment,
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
    let record = staged
        .commit()
        .map_err(|e| ToolError::from_store(NAME, e))?;
    session.clusters = Some(Arc::new(ClusterState {
        run_id: expected_run_id.clone(),
        cluster_assignment_id: record.cluster_assignment_id.clone(),
        labels: computed.labels.clone(),
        meta: computed.meta.clone(),
        evidence: computed.evidence.clone(),
        evidence_key: computed.evidence_key.clone(),
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
    let live = view.live.clone().ok_or_else(|| {
        ToolError::new(
            NAME,
            "MODEL_NOT_LOADED",
            format!("The fitted model of run {run_id} is no longer in memory."),
            "Re-run run_topological_sweep (the same config reuses cached projections), then retry.",
        )
        .detail("run_id", json!(run_id))
    })?;
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
    let (interpretation_threshold, threshold_source, threshold_inherited) =
        match (p.interpretation_edge_weight_threshold, p.method) {
            (None, MethodParam::Spectral) => (0.0, "spectral_default_full_affinity", false),
            (None, _) => (construction_threshold, "inherited_construction", true),
            (Some(t), _) => (t, "explicit", false),
        };
    let surface = threshold_surface_payload(
        construction_threshold,
        interpretation_threshold,
        threshold_inherited,
        threshold_source,
    );

    let exclude_columns = excluded_columns(&live, &p.exclude_columns);
    let dropped = live.fitted.config.preprocessing.drop_columns.clone();
    let job = Job {
        live,
        store: state.store.clone(),
        run_id: run_id.clone(),
        cached: state.session().clusters.clone(),
        method: p.method.into(),
        max_k: p.max_k,
        interpretation_threshold,
        threshold_source,
        construction_threshold,
        n_ball_maps: view.record.n_ball_maps,
        minhash_d: view.minhash_d(),
        exclude_columns,
        // Only characterize the largest clusters so a shattered graph
        // cannot stall the O(clusters x features) evidence pass.
        max_clusters_to_characterize: (p.max_clusters * 2).max(50),
        detail: p.detail,
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
    // is profiled but did not shape the groups.
    payload["profiled_features"] = computed
        .evidence
        .working_columns
        .iter()
        .map(|c| json!({"column": c, "used_for_grouping": !dropped.contains(c)}))
        .collect();
    payload["threshold_inherited"] = json!(threshold_inherited);
    payload["threshold_source"] = json!(threshold_source);
    payload["threshold_surface"] = surface.clone();
    payload["cluster_assignment"] = assignment_summary(&record);
    payload["cluster_assignment_id"] = json!(record.cluster_assignment_id);
    let text = match p.response_format {
        Format::Markdown => Some(prepend_threshold_markdown(&text, &surface)),
        Format::Json => None,
    };
    Ok(ok_result(payload, text))
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
        let err = publish(&state, &computed_on, &computed, stage()).unwrap_err();
        assert_eq!(err.error_code, "SESSION_REVISION_CONFLICT");
        assert!(state.session().clusters.is_none());
        assert_eq!(
            assignment_files(&root.0),
            0,
            "the staged files are discarded"
        );

        let record = publish(&state, &latest, &computed, stage()).unwrap();
        let published = state.session().clusters.clone().unwrap();
        assert_eq!(published.run_id, latest);
        assert_eq!(
            published.cluster_assignment_id,
            record.cluster_assignment_id
        );
        assert_eq!(assignment_files(&root.0), 2);
    }
}
