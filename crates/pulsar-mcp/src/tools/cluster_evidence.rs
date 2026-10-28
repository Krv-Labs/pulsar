//! The read-only cluster tools of `pulsar/mcp/tools/clustering.py`:
//! `get_cluster_profile`, `get_feature_signal`, `get_cluster_signal_matrix`
//! and `compare_clusters`. They read the labels and evidence index that
//! `generate_cluster_dossier` published into the session; nothing is
//! recomputed except the pairwise statistics of `compare_clusters`.
//!
//! Known divergences from Python:
//! - `compare_clusters` reads the data table of the live run. Python reloaded
//!   the table from disk when `run_id` named a persisted run; here the labels
//!   only exist for the session's latest run, so an evicted run returns
//!   MODEL_NOT_LOADED.

use std::sync::Arc;

use pulsar_analysis::cluster_payloads::{
    build_cluster_selection, cluster_profile_payload_to_markdown, cluster_selection_to_markdown,
    compact_cluster_result_payload, feature_signal_payload_to_markdown,
};
use pulsar_analysis::compare::{compare_clusters, comparison_to_markdown};
use pulsar_analysis::dossier_payloads::{
    cluster_profile_payload, feature_signal_payload, signal_matrix_payload,
};
use pulsar_analysis::plan::py_str;
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

use super::clustering::{surface_labels, DossierDetail};
use super::params::Format;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::{ClusterState, ServerState};

pub const PROFILE: &str = "get_cluster_profile";
pub const SIGNAL: &str = "get_feature_signal";
pub const MATRIX: &str = "get_cluster_signal_matrix";
pub const COMPARE: &str = "compare_clusters";

fn eight() -> usize {
    8
}

fn sixteen() -> usize {
    16
}

fn twenty() -> usize {
    20
}

fn yes() -> bool {
    true
}

fn standard() -> DossierDetail {
    DossierDetail::Standard
}

/// The labels published for the session, checked against `run_id` (empty:
/// against the session's latest run).
fn require_clusters(
    state: &ServerState,
    tool: &'static str,
    run_id: &str,
) -> Result<Arc<ClusterState>, ToolError> {
    let session = state.session();
    if run_id.is_empty() && session.latest_run_id.is_none() {
        return Err(ToolError::model_not_loaded(tool));
    }
    let Some(clusters) = session.clusters.clone() else {
        return Err(ToolError::new(
            tool,
            "CLUSTERS_NOT_BUILT",
            "No cluster evidence found in this session.",
            "Run generate_cluster_dossier first (detail='summary' is enough).",
        ));
    };
    if !run_id.is_empty() {
        if run_id != clusters.run_id.to_string() {
            return Err(ToolError::new(
                tool,
                "CLUSTERS_STALE",
                format!(
                    "Cluster labels in this session were built for run '{}', not the requested run '{run_id}'.",
                    clusters.run_id
                ),
                "Rebuild the dossier for that run: omit run_id here and call generate_cluster_dossier on the session's latest run, or pass run_id equal to clusters_run_id.",
            )
            .detail("requested_run_id", json!(run_id))
            .detail("clusters_run_id", json!(clusters.run_id))
            .detail("latest_run_id", json!(session.latest_run_id)));
        }
    } else if session.latest_run_id.as_ref() != Some(&clusters.run_id) {
        // A sweep clears the labels, so this only guards a future publish path.
        let current = session.latest_run_id.as_ref().map(|r| r.to_string());
        return Err(ToolError::new(
            tool,
            "CLUSTER_CACHE_STALE",
            format!(
                "Cached cluster state is stale: it was computed from run '{}', but the active fitted model is run '{}'. A newer sweep has run since the last dossier.",
                clusters.run_id,
                current.as_deref().unwrap_or("None"),
            ),
            "Re-run generate_cluster_dossier() to recompute clusters against the current fitted model before reading cluster evidence.",
        )
        .detail("cached_run_id", json!(clusters.run_id))
        .detail("current_run_id", json!(current)));
    }
    Ok(clusters)
}

fn at_least_one(tool: &'static str, name: &str, value: usize) -> Result<(), ToolError> {
    if value < 1 {
        return Err(ToolError::invalid_argument(
            tool,
            format!("{name} must be >= 1, got '{value}'"),
            format!("Pass {name} >= 1."),
        ));
    }
    Ok(())
}

/// `surface_labels` in full detail, a pointer to the guide otherwise.
fn add_surface_labels(payload: &mut Value, full: bool) {
    if full {
        payload["surface_labels"] = surface_labels();
    } else {
        payload["surface_labels_ref"] = json!("get_workflow_guide");
    }
}

fn ok_enum() -> Value {
    s::enumeration(&["ok"])
}

// ---- get_cluster_profile ---------------------------------------------------------------

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ProfileParams {
    /// Integer cluster ID as listed by generate_cluster_dossier (cluster_selection.shown_ids). IDs are stable within one dossier.
    pub cluster_id: i64,
    /// 'summary': headline features. 'standard' (default): ranked numeric and categorical evidence up to max_features. 'full': lossless evidence plus size telemetry.
    #[serde(default = "standard")]
    pub detail: DossierDetail,
    /// Maximum feature rows returned for this cluster (>= 1). Default 16.
    #[serde(default = "sixteen")]
    pub max_features: usize,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    #[serde(default)]
    pub response_format: Format,
    /// Optional provenance check: the run_id (from run_topological_sweep) the session's cluster labels must have been built from; returns CLUSTERS_STALE otherwise. Omit to use the current labels unchecked.
    #[serde(default)]
    pub run_id: String,
}

pub fn profile_spec() -> ToolSpec {
    ToolSpec {
        name: PROFILE,
        title: "Cluster Profile",
        description: "Evidence for one cluster (feature shifts versus the rest) from the cached dossier. Reads the session's cluster labels; no recomputation. Use after generate_cluster_dossier to drill into one cluster; use get_feature_signal to follow one feature across clusters.",
        input_schema: input_schema::<ProfileParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", ok_enum()),
                ("clusters_run_id", s::nullable(s::string())),
                ("cluster_result", s::open_object()),
                ("detail", s::enumeration(&["summary", "standard", "full"])),
                ("max_features", s::integer()),
                ("cluster", s::open_object()),
                ("surface_labels_ref", s::string()),
                ("surface_labels", s::open_object()),
            ],
            &["status", "clusters_run_id", "cluster_result", "detail", "cluster"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn profile(state: &ServerState, p: ProfileParams) -> Result<CallToolResult, ToolError> {
    at_least_one(PROFILE, "max_features", p.max_features)?;
    let clusters = require_clusters(state, PROFILE, &p.run_id)?;
    let full = p.detail == DossierDetail::Full;
    let cluster = cluster_profile_payload(
        &clusters.evidence,
        p.cluster_id,
        p.detail.as_str(),
        p.max_features,
    )
    .map_err(|_| {
        ToolError::new(
            PROFILE,
            "CLUSTER_ID_UNKNOWN",
            format!("Unknown cluster_id '{}'.", p.cluster_id),
            "Use generate_cluster_dossier first and inspect available cluster IDs.",
        )
    })?;
    let mut payload = json!({
        "status": "ok",
        "clusters_run_id": clusters.run_id,
        "cluster_result": compact_cluster_result_payload(&clusters.meta, full),
        "detail": p.detail.as_str(),
        "max_features": p.max_features,
        "cluster": cluster,
    });
    add_surface_labels(&mut payload, full);
    let text = match p.response_format {
        Format::Markdown => {
            let applied = py_str(&clusters.meta["interpretation_edge_weight_threshold_applied"]);
            Some(format!(
                "_Cluster membership and evidence are computed on the interpretation slice @ interpretation_edge_weight_threshold ({applied}) of the [0,1]-normalized weighted adjacency (sparsified only when enabled), not the persisted graph @ construction_threshold._\n\n{}",
                cluster_profile_payload_to_markdown(&payload)
            ))
        }
        Format::Json => None,
    };
    Ok(ok_result(payload, text))
}

// ---- get_feature_signal ----------------------------------------------------------------

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SignalParams {
    /// Numeric column names, or 'column=value' pairs for categorical levels (e.g. 'segment=alpha').
    pub feature_names: Vec<String>,
    /// Restrict to these cluster IDs. Omit to use the top max_clusters clusters ranked by signal.
    #[serde(default)]
    pub cluster_ids: Option<Vec<i64>>,
    /// 'summary' (default): core and supporting-tier features only, compact statistics. 'standard': also context-tier features; core/supporting rows carry every statistic. 'full': every feature with every statistic, plus full cluster sizes and surface_labels.
    #[serde(default)]
    pub detail: DossierDetail,
    /// Show at most this many clusters, ranked by signal (>= 1). Default 8.
    #[serde(default = "eight")]
    pub max_clusters: usize,
    /// Text view: 'markdown' (default) or 'json'. structured_content is identical.
    #[serde(default)]
    pub response_format: Format,
    /// Optional provenance check: the run_id (from run_topological_sweep) the session's cluster labels must have been built from; returns CLUSTERS_STALE otherwise. Omit to use the current labels unchecked.
    #[serde(default)]
    pub run_id: String,
}

pub fn signal_spec() -> ToolSpec {
    ToolSpec {
        name: SIGNAL,
        title: "Feature Signal by Cluster",
        description: "Cross-cluster evidence for specific features from the cached dossier: per requested feature, how strongly each cluster departs from the rest. Use to test a hypothesis about particular columns; use get_cluster_signal_matrix for an all-features overview.",
        input_schema: input_schema::<SignalParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", ok_enum()),
                ("clusters_run_id", s::nullable(s::string())),
                ("cluster_result", s::open_object()),
                ("cluster_selection", s::open_object()),
                ("signals", s::open_object()),
                ("surface_labels_ref", s::string()),
                ("surface_labels", s::open_object()),
            ],
            &[
                "status",
                "clusters_run_id",
                "cluster_result",
                "cluster_selection",
                "signals",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn signal(state: &ServerState, p: SignalParams) -> Result<CallToolResult, ToolError> {
    at_least_one(SIGNAL, "max_clusters", p.max_clusters)?;
    let clusters = require_clusters(state, SIGNAL, &p.run_id)?;
    let full = p.detail == DossierDetail::Full;
    let signals = feature_signal_payload(
        &clusters.evidence,
        &p.feature_names,
        p.cluster_ids.as_deref(),
        p.detail.as_str(),
        Some(p.max_clusters),
    );
    let shown: Vec<i64> = signals["clusters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["cluster_id"].as_i64())
        .collect();
    let mut payload = json!({
        "status": "ok",
        "clusters_run_id": clusters.run_id,
        "cluster_result": compact_cluster_result_payload(&clusters.meta, full),
        "cluster_selection": build_cluster_selection(&clusters.meta, &shown, "signal"),
        "signals": signals,
    });
    add_surface_labels(&mut payload, full);
    let text = match p.response_format {
        Format::Markdown => Some(feature_signal_payload_to_markdown(&payload["signals"])),
        Format::Json => None,
    };
    Ok(ok_result(payload, text))
}

// ---- get_cluster_signal_matrix ---------------------------------------------------------

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct MatrixParams {
    /// Restrict the matrix to these cluster IDs. Omit for all clusters, truncated to max_clusters by signal.
    #[serde(default)]
    pub cluster_ids: Option<Vec<i64>>,
    /// False (default) shows only core and supporting signal tiers. True also includes weaker 'context' tier rows.
    #[serde(default)]
    pub include_context_tier: bool,
    /// Show at most this many clusters, ranked by signal (>= 1). Presentation-only truncation. Default 8.
    #[serde(default = "eight")]
    pub max_clusters: usize,
    /// True (default): markdown matrix text (z-scores for numeric, prevalence percent for categorical). False: JSON text.
    #[serde(default = "yes")]
    pub return_markdown: bool,
    /// Optional provenance check: the run_id (from run_topological_sweep) the session's cluster labels must have been built from; returns CLUSTERS_STALE otherwise. Omit to use the current labels unchecked.
    #[serde(default)]
    pub run_id: String,
}

pub fn matrix_spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: MATRIX,
        title: "Cluster Signal Matrix",
        description: "Cluster-by-feature signal matrix from the cached dossier: which features distinguish which clusters in one dense table, ranked by signal. Use for a cross-cluster overview; use get_feature_signal for chosen features and get_cluster_profile for one cluster.",
        input_schema: input_schema::<MatrixParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", ok_enum()),
                ("clusters_run_id", s::nullable(s::string())),
                ("cluster_result", open.clone()),
                ("surface_labels_ref", s::string()),
                ("cluster_selection", open.clone()),
                (
                    "signal_matrix",
                    s::object(
                        &[
                            ("numeric_columns", s::array(s::string())),
                            ("categorical_values", s::array(open.clone())),
                            ("clusters_returned", s::integer()),
                            ("clusters_omitted", s::integer()),
                            ("omitted_clusters", s::array(open.clone())),
                            ("numeric_features_omitted", s::integer()),
                            ("categorical_features_omitted", s::integer()),
                            ("numeric_rows", s::array(open.clone())),
                            ("categorical_rows", s::array(open)),
                        ],
                        &["clusters_returned", "clusters_omitted"],
                        true,
                    ),
                ),
            ],
            &[
                "status",
                "clusters_run_id",
                "cluster_result",
                "cluster_selection",
                "signal_matrix",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

pub async fn matrix(state: &ServerState, p: MatrixParams) -> Result<CallToolResult, ToolError> {
    at_least_one(MATRIX, "max_clusters", p.max_clusters)?;
    let clusters = require_clusters(state, MATRIX, &p.run_id)?;
    let ids = p.cluster_ids.as_deref();
    let build = |markdown: bool| {
        signal_matrix_payload(
            &clusters.evidence,
            ids,
            p.include_context_tier,
            Some(p.max_clusters),
            markdown,
        )
    };
    let signal_matrix = build(false);
    let omitted: Vec<i64> = signal_matrix["omitted_clusters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["cluster_id"].as_i64())
        .collect();
    let universe: Vec<i64> = match ids {
        Some(ids) => ids.to_vec(),
        None => clusters.meta["cluster_sizes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| c["cluster_id"].as_i64())
            .collect(),
    };
    let shown: Vec<i64> = universe
        .into_iter()
        .filter(|id| !omitted.contains(id))
        .collect();
    let selection = build_cluster_selection(&clusters.meta, &shown, "signal");
    let text = p.return_markdown.then(|| {
        let report = build(true);
        cluster_selection_to_markdown(&selection) + report["markdown_report"].as_str().unwrap_or("")
    });
    let payload = json!({
        "status": "ok",
        "clusters_run_id": clusters.run_id,
        "cluster_result": compact_cluster_result_payload(&clusters.meta, false),
        "surface_labels_ref": "get_workflow_guide",
        "cluster_selection": selection,
        "signal_matrix": signal_matrix,
    });
    Ok(ok_result(payload, text))
}

// ---- compare_clusters ------------------------------------------------------------------

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct CompareParams {
    /// First cluster ID, as listed by generate_cluster_dossier.
    pub cluster_a: i64,
    /// Second cluster ID, as listed by generate_cluster_dossier.
    pub cluster_b: i64,
    /// Return at most this many numeric features, ranked by |Cohen's d| (>= 1); features_omitted reports the rest. Default 20.
    #[serde(default = "twenty")]
    pub max_features: usize,
    /// Optional provenance check: the run_id (from run_topological_sweep) the session's cluster labels must have been built from; returns CLUSTERS_STALE otherwise. Omit to use the current labels unchecked.
    #[serde(default)]
    pub run_id: String,
}

pub fn compare_spec() -> ToolSpec {
    let num = || s::nullable(s::number());
    ToolSpec {
        name: COMPARE,
        title: "Compare Clusters",
        description: "Pairwise statistical comparison of two clusters over numeric features: both means, Welch t-test and KS-test p-values, and Cohen's d, ranked by effect size. Use after generate_cluster_dossier to contrast two specific clusters; p-values are uncorrected for multiple comparisons.",
        input_schema: input_schema::<CompareParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", ok_enum()),
                ("clusters_run_id", s::nullable(s::string())),
                ("cluster_a", s::integer()),
                ("cluster_b", s::integer()),
                ("features_compared", s::integer()),
                ("features_omitted", s::integer()),
                (
                    "results",
                    s::array(s::object(
                        &[
                            ("column", s::string()),
                            ("mean_a", num()),
                            ("mean_b", num()),
                            ("p_val_t", num()),
                            ("p_val_ks", num()),
                            ("cohens_d", num()),
                        ],
                        &["column"],
                        false,
                    )),
                ),
            ],
            &[
                "status",
                "clusters_run_id",
                "cluster_a",
                "cluster_b",
                "features_compared",
                "features_omitted",
                "results",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

fn finite_or_null(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}

pub async fn compare(state: &ServerState, p: CompareParams) -> Result<CallToolResult, ToolError> {
    at_least_one(COMPARE, "max_features", p.max_features)?;
    let clusters = require_clusters(state, COMPARE, &p.run_id)?;
    let live = state.cache.get(&clusters.run_id).ok_or_else(|| {
        ToolError::new(
            COMPARE,
            "MODEL_NOT_LOADED",
            format!(
                "The data of run {} is no longer in memory.",
                clusters.run_id
            ),
            "Re-run run_topological_sweep (the same config reuses cached projections) and generate_cluster_dossier, then retry.",
        )
        .detail("run_id", json!(clusters.run_id))
    })?;
    let (a, b) = (p.cluster_a, p.cluster_b);
    let worker = clusters.clone();
    let results =
        tokio::task::spawn_blocking(move || compare_clusters(&live.table, &worker.labels, a, b))
            .await
            .map_err(|e| ToolError::worker_stopped(COMPARE, "COMPARE_FAILED", &e))?;
    if results.is_empty() {
        return Err(ToolError::new(
            COMPARE,
            "COMPARISON_EMPTY",
            format!("No results for comparison between clusters {a} and {b}."),
            "Both clusters need at least 2 rows and numeric columns with at least 2 non-missing values; check IDs with generate_cluster_dossier.",
        )
        .detail("cluster_a", json!(a))
        .detail("cluster_b", json!(b)));
    }
    let shown = &results[..p.max_features.min(results.len())];
    let omitted = results.len() - shown.len();
    let rows: Vec<Value> = shown
        .iter()
        .map(|r| {
            json!({
                "column": r.column,
                "mean_a": finite_or_null(r.mean_a),
                "mean_b": finite_or_null(r.mean_b),
                "p_val_t": finite_or_null(r.p_val_t),
                "p_val_ks": finite_or_null(r.p_val_ks),
                "cohens_d": finite_or_null(r.cohens_d),
            })
        })
        .collect();
    let payload = json!({
        "status": "ok",
        "clusters_run_id": clusters.run_id,
        "cluster_a": a,
        "cluster_b": b,
        "features_compared": results.len(),
        "features_omitted": omitted,
        "results": rows,
    });
    let mut text = comparison_to_markdown(a, b, shown);
    if omitted > 0 {
        text.push_str(&format!(
            "\n\n_{omitted} further features omitted; raise max_features to see them._"
        ));
    }
    Ok(ok_result(payload, Some(text)))
}
