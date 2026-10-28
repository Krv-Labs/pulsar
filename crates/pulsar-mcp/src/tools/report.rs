//! `export_html_report` (`pulsar/mcp/tools/reporting.py`): write the
//! standalone HTML report (`pulsar_report::html`) of the run the session's
//! clusters were published for.
//!
//! Known divergences from Python: none in the tool; the figure's layout for
//! graphs of 500+ nodes is documented in `pulsar_report::layout`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pulsar_analysis::dossier::build_dossier;
use pulsar_analysis::feature_evidence::build_feature_evidence_index;
use pulsar_analysis::GraphAnalysis;
use pulsar_report::{dossier_to_html, ReportGraph};
use pulsar_store::DatasetId;
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use super::export::normalize_cluster_names;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const NAME: &str = "export_html_report";

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct Params {
    /// Optional dataset handle; the report is written next to that file as <name>_report.html, else pulsar_report.html in the working directory.
    pub dataset_id: String,
    /// Column names to leave out of the feature evidence shown in the report. Omit to reuse the session's dossier exclusions.
    pub exclude_columns: Option<Vec<String>>,
    /// Optional cluster names: integer cluster ids as string keys, e.g. {"0": "Cohort A"}. Unnamed clusters get auto-names plus a warning.
    pub cluster_names: Option<Map<String, Value>>,
    /// Run from run_topological_sweep. Omit for the latest run. Must be the run the clusters were computed on.
    pub run_id: String,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: NAME,
        title: "Export HTML Report",
        description: "Write a self-contained HTML report of the fitted topology and its clusters. Returns `report_path` and a file:// `report_url`. Requires clusters from generate_cluster_dossier for the same run (else CLUSTERS_STALE or CLUSTERS_MISSING). Name clusters via cluster_names for a readable report.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("report_path", s::string()),
                ("report_url", s::string()),
                ("message", s::string()),
                ("warning", s::string()),
            ],
            &["status", "report_path", "report_url", "message"],
            false,
        )),
        read_only: false,
        idempotent: true,
    }
}

fn failed(reason: impl Into<String>) -> ToolError {
    ToolError::new(
        NAME,
        "EXPORT_REPORT_FAILED",
        reason,
        "Check the reason, confirm the output location is writable, and retry.",
    )
}

/// `<dataset stem>_report.html` beside the dataset, else
/// `pulsar_report.html` in the working directory.
fn report_path(state: &ServerState, dataset_id: &str) -> Result<PathBuf, ToolError> {
    let cwd = || {
        std::env::current_dir()
            .unwrap_or_default()
            .join("pulsar_report.html")
    };
    if dataset_id.is_empty() {
        return Ok(cwd());
    }
    let id = DatasetId::parse(dataset_id).map_err(|e| ToolError::from_store(NAME, e))?;
    let record = state
        .store
        .get_dataset(&id)
        .map_err(|e| ToolError::from_store(NAME, e))?;
    if record.path.is_file() {
        let stem = record
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(record.path.with_file_name(format!("{stem}_report.html")))
    } else {
        Ok(cwd())
    }
}

fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    let result = std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    let view = state.resolve_run(NAME, &p.run_id)?;
    let run_id = view.run_id().clone();
    let live = view.live.clone().ok_or_else(|| {
        ToolError::new(
            NAME,
            "MODEL_NOT_LOADED",
            format!("The fitted model of run {run_id} is no longer in memory."),
            "Re-run run_topological_sweep (the same config reuses cached projections), then retry.",
        )
        .detail("run_id", json!(run_id))
    })?;
    let Some(clusters) = state.session().clusters.clone() else {
        return Err(ToolError::new(
            NAME,
            "CLUSTERS_MISSING",
            "No clusters found. Run generate_cluster_dossier first.",
            "Run generate_cluster_dossier for this run, then retry.",
        ));
    };
    if clusters.run_id != run_id {
        return Err(ToolError::new(
            NAME,
            "CLUSTERS_STALE",
            format!(
                "Session cluster labels were computed from run '{}', not the requested run '{run_id}'.",
                clusters.run_id
            ),
            format!("Rebuild the dossier for this run with generate_cluster_dossier(run_id='{run_id}') (it must be the latest run), then retry."),
        )
        .detail("requested_run_id", json!(run_id))
        .detail("clusters_run_id", json!(clusters.run_id)));
    }
    let path = report_path(state, &p.dataset_id)?;
    let exclude = p
        .exclude_columns
        .clone()
        .unwrap_or_else(|| clusters.evidence_key.exclude_columns.clone());
    let config_yaml = view.record.config_yaml.clone();
    let (n_ball_maps, minhash_d) = (view.record.n_ball_maps, view.minhash_d());
    let names = p.cluster_names.clone();
    let worker_clusters = Arc::clone(&clusters);
    let html = tokio::task::spawn_blocking(move || -> Result<String, ToolError> {
        let fitted = &live.fitted;
        let labels = &worker_clusters.labels;
        let mut sorted_exclude = exclude.clone();
        sorted_exclude.sort();
        let evidence = if sorted_exclude == worker_clusters.evidence_key.exclude_columns {
            worker_clusters.evidence.clone()
        } else {
            Arc::new(
                build_feature_evidence_index(&live.table, labels, &fitted.graph.edges, &exclude, None)
                    .map_err(|e| failed(e.to_string()))?,
            )
        };
        let affinity = fitted.weighted_edges(Some(0.0));
        let metrics = GraphAnalysis::new(&fitted.graph)
            .ok()
            .map(|a| a.metrics(fitted.construction_threshold, n_ball_maps, Some(&affinity), minhash_d))
            .and_then(|m| serde_json::to_value(m).ok())
            .unwrap_or_else(|| json!({}));
        let mut dossier = build_dossier(&evidence, &live.table, labels, &fitted.graph.edges, "full", metrics)
            .map_err(|e| failed(e.to_string()))?;
        let valid: Vec<i64> = dossier.clusters.iter().map(|c| c.cluster_id).collect();
        let normalized = normalize_cluster_names(&names.unwrap_or_default(), &valid, false).map_err(|reason| {
            ToolError::new(
                NAME,
                "CLUSTER_NAMES_INVALID",
                reason,
                "Pass cluster_names as a flat object whose keys are existing integer cluster ids encoded as strings and whose values are non-empty names.",
            )
        })?;
        for profile in &mut dossier.clusters {
            if let Some(name) = normalized.get(&profile.cluster_id) {
                profile.semantic_name = name.clone();
            }
        }
        let graph = ReportGraph {
            graph: &fitted.graph,
            affinity: &affinity,
            stability: fitted.stability.as_ref(),
            resolved_threshold: fitted.construction_threshold,
        };
        Ok(dossier_to_html(&dossier, labels, Some(&graph), Some(&config_yaml)))
    })
    .await
    .map_err(|e| ToolError::worker_stopped(NAME, "EXPORT_REPORT_FAILED", &e))??;
    write_atomic(&path, &html).map_err(|e| failed(format!("{}: {e}", path.display())))?;
    let absolute = std::path::absolute(&path).unwrap_or_else(|_| path.clone());
    let mut payload = json!({
        "status": "ok",
        "report_path": path.display().to_string(),
        "report_url": format!("file://{}", absolute.display()),
        "message": "HTML report exported successfully. Open the report_url in your browser.",
    });
    if p.cluster_names.as_ref().is_none_or(Map::is_empty) {
        payload["warning"] = json!("cluster_names was not provided — clusters are labeled with robotic auto-names (e.g. '[Auto] High X, Low Y'). For a human-tractable report, re-run with cluster_names={\"0\": \"<scientific cohort name>\", ...}.");
    }
    Ok(ok_result(payload, None))
}
