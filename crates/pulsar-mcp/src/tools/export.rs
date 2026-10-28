//! Exports (`pulsar/mcp/tools/reporting.py`): `export_labeled_data` writes
//! the dataset with a persisted cluster assignment and the agent's cluster
//! names appended, as the CSV pandas would write; `export_dataset_bundle`
//! writes the fitted graph, node/edge/group tables and data as a Parquet
//! bundle (`pulsar_report::bundle`).
//!
//! Known divergences from Python:
//! - With a `dataset_id` on the assignment the rows are always re-read from
//!   the (unchanged) dataset file; Python reused an identical in-memory frame.
//! - Cluster-name keys parse as Rust integers (`"1_0"` is not `10`).

use pulsar_data::Column;
use pulsar_store::{ClusterAssignmentId, StoreError};
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use super::characterize::load;
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const LABELED: &str = "export_labeled_data";
pub const BUNDLE: &str = "export_dataset_bundle";

/// Python's `str(list)` of integers.
fn py_list_ints(ids: &[i64]) -> String {
    let items: Vec<String> = ids.iter().map(i64::to_string).collect();
    format!("[{}]", items.join(", "))
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct LabeledParams {
    /// Handle of a saved cluster assignment, emitted by generate_cluster_dossier.
    pub cluster_assignment_id: String,
    /// Names for EVERY cluster: integer cluster ids as string keys, non-empty values, e.g. {"0": "Cohort A"}. Pulsar does not infer domain meaning.
    pub cluster_names: Map<String, Value>,
    /// Absolute server-visible path of the CSV to write; an existing file is replaced.
    pub output_path: String,
    /// Optional run handle: labels must come from that run (else CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH); its in-memory data is used.
    #[serde(default)]
    pub run_id: String,
}

pub fn labeled_spec() -> ToolSpec {
    ToolSpec {
        name: LABELED,
        title: "Export Labeled Data",
        description: "Write the dataset to CSV with a topological cluster id and your cluster name per row. Returns the output path, row count and cluster ids. Use after generate_cluster_dossier once you have named the clusters; the original columns are preserved and two columns are appended.",
        input_schema: input_schema::<LabeledParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("output_path", s::string()),
                ("cluster_assignment_id", s::string()),
                ("run_id", s::nullable(s::string())),
                ("dataset_id", s::nullable(s::string())),
                ("row_count", s::integer()),
                ("cluster_ids", s::array(s::integer())),
            ],
            &["status", "output_path", "cluster_assignment_id", "run_id", "dataset_id", "row_count", "cluster_ids"],
            false,
        )),
        read_only: false,
        idempotent: true,
    }
}

/// `_normalize_cluster_names`: valid ids named once with non-empty strings
/// (every valid id when `require_all`).
pub fn normalize_cluster_names(
    names: &Map<String, Value>,
    valid: &[i64],
    require_all: bool,
) -> Result<std::collections::BTreeMap<i64, String>, String> {
    let mut out = std::collections::BTreeMap::new();
    for (raw_id, raw_name) in names {
        let key = raw_id.trim();
        if key.is_empty() {
            return Err("cluster_names contains an empty cluster ID key.".into());
        }
        let Ok(id) = key.parse::<i64>() else {
            return Err(format!(
                "cluster_names keys must be integer cluster IDs encoded as strings; got {}.",
                pulsar_analysis::plan::py_repr(raw_id)
            ));
        };
        if out.contains_key(&id) {
            return Err(format!("cluster_names contains duplicate cluster ID {id}."));
        }
        if !valid.contains(&id) {
            return Err(format!(
                "cluster_names contains unknown cluster ID {id}. Available IDs: {}.",
                py_list_ints(valid)
            ));
        }
        match raw_name.as_str().map(str::trim) {
            Some(name) if !name.is_empty() => {
                out.insert(id, name.to_owned());
            }
            _ => {
                return Err(format!(
                    "cluster_names[{}] must be a non-empty string name.",
                    pulsar_analysis::plan::py_repr(key)
                ))
            }
        }
    }
    let missing: Vec<i64> = valid
        .iter()
        .copied()
        .filter(|id| !out.contains_key(id))
        .collect();
    if require_all && !missing.is_empty() {
        return Err(format!(
            "cluster IDs {} have no name mapping. Provide all {} cluster IDs.",
            py_list_ints(&missing),
            valid.len()
        ));
    }
    Ok(out)
}

pub async fn labeled(state: &ServerState, p: LabeledParams) -> Result<CallToolResult, ToolError> {
    let store_err = |e: StoreError| ToolError::from_store(LABELED, e);
    let id = ClusterAssignmentId::parse(&p.cluster_assignment_id).map_err(store_err)?;
    let assignment = state.store.get_cluster_assignment(&id).map_err(store_err)?;

    let view = if p.run_id.is_empty() {
        None
    } else {
        let view = state.resolve_run(LABELED, &p.run_id)?;
        if assignment.run_id.as_ref() != Some(view.run_id()) {
            return Err(ToolError::new(
                LABELED,
                "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
                "The cluster assignment was computed from a different run than run_id.",
                "Pass the run_id the assignment came from, or omit run_id, or regenerate the dossier for the run you want.",
            )
            .detail("cluster_assignment_id", json!(p.cluster_assignment_id))
            .detail("assignment_run_id", json!(assignment.run_id))
            .detail("requested_run_id", json!(p.run_id)));
        }
        Some(view)
    };

    let mut table = if let Some(dataset_id) = &assignment.dataset_id {
        let record = state.store.check_dataset(dataset_id).map_err(|_| {
            ToolError::new(
                LABELED,
                "CLUSTER_ASSIGNMENT_DATASET_STALE",
                "The dataset backing this cluster assignment is gone or has changed since the labels were computed, so the labels no longer align with its rows.",
                "Re-ingest the dataset and rerun generate_cluster_dossier, then export with the new cluster_assignment_id.",
            )
            .detail("cluster_assignment_id", json!(p.cluster_assignment_id))
            .detail("dataset_id", json!(dataset_id))
        })?;
        load(LABELED, &record.path.display().to_string()).await?
    } else {
        let live = view.as_ref().and_then(|v| v.live.clone()).or_else(|| {
            let latest = state.session().latest_run_id.clone();
            match (&latest, &assignment.run_id) {
                (Some(l), Some(a)) if l == a => state.cache.get(l),
                _ => None,
            }
        });
        match live.filter(|l| l.table.n_rows() == assignment.label_count) {
            Some(l) => l.table.clone(),
            None => {
                return Err(ToolError::new(
                    LABELED,
                    "CLUSTER_ASSIGNMENT_DATA_UNAVAILABLE",
                    "Cluster assignment has no durable dataset_id and the matching data is not loaded in this session.",
                    "Use a cluster_assignment_id emitted from a dossier generated after ingest_dataset/run_topological_sweep(dataset_id=...), or regenerate the dossier with the source data loaded in this MCP session.",
                )
                .detail("cluster_assignment_id", json!(p.cluster_assignment_id))
                .detail("run_id", json!(assignment.run_id))
                .detail("dataset_id", json!(assignment.dataset_id))
                .detail("label_count", json!(assignment.label_count)))
            }
        }
    };

    if table.n_rows() != assignment.label_count {
        return Err(ToolError::new(
            LABELED,
            "CLUSTER_ASSIGNMENT_ROW_MISMATCH",
            "Cluster assignment label count does not match the dataset row count.",
            "Regenerate generate_cluster_dossier for the dataset you want to export and use the new cluster_assignment_id.",
        )
        .detail("cluster_assignment_id", json!(p.cluster_assignment_id))
        .detail("dataset_id", json!(assignment.dataset_id))
        .detail("label_count", json!(assignment.label_count))
        .detail("row_count", json!(table.n_rows())));
    }

    let names = normalize_cluster_names(&p.cluster_names, &assignment.cluster_ids, true).map_err(|reason| {
        ToolError::new(
            LABELED,
            "CLUSTER_NAMES_INVALID",
            reason,
            format!(
                "Pass cluster_names with every cluster id as a string key ({}) and a non-empty name for each.",
                py_list_ints(&assignment.cluster_ids)
            ),
        )
        .detail("valid_cluster_ids", json!(assignment.cluster_ids))
    })?;

    let labels = state.store.cluster_labels(&id).map_err(store_err)?;
    let label_names: Vec<Option<String>> = labels.iter().map(|l| names.get(l).cloned()).collect();
    table.names.push("topological_cluster_id".into());
    table.columns.push(Column::Int(labels));
    table.names.push("topological_cluster_name".into());
    table.columns.push(Column::Text(label_names));
    let row_count = table.n_rows();
    let output_path = p.output_path.clone();
    tokio::task::spawn_blocking(move || {
        pulsar_data::write_csv(&table, &output_path).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| ToolError::worker_stopped(LABELED, "EXPORT_WRITE_FAILED", &e))?
    .map_err(|e| {
        ToolError::new(
            LABELED,
            "EXPORT_WRITE_FAILED",
            e,
            "Check that output_path is writable on the host, then retry.",
        )
        .detail("output_path", json!(p.output_path))
    })?;
    Ok(ok_result(
        json!({
            "status": "ok",
            "output_path": p.output_path,
            "cluster_assignment_id": p.cluster_assignment_id,
            "run_id": assignment.run_id,
            "dataset_id": assignment.dataset_id,
            "row_count": row_count,
            "cluster_ids": assignment.cluster_ids,
        }),
        None,
    ))
}

// ---- export_dataset_bundle ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LayoutParam {
    #[default]
    Projection,
    Spectral,
    Zeros,
}

fn yes() -> bool {
    true
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct BundleParams {
    /// Absolute host-visible directory (on the MCP server machine) under which the bundle folder is created.
    pub output_dir: String,
    /// Bundle folder name created under output_dir, e.g. project_alpha.
    pub slug: String,
    /// Optional cluster_assignment_id from generate_cluster_dossier. Omit to use the session's clusters (must be the same run).
    #[serde(default)]
    pub cluster_assignment_id: Option<String>,
    /// Optional cluster names: keys are integer cluster ids as strings, e.g. {"0": "Cohort A"}; unnamed clusters keep default labels.
    #[serde(default)]
    pub cluster_names: Option<Map<String, Value>>,
    /// Optional free-text cluster descriptions keyed like cluster_names (integer cluster ids as strings).
    #[serde(default)]
    pub cluster_descriptions: Option<Map<String, Value>>,
    /// Edge weight cut-off in [0, 1] for graph/edges.parquet. Omit to use the model's resolved construction threshold.
    #[serde(default)]
    pub edges_threshold: Option<f64>,
    /// Node coordinates in graph/nodes.parquet: projection (from the fitted embedding), spectral (graph eigenvectors), or zeros.
    #[serde(default)]
    pub layout: LayoutParam,
    /// Also write the preprocessed (imputed) data to tabular/clean.parquet.
    #[serde(default)]
    pub include_clean: bool,
    /// Write export_manifest.json with metadata and counts.
    #[serde(default = "yes")]
    pub write_manifest: bool,
    /// Run from run_topological_sweep. Omit for the latest run. Must be the run the clusters were computed on.
    #[serde(default)]
    pub run_id: String,
}

pub fn bundle_spec() -> ToolSpec {
    ToolSpec {
        name: BUNDLE,
        title: "Export Dataset Bundle",
        description: "Export the fitted graph, node/edge tables, clusters and data as a Parquet bundle. Writes `{output_dir}/{slug}/` (tables, graph, groups, manifest) and returns the paths and manifest. Needs clusters for the same run (generate_cluster_dossier). For one labeled CSV use export_labeled_data.",
        input_schema: input_schema::<BundleParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("output_dir", s::string()),
                ("slug", s::string()),
                ("bundle_path", s::string()),
                ("manifest", s::open_object()),
            ],
            &["status", "output_dir", "slug", "bundle_path", "manifest"],
            false,
        )),
        read_only: false,
        idempotent: false,
    }
}

/// Python `str()` of a JSON value, for free-text metadata.
fn py_str_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        other => other.to_string(),
    }
}

/// `_normalize_cluster_metadata`: optional names or descriptions keyed by
/// existing cluster ids; values are kept as trimmed text.
pub fn normalize_cluster_metadata(
    metadata: Option<&Map<String, Value>>,
    valid: &[i64],
    name: &str,
) -> Result<std::collections::BTreeMap<i64, String>, String> {
    let mut out = std::collections::BTreeMap::new();
    for (raw_id, val) in metadata.into_iter().flatten() {
        let key = raw_id.trim();
        if key.is_empty() {
            return Err(format!("{name} contains an empty cluster ID key."));
        }
        let Ok(id) = key.parse::<i64>() else {
            return Err(format!(
                "{name} keys must be integer cluster IDs encoded as strings; got {}.",
                pulsar_analysis::plan::py_repr(raw_id)
            ));
        };
        if !valid.contains(&id) {
            return Err(format!(
                "{name} contains unknown cluster ID {id}. Available IDs: {}.",
                py_list_ints(valid)
            ));
        }
        out.insert(id, py_str_value(val).trim().to_owned());
    }
    Ok(out)
}

pub async fn bundle(state: &ServerState, p: BundleParams) -> Result<CallToolResult, ToolError> {
    let view = state.resolve_run(BUNDLE, &p.run_id)?;
    let run_id = view.run_id().clone();
    let live = view.live.clone().ok_or_else(|| {
        ToolError::new(
            BUNDLE,
            "MODEL_NOT_LOADED",
            format!("The fitted model of run {run_id} is no longer in memory."),
            "Re-run run_topological_sweep (the same config reuses cached projections), then retry.",
        )
        .detail("run_id", json!(run_id))
    })?;
    if let Some(t) = p.edges_threshold {
        if !(0.0..=1.0).contains(&t) {
            return Err(ToolError::new(
                BUNDLE,
                "INVALID_EDGES_THRESHOLD",
                "edges_threshold must be a float between 0.0 and 1.0",
                "Pass edges_threshold in [0, 1], or omit it.",
            ));
        }
    }

    let (labels, valid): (Vec<i64>, Vec<i64>) = if let Some(ca) = &p.cluster_assignment_id {
        let store_err = |e: StoreError| ToolError::from_store(BUNDLE, e);
        let id = ClusterAssignmentId::parse(ca).map_err(store_err)?;
        let assignment = state.store.get_cluster_assignment(&id).map_err(store_err)?;
        if assignment.run_id.as_ref() != Some(&run_id) {
            return Err(ToolError::new(
                BUNDLE,
                "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
                "Cluster assignment was computed from a different run than the active fitted model.",
                "Regenerate generate_cluster_dossier for the active model and use the new cluster_assignment_id.",
            )
            .detail("cluster_assignment_id", json!(ca))
            .detail("assignment_run_id", json!(assignment.run_id))
            .detail("current_run_id", json!(run_id)));
        }
        if let Some(dataset_id) = &assignment.dataset_id {
            if state.store.check_dataset(dataset_id).is_err() {
                return Err(ToolError::new(
                    BUNDLE,
                    "CLUSTER_ASSIGNMENT_DATASET_STALE",
                    "The dataset backing this cluster assignment is gone or has changed since the labels were computed, so the labels no longer align with its rows.",
                    "Re-ingest the dataset and rerun generate_cluster_dossier, then export with the new cluster_assignment_id.",
                )
                .detail("cluster_assignment_id", json!(ca))
                .detail("dataset_id", json!(dataset_id)));
            }
            if live
                .dataset_id
                .as_ref()
                .is_some_and(|current| current != dataset_id)
            {
                return Err(ToolError::new(
                    BUNDLE,
                    "CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
                    "Cluster assignment dataset does not match the active fitted model's dataset.",
                    "Export using a cluster_assignment_id generated for the active fitted model.",
                )
                .detail("cluster_assignment_id", json!(ca))
                .detail("assignment_dataset_id", json!(dataset_id))
                .detail("current_dataset_id", json!(live.dataset_id)));
            }
        }
        if assignment.label_count != live.table.n_rows() {
            return Err(ToolError::new(
                BUNDLE,
                "CLUSTER_ASSIGNMENT_ROW_MISMATCH",
                "Cluster assignment label count does not match the active model's dataset row count.",
                "Regenerate generate_cluster_dossier for this dataset and use the new cluster_assignment_id.",
            )
            .detail("cluster_assignment_id", json!(ca))
            .detail("label_count", json!(assignment.label_count))
            .detail("row_count", json!(live.table.n_rows())));
        }
        (
            state.store.cluster_labels(&id).map_err(store_err)?,
            assignment.cluster_ids,
        )
    } else {
        let Some(clusters) = state.session().clusters.clone() else {
            return Err(ToolError::new(
                BUNDLE,
                "CLUSTERS_MISSING",
                "No cluster assignment ID was provided and no clusters are active in this session.",
                "Run generate_cluster_dossier first or provide a cluster_assignment_id.",
            ));
        };
        if clusters.run_id != run_id {
            let (code, reason, action) = if p.run_id.is_empty() {
                (
                    "CLUSTER_CACHE_STALE",
                    format!("Cached cluster state is stale. It was computed from run '{}', but the active fitted model is run '{run_id}'.", clusters.run_id),
                    "Re-run generate_cluster_dossier() to recompute clusters before exporting.".to_owned(),
                )
            } else {
                (
                    "CLUSTERS_STALE",
                    format!("Session cluster labels were computed from run '{}', not the requested run '{run_id}'.", clusters.run_id),
                    format!("Rebuild the dossier for this run with generate_cluster_dossier(run_id='{run_id}') (it must be the latest run), then retry."),
                )
            };
            return Err(ToolError::new(BUNDLE, code, reason, action)
                .detail("clusters_run_id", json!(clusters.run_id))
                .detail("requested_run_id", json!(run_id)));
        }
        let mut valid = clusters.labels.clone();
        valid.sort_unstable();
        valid.dedup();
        (clusters.labels.clone(), valid)
    };

    let invalid_names = |reason: String| {
        ToolError::new(
            BUNDLE,
            "CLUSTER_NAMES_INVALID",
            reason,
            format!(
                "Pass cluster_names / cluster_descriptions as flat objects keyed by existing integer cluster ids as strings ({}).",
                py_list_ints(&valid)
            ),
        )
        .detail("valid_cluster_ids", json!(valid))
    };
    let names = normalize_cluster_metadata(p.cluster_names.as_ref(), &valid, "cluster_names")
        .map_err(invalid_names)?;
    let descs = normalize_cluster_metadata(
        p.cluster_descriptions.as_ref(),
        &valid,
        "cluster_descriptions",
    )
    .map_err(invalid_names)?;

    let layout = match p.layout {
        LayoutParam::Projection => pulsar_report::Layout::Projection,
        LayoutParam::Spectral => pulsar_report::Layout::Spectral,
        LayoutParam::Zeros => pulsar_report::Layout::Zeros,
    };
    let (output_dir, slug) = (std::path::PathBuf::from(&p.output_dir), p.slug.clone());
    let (edges_threshold, include_clean, write_manifest) =
        (p.edges_threshold, p.include_clean, p.write_manifest);
    let manifest = tokio::task::spawn_blocking(move || {
        let opts = pulsar_report::BundleOptions {
            labels: &labels,
            cluster_names: &names,
            cluster_descriptions: &descs,
            edges_threshold,
            layout,
            include_clean,
            write_manifest,
            pulsar_version: env!("CARGO_PKG_VERSION"),
        };
        pulsar_report::export_dataset_bundle(&live.fitted, &live.table, &output_dir, &slug, &opts)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| ToolError::worker_stopped(BUNDLE, "EXPORT_BUNDLE_FAILED", &e))?
    .map_err(|e| {
        ToolError::new(
            BUNDLE,
            "EXPORT_BUNDLE_FAILED",
            e,
            "Check that output_dir is writable on the host and retry.",
        )
    })?;
    let resolve =
        |p: std::path::PathBuf| std::fs::canonicalize(&p).unwrap_or(p).display().to_string();
    let output_dir = resolve(std::path::PathBuf::from(&p.output_dir));
    let bundle_path = resolve(std::path::Path::new(&p.output_dir).join(&p.slug));
    Ok(ok_result(
        json!({
            "status": "ok",
            "output_dir": output_dir,
            "slug": p.slug,
            "bundle_path": bundle_path,
            "manifest": manifest,
        }),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn cluster_names_errors_match_python() {
        let valid = [0, 1, 2];
        let err = |v: Value| normalize_cluster_names(&names(v), &valid, true).unwrap_err();
        assert_eq!(
            err(json!({" ": "x"})),
            "cluster_names contains an empty cluster ID key."
        );
        assert_eq!(
            err(json!({"a": "x"})),
            "cluster_names keys must be integer cluster IDs encoded as strings; got 'a'."
        );
        assert_eq!(
            err(json!({"0": "a", " 0": "b"})),
            "cluster_names contains duplicate cluster ID 0."
        );
        assert_eq!(
            err(json!({"7": "x"})),
            "cluster_names contains unknown cluster ID 7. Available IDs: [0, 1, 2]."
        );
        assert_eq!(
            err(json!({"1": "  "})),
            "cluster_names['1'] must be a non-empty string name."
        );
        assert_eq!(
            err(json!({"1": 3})),
            "cluster_names['1'] must be a non-empty string name."
        );
        assert_eq!(
            err(json!({"1": "x"})),
            "cluster IDs [0, 2] have no name mapping. Provide all 3 cluster IDs."
        );
        let ok = normalize_cluster_names(
            &names(json!({"0": " A ", "1": "B", "2": "C"})),
            &valid,
            true,
        )
        .unwrap();
        assert_eq!(ok[&0], "A");
    }
}
