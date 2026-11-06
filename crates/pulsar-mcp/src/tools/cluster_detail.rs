//! `get_cluster_detail`: per-cluster evidence sections for any persisted
//! cluster assignment, paged in size order within the response budget, and
//! the assignment resolution the cluster read tools share.
//!
//! An assignment is read from the store (its labels) and the fitted model of
//! its run (table and graph), so any assignment whose run is still loaded
//! works, including one that is no longer live (an earlier dossier, or a
//! merged assignment with method `merged:<orig>`). Evidence is built only for
//! the clusters of the page. Nothing here writes the session's live labels.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use pulsar_analysis::characterization::GROUP_SUMMARY_DEFAULTS;
use pulsar_analysis::cluster_payloads::{
    build_evidence_payload, build_summary_evidence_payload, cluster_result_payload,
};
use pulsar_analysis::dossier::{build_dossier, build_dossier_with_central_rows};
use pulsar_analysis::feature_evidence::{
    build_feature_evidence_index, build_feature_evidence_index_for_clusters, FeatureEvidenceIndex,
};
use pulsar_store::{ClusterAssignmentId, ClusterAssignmentRecord};
use rmcp::model::CallToolResult;
use serde_json::{json, Map, Value};

use super::dossier_budget::{
    self, cap_sections, next_ids, next_page, size_order_of_labels, truncation, truncation_line,
};
use super::dossier_render::section_markdown;
use super::params::Format;
use super::{clustering, ToolSpec};
use crate::budget;
use crate::envelope::{ok_result, ToolError};
use crate::release;
use crate::schema::{self as s, input_schema};
use crate::state::{ClusterState, EvidenceKey, LiveRun, ServerState};

pub const NAME: &str = "get_cluster_detail";

/// Clusters characterized per page at most; `next` continues.
const PAGE_MAX: usize = 25;

/// Clusters characterized for a non-live assignment read by the profile
/// tools (as many as a default dossier).
const CHARACTERIZE_MAX: usize = 50;

// ---- assignment resolution -------------------------------------------------------------

/// A persisted assignment with what its evidence is built from.
pub(crate) struct Loaded {
    pub record: ClusterAssignmentRecord,
    pub labels: Arc<Vec<i64>>,
    pub live_run: Arc<LiveRun>,
    pub exclude_columns: Vec<String>,
    /// The session's live labels are this assignment.
    pub live: Option<Arc<ClusterState>>,
}

fn live_clusters(state: &ServerState) -> Option<Arc<ClusterState>> {
    state.session().clusters.clone()
}

/// The live assignment and its clusters, for error text.
fn live_note(state: &ServerState) -> String {
    match live_clusters(state) {
        Some(c) => {
            let mut ids: Vec<i64> = size_order_of_labels(&c.labels)
                .iter()
                .map(|(id, _)| *id)
                .collect();
            format!(
                "The live assignment is {} (cluster ids {}).",
                c.cluster_assignment_id,
                id_list(&mut ids)
            )
        }
        None => "No assignment is live in this session (run generate_cluster_dossier).".into(),
    }
}

/// Ids as ranges, bounded for error text.
pub(crate) fn id_list(ids: &mut [i64]) -> String {
    let n = ids.len();
    let text = dossier_budget::ranges(ids);
    let shown = budget::cut_text(&text, 300);
    if shown.len() < text.len() {
        format!("{shown}…; {n} in all")
    } else {
        text
    }
}

/// Resolve `id` (empty: the live assignment) to its record, labels and run.
pub(crate) fn load(
    state: &ServerState,
    tool: &'static str,
    id: Option<&str>,
) -> Result<Loaded, ToolError> {
    let live = live_clusters(state);
    let id = match id.filter(|s| !s.is_empty()) {
        Some(id) => ClusterAssignmentId::parse(id).map_err(|e| ToolError::from_store(tool, e))?,
        None => match &live {
            Some(c) => c.cluster_assignment_id.clone(),
            None => {
                return Err(ToolError::new(
                    tool,
                    "CLUSTERS_NOT_BUILT",
                    "No cluster assignment is live in this session and none was named.",
                    "Run generate_cluster_dossier, or pass the cluster_assignment_id of a persisted assignment.",
                ))
            }
        },
    };
    let record = state.store.get_cluster_assignment(&id).map_err(|e| {
        let mut err = ToolError::from_store(tool, e);
        err.agent_action = format!(
            "Pass a cluster_assignment_id returned by generate_cluster_dossier (or a merge), or omit it for the live labels. {}",
            live_note(state)
        );
        err
    })?;
    let live = live.filter(|c| c.cluster_assignment_id == id);
    let Some(run_id) = record.run_id.clone() else {
        return Err(ToolError::new(
            tool,
            "ASSIGNMENT_WITHOUT_RUN",
            format!("Assignment {id} names no run, so its evidence cannot be built."),
            "Use export_labeled_data for its labels, or an assignment from generate_cluster_dossier.",
        )
        .detail("cluster_assignment_id", json!(id)));
    };
    let live_run = state.cache.get(&run_id).ok_or_else(|| {
        ToolError::new(
            tool,
            "MODEL_NOT_LOADED",
            format!("The fitted model of run {run_id}, which assignment {id} labels, is no longer in memory."),
            format!("Evidence for this assignment needs its run loaded; export_labeled_data still exports its labels. {}", live_note(state)),
        )
        .detail("cluster_assignment_id", json!(id))
        .detail("run_id", json!(run_id))
    })?;
    let labels = match &live {
        Some(c) => Arc::new(c.labels.clone()),
        None => Arc::new(
            state
                .store
                .cluster_labels(&id)
                .map_err(|e| ToolError::from_store(tool, e))?,
        ),
    };
    if labels.len() != live_run.table.n_rows() {
        return Err(ToolError::new(
            tool,
            "ASSIGNMENT_MISMATCH",
            format!(
                "Assignment {id} has {} labels but run {run_id} has {} rows.",
                labels.len(),
                live_run.table.n_rows()
            ),
            "Use an assignment produced for this run.",
        ));
    }
    let exclude_columns = match &live {
        Some(c) => c.evidence_key.exclude_columns.clone(),
        None => clustering::excluded_columns(&live_run, &[]),
    };
    Ok(Loaded {
        record,
        labels,
        live_run,
        exclude_columns,
        live,
    })
}

fn meta_of(record: &ClusterAssignmentRecord, labels: &[i64]) -> Value {
    cluster_result_payload(
        labels,
        &record.method,
        record.cluster_ids.len(),
        None,
        record.interpretation_edge_weight_threshold,
        &Value::Null,
        None,
    )
}

/// The clustering a read tool uses: the live labels (checked against
/// `run_id`), or the persisted assignment `assignment_id`. A non-live one
/// is cached in the session beside the live labels, never in their place.
pub(crate) async fn resolve(
    state: &ServerState,
    tool: &'static str,
    run_id: &str,
    assignment_id: Option<&str>,
) -> Result<Arc<ClusterState>, ToolError> {
    let wanted = assignment_id.filter(|s| !s.is_empty());
    let live_id = live_clusters(state).map(|c| c.cluster_assignment_id.to_string());
    let Some(wanted) = wanted.filter(|w| Some(*w) != live_id.as_deref()) else {
        return super::cluster_evidence::require_clusters(state, tool, run_id);
    };
    if let Some(cached) = state
        .session()
        .persisted_clusters
        .clone()
        .filter(|c| c.cluster_assignment_id.as_str() == wanted)
    {
        return Ok(cached);
    }
    let loaded = load(state, tool, Some(wanted))?;
    let run_id_of = loaded.live_run.run_id.clone();
    if !run_id.is_empty() && run_id != run_id_of.to_string() {
        return Err(ToolError::new(
            tool,
            "CLUSTERS_STALE",
            format!(
                "Assignment {wanted} labels run '{run_id_of}', not the requested run '{run_id}'."
            ),
            "Omit run_id, or pass the run_id the assignment was built on.",
        ));
    }
    let (labels, table) = (loaded.labels.clone(), loaded.live_run.clone());
    let exclude = loaded.exclude_columns.clone();
    let method = loaded.record.method.clone();
    let cut_off = loaded.record.interpretation_edge_weight_threshold;
    let (evidence, group_summary) = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut evidence = build_feature_evidence_index(
            &table.table,
            &labels,
            &table.fitted.graph.edges,
            &exclude,
            Some(CHARACTERIZE_MAX),
        )
        .map_err(|e| e.to_string())?;
        let summary = clustering::apply_summary_names(
            &mut evidence,
            &table,
            &labels,
            &exclude,
            &method,
            cut_off,
            &GROUP_SUMMARY_DEFAULTS,
        )?;
        Ok((evidence, summary))
    })
    .await
    .map_err(|e| ToolError::worker_stopped(tool, "EVIDENCE_FAILED", &e))?
    .map_err(|e| {
        ToolError::new(
            tool,
            "EVIDENCE_FAILED",
            e,
            "Check the assignment, then retry.",
        )
    })?;
    let record = &loaded.record;
    let clusters = Arc::new(ClusterState {
        run_id: run_id_of,
        cluster_assignment_id: record.cluster_assignment_id.clone(),
        labels: loaded.labels.to_vec(),
        meta: meta_of(record, &loaded.labels),
        evidence: Arc::new(evidence),
        evidence_key: EvidenceKey {
            method: record.method.clone(),
            interpretation_edge_weight_threshold: record.interpretation_edge_weight_threshold,
            exclude_columns: loaded.exclude_columns,
            max_clusters_to_characterize: CHARACTERIZE_MAX,
        },
        group_summary,
        summary_thresholds: GROUP_SUMMARY_DEFAULTS,
        resolution: None,
        dossiers: Mutex::new(HashMap::new()),
    });
    state.session().persisted_clusters = Some(clusters.clone());
    Ok(clusters)
}

/// Evidence for one cluster that `clusters.evidence` did not characterize
/// (the dossier characterizes only the largest ones), or `None` when the
/// cluster is not in the labels or the run is gone.
pub(crate) async fn evidence_for(
    state: &ServerState,
    clusters: &ClusterState,
    cluster_ids: &[i64],
) -> Option<Arc<FeatureEvidenceIndex>> {
    let present: BTreeSet<i64> = clusters.labels.iter().copied().collect();
    if !cluster_ids.iter().all(|id| present.contains(id)) {
        return None;
    }
    let run = state.cache.get(&clusters.run_id)?;
    let labels = clusters.labels.clone();
    let exclude = clusters.evidence_key.exclude_columns.clone();
    let summary = clusters.group_summary.clone();
    let thresholds = clusters.summary_thresholds;
    let method = clusters.evidence_key.method.clone();
    let cut_off = clusters.evidence_key.interpretation_edge_weight_threshold;
    let ids = cluster_ids.to_vec();
    tokio::task::spawn_blocking(move || {
        let mut evidence = build_feature_evidence_index_for_clusters(
            &run.table,
            &labels,
            &run.fitted.graph.edges,
            &exclude,
            &ids,
        )
        .ok()?;
        if summary.is_null() {
            let _ = clustering::apply_summary_names(
                &mut evidence,
                &run,
                &labels,
                &exclude,
                &method,
                cut_off,
                &thresholds,
            );
        } else {
            evidence.rename(&pulsar_analysis::characterization::auto_names(&summary));
        }
        Some(evidence)
    })
    .await
    .ok()
    .flatten()
    .map(Arc::new)
}

/// `CLUSTER_ID_UNKNOWN` naming the assignment read, its clusters, the live
/// assignment and how to read another.
pub(crate) fn unknown_cluster(
    state: &ServerState,
    tool: &'static str,
    cluster_id: i64,
    clusters: &ClusterState,
) -> ToolError {
    let mut ids: Vec<i64> = size_order_of_labels(&clusters.labels)
        .iter()
        .map(|(id, _)| *id)
        .collect();
    let live = live_clusters(state)
        .is_some_and(|c| c.cluster_assignment_id == clusters.cluster_assignment_id);
    ToolError::new(
        tool,
        "CLUSTER_ID_UNKNOWN",
        format!(
            "Cluster {cluster_id} is not in assignment {} ({}), whose cluster ids are {}.",
            clusters.cluster_assignment_id,
            if live { "the live labels" } else { "a persisted assignment" },
            id_list(&mut ids),
        ),
        format!(
            "Pick one of those ids, or pass cluster_assignment_id to read another persisted assignment (get_cluster_detail lists any assignment's clusters). {}",
            live_note(state)
        ),
    )
    .detail("cluster_id", json!(cluster_id))
    .detail("cluster_assignment_id", json!(clusters.cluster_assignment_id))
}

// ---- get_cluster_detail ----------------------------------------------------------------

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    TopologicalNeighbors,
    TierCounts,
    FeaturePreview,
    NumericFeatures,
    CategoricalFeatures,
    CentralRows,
}

const DEFAULT_SECTIONS: [Section; 4] = [
    Section::TopologicalNeighbors,
    Section::TierCounts,
    Section::NumericFeatures,
    Section::CategoricalFeatures,
];

impl Section {
    fn keys(self) -> &'static [&'static str] {
        match self {
            Self::TopologicalNeighbors => &["topological_neighbors"],
            Self::TierCounts => &["numeric_tier_counts", "categorical_tier_counts"],
            Self::FeaturePreview => &[
                "feature_preview",
                "feature_preview_limit",
                "features_previewed",
                "features_omitted",
            ],
            Self::NumericFeatures => &["numeric_features"],
            Self::CategoricalFeatures => &["categorical_features"],
            Self::CentralRows => &["central_rows"],
        }
    }

    fn standard(self) -> bool {
        matches!(
            self,
            Self::NumericFeatures | Self::CategoricalFeatures | Self::CentralRows
        )
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema, Default)]
#[serde(default)]
pub struct DetailParams {
    /// Persisted assignment to read (from generate_cluster_dossier or a merge). Omit for the live labels.
    pub cluster_assignment_id: Option<String>,
    /// Clusters to show, in this order. Omit to page all clusters by size with cursor.
    pub cluster_ids: Option<Vec<i64>>,
    /// Continue here: the cursor in a previous truncation.next. Omit to start at the largest cluster.
    pub cursor: Option<String>,
    /// Sections per cluster, at the dossier's clusters[] paths. Default: topological_neighbors, tier_counts, numeric_features, categorical_features.
    pub sections: Option<Vec<Section>>,
    /// Feature rows per list section (>= 1). Default 8.
    pub max_features: Option<usize>,
    /// Text view: 'markdown' (default) or 'json'.
    pub response_format: Format,
}

pub fn spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: NAME,
        title: "Cluster Detail",
        description: "Per-cluster evidence for any persisted cluster assignment, including merged ones (default: the live labels), paged largest first within the response budget. Continue a dossier or page via its `truncation.next` cursor; cluster_ids picks clusters. Read-only: never changes the live labels.",
        input_schema: input_schema::<DetailParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::enumeration(&["ok"])),
                ("cluster_assignment_id", s::string()),
                ("run_id", s::string()),
                ("method", s::string()),
                ("interpretation_edge_weight_threshold", s::number()),
                ("live", s::boolean()),
                ("total_clusters", s::integer()),
                ("cursor", s::nullable(s::integer())),
                ("sections", s::array(s::string())),
                ("clusters", s::array(open.clone())),
                ("truncation", s::truncation()),
                (release::KEY, open),
            ],
            &[
                "status",
                "cluster_assignment_id",
                "run_id",
                "method",
                "live",
                "total_clusters",
                "clusters",
                "truncation",
            ],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

fn invalid(reason: impl Into<String>, action: impl Into<String>) -> ToolError {
    ToolError::invalid_argument(NAME, reason, action)
}

/// One cluster entry per id with the requested sections, at the dossier's
/// paths.
fn entries(
    loaded: &Loaded,
    ids: &[i64],
    sections: &[Section],
    max_features: usize,
) -> Result<Vec<Value>, String> {
    let run = &loaded.live_run;
    let edges = &run.fitted.graph.edges;
    let reuse_live = loaded.live.as_ref().filter(|c| {
        ids.len() == c.evidence.cluster_bundles.len()
            && ids
                .iter()
                .all(|id| c.evidence.cluster_bundles.contains_key(id))
    });
    let evidence = match reuse_live {
        Some(c) => c.evidence.clone(),
        None => {
            let mut evidence = build_feature_evidence_index_for_clusters(
                &run.table,
                &loaded.labels,
                edges,
                &loaded.exclude_columns,
                ids,
            )
            .map_err(|e| e.to_string())?;
            match &loaded.live {
                Some(c) => {
                    evidence.rename(&pulsar_analysis::characterization::auto_names(
                        &c.group_summary,
                    ));
                }
                None => {
                    clustering::apply_summary_names(
                        &mut evidence,
                        run,
                        &loaded.labels,
                        &loaded.exclude_columns,
                        &loaded.record.method,
                        loaded.record.interpretation_edge_weight_threshold,
                        &GROUP_SUMMARY_DEFAULTS,
                    )?;
                }
            }
            Arc::new(evidence)
        }
    };
    let meta = meta_of(&loaded.record, &loaded.labels);
    let summary =
        build_summary_evidence_payload(&evidence, &meta, &json!({}), None, max_features, 0);
    let mut by_id: std::collections::BTreeMap<i64, Map<String, Value>> = summary["clusters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| Some((c["cluster_id"].as_i64()?, c.as_object()?.clone())))
        .collect();
    if sections.iter().any(|s| s.standard()) {
        let dossier = match reuse_live {
            Some(c) => {
                let mut guard = c.dossiers.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(d) = guard.get("standard") {
                    d.clone()
                } else {
                    let existing_central: Option<HashMap<i64, Vec<Value>>> =
                        guard.values().next().map(|d| {
                            d.clusters
                                .iter()
                                .map(|cl| (cl.cluster_id, cl.central_rows.clone()))
                                .collect()
                        });
                    let built = build_dossier_with_central_rows(
                        &evidence,
                        &run.table,
                        &loaded.labels,
                        edges,
                        "standard",
                        json!({}),
                        existing_central.as_ref(),
                    )
                    .map_err(|e| e.to_string())?;
                    guard.insert("standard".to_owned(), built.clone());
                    built
                }
            }
            None => build_dossier(
                &evidence,
                &run.table,
                &loaded.labels,
                edges,
                "standard",
                json!({}),
            )
            .map_err(|e| e.to_string())?,
        };
        let standard = build_evidence_payload(&dossier, &meta, "standard", None, max_features);
        for c in standard["clusters"].as_array().into_iter().flatten() {
            if let (Some(id), Some(map)) = (c["cluster_id"].as_i64(), c.as_object()) {
                let entry = by_id.entry(id).or_default();
                for key in ["numeric_features", "categorical_features", "central_rows"] {
                    if let Some(v) = map.get(key) {
                        entry.insert(key.into(), v.clone());
                    }
                }
            }
        }
    }
    let wanted: Vec<&str> = ["cluster_id", "size", "size_pct", "semantic_name"]
        .into_iter()
        .chain(sections.iter().flat_map(|s| s.keys().iter().copied()))
        .collect();
    Ok(ids
        .iter()
        .filter_map(|id| by_id.remove(id))
        .map(|mut m| {
            m.retain(|k, _| wanted.contains(&k.as_str()));
            Value::Object(m)
        })
        .collect())
}

pub async fn call(state: &ServerState, p: DetailParams) -> Result<CallToolResult, ToolError> {
    if p.cluster_ids.is_some() && p.cursor.is_some() {
        return Err(invalid(
            "cluster_ids and cursor are exclusive",
            "Pass cluster_ids to pick clusters, or cursor to continue paging, not both.",
        ));
    }
    let max_features = p.max_features.unwrap_or(8);
    if max_features < 1 {
        return Err(invalid(
            format!("max_features must be >= 1, got '{max_features}'"),
            "Pass max_features >= 1.",
        ));
    }
    let mut sections: Vec<Section> = p
        .sections
        .clone()
        .unwrap_or_else(|| DEFAULT_SECTIONS.to_vec());
    sections.dedup();
    let loaded = load(state, NAME, p.cluster_assignment_id.as_deref())?;
    let assignment_id = loaded.record.cluster_assignment_id.to_string();
    let order = size_order_of_labels(&loaded.labels);
    let total = order.len();

    // The ids of this request, in order, and where a cursor page starts.
    let (scope, start): (Vec<i64>, Option<usize>) = match &p.cluster_ids {
        Some(ids) => {
            let present: BTreeSet<i64> = order.iter().map(|(id, _)| *id).collect();
            let missing: Vec<i64> = ids
                .iter()
                .copied()
                .filter(|id| !present.contains(id))
                .collect();
            if let Some(&first) = missing.first() {
                let clusters = ClusterState {
                    run_id: loaded.live_run.run_id.clone(),
                    cluster_assignment_id: loaded.record.cluster_assignment_id.clone(),
                    labels: loaded.labels.to_vec(),
                    meta: Value::Null,
                    evidence: Arc::new(FeatureEvidenceIndex::default()),
                    evidence_key: EvidenceKey {
                        method: String::new(),
                        interpretation_edge_weight_threshold: 0.0,
                        exclude_columns: vec![],
                        max_clusters_to_characterize: 0,
                    },
                    group_summary: Value::Null,
                    summary_thresholds: GROUP_SUMMARY_DEFAULTS,
                    resolution: None,
                    dossiers: Mutex::new(HashMap::new()),
                };
                return Err(unknown_cluster(state, NAME, first, &clusters)
                    .detail("unknown_cluster_ids", json!(missing)));
            }
            let mut seen = BTreeSet::new();
            (
                ids.iter().copied().filter(|id| seen.insert(*id)).collect(),
                None,
            )
        }
        None => {
            let offset = match p.cursor.as_deref() {
                None | Some("") => 0,
                Some(c) => c.parse::<usize>().map_err(|_| {
                    invalid(
                        format!("cursor '{c}' is not a cursor from truncation.next"),
                        "Pass the cursor exactly as a previous truncation.next gave it, or omit it to start at the largest cluster.",
                    )
                })?,
            };
            let ids = order.iter().skip(offset).map(|(id, _)| *id).collect();
            (ids, Some(offset.min(total)))
        }
    };
    let page: Vec<i64> = scope.iter().copied().take(PAGE_MAX).collect();

    let worker_sections = sections.clone();
    let worker_page = page.clone();
    let (loaded, built) = tokio::task::spawn_blocking(move || {
        let built = entries(&loaded, &worker_page, &worker_sections, max_features);
        (loaded, built)
    })
    .await
    .map_err(|e| ToolError::worker_stopped(NAME, "EVIDENCE_FAILED", &e))?;
    let built = built.map_err(|e| {
        ToolError::new(
            NAME,
            "EVIDENCE_FAILED",
            e,
            "Check the assignment and arguments, then retry.",
        )
    })?;

    let section_names: Vec<String> = sections
        .iter()
        .map(|s| {
            serde_json::to_value(s)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_default()
        })
        .collect();
    let mut extra = Map::new();
    if p.sections.is_some() {
        extra.insert("sections".into(), json!(section_names));
    }
    if p.max_features.is_some() {
        extra.insert("max_features".into(), json!(max_features));
    }
    let budget = budget::render_budget(state.options.max_response_bytes);
    let head = json!({
        "status": "ok",
        "cluster_assignment_id": assignment_id,
        "run_id": loaded.live_run.run_id,
        "method": loaded.record.method,
        "interpretation_edge_weight_threshold": loaded.record.interpretation_edge_weight_threshold,
        "live": loaded.live.is_some(),
        "total_clusters": total,
        "cursor": start,
        "sections": section_names,
    });
    let render = |list: &[Value], n: usize, trimmed: bool| -> (Value, String) {
        let mut payload = head.clone();
        payload["clusters"] = json!(list[..n]);
        let rest: Vec<i64> = scope[n..].to_vec();
        let next = match start {
            Some(offset) if !rest.is_empty() => Some(next_page(&assignment_id, offset + n, &extra)),
            None if !rest.is_empty() => Some(next_ids(&assignment_id, &rest, &extra)),
            _ => None,
        };
        let t = truncation(n, scope.len(), &rest, trimmed, budget, next);
        let line = truncation_line(&t);
        payload["truncation"] = t;
        let text = match p.response_format {
            Format::Json => serde_json::to_string(&payload).unwrap_or_default(),
            Format::Markdown => {
                let body: Vec<String> = list[..n].iter().map(section_markdown).collect();
                format!(
                    "# Cluster Detail\n\n- cluster_assignment_id: `{assignment_id}` ({}; method {})\n- {line}\n\n{}",
                    if loaded.live.is_some() { "live labels" } else { "persisted, not live" },
                    loaded.record.method,
                    body.join("\n\n")
                )
            }
        };
        release::attach_detail(state, &loaded.live_run.run_id, &loaded.labels, &mut payload);
        (payload, text)
    };
    // The release block grows with the page (labels per cluster entry), so
    // it is counted in the fit, though the gateway strips it.
    let fits =
        |pair: &(Value, String)| budget::json_chars(&pair.0) + pair.1.chars().count() <= budget;

    let finish = |(payload, text): (Value, String)| Ok(ok_result(payload, Some(text)));

    let mut list = built;
    let n = list.len();
    let full = render(&list, n, false);
    if fits(&full) {
        return finish(full);
    }
    for cap in [3, 1] {
        for c in &mut list {
            cap_sections(c, cap.min(max_features));
        }
        let pair = render(&list, n, true);
        if fits(&pair) {
            return finish(pair);
        }
    }
    // Fewer clusters; at least one, so paging always moves.
    let (mut lo, mut hi) = (1, n);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(&render(&list, mid, true)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let n = lo.min(n);
    let mut final_page = render(&list, n, true);
    if n == 1 && !fits(&final_page) {
        budget::trim_value(&mut list[0], budget / 3);
        final_page = render(&list, 1, true);
    }
    if !fits(&final_page) {
        return Err(page_too_large(
            &final_page.0,
            &final_page.1,
            budget,
            &assignment_id,
        ));
    }
    finish(final_page)
}

/// `RESPONSE_TOO_LARGE` for a page already cut to one cluster: names the
/// sizes and the one request that can still help, so the agent adapts rather
/// than retrying. The release block (the page's row sets, as ranges or
/// `bitmap_b64`, whichever is shorter) is counted in the page's size.
fn page_too_large(payload: &Value, text: &str, budget: usize, assignment_id: &str) -> ToolError {
    let size = budget::json_chars(payload) + text.chars().count();
    let release_chars = payload.get(release::KEY).map_or(0, budget::json_chars);
    let evidence_chars = size - release_chars;
    let cluster = payload["clusters"][0]["cluster_id"].as_i64();
    let which = cluster.map_or_else(String::new, |id| format!(" (cluster {id})"));
    let (reason, action) = if release_chars >= budget {
        (
            format!(
                "Even a one-cluster page{which} is {size} characters, over the {budget}-character budget: its release block (row sets, already in the shorter of ranges and bitmap_b64) is {release_chars} characters on its own, so no section or page setting shrinks it."
            ),
            format!(
                "Raise PULSAR_MCP_MAX_RESPONSE_BYTES to at least {size}, or read this cluster with get_cluster_profile. Fewer sections, a lower max_features or another page do not help."
            ),
        )
    } else {
        (
            format!(
                "Even a one-cluster page{which} is {size} characters, over the {budget}-character budget: {evidence_chars} of evidence plus a {release_chars}-character release block (row sets, already in the shorter of ranges and bitmap_b64). The page cannot get smaller than one cluster."
            ),
            format!(
                "Re-call get_cluster_detail with cluster_ids=[{}], sections=['tier_counts'] and max_features=1 (evidence must fit in {} characters); or use get_cluster_profile; or raise PULSAR_MCP_MAX_RESPONSE_BYTES to at least {size}.",
                cluster.map_or_else(|| "<one id>".to_owned(), |id| id.to_string()),
                budget - release_chars
            ),
        )
    };
    ToolError::new(NAME, "RESPONSE_TOO_LARGE", reason, action)
        .detail("size_chars", json!(size))
        .detail("limit_chars", json!(budget))
        .detail("release_chars", json!(release_chars))
        .detail("evidence_chars", json!(evidence_chars))
        .detail("cluster_assignment_id", json!(assignment_id))
}
