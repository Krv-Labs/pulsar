//! Process-wide server state: the store, the bounded model cache, the
//! single-client session, and run resolution for tools.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use pulsar_analysis::characterization::GroupSummaryThresholds;
use pulsar_analysis::clustering::resolve::ResolveMethod;
use pulsar_analysis::dossier::TopologicalDossier;
use pulsar_analysis::feature_evidence::FeatureEvidenceIndex;
use pulsar_config::PulsarConfig;
use pulsar_core::TemporalCosmicGraph;
use pulsar_data::Key;
use pulsar_data::Table;
use pulsar_pipeline::CosmicTrajectory;
use pulsar_pipeline::{Fitted, Graph};
use pulsar_store::{
    ClusterAssignmentId, DatasetId, LongitudinalId, ModelCache, RunId, RunRecord, Store, StoreError,
};
use rmcp::task_manager::TaskManager;

use crate::envelope::ToolError;

/// Default `PULSAR_MCP_MAX_RESPONSE_BYTES`, the hard limit in agent-visible
/// characters: under Claude Code's 50,000-character spill line (see
/// [`crate::budget`]).
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = crate::budget::DEFAULT_HARD_LIMIT;

/// A fitted model held in memory with what it was fitted on.
pub struct LiveRun {
    pub run_id: RunId,
    pub fitted: Fitted,
    pub table: Table,
    pub dataset_id: Option<DatasetId>,
    pub data_path: Option<PathBuf>,
    /// Projection fingerprint of the embeddings (cache reuse across retries).
    pub projection_fingerprint: String,
    /// Columns the run's preprocessing plan names as identifiers (empty
    /// when the run matches no stored plan).
    pub identifier_columns: Vec<String>,
}

impl LiveRun {
    /// Approximate bytes pinned by this entry (table, embeddings, affinity).
    pub fn memory_bytes(&self) -> usize {
        let table = self.table.n_rows() * self.table.names.len() * 8;
        let embeddings: usize = self.fitted.embeddings.iter().map(|e| e.len() * 8).sum();
        let affinity = self.fitted.cosmic.weighted_edges().len() * 24;
        table + embeddings + affinity
    }
}

/// Session state of the (stdio) client.
#[derive(Debug, Default, Clone)]
pub struct Session {
    pub latest_dataset_id: Option<DatasetId>,
    pub latest_run_id: Option<RunId>,
    pub active_config_yaml: Option<String>,
    pub active_config_dataset_id: Option<DatasetId>,
    /// The preprocessing plan most recently drafted or edited.
    pub latest_plan_id: Option<pulsar_store::PlanId>,
    /// Labels and evidence published by `generate_cluster_dossier`; a new
    /// sweep clears them.
    pub clusters: Option<Arc<ClusterState>>,
    /// The last persisted (non-live) assignment a cluster read tool loaded
    /// by `cluster_assignment_id`; never the live labels.
    pub persisted_clusters: Option<Arc<ClusterState>>,
    /// Longitudinal panels built in this session, oldest first (at most
    /// [`MAX_LONGITUDINAL_ARTIFACTS`]).
    pub longitudinal: Vec<Arc<LongitudinalArtifact>>,
    /// Graph artifacts from `create_graph_artifact`, oldest first (at most
    /// [`MAX_GRAPH_ARTIFACTS`]). A new sweep marks them stale, not gone.
    pub graph_artifacts: Vec<Arc<GraphArtifact>>,
    /// True iff the client declared experimental `io.phirewall/release` v1 at initialize.
    pub phirewall_release: bool,
}

/// Graph artifacts kept in the session.
pub const MAX_GRAPH_ARTIFACTS: usize = 8;

/// A derived graph (spectral sparsifier) of one run, with its measurements.
#[derive(Debug)]
pub struct GraphArtifact {
    /// `graph_<12 hex>`.
    pub artifact_id: String,
    pub run_id: RunId,
    pub kind: String,
    pub params: serde_json::Value,
    /// Weights on `[0, 1]`.
    pub graph: Graph,
    pub metrics: pulsar_analysis::GraphMetrics,
}

/// Longitudinal panels kept in memory; older ones keep only their store
/// record (enough to rebuild).
pub const MAX_LONGITUDINAL_ARTIFACTS: usize = 4;

/// One built longitudinal panel and the representations it was built with.
#[derive(Debug)]
pub struct LongitudinalArtifact {
    pub longitudinal_id: LongitudinalId,
    pub dataset_id: Option<DatasetId>,
    pub representation: &'static str,
    pub trajectory: Option<CosmicTrajectory>,
    pub temporal: Option<TemporalCosmicGraph>,
    pub entity_column: String,
    pub time_column: String,
    pub feature_columns: Vec<String>,
    pub times: Vec<Key>,
    /// The pivot report.
    pub report: serde_json::Value,
    pub config_yaml: String,
}

/// Cached graph-level metrics and sorted structures for one `run_id`,
/// invalidated when a new sweep replaces the session's run.
#[derive(Debug, Clone)]
pub struct RunGraphCache {
    pub run_id: RunId,
    pub affinity: Arc<Graph>,
    pub threshold_graph: Arc<pulsar_analysis::ThresholdGraph>,
    pub stability: Option<Arc<pulsar_core::StabilityResult>>,
    pub metrics: Option<pulsar_analysis::GraphMetrics>,
    pub summary: Option<Arc<pulsar_analysis::GraphSummary>>,
}

impl RunGraphCache {
    pub fn build(
        run_id: RunId,
        graph: &Graph,
        affinity: Graph,
        resolved_construction_threshold: f64,
        n_ball_maps: usize,
        minhash_d: Option<usize>,
        fitted_stability: Option<&pulsar_core::StabilityResult>,
    ) -> Self {
        let threshold_graph = Arc::new(pulsar_analysis::ThresholdGraph::from_edges(
            affinity.n,
            &affinity.edges,
        ));
        let (metrics, summary) = match pulsar_analysis::GraphAnalysis::new(graph) {
            Ok(analysis) => {
                let m = analysis.metrics(
                    resolved_construction_threshold,
                    n_ball_maps,
                    Some(&affinity.edges),
                    minhash_d,
                );
                let s = Arc::new(analysis.summary(resolved_construction_threshold));
                (Some(m), Some(s))
            }
            Err(_) => (None, None),
        };
        let stability = match fitted_stability {
            Some(s) => Some(Arc::new(s.clone())),
            None => pulsar_core::find_stable_thresholds_from_edges(
                affinity.n,
                &affinity.edges,
                pulsar_core::DEFAULT_NUM_BINS,
            )
            .ok()
            .map(Arc::new),
        };
        Self {
            run_id,
            affinity: Arc::new(affinity),
            threshold_graph,
            stability,
            metrics,
            summary,
        }
    }

    pub fn from_live(live: &LiveRun, record: &RunRecord) -> Self {
        let fitted = &live.fitted;
        let affinity = Graph {
            n: fitted.n,
            edges: fitted.weighted_edges(Some(0.0)),
        };
        let minhash_d = match fitted.config.cosmic_graph.construction {
            pulsar_config::Construction::Minhash => {
                Some(fitted.config.cosmic_graph.minhash_d as usize)
            }
            pulsar_config::Construction::Exact => None,
        };
        Self::build(
            record.run_id.clone(),
            &fitted.graph,
            affinity,
            record.resolved_construction_threshold,
            record.n_ball_maps,
            minhash_d,
            fitted.stability.as_ref(),
        )
    }
}

/// Cached clustering resolution for the live `ClusterState`.
#[derive(Debug, Clone)]
pub struct CachedResolution {
    pub method: ResolveMethod,
    pub max_k: usize,
    pub interpretation_threshold: f64,
    pub method_used: String,
    pub n_clusters: usize,
    pub silhouette: Option<f64>,
    pub threshold_applied: f64,
    pub plateaus: serde_json::Value,
    pub failure_reason: Option<String>,
}

/// One published clustering of the session's latest run: what the cluster
/// tools read, and the evidence cache a repeated dossier reuses.
#[derive(Debug)]
pub struct ClusterState {
    pub run_id: RunId,
    pub cluster_assignment_id: ClusterAssignmentId,
    pub labels: Vec<i64>,
    /// `cluster_result_payload` of the clustering.
    pub meta: serde_json::Value,
    pub evidence: Arc<FeatureEvidenceIndex>,
    /// Everything besides the run and labels that `evidence` depends on.
    pub evidence_key: EvidenceKey,
    /// The dossier's group_summary, reused while the evidence and the
    /// thresholds match.
    pub group_summary: serde_json::Value,
    pub summary_thresholds: GroupSummaryThresholds,
    /// Cached clustering cut when computed via `resolve_clusters`.
    pub resolution: Option<CachedResolution>,
    /// Cached `TopologicalDossier` by detail (`"standard"` / `"full"`).
    pub dossiers: Mutex<HashMap<String, TopologicalDossier>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceKey {
    pub method: String,
    pub interpretation_edge_weight_threshold: f64,
    /// Sorted.
    pub exclude_columns: Vec<String>,
    pub max_clusters_to_characterize: usize,
}

#[derive(Debug, Clone)]
pub struct ServerOptions {
    /// `None` disables the response-size backstop.
    pub max_response_bytes: Option<usize>,
    pub transport: String,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            max_response_bytes: max_response_bytes_from_env(),
            transport: "stdio".into(),
        }
    }
}

fn max_response_bytes_from_env() -> Option<usize> {
    match std::env::var("PULSAR_MCP_MAX_RESPONSE_BYTES") {
        Ok(raw) => match raw.trim().parse::<i64>() {
            Ok(v) if v > 0 => Some(v as usize),
            Ok(_) => None,
            Err(_) => Some(DEFAULT_MAX_RESPONSE_BYTES),
        },
        Err(_) => Some(DEFAULT_MAX_RESPONSE_BYTES),
    }
}

pub struct ServerState {
    pub store: Store,
    pub cache: ModelCache<LiveRun>,
    pub tasks: TaskManager,
    pub options: ServerOptions,
    session: Mutex<Session>,
    run_graph_cache: Arc<Mutex<HashMap<RunId, Arc<RunGraphCache>>>>,
}

impl ServerState {
    /// Open the state over `store`. Jobs a previous process left running are
    /// failed as `INTERRUPTED` here, before any new job can be submitted.
    pub fn new(store: Store, options: ServerOptions) -> Result<Self, StoreError> {
        store.recover_interrupted()?;
        Ok(Self {
            store,
            cache: ModelCache::from_env(),
            tasks: TaskManager::new(),
            options,
            session: Mutex::new(Session::default()),
            run_graph_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn session(&self) -> MutexGuard<'_, Session> {
        self.session.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn session_snapshot(&self) -> Session {
        self.session().clone()
    }

    pub fn phirewall_release(&self) -> bool {
        self.session().phirewall_release
    }

    /// Replace the cached graph metrics for the newly fitted run and
    /// invalidate any superseded run's cached graph metrics.
    pub fn set_run_graph_cache(&self, replaced: Option<&RunId>, entry: Arc<RunGraphCache>) {
        let mut guard = self
            .run_graph_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(prev) = replaced {
            guard.remove(prev);
        }
        guard.insert(entry.run_id.clone(), entry);
    }

    /// Return the cached `RunGraphCache` for `record.run_id`, building and
    /// inserting it from `live` on a miss.
    pub fn graph_cache_for_live(&self, live: &LiveRun, record: &RunRecord) -> Arc<RunGraphCache> {
        if let Some(entry) = self
            .run_graph_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&record.run_id)
            .cloned()
        {
            return entry;
        }
        let arc = Arc::new(RunGraphCache::from_live(live, record));
        self.run_graph_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(record.run_id.clone(), arc.clone());
        arc
    }

    /// Resolve `run_id` (empty = the session's latest run) to its persisted
    /// record plus the live model when it is still cached.
    pub fn resolve_run(&self, tool: &'static str, run_id: &str) -> Result<RunView, ToolError> {
        let id = if run_id.is_empty() {
            self.session()
                .latest_run_id
                .clone()
                .ok_or_else(|| ToolError::model_not_loaded(tool))?
        } else {
            RunId::parse(run_id).map_err(|e| ToolError::from_store(tool, e))?
        };
        let record = self
            .store
            .get_run(&id)
            .map_err(|e| ToolError::from_store(tool, e))?;
        Ok(RunView {
            live: self.cache.get(&id),
            record,
            store: self.store.clone(),
            run_graph_cache: self.run_graph_cache.clone(),
        })
    }

    /// `model_cache` block of `get_runtime_context`.
    pub fn cache_stats(&self) -> serde_json::Value {
        let ids = self.cache.run_ids();
        let entries: Vec<serde_json::Value> = ids
            .iter()
            .map(|id| {
                let bytes = self.cache.get(id).map(|r| r.memory_bytes()).unwrap_or(0);
                serde_json::json!({"run_id": id, "memory_bytes": bytes})
            })
            .collect();
        let total: u64 = entries
            .iter()
            .map(|e| e["memory_bytes"].as_u64().unwrap_or(0))
            .sum();
        serde_json::json!({
            "size": ids.len(),
            "capacity": self.cache.capacity(),
            "run_ids": ids,
            "memory_bytes": total,
            "entries": entries,
            "model_bytes_estimated": true,
        })
    }
}

/// One resolved run: its record and, when cached, the fitted model.
pub struct RunView {
    pub record: RunRecord,
    pub live: Option<Arc<LiveRun>>,
    store: Store,
    run_graph_cache: Arc<Mutex<HashMap<RunId, Arc<RunGraphCache>>>>,
}

impl RunView {
    pub fn run_id(&self) -> &RunId {
        &self.record.run_id
    }

    /// Cached graph metrics, sorted edge structures and stability curve for
    /// this `run_id`, computed once on demand and reused across tools.
    pub fn graph_cache(&self) -> Result<Arc<RunGraphCache>, StoreError> {
        if let Some(entry) = self
            .run_graph_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&self.record.run_id)
            .cloned()
        {
            return Ok(entry);
        }
        let built = match &self.live {
            Some(live) => RunGraphCache::from_live(live, &self.record),
            None => {
                let graph = self.store.run_graph(&self.record.run_id)?;
                let affinity = self.store.run_affinity(&self.record.run_id)?;
                RunGraphCache::build(
                    self.record.run_id.clone(),
                    &graph,
                    affinity,
                    self.record.resolved_construction_threshold,
                    self.record.n_ball_maps,
                    self.minhash_d(),
                    None,
                )
            }
        };
        let arc = Arc::new(built);
        self.run_graph_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(self.record.run_id.clone(), arc.clone());
        Ok(arc)
    }

    /// The constructed graph (live model or the `graph.arrow` sidecar).
    pub fn graph(&self) -> Result<Graph, StoreError> {
        match &self.live {
            Some(live) => Ok(live.fitted.graph.clone()),
            None => self.store.run_graph(&self.record.run_id),
        }
    }

    /// The full weighted affinity (live model or the `affinity.arrow` sidecar).
    pub fn affinity(&self) -> Result<Graph, StoreError> {
        if let Some(entry) = self
            .run_graph_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&self.record.run_id)
            .cloned()
        {
            return Ok((*entry.affinity).clone());
        }
        match &self.live {
            Some(live) => Ok(Graph {
                n: live.fitted.n,
                edges: live.fitted.weighted_edges(Some(0.0)),
            }),
            None => self.store.run_affinity(&self.record.run_id),
        }
    }

    /// The run's resolved config (from the persisted YAML).
    pub fn config(&self) -> Option<PulsarConfig> {
        pulsar_config::from_yaml_str(&self.record.config_yaml).ok()
    }

    /// MinHash depth when the run was built that way.
    pub fn minhash_d(&self) -> Option<usize> {
        let cfg = self.config()?;
        match cfg.cosmic_graph.construction {
            pulsar_config::Construction::Minhash => Some(cfg.cosmic_graph.minhash_d as usize),
            pulsar_config::Construction::Exact => None,
        }
    }
}
