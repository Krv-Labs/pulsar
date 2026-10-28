//! The persisted records. JSON metadata with a `schema` version; the
//! unbounded arrays live in Arrow IPC sidecars next to them.

use std::path::PathBuf;

use pulsar_config::{Construction, ProjectionMethod, PulsarConfig, Threshold};
use pulsar_pipeline::{Fitted, Graph};
use serde::{Deserialize, Serialize};

use crate::handle::{ClusterAssignmentId, DatasetId, LongitudinalId, PlanId, RunId, UploadId};

/// Bumped whenever a record's JSON shape changes incompatibly. Readers refuse
/// other versions with `STALE_RECORD` instead of guessing.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSource {
    /// Registered in place; the file stays where the user keeps it.
    Path,
    /// Copied into the store from an upload.
    Upload,
    /// Written by `sample_longitudinal_entities` from another dataset.
    EntitySample,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub schema: u32,
    pub dataset_id: DatasetId,
    pub path: PathBuf,
    pub name: String,
    pub size_bytes: u64,
    pub mtime_ns: u64,
    pub ingested_at: f64,
    pub source: DatasetSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UploadRecord {
    pub schema: u32,
    pub upload_id: UploadId,
    pub filename: String,
    pub media_type: String,
    pub created_at: f64,
    pub bytes_received: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdPolicy {
    Auto,
    Fixed,
}

/// The parts of a config that sweep history and comparison read, so no
/// YAML is re-parsed on the read path. `config_yaml` keeps the whole thing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigSummary {
    pub run_name: String,
    pub projection_method: ProjectionMethod,
    pub projection_dimensions: Vec<usize>,
    pub n_seeds: usize,
    pub n_epsilons: usize,
    pub construction: Construction,
    pub threshold_policy: ThresholdPolicy,
    /// The fixed construction threshold the config asked for; `None` on auto.
    pub requested_threshold: Option<f64>,
    pub sparsify: bool,
}

impl ConfigSummary {
    pub fn from_config(cfg: &PulsarConfig) -> Self {
        let (threshold_policy, requested_threshold) = match cfg.cosmic_graph.construction_threshold
        {
            Threshold::Auto => (ThresholdPolicy::Auto, None),
            Threshold::Value(v) => (ThresholdPolicy::Fixed, Some(v)),
        };
        Self {
            run_name: cfg.run_name.clone(),
            projection_method: cfg.projection.method,
            projection_dimensions: cfg.projection.dimensions.clone(),
            n_seeds: cfg.projection.seeds.len(),
            n_epsilons: cfg.ball_mapper.epsilons.len(),
            construction: cfg.cosmic_graph.construction,
            threshold_policy,
            requested_threshold,
            sparsify: cfg.cosmic_graph.sparsify,
        }
    }

    /// `"auto"` or the requested number, as the Python history labelled it.
    pub fn threshold_label(&self) -> String {
        match self.requested_threshold {
            Some(v) => v.to_string(),
            None => "auto".to_owned(),
        }
    }
}

/// Component-level metrics of the constructed graph: what sweep history,
/// comparison and the health label need. Weight percentiles and advisories
/// belong to the diagnostics port (#64).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetrics {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub density: f64,
    pub avg_degree: f64,
    pub component_count: usize,
    /// Descending.
    pub component_sizes: Vec<usize>,
    pub giant_fraction: f64,
    pub singleton_count: usize,
    pub singleton_fraction: f64,
}

impl RunMetrics {
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.n;
        let n_edges = graph.n_edges();
        let max_pairs = n * n.saturating_sub(1) / 2;
        let sizes = graph.component_sizes();
        let singleton_count = sizes.iter().filter(|&&s| s == 1).count();
        let frac = |count: usize| if n > 0 { count as f64 / n as f64 } else { 0.0 };
        Self {
            n_nodes: n,
            n_edges,
            density: if max_pairs > 0 {
                n_edges as f64 / max_pairs as f64
            } else {
                0.0
            },
            avg_degree: if n > 0 {
                2.0 * n_edges as f64 / n as f64
            } else {
                0.0
            },
            component_count: sizes.len(),
            giant_fraction: frac(sizes.first().copied().unwrap_or(0)),
            singleton_count,
            singleton_fraction: frac(singleton_count),
            component_sizes: sizes,
        }
    }

    /// The health label sweep history reports (`history.py::_health_label`).
    pub fn health_label(&self) -> &'static str {
        const HAIRBALL_DENSITY: f64 = 0.8;
        const SPARSE_DENSITY: f64 = 0.05;
        const FRAGMENTED_SINGLETON_FRACTION: f64 = 0.25;
        const DOMINANT_GIANT: f64 = 0.95;
        if self.n_edges == 0 {
            "empty"
        } else if self.density > HAIRBALL_DENSITY {
            "hairball"
        } else if self.singleton_fraction > FRAGMENTED_SINGLETON_FRACTION {
            "fragmented"
        } else if self.density < SPARSE_DENSITY {
            "sparse"
        } else if self.giant_fraction > DOMINANT_GIANT {
            "giant_dominant"
        } else {
            "connected"
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlateauSummary {
    pub start: f64,
    pub end: f64,
    pub midpoint: f64,
    pub length: f64,
    pub component_count: usize,
    pub singleton_count: usize,
    pub singleton_fraction: f64,
}

/// The compact stability block of a run that used `construction_threshold:
/// auto` (`diagnostics.py::_threshold_stability_summary`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilitySummary {
    pub selected_threshold: f64,
    pub top_plateaus: Vec<PlateauSummary>,
    pub warning: Option<String>,
}

impl StabilitySummary {
    const TOP_K: usize = 3;

    /// `None` when the run used a fixed threshold.
    pub fn from_fitted(fitted: &Fitted, metrics: &RunMetrics) -> Option<Self> {
        let stability = fitted.stability.as_ref()?;
        let n = fitted.n;
        let mut row_max = vec![0.0_f64; n];
        for (i, j, w) in fitted.weighted_edges(Some(0.0)) {
            row_max[i] = row_max[i].max(w);
            row_max[j] = row_max[j].max(w);
        }
        let singletons_at = |t: f64| row_max.iter().filter(|&&v| v <= t).count();
        let top_plateaus = stability
            .plateaus
            .iter()
            .take(Self::TOP_K)
            .map(|p| {
                let singleton_count = singletons_at(p.midpoint());
                PlateauSummary {
                    start: p.start_threshold,
                    end: p.end_threshold,
                    midpoint: p.midpoint(),
                    length: p.length(),
                    component_count: p.component_count,
                    singleton_count,
                    singleton_fraction: round4(singleton_count as f64 / n.max(1) as f64),
                }
            })
            .collect();
        let warning = if metrics.n_edges == 0 {
            Some(
                "Auto threshold landed in an empty-graph regime. Pick a lower plateau midpoint \
                 and re-run, or fix construction_threshold."
                    .to_owned(),
            )
        } else if metrics.singleton_fraction > 0.8 {
            Some(
                "Auto threshold left >80% of nodes as singletons. Try a lower plateau midpoint \
                 to retain more edges."
                    .to_owned(),
            )
        } else {
            None
        };
        Some(Self {
            selected_threshold: stability.optimal_threshold,
            top_plateaus,
            warning,
        })
    }
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// A completed run. The constructed graph and the full weighted affinity are
/// sidecars (`graph.arrow`, `affinity.arrow`) in the run directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub schema: u32,
    pub run_id: RunId,
    pub dataset_id: Option<DatasetId>,
    pub data_path: Option<String>,
    pub config_yaml: String,
    pub config: ConfigSummary,
    pub n_rows: usize,
    pub n_features: usize,
    pub n_embeddings: usize,
    pub n_ball_maps: usize,
    pub resolved_construction_threshold: f64,
    pub weight_scale: f64,
    pub metrics: RunMetrics,
    pub threshold_stability_summary: Option<StabilitySummary>,
    pub created_at: f64,
}

impl RunRecord {
    /// Describe a fitted model. `created_at` is set here; the store stamps
    /// nothing else.
    pub fn from_fitted(
        run_id: RunId,
        fitted: &Fitted,
        dataset_id: Option<DatasetId>,
        data_path: Option<&str>,
    ) -> Self {
        let metrics = RunMetrics::from_graph(&fitted.graph);
        let threshold_stability_summary = StabilitySummary::from_fitted(fitted, &metrics);
        Self {
            schema: SCHEMA_VERSION,
            run_id,
            dataset_id,
            data_path: data_path.map(str::to_owned),
            config_yaml: pulsar_config::to_yaml(&fitted.config),
            config: ConfigSummary::from_config(&fitted.config),
            n_rows: fitted.n,
            n_features: fitted.layout.feature_names.len(),
            n_embeddings: fitted.embeddings.len(),
            n_ball_maps: fitted.ball_maps.len(),
            resolved_construction_threshold: fitted.construction_threshold,
            weight_scale: fitted.weight_scale,
            metrics,
            threshold_stability_summary,
            created_at: crate::atomic::now_seconds(),
        }
    }
}

/// The edge lists persisted with a run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunArtifacts {
    /// The constructed graph: `[0, 1]` weights strictly above the
    /// construction threshold.
    pub graph: Graph,
    /// The full weighted affinity (every positive `[0, 1]` weight), so
    /// interpretation thresholds can be re-applied without a refit.
    pub affinity: Graph,
}

impl RunArtifacts {
    pub fn from_fitted(fitted: &Fitted) -> Self {
        Self {
            graph: fitted.graph.clone(),
            affinity: Graph {
                n: fitted.n,
                edges: fitted.weighted_edges(Some(0.0)),
            },
        }
    }
}

/// Input for [`crate::Store::save_cluster_assignment`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterAssignment<'a> {
    pub run_id: Option<RunId>,
    pub dataset_id: Option<DatasetId>,
    pub method: String,
    pub interpretation_edge_weight_threshold: f64,
    pub threshold_source: String,
    pub construction_threshold: f64,
    /// One label per dataset row.
    pub labels: &'a [i64],
}

/// A persisted label vector. The labels are the `labels.arrow` sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterAssignmentRecord {
    pub schema: u32,
    pub cluster_assignment_id: ClusterAssignmentId,
    pub run_id: Option<RunId>,
    pub dataset_id: Option<DatasetId>,
    pub method: String,
    pub interpretation_edge_weight_threshold: f64,
    pub threshold_source: String,
    pub construction_threshold: f64,
    pub label_count: usize,
    /// Sorted distinct labels.
    pub cluster_ids: Vec<i64>,
    pub created_at: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    Trajectory,
    Temporal,
    Both,
}

/// Metadata of a built longitudinal panel. The matrices stay in the model
/// cache, as in v0.5; this record makes the handle survive a restart with
/// enough context to rebuild.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalRecord {
    pub schema: u32,
    pub longitudinal_id: LongitudinalId,
    pub dataset_id: Option<DatasetId>,
    pub representation: Representation,
    pub config_yaml: String,
    /// Panel summary (entity / time / column counts); shape owned by #66.
    pub panel: serde_json::Value,
    pub created_at: f64,
}

/// A preprocessing plan. The plan body is opaque JSON owned by the analysis
/// layer; the store only needs the handle, its dataset and the revision it
/// compares and swaps on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanRecord {
    pub schema: u32,
    pub plan_id: PlanId,
    /// The dataset the plan was drafted for (as given when it was saved).
    pub dataset_id: String,
    pub revision: u32,
    pub plan: serde_json::Value,
    pub created_at: f64,
    pub updated_at: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_and_health_label_follow_the_python_rules() {
        // Two triangles plus two singletons on 8 nodes.
        let graph = Graph {
            n: 8,
            edges: vec![
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 2, 1.0),
                (3, 4, 1.0),
                (3, 5, 1.0),
                (4, 5, 1.0),
            ],
        };
        let m = RunMetrics::from_graph(&graph);
        assert_eq!(m.n_nodes, 8);
        assert_eq!(m.n_edges, 6);
        assert_eq!(m.component_sizes, vec![3, 3, 1, 1]);
        assert_eq!(m.component_count, 4);
        assert_eq!(m.singleton_count, 2);
        assert!((m.singleton_fraction - 0.25).abs() < 1e-12);
        assert!((m.giant_fraction - 0.375).abs() < 1e-12);
        assert!((m.density - 6.0 / 28.0).abs() < 1e-12);
        assert!((m.avg_degree - 1.5).abs() < 1e-12);
        assert_eq!(m.health_label(), "connected");

        let empty = RunMetrics::from_graph(&Graph {
            n: 4,
            edges: vec![],
        });
        assert_eq!(empty.health_label(), "empty");
        assert_eq!(empty.component_sizes, vec![1, 1, 1, 1]);

        let full = RunMetrics::from_graph(&Graph {
            n: 3,
            edges: vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)],
        });
        assert_eq!(full.health_label(), "hairball");

        let mut sparse = full.clone();
        sparse.density = 0.01;
        assert_eq!(sparse.health_label(), "sparse");
        let mut giant = full.clone();
        giant.density = 0.5;
        giant.giant_fraction = 0.99;
        assert_eq!(giant.health_label(), "giant_dominant");
        let mut fragmented = giant.clone();
        fragmented.singleton_fraction = 0.3;
        assert_eq!(fragmented.health_label(), "fragmented");
    }
}
