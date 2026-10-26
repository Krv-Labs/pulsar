//! Graph analysis for the v1.0 port (#64): clustering, threshold mass
//! dynamics, diagnostics and the point-level Ball Mapper hooks.
//!
//! Ports of `pulsar/analysis/hooks.py`, `pulsar/mcp/thresholds.py`,
//! `pulsar/mcp/diagnostics.py` and `pulsar/mcp/minhash_advisor.py`.
//! scikit-learn and SciPy replacements:
//!
//! | Python | Rust |
//! |---|---|
//! | `AgglomerativeClustering(linkage="average", metric="precomputed")` | `kodama` average linkage (same fastcluster algorithm; ARI 1.0) |
//! | `SpectralClustering(affinity="precomputed")` | normalized Laplacian + dense or LOBPCG eigensolver + `pulsar_core::kmeans` (ARI ≥ 0.9) |
//! | `scipy.sparse.csgraph.connected_components` | `pulsar_core::UnionFind` over the sorted edge list |
//! | `sklearn.metrics.silhouette_score(metric="precomputed")` | [`silhouette_score`] |
//! | `networkx.articulation_points` | iterative Tarjan |
//! | `numpy.histogram` / sparkline, percentiles | [`diagnostics`] |
//!
//! Every rounded number is rounded with Python semantics ([`pyround`]) so
//! the `threshold_options` and `breakpoints` golden fixtures reproduce.

pub mod clustering;
pub mod components;
pub mod diagnostics;
pub mod error;
pub mod evidence;
pub mod hooks;
pub mod minhash;
pub mod payloads;
pub mod pyround;
pub mod silhouette;
pub mod thresholds;

pub use clustering::{
    cluster_agglomerative, cluster_spectral, cosmic_clusters, spectral_embedding, Method,
};
pub use components::{ComponentState, ThresholdGraph};
pub use diagnostics::{
    articulation_points, distribution_sparkline, finalization_gate, skeleton_payload,
    GraphAnalysis, GraphMetrics, GraphSummary,
};
pub use error::AnalysisError;
pub use evidence::{
    measure_preprocessing_evidence, neighborhood_overlap, ColumnEvidence, ColumnKind,
    ColumnRelation, EvidenceError, EvidenceOptions, MaskDependency, MissingGroup,
    PreprocessingEvidence, RelationKind,
};
pub use hooks::{label_points, membership_matrix, unclustered_points};
pub use silhouette::{silhouette_samples, silhouette_score};
pub use thresholds::{
    agent_threshold_options, component_mass_profile, first_report_ready_candidate,
    plateau_threshold_candidates, structural_breakpoints, threshold_morphology_profile,
    transition_adjacent_candidates, Breakpoint, Candidate, Policy, ThresholdOptions,
};
