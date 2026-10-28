//! Run registry, handles, and the job model for long-running work (#62).
//!
//! The Rust replacement for `pulsar/mcp/registry.py`, the persisted half of
//! `pulsar/mcp/session.py`, and `pulsar/mcp/history.py`:
//!
//! - [`Store`]: one directory of atomically written JSON records with Arrow
//!   IPC sidecars for edge lists and label vectors. Default location per
//!   [`paths::default_root`], overridable with `PULSAR_STORE_DIR`.
//! - Typed handles ([`DatasetId`], [`RunId`], [`ClusterAssignmentId`],
//!   [`LongitudinalId`]) that validate what an agent passes back, and
//!   [`StoreError`] codes such as `RUN_ID_UNKNOWN` for the error envelope.
//! - The job model: [`Store::submit`] creates a `queued` job whose id is the
//!   future `run_id`; the worker drives it `running` with progress, then
//!   `complete` (the [`RunRecord`] becomes readable) or `failed`. Runs and
//!   jobs survive a restart; [`Store::recover_interrupted`] fails whatever a
//!   dead process left behind.
//! - [`ModelCache`]: the bounded in-process LRU of live fitted models.
//! - [`summarize_history`]: sweep-history observations over persisted runs.

pub mod atomic;
pub mod cache;
pub mod error;
pub mod handle;
pub mod history;
pub mod job;
pub mod paths;
pub mod record;
pub mod sidecar;
pub mod store;

pub use cache::{ModelCache, ENV_CAPACITY};
pub use error::StoreError;
pub use handle::{
    ClusterAssignmentId, DatasetId, HandleKind, LongitudinalId, PlanId, RunId, UploadId,
};
pub use history::{summarize_history, ComponentProfile, FragmentationTrend, HistorySummary};
pub use job::{JobError, JobHandle, JobKind, JobRecord, JobState, ABANDONED, INTERRUPTED};
pub use paths::{default_root, APP_DIR, ENV_ROOT};
pub use record::{
    ClusterAssignment, ClusterAssignmentRecord, ConfigSummary, DatasetRecord, DatasetSource,
    LongitudinalRecord, PlanRecord, PlateauSummary, Representation, RunArtifacts, RunMetrics,
    RunRecord, StabilitySummary, ThresholdPolicy, UploadRecord, SCHEMA_VERSION,
};
pub use store::{RunListing, StagedClusterAssignment, Store, StoreOptions};

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::{Path, PathBuf};

    use pulsar_config::{Construction, ProjectionMethod};

    use crate::handle::{random_hex, RunId};
    use crate::record::{ConfigSummary, RunMetrics, RunRecord, ThresholdPolicy, SCHEMA_VERSION};

    /// A fresh directory under the system temp dir, removed on drop.
    pub struct TempRoot(PathBuf);

    impl TempRoot {
        pub fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "pulsar-store-{tag}-{}-{}",
                std::process::id(),
                random_hex()
            ));
            std::fs::create_dir_all(&path).expect("temp dir");
            Self(path)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub fn run_record(metrics: RunMetrics) -> RunRecord {
        RunRecord {
            schema: SCHEMA_VERSION,
            run_id: RunId::random(),
            dataset_id: None,
            data_path: None,
            config_yaml: String::new(),
            config: ConfigSummary {
                run_name: "test".into(),
                projection_method: ProjectionMethod::Jl,
                projection_dimensions: vec![2],
                n_seeds: 1,
                n_epsilons: 1,
                construction: Construction::Minhash,
                threshold_policy: ThresholdPolicy::Auto,
                requested_threshold: None,
                sparsify: false,
            },
            n_rows: metrics.n_nodes,
            n_features: 1,
            n_embeddings: 1,
            n_ball_maps: 1,
            resolved_construction_threshold: 0.0,
            weight_scale: 1.0,
            metrics,
            threshold_stability_summary: None,
            created_at: crate::atomic::now_seconds(),
        }
    }
}
