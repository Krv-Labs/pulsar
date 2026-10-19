//! Structured store errors with stable codes for the MCP error envelope.

use std::path::PathBuf;

use thiserror::Error;

use crate::handle::HandleKind;
use crate::job::{JobError, JobState};

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("{op} {}: {source}", path.display())]
    Io {
        op: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "no store location: set {} (or HOME / XDG_DATA_HOME)",
        crate::paths::ENV_ROOT
    )]
    NoHome,
    #[error(
        "invalid {kind} {value:?}: expected `{}_` followed by 12 hex characters",
        kind.prefix()
    )]
    InvalidHandle { kind: HandleKind, value: String },
    #[error("unknown {kind} {id}")]
    UnknownHandle { kind: HandleKind, id: String },
    #[error("run {run_id} is {state}{}", match error {
        Some(e) => format!(": {} ({})", e.message, e.code),
        None => String::new(),
    })]
    RunNotComplete {
        run_id: String,
        state: JobState,
        error: Option<JobError>,
    },
    #[error(
        "record {} has schema {}; this build reads schema {expected}",
        path.display(),
        found.map_or_else(|| "<missing>".to_owned(), |v| v.to_string())
    )]
    StaleRecord {
        path: PathBuf,
        found: Option<u32>,
        expected: u32,
    },
    #[error("record {} is not valid: {message}", path.display())]
    CorruptRecord { path: PathBuf, message: String },
    #[error("sidecar {} is not a valid {what}: {message}", path.display())]
    CorruptSidecar {
        path: PathBuf,
        what: &'static str,
        message: String,
    },
    #[error(
        "dataset {dataset_id} no longer matches {}: re-register it to get a new dataset_id",
        path.display()
    )]
    DatasetChanged { dataset_id: String, path: PathBuf },
    #[error("dataset path {} does not exist or is not readable", path.display())]
    DatasetNotFound { path: PathBuf },
    #[error("job {job_id} cannot {action} while {state}")]
    JobState {
        job_id: String,
        state: JobState,
        action: &'static str,
    },
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl StoreError {
    /// Stable code for error envelopes.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Io { .. } => "IO",
            Self::NoHome => "NO_STORE_LOCATION",
            Self::InvalidHandle { .. } => "INVALID_HANDLE",
            Self::UnknownHandle { kind, .. } => kind.unknown_code(),
            Self::RunNotComplete { .. } => "RUN_NOT_COMPLETE",
            Self::StaleRecord { .. } => "STALE_RECORD",
            Self::CorruptRecord { .. } => "RECORD_CORRUPT",
            Self::CorruptSidecar { .. } => "SIDECAR_CORRUPT",
            Self::DatasetChanged { .. } => "DATASET_CHANGED",
            Self::DatasetNotFound { .. } => "DATASET_NOT_FOUND",
            Self::JobState { .. } => "JOB_STATE",
            Self::InvalidArgument(_) => "INVALID_ARGUMENT",
        }
    }

    /// What an agent should do next; the `agent_action` of the envelope.
    pub fn agent_action(&self) -> String {
        match self {
            Self::UnknownHandle { kind, .. } => match kind {
                HandleKind::Run => {
                    "The run_id is not in the store. Re-run the sweep to get a new run_id.".into()
                }
                HandleKind::Dataset => {
                    "The dataset_id is not in the store. Ingest the dataset again.".into()
                }
                HandleKind::ClusterAssignment => {
                    "The cluster_assignment_id is not in the store. Generate the dossier again."
                        .into()
                }
                HandleKind::Longitudinal => {
                    "The longitudinal_id is not in the store. Build the longitudinal graph again."
                        .into()
                }
                HandleKind::Upload => "The upload_id is not in the store. Start a new upload.".into(),
            },
            Self::RunNotComplete { state, .. } => match state {
                JobState::Queued | JobState::Running => {
                    "The run has not finished. Poll the job until it is complete, then retry."
                        .into()
                }
                JobState::Failed => "The run failed; fix the reported error and resubmit.".into(),
                JobState::Complete => "Retry the read; the run just completed.".into(),
            },
            Self::DatasetChanged { .. } => {
                "The file behind this dataset_id changed. Ingest it again to get a fresh handle."
                    .into()
            }
            Self::StaleRecord { .. } => {
                "The record was written by another Pulsar version. Re-run the step that produced it."
                    .into()
            }
            Self::InvalidHandle { kind, .. } => {
                format!("Pass a {kind} exactly as an earlier tool result returned it.")
            }
            _ => "Report the error; the store itself is unhealthy or the input is invalid.".into(),
        }
    }
}

pub(crate) fn io(
    op: &'static str,
    path: impl Into<PathBuf>,
) -> impl FnOnce(std::io::Error) -> StoreError {
    let path = path.into();
    move |source| StoreError::Io { op, path, source }
}
