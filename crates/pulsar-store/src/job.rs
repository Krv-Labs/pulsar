//! The job model: a run is `queued` when submitted, `running` while the
//! worker reports progress, and ends `complete` (its [`RunRecord`] is now
//! readable) or `failed`. The job and the run share one id, so the MCP
//! server can hand out `run_id` at submit time and poll it (#71).
//!
//! Job records live at `jobs/<run_id>.json` and are rewritten atomically on
//! every transition, so another process (or the same server after a
//! restart) can read them. Progress rewrites are throttled; terminal
//! transitions are always written and fsynced.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::atomic::{now_seconds, write_json};
use crate::error::StoreError;
use crate::handle::{DatasetId, RunId};
use crate::record::{RunArtifacts, RunRecord, SCHEMA_VERSION};
use crate::store::Store;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Complete,
    Failed,
}

impl JobState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Failed)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Complete => "complete",
            Self::Failed => "failed",
        }
    }
}

impl fmt::Display for JobState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What kind of work a job does. Only `Sweep` produces a [`RunRecord`];
/// the others reuse the model so later issues can attach their own outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Sweep,
    Dossier,
    Sparsify,
    Longitudinal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobError {
    pub code: String,
    pub message: String,
}

/// Code used when [`Store::recover_interrupted`] fails a job whose process
/// went away.
pub const INTERRUPTED: &str = "INTERRUPTED";
/// Code used when the last [`JobHandle`] is dropped before a terminal state.
pub const ABANDONED: &str = "ABANDONED";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRecord {
    pub schema: u32,
    pub job_id: RunId,
    pub kind: JobKind,
    pub dataset_id: Option<DatasetId>,
    pub state: JobState,
    /// Latest progress stage label.
    pub stage: String,
    /// Cumulative progress in `[0, 1]`.
    pub progress: f64,
    pub created_at: f64,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    pub error: Option<JobError>,
    pub cancel_requested: bool,
    /// Process that owns the job; informational.
    pub pid: u32,
}

impl JobRecord {
    pub(crate) fn new(job_id: RunId, kind: JobKind, dataset_id: Option<DatasetId>) -> Self {
        Self {
            schema: SCHEMA_VERSION,
            job_id,
            kind,
            dataset_id,
            state: JobState::Queued,
            stage: "queued".to_owned(),
            progress: 0.0,
            created_at: now_seconds(),
            started_at: None,
            finished_at: None,
            error: None,
            cancel_requested: false,
            pid: std::process::id(),
        }
    }

    pub(crate) fn fail(&mut self, code: &str, message: &str) {
        self.state = JobState::Failed;
        self.error = Some(JobError {
            code: code.to_owned(),
            message: message.to_owned(),
        });
        self.finished_at = Some(now_seconds());
    }
}

struct Live {
    record: JobRecord,
    last_flush: Instant,
}

struct Inner {
    store: Store,
    path: PathBuf,
    live: Mutex<Live>,
    cancel: AtomicBool,
    flush_interval: Duration,
}

/// The worker's handle on one job. Cloneable; all clones share state.
#[derive(Clone)]
pub struct JobHandle {
    inner: Arc<Inner>,
}

impl fmt::Debug for JobHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let live = self.lock();
        f.debug_struct("JobHandle")
            .field("job_id", &live.record.job_id)
            .field("state", &live.record.state)
            .field("progress", &live.record.progress)
            .finish()
    }
}

impl JobHandle {
    pub(crate) fn create(
        store: Store,
        path: PathBuf,
        record: JobRecord,
    ) -> Result<Self, StoreError> {
        write_json(&path, &record, true)?;
        Ok(Self {
            inner: Arc::new(Inner {
                flush_interval: store.options().progress_flush_interval,
                store,
                path,
                live: Mutex::new(Live {
                    record,
                    last_flush: Instant::now(),
                }),
                cancel: AtomicBool::new(false),
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Live> {
        self.inner.live.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn id(&self) -> RunId {
        self.lock().record.job_id.clone()
    }

    pub fn kind(&self) -> JobKind {
        self.lock().record.kind
    }

    pub fn state(&self) -> JobState {
        self.lock().record.state
    }

    /// A snapshot of the record.
    pub fn record(&self) -> JobRecord {
        self.lock().record.clone()
    }

    fn write(live: &mut Live, path: &Path, durable: bool) -> Result<(), StoreError> {
        write_json(path, &live.record, durable)?;
        live.last_flush = Instant::now();
        Ok(())
    }

    fn require(live: &Live, allowed: &[JobState], action: &'static str) -> Result<(), StoreError> {
        if allowed.contains(&live.record.state) {
            Ok(())
        } else {
            Err(StoreError::JobState {
                job_id: live.record.job_id.to_string(),
                state: live.record.state,
                action,
            })
        }
    }

    /// `queued` → `running`.
    pub fn start(&self) -> Result<(), StoreError> {
        let mut live = self.lock();
        Self::require(&live, &[JobState::Queued], "start")?;
        live.record.state = JobState::Running;
        live.record.stage = "running".to_owned();
        live.record.started_at = Some(now_seconds());
        Self::write(&mut live, &self.inner.path, true)
    }

    /// Report progress while `running`. Persisted at most once per
    /// `progress_flush_interval`; call [`flush`](Self::flush) to force it.
    pub fn progress(&self, stage: &str, fraction: f64) -> Result<(), StoreError> {
        if !fraction.is_finite() {
            return Err(StoreError::InvalidArgument(format!(
                "progress fraction must be finite, got {fraction}"
            )));
        }
        let mut live = self.lock();
        Self::require(&live, &[JobState::Running], "report progress")?;
        live.record.stage = stage.to_owned();
        live.record.progress = fraction.clamp(0.0, 1.0);
        if live.last_flush.elapsed() >= self.inner.flush_interval {
            Self::write(&mut live, &self.inner.path, false)?;
        }
        Ok(())
    }

    /// Persist the current record now.
    pub fn flush(&self) -> Result<(), StoreError> {
        let mut live = self.lock();
        Self::write(&mut live, &self.inner.path, false)
    }

    /// A progress callback for the pipeline. Persistence errors are dropped:
    /// progress is best-effort and the terminal transition still reports.
    pub fn progress_sink(&self) -> impl FnMut(&str, f64) + '_ {
        move |stage: &str, fraction: f64| {
            let _ = self.progress(stage, fraction);
        }
    }

    /// Write the run (record and sidecars), then mark the job `complete`.
    /// Allowed from `queued` as well, for synchronous callers.
    pub fn complete(&self, record: &RunRecord, artifacts: &RunArtifacts) -> Result<(), StoreError> {
        let mut live = self.lock();
        Self::require(&live, &[JobState::Queued, JobState::Running], "complete")?;
        if record.run_id != live.record.job_id {
            return Err(StoreError::InvalidArgument(format!(
                "run record {} does not belong to job {}",
                record.run_id, live.record.job_id
            )));
        }
        self.inner.store.write_run(record, artifacts)?;
        live.record.state = JobState::Complete;
        live.record.stage = "complete".to_owned();
        live.record.progress = 1.0;
        live.record.finished_at = Some(now_seconds());
        Self::write(&mut live, &self.inner.path, true)
    }

    /// Mark the job `failed` with a stable code and a message.
    pub fn fail(&self, code: &str, message: &str) -> Result<(), StoreError> {
        let mut live = self.lock();
        Self::require(&live, &[JobState::Queued, JobState::Running], "fail")?;
        live.record.fail(code, message);
        Self::write(&mut live, &self.inner.path, true)
    }

    /// Ask the worker to stop at its next checkpoint. The pipeline has no
    /// checkpoints yet; this records the request so #71 can honour it.
    pub fn request_cancel(&self) -> Result<(), StoreError> {
        self.inner.cancel.store(true, Ordering::SeqCst);
        let mut live = self.lock();
        live.record.cancel_requested = true;
        Self::write(&mut live, &self.inner.path, false)
    }

    pub fn cancel_requested(&self) -> bool {
        self.inner.cancel.load(Ordering::SeqCst)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let live = self.live.get_mut().unwrap_or_else(|e| e.into_inner());
        if live.record.state.is_terminal() {
            return;
        }
        live.record
            .fail(ABANDONED, "job handle dropped before the job finished");
        let _ = write_json(&self.path, &live.record, true);
    }
}
