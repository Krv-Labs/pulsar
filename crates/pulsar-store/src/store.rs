//! The on-disk registry.
//!
//! ```text
//! <root>/
//!   datasets/<dataset_id>.json          one record per dataset (no shared index)
//!   datasets/files/<dataset_id>_<name>  uploaded copies
//!   uploads/<upload_id>.{json,bin}      in-progress uploads
//!   jobs/<run_id>.json                  job state and progress
//!   runs/<run_id>/record.json           the run, written last
//!   runs/<run_id>/graph.arrow           constructed graph edges
//!   runs/<run_id>/affinity.arrow        full weighted affinity edges
//!   cluster_assignments/<id>.json + <id>.labels.arrow
//!   longitudinal/<id>.json
//! ```
//!
//! Every record is its own file written atomically, so there is no index to
//! lock: writers never block readers, and a reader sees a record only once
//! it is whole. Listings are directory scans.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pulsar_pipeline::Graph;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::atomic::{
    list_json_stems, list_subdirs, now_seconds, read_record, stage_json, sweep_orphans, write_json,
    StagedFile,
};
use crate::error::{io, StoreError};
use crate::handle::{
    ClusterAssignmentId, DatasetId, HandleKind, LongitudinalId, PlanId, RunId, UploadId, HEX_LEN,
};
use crate::job::{JobHandle, JobKind, JobRecord, INTERRUPTED};
use crate::paths::{default_root, Dirs};
use crate::record::{
    ClusterAssignment, ClusterAssignmentRecord, DatasetRecord, DatasetSource, LongitudinalRecord,
    PlanRecord, Representation, RunArtifacts, RunRecord, UploadRecord, SCHEMA_VERSION,
};
use crate::sidecar;

const RECORD_JSON: &str = "record.json";
const GRAPH_ARROW: &str = "graph.arrow";
const AFFINITY_ARROW: &str = "affinity.arrow";
const UTF8_BOM: [u8; 3] = [0xef, 0xbb, 0xbf];

#[derive(Debug, Clone)]
pub struct StoreOptions {
    /// Minimum interval between persisted progress updates of a job.
    pub progress_flush_interval: Duration,
    /// Temp files older than this are removed when the store opens.
    pub orphan_max_age: Duration,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            progress_flush_interval: Duration::from_millis(200),
            orphan_max_age: Duration::from_secs(3600),
        }
    }
}

struct Inner {
    dirs: Dirs,
    options: StoreOptions,
    /// Orders plan writers within this process; [`Store::plan_guard`] adds
    /// a file lock for writers in other processes.
    plan_lock: Mutex<()>,
}

/// A handle on one store root. Cheap to clone; safe to share across threads.
#[derive(Clone)]
pub struct Store {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("root", &self.inner.dirs.root)
            .finish()
    }
}

/// Result of [`Store::list_runs`].
#[derive(Debug, Clone, PartialEq)]
pub struct RunListing {
    /// Oldest first.
    pub runs: Vec<RunRecord>,
    /// Run directories without a readable current-schema record (jobs still
    /// running or failed, or records from another Pulsar version).
    pub skipped: usize,
}

/// A cluster assignment written and fsynced under temp names, not yet
/// readable by id. [`commit`](Self::commit) publishes it; dropping it
/// uncommitted discards it, so a publish refused or cancelled after staging
/// persists nothing.
#[derive(Debug)]
#[must_use = "a staged assignment is discarded unless committed"]
pub struct StagedClusterAssignment {
    record: ClusterAssignmentRecord,
    labels: StagedFile,
    meta: StagedFile,
}

impl StagedClusterAssignment {
    pub fn record(&self) -> &ClusterAssignmentRecord {
        &self.record
    }

    /// Make the assignment visible. Labels land first; the metadata rename
    /// is the single step that makes it readable.
    pub fn commit(self) -> Result<ClusterAssignmentRecord, StoreError> {
        let labels_path = self.labels.path().to_path_buf();
        self.labels.commit()?;
        if let Err(e) = self.meta.commit() {
            let _ = fs::remove_file(&labels_path);
            return Err(e);
        }
        Ok(self.record)
    }
}

fn digest12(hasher: Sha256) -> String {
    let hex = format!("{:x}", hasher.finalize());
    hex[..HEX_LEN].to_owned()
}

fn unknown(kind: HandleKind, id: &str) -> StoreError {
    StoreError::UnknownHandle {
        kind,
        id: id.to_owned(),
    }
}

fn safe_filename(filename: &str) -> String {
    Path::new(filename)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "uploaded.csv".to_owned())
}

fn mtime_ns(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

impl Store {
    /// Open (creating if needed) the default root; see [`default_root`].
    pub fn open() -> Result<Self, StoreError> {
        Self::open_with(default_root()?, StoreOptions::default())
    }

    pub fn open_at(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with(root, StoreOptions::default())
    }

    pub fn open_with(root: impl AsRef<Path>, options: StoreOptions) -> Result<Self, StoreError> {
        let dirs = Dirs::new(root.as_ref());
        for dir in dirs.all() {
            fs::create_dir_all(dir).map_err(io("create", dir))?;
        }
        sweep_orphans(&dirs.all(), options.orphan_max_age);
        Ok(Self {
            inner: Arc::new(Inner {
                dirs,
                options,
                plan_lock: Mutex::new(()),
            }),
        })
    }

    pub fn root(&self) -> &Path {
        &self.inner.dirs.root
    }

    pub fn options(&self) -> &StoreOptions {
        &self.inner.options
    }

    fn dirs(&self) -> &Dirs {
        &self.inner.dirs
    }

    // ----------------------------------------------------------------- datasets

    fn dataset_path(&self, id: &DatasetId) -> PathBuf {
        self.dirs().datasets.join(format!("{id}.json"))
    }

    /// Register a file in place. The id is content-addressed by resolved
    /// path, size and mtime, exactly as v0.5 did, so re-registering an
    /// unchanged file returns the same `dataset_id`.
    pub fn register_dataset(&self, path: impl AsRef<Path>) -> Result<DatasetRecord, StoreError> {
        let path = path.as_ref();
        let resolved = fs::canonicalize(path).map_err(|_| StoreError::DatasetNotFound {
            path: path.to_path_buf(),
        })?;
        let meta = fs::metadata(&resolved).map_err(io("stat", &resolved))?;
        if !meta.is_file() {
            return Err(StoreError::DatasetNotFound { path: resolved });
        }
        let mtime = mtime_ns(&meta);
        let mut hasher = Sha256::new();
        hasher.update(format!("{}:{}:{}", resolved.display(), meta.len(), mtime).as_bytes());
        let record = DatasetRecord {
            schema: SCHEMA_VERSION,
            dataset_id: DatasetId::from_hex(&digest12(hasher)),
            name: resolved
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            path: resolved,
            size_bytes: meta.len(),
            mtime_ns: mtime,
            ingested_at: now_seconds(),
            source: DatasetSource::Path,
        };
        write_json(&self.dataset_path(&record.dataset_id), &record, true)?;
        Ok(record)
    }

    /// Move a file the caller owns into the store (uploads land here). The id
    /// is content-addressed by `source` and the bytes. A UTF-8 BOM is
    /// stripped first, as v0.5 did.
    pub fn register_dataset_from_file(
        &self,
        filename: &str,
        path: &Path,
        source: DatasetSource,
    ) -> Result<DatasetRecord, StoreError> {
        strip_utf8_bom(path)?;
        let source_tag = match source {
            DatasetSource::Path => "path",
            DatasetSource::Upload => "upload",
            DatasetSource::EntitySample => "entity_sample",
        };
        let mut hasher = Sha256::new();
        hasher.update(source_tag.as_bytes());
        hasher.update(b"\0");
        let mut file = fs::File::open(path).map_err(io("open", path))?;
        let mut buf = vec![0_u8; 1 << 16];
        loop {
            let n = file.read(&mut buf).map_err(io("read", path))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        drop(file);
        let dataset_id = DatasetId::from_hex(&digest12(hasher));
        let name = safe_filename(filename);
        let stored = self
            .dirs()
            .dataset_files
            .join(format!("{dataset_id}_{name}"));
        move_file(path, &stored)?;
        let meta = fs::metadata(&stored).map_err(io("stat", &stored))?;
        let record = DatasetRecord {
            schema: SCHEMA_VERSION,
            dataset_id,
            path: stored,
            name,
            size_bytes: meta.len(),
            mtime_ns: mtime_ns(&meta),
            ingested_at: now_seconds(),
            source,
        };
        write_json(&self.dataset_path(&record.dataset_id), &record, true)?;
        Ok(record)
    }

    pub fn get_dataset(&self, id: &DatasetId) -> Result<DatasetRecord, StoreError> {
        read_record(&self.dataset_path(id))?
            .ok_or_else(|| unknown(HandleKind::Dataset, id.as_str()))
    }

    /// Every dataset, by id.
    pub fn list_datasets(&self) -> Result<Vec<DatasetRecord>, StoreError> {
        let mut out = Vec::new();
        for stem in list_json_stems(&self.dirs().datasets)? {
            if let Ok(id) = DatasetId::parse(&stem) {
                if let Some(record) = read_record::<DatasetRecord>(&self.dataset_path(&id))? {
                    out.push(record);
                }
            }
        }
        Ok(out)
    }

    /// True while the file behind a dataset still has the registered size
    /// and mtime, so persisted labels still align with its rows.
    pub fn dataset_matches_disk(record: &DatasetRecord) -> bool {
        fs::metadata(&record.path)
            .map(|m| m.len() == record.size_bytes && mtime_ns(&m) == record.mtime_ns)
            .unwrap_or(false)
    }

    /// `get_dataset`, then `DATASET_CHANGED` if the file drifted.
    pub fn check_dataset(&self, id: &DatasetId) -> Result<DatasetRecord, StoreError> {
        let record = self.get_dataset(id)?;
        if Self::dataset_matches_disk(&record) {
            Ok(record)
        } else {
            Err(StoreError::DatasetChanged {
                dataset_id: id.to_string(),
                path: record.path,
            })
        }
    }

    // ------------------------------------------------------------------ uploads

    fn upload_meta(&self, id: &UploadId) -> PathBuf {
        self.dirs().uploads.join(format!("{id}.json"))
    }

    fn upload_bin(&self, id: &UploadId) -> PathBuf {
        self.dirs().uploads.join(format!("{id}.bin"))
    }

    pub fn begin_upload(
        &self,
        filename: &str,
        media_type: &str,
    ) -> Result<UploadRecord, StoreError> {
        let record = UploadRecord {
            schema: SCHEMA_VERSION,
            upload_id: UploadId::random(),
            filename: safe_filename(filename),
            media_type: media_type.to_owned(),
            created_at: now_seconds(),
            bytes_received: 0,
        };
        let bin = self.upload_bin(&record.upload_id);
        fs::write(&bin, b"").map_err(io("create", &bin))?;
        write_json(&self.upload_meta(&record.upload_id), &record, true)?;
        Ok(record)
    }

    pub fn get_upload(&self, id: &UploadId) -> Result<UploadRecord, StoreError> {
        read_record(&self.upload_meta(id))?.ok_or_else(|| unknown(HandleKind::Upload, id.as_str()))
    }

    pub fn append_upload_chunk(
        &self,
        id: &UploadId,
        chunk: &[u8],
    ) -> Result<UploadRecord, StoreError> {
        let mut record = self.get_upload(id)?;
        let bin = self.upload_bin(id);
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&bin)
            .map_err(io("open", &bin))?;
        file.write_all(chunk).map_err(io("append", &bin))?;
        record.bytes_received += chunk.len() as u64;
        write_json(&self.upload_meta(id), &record, false)?;
        Ok(record)
    }

    /// Turn the uploaded bytes into a dataset and forget the upload.
    pub fn finalize_upload(&self, id: &UploadId) -> Result<DatasetRecord, StoreError> {
        let record = self.get_upload(id)?;
        let dataset = self.register_dataset_from_file(
            &record.filename,
            &self.upload_bin(id),
            DatasetSource::Upload,
        )?;
        let meta = self.upload_meta(id);
        fs::remove_file(&meta).map_err(io("remove", &meta))?;
        Ok(dataset)
    }

    // --------------------------------------------------------------------- jobs

    fn job_path(&self, id: &RunId) -> PathBuf {
        self.dirs().jobs.join(format!("{id}.json"))
    }

    /// Create a `queued` job and hand back its handle. The job id is the
    /// `run_id` the run will have.
    pub fn submit(
        &self,
        kind: JobKind,
        dataset_id: Option<DatasetId>,
    ) -> Result<JobHandle, StoreError> {
        let record = JobRecord::new(RunId::random(), kind, dataset_id);
        let path = self.job_path(&record.job_id);
        JobHandle::create(self.clone(), path, record)
    }

    pub fn get_job(&self, id: &RunId) -> Result<JobRecord, StoreError> {
        read_record(&self.job_path(id))?.ok_or_else(|| unknown(HandleKind::Run, id.as_str()))
    }

    /// Every job, oldest first. Records from another schema are skipped.
    pub fn list_jobs(&self) -> Result<Vec<JobRecord>, StoreError> {
        let mut out = Vec::new();
        for stem in list_json_stems(&self.dirs().jobs)? {
            let Ok(id) = RunId::parse(&stem) else {
                continue;
            };
            match read_record::<JobRecord>(&self.job_path(&id)) {
                Ok(Some(record)) => out.push(record),
                Ok(None)
                | Err(StoreError::StaleRecord { .. })
                | Err(StoreError::CorruptRecord { .. }) => {}
                Err(e) => return Err(e),
            }
        }
        out.sort_by(|a, b| {
            a.created_at
                .total_cmp(&b.created_at)
                .then_with(|| a.job_id.cmp(&b.job_id))
        });
        Ok(out)
    }

    /// Fail every `queued` or `running` job with `INTERRUPTED`. Call once at
    /// server start, before any job is submitted, when no other process can
    /// own a job in this store.
    pub fn recover_interrupted(&self) -> Result<Vec<RunId>, StoreError> {
        let mut failed = Vec::new();
        for mut record in self.list_jobs()? {
            if record.state.is_terminal() {
                continue;
            }
            record.fail(
                INTERRUPTED,
                "the process running this job exited before it finished",
            );
            write_json(&self.job_path(&record.job_id), &record, true)?;
            failed.push(record.job_id);
        }
        Ok(failed)
    }

    // --------------------------------------------------------------------- runs

    /// Directory holding a run's record and sidecars.
    pub fn run_dir(&self, id: &RunId) -> PathBuf {
        self.dirs().run_dir(id.as_str())
    }

    fn run_blob_path(&self, id: &RunId, name: &str) -> Result<PathBuf, StoreError> {
        let valid = !name.is_empty()
            && name.ends_with(".json")
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
            && !name.starts_with('.');
        if !valid {
            return Err(StoreError::InvalidArgument(format!(
                "run blob name must be a plain `<name>.json`, got {name:?}"
            )));
        }
        Ok(self.run_dir(id).join(name))
    }

    /// Attach an extra JSON document (for example a tool response) to a
    /// completed run, atomically. `name` must be `<name>.json`.
    pub fn write_run_json<T: Serialize>(
        &self,
        id: &RunId,
        name: &str,
        value: &T,
    ) -> Result<(), StoreError> {
        self.get_run(id)?;
        let path = self.run_blob_path(id, name)?;
        let mut bytes = serde_json::to_vec(value).map_err(|e| StoreError::CorruptRecord {
            path: path.clone(),
            message: e.to_string(),
        })?;
        bytes.push(b'\n');
        crate::atomic::write_atomic(&path, &bytes, true)
    }

    /// Read a JSON document attached with [`Self::write_run_json`];
    /// `Ok(None)` when the run exists but has no such document.
    pub fn read_run_json(
        &self,
        id: &RunId,
        name: &str,
    ) -> Result<Option<serde_json::Value>, StoreError> {
        self.get_run(id)?;
        let path = self.run_blob_path(id, name)?;
        match fs::read(&path) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes)
                    .map(Some)
                    .map_err(|e| StoreError::CorruptRecord {
                        path,
                        message: e.to_string(),
                    })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io("read", path)(e)),
        }
    }

    pub(crate) fn write_run(
        &self,
        record: &RunRecord,
        artifacts: &RunArtifacts,
    ) -> Result<(), StoreError> {
        let dir = self.run_dir(&record.run_id);
        fs::create_dir_all(&dir).map_err(io("create", &dir))?;
        sidecar::write_edges(&dir.join(GRAPH_ARROW), &artifacts.graph)?;
        sidecar::write_edges(&dir.join(AFFINITY_ARROW), &artifacts.affinity)?;
        // The record goes last: a visible record implies complete sidecars.
        write_json(&dir.join(RECORD_JSON), record, true)
    }

    /// The run record. A run whose job has not completed is reported as
    /// `RUN_NOT_COMPLETE` with the job state; an unknown id as
    /// `RUN_ID_UNKNOWN`.
    pub fn get_run(&self, id: &RunId) -> Result<RunRecord, StoreError> {
        if let Some(record) = read_record::<RunRecord>(&self.run_dir(id).join(RECORD_JSON))? {
            return Ok(record);
        }
        match self.get_job(id) {
            Ok(job) => Err(StoreError::RunNotComplete {
                run_id: id.to_string(),
                state: job.state,
                error: job.error,
            }),
            Err(e) => Err(e),
        }
    }

    /// The constructed graph (weights above the construction threshold).
    pub fn run_graph(&self, id: &RunId) -> Result<Graph, StoreError> {
        self.get_run(id)?;
        sidecar::read_edges(&self.run_dir(id).join(GRAPH_ARROW))
    }

    /// The full weighted affinity (every positive weight), for re-applying
    /// an interpretation threshold without a refit.
    pub fn run_affinity(&self, id: &RunId) -> Result<Graph, StoreError> {
        self.get_run(id)?;
        sidecar::read_edges(&self.run_dir(id).join(AFFINITY_ARROW))
    }

    /// Completed runs, oldest first, optionally only those of one dataset.
    /// This is the sweep history.
    pub fn list_runs(&self, dataset_id: Option<&DatasetId>) -> Result<RunListing, StoreError> {
        let mut runs = Vec::new();
        let mut skipped = 0;
        for name in list_subdirs(&self.dirs().runs)? {
            let Ok(id) = RunId::parse(&name) else {
                continue;
            };
            match read_record::<RunRecord>(&self.run_dir(&id).join(RECORD_JSON)) {
                Ok(Some(record)) => {
                    if dataset_id.is_none_or(|want| record.dataset_id.as_ref() == Some(want)) {
                        runs.push(record);
                    }
                }
                Ok(None)
                | Err(StoreError::StaleRecord { .. })
                | Err(StoreError::CorruptRecord { .. }) => {
                    skipped += 1;
                }
                Err(e) => return Err(e),
            }
        }
        runs.sort_by(|a, b| {
            a.created_at
                .total_cmp(&b.created_at)
                .then_with(|| a.run_id.cmp(&b.run_id))
        });
        Ok(RunListing { runs, skipped })
    }

    // ------------------------------------------------------- cluster assignments

    fn assignment_meta(&self, id: &ClusterAssignmentId) -> PathBuf {
        self.dirs().cluster_assignments.join(format!("{id}.json"))
    }

    fn assignment_labels(&self, id: &ClusterAssignmentId) -> PathBuf {
        self.dirs()
            .cluster_assignments
            .join(format!("{id}.labels.arrow"))
    }

    /// Persist a label vector (stage, then commit). If `run_id` is given it
    /// must name a completed run with as many rows as there are labels.
    pub fn save_cluster_assignment(
        &self,
        input: ClusterAssignment<'_>,
    ) -> Result<ClusterAssignmentRecord, StoreError> {
        self.stage_cluster_assignment(input)?.commit()
    }

    /// Write and fsync an assignment without making it visible. This is the
    /// O(rows) part, so run it off the async runtime; the
    /// [`StagedClusterAssignment::commit`] that publishes it is two renames.
    /// Validation is the same as [`Store::save_cluster_assignment`].
    pub fn stage_cluster_assignment(
        &self,
        input: ClusterAssignment<'_>,
    ) -> Result<StagedClusterAssignment, StoreError> {
        if let Some(run_id) = &input.run_id {
            let run = self.get_run(run_id)?;
            if run.n_rows != input.labels.len() {
                return Err(StoreError::InvalidArgument(format!(
                    "{} labels for run {run_id} with {} rows",
                    input.labels.len(),
                    run.n_rows
                )));
            }
        }
        let mut cluster_ids: Vec<i64> = input.labels.to_vec();
        cluster_ids.sort_unstable();
        cluster_ids.dedup();
        let record = ClusterAssignmentRecord {
            schema: SCHEMA_VERSION,
            cluster_assignment_id: ClusterAssignmentId::random(),
            run_id: input.run_id,
            dataset_id: input.dataset_id,
            method: input.method,
            interpretation_edge_weight_threshold: input.interpretation_edge_weight_threshold,
            threshold_source: input.threshold_source,
            construction_threshold: input.construction_threshold,
            label_count: input.labels.len(),
            cluster_ids,
            created_at: now_seconds(),
        };
        let labels = sidecar::stage_labels(
            &self.assignment_labels(&record.cluster_assignment_id),
            input.labels,
        )?;
        let meta = stage_json(
            &self.assignment_meta(&record.cluster_assignment_id),
            &record,
            true,
        )?;
        Ok(StagedClusterAssignment {
            record,
            labels,
            meta,
        })
    }

    pub fn get_cluster_assignment(
        &self,
        id: &ClusterAssignmentId,
    ) -> Result<ClusterAssignmentRecord, StoreError> {
        read_record(&self.assignment_meta(id))?
            .ok_or_else(|| unknown(HandleKind::ClusterAssignment, id.as_str()))
    }

    /// The per-row labels of an assignment.
    pub fn cluster_labels(&self, id: &ClusterAssignmentId) -> Result<Vec<i64>, StoreError> {
        self.get_cluster_assignment(id)?;
        sidecar::read_labels(&self.assignment_labels(id))
    }

    // ------------------------------------------------------------- longitudinal

    fn longitudinal_path(&self, id: &LongitudinalId) -> PathBuf {
        self.dirs().longitudinal.join(format!("{id}.json"))
    }

    pub fn save_longitudinal(
        &self,
        dataset_id: Option<DatasetId>,
        representation: Representation,
        config_yaml: &str,
        panel: serde_json::Value,
    ) -> Result<LongitudinalRecord, StoreError> {
        let record = LongitudinalRecord {
            schema: SCHEMA_VERSION,
            longitudinal_id: LongitudinalId::random(),
            dataset_id,
            representation,
            config_yaml: config_yaml.to_owned(),
            panel,
            created_at: now_seconds(),
        };
        write_json(
            &self.longitudinal_path(&record.longitudinal_id),
            &record,
            true,
        )?;
        Ok(record)
    }

    pub fn get_longitudinal(&self, id: &LongitudinalId) -> Result<LongitudinalRecord, StoreError> {
        read_record(&self.longitudinal_path(id))?
            .ok_or_else(|| unknown(HandleKind::Longitudinal, id.as_str()))
    }

    // ------------------------------------------------------------------- plans

    fn plan_path(&self, id: &PlanId) -> PathBuf {
        self.dirs().plans.join(format!("{id}.json"))
    }

    /// Serializes plan read-modify-write so a revision check and the write
    /// that follows it cannot interleave with another writer: the mutex
    /// orders threads of this process, the exclusive lock on `plans/.lock`
    /// orders the other processes sharing the store. Both release on drop.
    fn plan_guard(&self) -> Result<(fs::File, std::sync::MutexGuard<'_, ()>), StoreError> {
        let guard = self
            .inner
            .plan_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let path = self.dirs().plans.join(".lock");
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(io("open", &path))?;
        file.lock().map_err(io("lock", &path))?;
        Ok((file, guard))
    }

    /// Create a plan record. An existing plan is never replaced
    /// ([`StoreError::PlanExists`]); later changes go through
    /// [`Store::update_plan`], which checks the revision.
    pub fn save_plan(
        &self,
        id: &PlanId,
        dataset_id: &str,
        revision: u32,
        plan: serde_json::Value,
    ) -> Result<PlanRecord, StoreError> {
        let _guard = self.plan_guard()?;
        let path = self.plan_path(id);
        if path.try_exists().map_err(io("stat", &path))? {
            return Err(StoreError::PlanExists {
                plan_id: id.to_string(),
            });
        }
        let now = now_seconds();
        let record = PlanRecord {
            schema: SCHEMA_VERSION,
            plan_id: id.clone(),
            dataset_id: dataset_id.to_owned(),
            revision,
            plan,
            previous_plan: None,
            created_at: now,
            updated_at: now,
        };
        write_json(&path, &record, true)?;
        Ok(record)
    }

    pub fn get_plan(&self, id: &PlanId) -> Result<PlanRecord, StoreError> {
        read_record(&self.plan_path(id))?.ok_or_else(|| unknown(HandleKind::Plan, id.as_str()))
    }

    /// Replace a plan only if it is still at `expected_revision`; otherwise
    /// [`StoreError::RevisionConflict`] names the revision it is at now.
    pub fn update_plan(
        &self,
        id: &PlanId,
        expected_revision: u32,
        new_revision: u32,
        plan: serde_json::Value,
    ) -> Result<PlanRecord, StoreError> {
        let _guard = self.plan_guard()?;
        let mut record = self.get_plan(id)?;
        if record.revision != expected_revision {
            return Err(StoreError::RevisionConflict {
                plan_id: id.to_string(),
                current: record.revision,
                expected: expected_revision,
            });
        }
        record.revision = new_revision;
        record.previous_plan = Some(std::mem::replace(&mut record.plan, plan));
        record.updated_at = now_seconds();
        write_json(&self.plan_path(id), &record, true)?;
        Ok(record)
    }

    /// Plans for `dataset_id` (or all), least recently updated first, so the
    /// last one is the plan most recently worked on. Unreadable records are
    /// skipped.
    pub fn list_plans(&self, dataset_id: Option<&str>) -> Result<Vec<PlanRecord>, StoreError> {
        let mut plans: Vec<PlanRecord> = list_json_stems(&self.dirs().plans)?
            .iter()
            .filter_map(|stem| PlanId::parse(stem).ok())
            .filter_map(|id| self.get_plan(&id).ok())
            .filter(|p| dataset_id.is_none_or(|d| p.dataset_id == d))
            .collect();
        plans.sort_by(|a, b| a.updated_at.total_cmp(&b.updated_at));
        Ok(plans)
    }
}

fn strip_utf8_bom(path: &Path) -> Result<(), StoreError> {
    let mut file = fs::File::open(path).map_err(io("open", path))?;
    let mut prefix = [0_u8; 3];
    let n = file.read(&mut prefix).map_err(io("read", path))?;
    if n < 3 || prefix != UTF8_BOM {
        return Ok(());
    }
    let mut rest = Vec::new();
    file.read_to_end(&mut rest).map_err(io("read", path))?;
    drop(file);
    crate::atomic::write_atomic(path, &rest, false)
}

fn move_file(from: &Path, to: &Path) -> Result<(), StoreError> {
    if fs::rename(from, to).is_ok() {
        return Ok(());
    }
    fs::copy(from, to).map_err(io("copy", to))?;
    fs::remove_file(from).map_err(io("remove", from))
}
