//! Running a sweep as a store job: the blocking fit on a worker thread,
//! progress persisted through the [`JobHandle`] and forwarded to the client
//! as MCP progress notifications, then the run committed and cached.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use pulsar_config::PulsarConfig;
use pulsar_data::Table;
use pulsar_pipeline::{fit, FitOptions, ProgressEvent};
use pulsar_store::{DatasetId, JobHandle, JobKind, RunArtifacts, RunRecord};
use rmcp::model::{NotificationMetaObject, ProgressNotificationParam, ProgressToken};
use rmcp::service::{Peer, RoleServer};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::envelope::ToolError;
use crate::state::{LiveRun, ServerState};

/// `_meta` key for the Phirewall structured progress contract (`io.phirewall/progress` v1).
pub const PHIREWALL_PROGRESS_META_KEY: &str = "io.phirewall/progress";

const CONTRACT_VERSION: u8 = 1;
const MAX_COUNTS_ENTRIES: usize = 16;

/// Maps `ProgressTracker` stage keys (after stripping any `Dataset i/N: `
/// prefix) onto `(contract_stage_id, inner_count_unit)` per SPEC §2b.
///
/// `graph` maps to `"stability"` because building the thresholded graph is the
/// tail of the steadiness check (`"finalize"` stays in Phirewall's profile for
/// Pulsar 0.6 labels and is not emitted on 0.7).
const STAGE_MAP: &[(&str, &str, Option<&str>)] = &[
    ("load", "load", None),
    ("preprocess", "preprocess", None),
    ("scale", "scale", None),
    ("projection", "project", Some("projection")),
    ("sweep_batches", "sweep", Some("batch")),
    ("cosmic_graph", "consensus", None),
    ("stability", "stability", None),
    ("graph", "stability", None),
];

/// Enclosing loop counter (`outer` in `io.phirewall/progress` v1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OuterCount {
    pub done: u64,
    pub total: u64,
    pub unit: String,
}

/// Structured progress payload attached at `params._meta["io.phirewall/progress"]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhirewallProgress {
    pub v: u8,
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer: Option<OuterCount>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counts: BTreeMap<String, u64>,
}

/// Contract identifier rule from SPEC §1: `^[a-z][a-z0-9_]{0,31}$`.
pub fn is_valid_contract_id(s: &str) -> bool {
    let bytes = s.as_bytes();
    matches!(bytes.first(), Some(b'a'..=b'z'))
        && bytes.len() <= 32
        && bytes[1..]
            .iter()
            .all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_'))
}

fn strip_dataset_prefix(key: &str) -> Option<(Option<OuterCount>, &str)> {
    let Some(rest) = key.strip_prefix("Dataset ") else {
        return Some((None, key));
    };
    let (ratio, bare) = rest.split_once(": ")?;
    let (done_s, total_s) = ratio.split_once('/')?;
    let done: u64 = done_s.parse().ok()?;
    let total: u64 = total_s.parse().ok()?;
    if done == 0 || total == 0 || done > total {
        return None;
    }
    Some((
        Some(OuterCount {
            done,
            total,
            unit: "dataset".to_owned(),
        }),
        bare,
    ))
}

impl PhirewallProgress {
    /// Map a [`ProgressEvent`] and optional `(rows, columns)` table dimensions
    /// onto the `io.phirewall/progress` v1 contract. Unknown keys or invalid
    /// counts return `None` so no contract object is attached.
    pub fn from_event(
        event: ProgressEvent<'_>,
        table_dims: Option<(usize, usize)>,
    ) -> Option<Self> {
        let (outer, bare_key) = strip_dataset_prefix(event.key)?;
        let &(_, stage_id, unit) = STAGE_MAP.iter().find(|&&(k, _, _)| k == bare_key)?;
        let (done, total, unit) = match (event.count, unit) {
            (Some((d, t)), Some(u)) => (Some(d), Some(t), Some(u.to_owned())),
            (None, _) | (Some(_), None) => (None, None, None),
        };
        let mut counts = BTreeMap::new();
        if let Some((rows, cols)) = table_dims {
            counts.insert("rows".to_owned(), rows as u64);
            counts.insert("columns".to_owned(), cols as u64);
        }
        Self {
            v: CONTRACT_VERSION,
            stage: stage_id.to_owned(),
            done,
            total,
            unit,
            outer,
            counts,
        }
        .validated()
    }

    /// Check every invariant of the `io.phirewall/progress` v1 contract (SPEC §1).
    pub fn is_valid(&self) -> bool {
        if self.v != CONTRACT_VERSION || !is_valid_contract_id(&self.stage) {
            return false;
        }
        let inner_ok = match (self.done, self.total, self.unit.as_deref()) {
            (None, None, None) => true,
            (Some(done), Some(total), Some(unit)) => {
                total >= 1 && done <= total && is_valid_contract_id(unit)
            }
            _ => false,
        };
        if !inner_ok {
            return false;
        }
        if let Some(outer) = &self.outer {
            if outer.total < 1 || outer.done > outer.total || !is_valid_contract_id(&outer.unit) {
                return false;
            }
        }
        if self.counts.len() > MAX_COUNTS_ENTRIES
            || !self.counts.keys().all(|k| is_valid_contract_id(k))
        {
            return false;
        }
        true
    }

    /// Validate and return `Some(self)` when every SPEC §1 rule holds.
    pub fn validated(self) -> Option<Self> {
        self.is_valid().then_some(self)
    }

    /// Wrap this contract object in an `rmcp` [`NotificationMetaObject`] under
    /// [`PHIREWALL_PROGRESS_META_KEY`].
    pub fn into_meta(self) -> NotificationMetaObject {
        debug_assert!(self.is_valid(), "invalid progress contract: {self:?}");
        let mut meta = NotificationMetaObject::new();
        if let Ok(value) = serde_json::to_value(self) {
            meta.insert(PHIREWALL_PROGRESS_META_KEY.to_owned(), value);
        }
        meta
    }
}

/// Everything a sweep needs, resolved before the job is queued.
pub struct SweepRequest {
    pub cfg: PulsarConfig,
    pub table: Table,
    pub dataset_id: Option<DatasetId>,
    pub data_path: Option<PathBuf>,
    pub precomputed: Option<Vec<Array2<f64>>>,
    pub projection_fingerprint: String,
}

/// Where progress goes besides the job record.
#[derive(Clone)]
pub struct ProgressTarget {
    pub peer: Peer<RoleServer>,
    pub token: ProgressToken,
}

/// Queue the job for `req`. Returns the handle so the caller can hand the
/// `run_id` out before the fit starts.
pub fn submit(state: &ServerState, req: &SweepRequest) -> Result<JobHandle, ToolError> {
    state
        .store
        .submit(JobKind::Sweep, req.dataset_id.clone())
        .map_err(|e| ToolError::from_store("run_topological_sweep", e))
}

/// Run the fit for a queued job and cache the live model.
///
/// Returns `(live, record, artifacts)` so the caller can build the tool
/// response payload before marking the [`JobHandle`] complete.
pub async fn run_sweep(
    state: Arc<ServerState>,
    req: SweepRequest,
    job: &JobHandle,
    progress: Option<ProgressTarget>,
) -> Result<(Arc<LiveRun>, RunRecord, RunArtifacts), ToolError> {
    const TOOL: &str = "run_topological_sweep";
    job.start().map_err(|e| ToolError::from_store(TOOL, e))?;
    let (tx, mut rx) = mpsc::unbounded_channel::<(String, f64, Option<PhirewallProgress>)>();
    let run_id = job.id();

    let worker_job = job.clone();
    let worker = tokio::task::spawn_blocking(move || {
        let SweepRequest {
            cfg,
            table,
            dataset_id,
            data_path,
            precomputed,
            projection_fingerprint,
        } = req;
        let table_dims = (table.n_rows(), table.n_cols());
        let mut on_progress = |ev: ProgressEvent<'_>| {
            let _ = worker_job.progress(ev.label, ev.fraction);
            let contract = PhirewallProgress::from_event(ev, Some(table_dims));
            let _ = tx.send((ev.label.to_owned(), ev.fraction, contract));
        };
        let opts = FitOptions {
            precomputed_embeddings: precomputed,
            ..FitOptions::default()
        };
        let fitted = fit(&cfg, &table, opts, Some(&mut on_progress))?;
        Ok::<_, pulsar_pipeline::PipelineError>((
            fitted,
            table,
            dataset_id,
            data_path,
            projection_fingerprint,
        ))
    });

    // Forward progress while the fit runs; the channel closes with the worker.
    let forwarder = async {
        while let Some((stage, fraction, contract)) = rx.recv().await {
            if let Some(target) = &progress {
                let mut param = ProgressNotificationParam::new(target.token.clone(), fraction)
                    .with_total(1.0)
                    .with_message(stage);
                if let Some(contract) = contract {
                    param.meta = Some(contract.into_meta());
                }
                let _ = target.peer.notify_progress(param).await;
            }
        }
    };
    let (outcome, ()) = tokio::join!(worker, forwarder);

    let (fitted, table, dataset_id, data_path, projection_fingerprint) = match outcome {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            let _ = job.fail(e.code(), &e.to_string());
            return Err(ToolError::new(
                TOOL,
                e.code(),
                e.to_string(),
                "Fix the config or data problem the error names, then re-run the sweep.",
            )
            .detail("run_id", serde_json::json!(run_id)));
        }
        Err(join) => {
            let _ = job.fail("FIT_PANICKED", &join.to_string());
            return Err(ToolError::new(
                TOOL,
                "FIT_PANICKED",
                format!("the sweep worker stopped unexpectedly: {join}"),
                "An internal computation panicked: this is a defect, and the message is the evidence. Report it with the message. Whether the config contributed is not known: the same panic may or may not recur with different parameters or data.",
            )
            .detail("run_id", serde_json::json!(run_id)));
        }
    };

    let data_path_text = data_path.as_ref().map(|p| p.to_string_lossy().into_owned());
    let record = RunRecord::from_fitted(
        run_id.clone(),
        &fitted,
        dataset_id.clone(),
        data_path_text.as_deref(),
    );
    let artifacts = RunArtifacts::from_fitted(&fitted);

    let live = Arc::new(LiveRun {
        run_id: run_id.clone(),
        fitted,
        table,
        dataset_id,
        data_path,
        projection_fingerprint,
    });
    state.cache.put(run_id.clone(), live.clone());
    {
        let mut session = state.session();
        session.latest_run_id = Some(run_id);
        session.clusters = None;
        if let Some(ds) = &live.dataset_id {
            session.latest_dataset_id = Some(ds.clone());
        }
    }
    Ok((live, record, artifacts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event<'a>(key: &'a str, label: &'a str, count: Option<(u64, u64)>) -> ProgressEvent<'a> {
        ProgressEvent {
            key,
            label,
            fraction: 0.5,
            count,
        }
    }

    #[test]
    fn maps_every_single_dataset_tracker_key() {
        let dims = Some((1200, 34));
        for (key, expected_stage) in [
            ("load", "load"),
            ("preprocess", "preprocess"),
            ("scale", "scale"),
            ("cosmic_graph", "consensus"),
            ("stability", "stability"),
            ("graph", "stability"),
        ] {
            let c = PhirewallProgress::from_event(event(key, key, None), dims).expect(key);
            assert!(c.is_valid());
            assert_eq!(c.v, 1);
            assert_eq!(c.stage, expected_stage);
            assert_eq!(c.done, None);
            assert_eq!(c.total, None);
            assert_eq!(c.unit, None);
            assert_eq!(c.outer, None);
            assert_eq!(c.counts.get("rows"), Some(&1200));
            assert_eq!(c.counts.get("columns"), Some(&34));
        }

        let proj = PhirewallProgress::from_event(
            event("projection", "projection variant 2/4", Some((2, 4))),
            dims,
        )
        .expect("projection variant");
        assert!(proj.is_valid());
        assert_eq!(proj.stage, "project");
        assert_eq!(proj.done, Some(2));
        assert_eq!(proj.total, Some(4));
        assert_eq!(proj.unit.as_deref(), Some("projection"));
        assert_eq!(proj.outer, None);

        let proj_cached =
            PhirewallProgress::from_event(event("projection", "projection (cached)", None), None)
                .expect("cached projection");
        assert!(proj_cached.is_valid());
        assert_eq!(proj_cached.stage, "project");
        assert_eq!(proj_cached.done, None);
        assert_eq!(proj_cached.total, None);
        assert_eq!(proj_cached.unit, None);
        assert!(proj_cached.counts.is_empty());

        let sweep = PhirewallProgress::from_event(
            event(
                "sweep_batches",
                "sweep batch 3/8: ball_mapper + construction",
                Some((3, 8)),
            ),
            dims,
        )
        .expect("sweep_batches");
        assert!(sweep.is_valid());
        assert_eq!(sweep.stage, "sweep");
        assert_eq!(sweep.done, Some(3));
        assert_eq!(sweep.total, Some(8));
        assert_eq!(sweep.unit.as_deref(), Some("batch"));
        assert_eq!(sweep.outer, None);
    }

    #[test]
    fn maps_multi_dataset_prefix_into_outer_counter() {
        let expected_outer = Some(OuterCount {
            done: 2,
            total: 3,
            unit: "dataset".to_owned(),
        });
        for (key, expected_stage) in [
            ("Dataset 2/3: load", "load"),
            ("Dataset 2/3: preprocess", "preprocess"),
            ("Dataset 2/3: scale", "scale"),
            ("Dataset 2/3: projection", "project"),
        ] {
            let c = PhirewallProgress::from_event(event(key, key, None), None).expect(key);
            assert!(c.is_valid());
            assert_eq!(c.stage, expected_stage);
            assert_eq!(c.outer, expected_outer);
            assert_eq!(c.done, None);
        }

        let proj = PhirewallProgress::from_event(
            event(
                "Dataset 2/3: projection",
                "Dataset 2/3: projection variant 1/2",
                Some((1, 2)),
            ),
            None,
        )
        .expect("multi-dataset projection variant");
        assert_eq!(proj.stage, "project");
        assert_eq!(proj.done, Some(1));
        assert_eq!(proj.total, Some(2));
        assert_eq!(proj.unit.as_deref(), Some("projection"));
        assert_eq!(proj.outer, expected_outer);

        let proj_cached = PhirewallProgress::from_event(
            event(
                "Dataset 2/3: projection",
                "Dataset 2/3: projection (cached)",
                None,
            ),
            None,
        )
        .expect("multi-dataset cached projection");
        assert_eq!(proj_cached.stage, "project");
        assert_eq!(proj_cached.done, None);
        assert_eq!(proj_cached.outer, expected_outer);

        let sweep = PhirewallProgress::from_event(
            event(
                "Dataset 2/3: sweep_batches",
                "Dataset 2/3: sweep batch 4/8: ball_mapper + construction",
                Some((4, 8)),
            ),
            None,
        )
        .expect("multi-dataset sweep_batches");
        assert_eq!(sweep.stage, "sweep");
        assert_eq!(sweep.done, Some(4));
        assert_eq!(sweep.total, Some(8));
        assert_eq!(sweep.unit.as_deref(), Some("batch"));
        assert_eq!(sweep.outer, expected_outer);
    }

    #[test]
    fn unknown_keys_and_malformed_inputs_produce_no_contract() {
        for bad_key in [
            "",
            "unknown",
            "finalize",
            "Dataset 1/2: unknown",
            "Dataset 0/2: load",
            "Dataset 3/2: load",
            "Dataset 1/0: load",
            "Dataset a/b: load",
            "Dataset 1: load",
        ] {
            assert_eq!(
                PhirewallProgress::from_event(event(bad_key, bad_key, None), None),
                None,
                "expected None for {bad_key:?}"
            );
        }

        // done > total or total == 0 on inner count must fail validation.
        assert_eq!(
            PhirewallProgress::from_event(event("sweep_batches", "sweep", Some((9, 8))), None),
            None
        );
        assert_eq!(
            PhirewallProgress::from_event(event("sweep_batches", "sweep", Some((0, 0))), None),
            None
        );
    }

    #[test]
    fn validates_contract_invariants() {
        assert!(is_valid_contract_id("a"));
        assert!(is_valid_contract_id("sweep_batches_1"));
        assert!(is_valid_contract_id(&"a".repeat(32)));
        assert!(!is_valid_contract_id(""));
        assert!(!is_valid_contract_id(&"a".repeat(33)));
        assert!(!is_valid_contract_id("1abc"));
        assert!(!is_valid_contract_id("_abc"));
        assert!(!is_valid_contract_id("Sweep"));
        assert!(!is_valid_contract_id("sweep-batches"));

        let base = PhirewallProgress::from_event(
            event("sweep_batches", "sweep batch 0/4", Some((0, 4))),
            Some((100, 10)),
        )
        .expect("done=0 is valid per SPEC §1 (0 <= done <= total)");
        assert!(base.is_valid());

        let mut bad_v = base.clone();
        bad_v.v = 2;
        assert!(!bad_v.is_valid());

        let mut bad_stage = base.clone();
        bad_stage.stage = "BadStage".into();
        assert!(!bad_stage.is_valid());

        let mut partial_inner = base.clone();
        partial_inner.unit = None;
        assert!(!partial_inner.is_valid());

        let mut bad_outer = base.clone();
        bad_outer.outer = Some(OuterCount {
            done: 3,
            total: 2,
            unit: "dataset".into(),
        });
        assert!(!bad_outer.is_valid());

        let mut bad_count_key = base.clone();
        bad_count_key.counts.insert("Bad-Key".into(), 1);
        assert!(!bad_count_key.is_valid());

        let mut too_many_counts = base;
        for i in 0..17 {
            too_many_counts.counts.insert(format!("k_{i}"), i);
        }
        assert!(!too_many_counts.is_valid());
    }
}
