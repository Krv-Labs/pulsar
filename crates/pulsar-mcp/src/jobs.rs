//! Running a sweep as a store job: the blocking fit on a worker thread,
//! progress persisted through the [`JobHandle`] and forwarded to the client
//! as MCP progress notifications, then the run committed and cached.

use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use pulsar_config::PulsarConfig;
use pulsar_data::Table;
use pulsar_pipeline::{fit, FitOptions};
use pulsar_store::{DatasetId, JobHandle, JobKind, RunArtifacts, RunRecord};
use rmcp::model::{ProgressNotificationParam, ProgressToken};
use rmcp::service::{Peer, RoleServer};
use tokio::sync::mpsc;

use crate::envelope::ToolError;
use crate::state::{LiveRun, ServerState};

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
    let (tx, mut rx) = mpsc::unbounded_channel::<(String, f64)>();
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
        let mut on_progress = |stage: &str, fraction: f64| {
            let _ = worker_job.progress(stage, fraction);
            let _ = tx.send((stage.to_owned(), fraction));
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
        while let Some((stage, fraction)) = rx.recv().await {
            if let Some(target) = &progress {
                let _ = target
                    .peer
                    .notify_progress(
                        ProgressNotificationParam::new(target.token.clone(), fraction)
                            .with_total(1.0)
                            .with_message(stage),
                    )
                    .await;
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
                "Report the error; retry with a smaller grid if it recurs.",
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
        if let Some(ds) = &live.dataset_id {
            session.latest_dataset_id = Some(ds.clone());
        }
    }
    Ok((live, record, artifacts))
}
