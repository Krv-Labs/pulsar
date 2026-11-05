//! The `io.phirewall/release` gate: the client's declared capability, and
//! the dataset path a run's block is resolved against.

use pulsar_store::{DatasetSource, RunId};
use rmcp::model::ClientCapabilities;
use serde_json::Value;

use super::KEY;
use crate::state::ServerState;

/// True iff the client declared experimental `io.phirewall/release` with `{"version": 1}`.
pub fn declared_by(capabilities: &ClientCapabilities) -> bool {
    capabilities
        .experimental
        .as_ref()
        .and_then(|exp| exp.get(KEY))
        .and_then(|v| v.get("version"))
        .and_then(Value::as_u64)
        == Some(1)
}

/// The dataset path a block for `run_id` names, or `None` when the session
/// did not ask for blocks or the run's dataset has no host path.
pub(super) fn released_dataset_path(state: &ServerState, run_id: &RunId) -> Option<String> {
    if !state.phirewall_release() {
        return None;
    }
    dataset_path_of(state, run_id)
}

fn dataset_path_of(state: &ServerState, run_id: &RunId) -> Option<String> {
    let run = state.store.get_run(run_id).ok()?;
    let ds_id = run.dataset_id?;
    let ds = state.store.get_dataset(&ds_id).ok()?;
    if ds.source != DatasetSource::Path {
        return None;
    }
    Some(ds.path.display().to_string())
}
