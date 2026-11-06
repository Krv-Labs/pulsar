//! The `io.phirewall/summary` v1 object: one key in a tool result's
//! structured content that the Phirewall gateway renders as its analysis
//! panel, beside Pulsar's own fields. Its schema is Phirewall's
//! `docs/contracts/summary-v1.json` (vendored for the contract tests as
//! `tests/fixtures/phirewall-summary-v1.json`).
//!
//! Each tool sets the parts it has: `create_config` its `inputs`, a sweep
//! its `thresholds`, `changes` (and `inputs` under a stored plan), the
//! dossier its `grouping`, the curve its `thresholds`. The parts are built
//! by the analysis code; this module adds the envelope.

use pulsar_analysis::characterization::GROUP_SUMMARY_DEFAULTS;
use pulsar_analysis::disclosure::RowCount;
use serde_json::{json, Map, Value};

/// The structured-content key.
pub const KEY: &str = "io.phirewall/summary";

/// The fewest rows any disclosed count stands for: the dossier's own floor.
pub const ROW_FLOOR: usize = GROUP_SUMMARY_DEFAULTS.min_rows;

/// The object for the run `run_id` (or, before a run, the dataset
/// `dataset_id`) of `rows` rows, with `parts`. At least one id is required.
pub fn summary(
    run_id: Option<String>,
    dataset_id: Option<String>,
    rows: usize,
    parts: impl IntoIterator<Item = (&'static str, Value)>,
) -> Value {
    debug_assert!(run_id.is_some() || dataset_id.is_some());
    let mut run = Map::new();
    if let Some(id) = run_id {
        run.insert("run_id".into(), json!(id));
    }
    if let Some(id) = dataset_id {
        run.insert("dataset_id".into(), json!(id));
    }
    run.insert("rows".into(), json!(RowCount::of(rows, ROW_FLOOR)));
    let mut out = Map::new();
    out.insert("version".into(), json!(1));
    out.insert("row_floor".into(), json!(ROW_FLOOR));
    out.insert("run".into(), Value::Object(run));
    out.extend(parts.into_iter().map(|(k, v)| (k.to_owned(), v)));
    Value::Object(out)
}
