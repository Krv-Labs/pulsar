//! The plan itself: per-column entries, open questions and the evidence they
//! rest on (`PreprocessingPlan` in `preprocessing_plan/model.py`).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::model::{PlanEntry, PlanQuestion};
use crate::evidence::PreprocessingEvidence;

/// A processed marker or dummy column that overlaps a target (or event flag)
/// so strongly it could let the map encode the outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetProxy {
    pub column: String,
    pub target: String,
    pub marker_rows: usize,
    pub share_with_target: f64,
    pub target_base_rate: f64,
}

/// Serializes losslessly (`IndexMap`s keep column and question order), so a
/// plan can be stored and reloaded: `from_value(to_value(plan)) == plan`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingPlan {
    pub plan_id: String,
    pub dataset_id: String,
    pub revision: u32,
    pub entries: IndexMap<String, PlanEntry>,
    pub questions: IndexMap<String, PlanQuestion>,
    /// `mg1 -> [columns]` from the evidence, in group order.
    pub missing_groups: IndexMap<String, Vec<String>>,
    pub evidence: PreprocessingEvidence,
    pub created_at: String,
    /// Timings and counters (`rows`, `evidence_ms`, `stakes_ms`, ...).
    pub cost: Map<String, Value>,
    pub target_proxies: Vec<TargetProxy>,
    pub n_imputations: u32,
}

impl PreprocessingPlan {
    /// The plan as JSON, for the store.
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("a plan is always representable as JSON")
    }

    /// Rebuild a stored plan; the error says which field is wrong.
    pub fn from_value(value: Value) -> Result<Self, String> {
        serde_json::from_value(value).map_err(|e| e.to_string())
    }
}
