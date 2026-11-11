//! Preprocessing plan (port of `pulsar/mcp/preprocessing_plan/`, PR 2 of the
//! plan port): an evidence-backed, editable decision per column.
//!
//! This slice holds the data model, wording helpers and the default-entry
//! rules; questions, stakes, the processed-space dry run and edits follow.

mod defaults;
mod draft;
mod edits;
mod model;
mod payload;
#[allow(clippy::module_inception)]
mod plan;
mod proxies;
mod questions;
mod render;
mod stakes;
mod text;

pub use defaults::{populate_default_entries, validate_role_designations, DraftContext};
pub use draft::{draft_plan, DraftError, DraftOptions};
pub use edits::{
    apply_answers_and_edits, diff_plan_entries, VALID_ENCODE_METHODS, VALID_IMPUTE_METHODS,
    VALID_ROLES,
};
pub use model::{
    edit, DecidedBy, Edit, EntryStatus, ImputeSpec, PendingQuestion, PlanEditError, PlanEntry,
    PlanQuestion, QuestionOption, QuestionStatus, QuestionTopic, Role,
};
pub use payload::{
    build_draft_plan_payload, build_edit_plan_payload, draft_plan_payload_to_markdown,
    edit_plan_payload_to_markdown, Detail, Limits, EVENT_FLAG_HINT_DESCRIPTION, HOW_TO_USE,
};
pub use plan::{PreprocessingPlan, TargetProxy};
pub use proxies::{calibrate_processed_space, dry_run_processed_space, DryRun, ProcessedSpace};
pub use questions::build_pending_questions;
pub use render::{
    build_marker_legend, plan_fingerprint, plan_to_preprocessing_dict, plan_to_preprocessing_yaml,
    py_canonical_json, py_json_dumps, summarize_plan_counts,
};
pub use stakes::{compute_stakes, StakesOptions};
pub use text::{
    format_col_list, format_missing_reason, is_count_like_zero_free, plural, py_repr, py_repr_list,
    py_repr_value, py_str, thousands,
};
