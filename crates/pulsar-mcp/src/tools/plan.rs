//! The preprocessing-plan tools: `draft_preprocessing_plan`,
//! `edit_preprocessing_plan` and `validate_preprocessing_config`
//! (`pulsar/mcp/tools/preprocessing.py`).
//!
//! Plans are persisted in the store under a `plan_id` handle, so they work on
//! any transport and across restarts, and edits swap the revision atomically
//! (`SESSION_REVISION_CONFLICT`). Blocking work (reading data, measuring
//! evidence, stakes, dry runs) runs on the blocking pool.

use std::collections::HashSet;

use pulsar_analysis::plan::{
    apply_answers_and_edits_guarded, build_draft_plan_payload, build_edit_plan_payload,
    build_preprocessing_summary, diff_plan_entries, draft_plan, draft_plan_payload_to_markdown,
    dry_run_processed_space, edit_plan_payload_to_markdown, processed_feature_names, DecidedBy,
    Detail as PlanDetail, DraftError, DraftOptions, HeldOneHot, ImportedDecisions, Limits,
    PlanEditError, PreprocessingPlan,
};
use pulsar_data::Table;
use pulsar_store::{PlanId, StoreError};
use rmcp::model::CallToolResult;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::config::resolve_dataset;
use super::params::{Detail, Format};
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::phirewall;
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const DRAFT: &str = "draft_preprocessing_plan";
pub const EDIT: &str = "edit_preprocessing_plan";
pub const VALIDATE: &str = "validate_preprocessing_config";

// ---- shared ------------------------------------------------------------------

fn compact(payload: &Value) -> String {
    // Plan payloads are large; keep the JSON text view token-lean.
    serde_json::to_string(payload).unwrap_or_default()
}

/// Keep only next-step tools this server actually registers.
/// The plan's `inputs` part for Phirewall's panel, so a plan shows as its own
/// card before any sweep runs; `previous` is the revision it replaced.
fn attach_summary(
    payload: &mut Value,
    table: &Table,
    plan: &PreprocessingPlan,
    previous: Option<&PreprocessingPlan>,
) {
    let names = processed_feature_names(table, plan).ok();
    let inputs =
        build_preprocessing_summary(plan, previous, names.as_deref(), phirewall::ROW_FLOOR);
    payload[phirewall::KEY] = phirewall::summary(
        None,
        Some(plan.dataset_id.clone()),
        table.n_rows(),
        [("inputs", inputs)],
    );
}

fn only_registered(payload: &mut Value) {
    let names: HashSet<&'static str> = super::catalog().iter().map(|t| t.name).collect();
    if let Some(list) = payload.get_mut("next_tools").and_then(Value::as_array_mut) {
        list.retain(|t| t.as_str().is_some_and(|n| names.contains(n)));
    }
}

async fn read_table(tool: &'static str, path: &str) -> Result<Table, ToolError> {
    let path = std::path::PathBuf::from(path);
    let shown = path.display().to_string();
    tokio::task::spawn_blocking(move || pulsar_data::read_table(&path))
        .await
        .map_err(|e| ToolError::worker_stopped(tool, "INGEST_FAILED", &e))?
        .map_err(|e| {
            ToolError::new(
                tool,
                e.code(),
                e.to_string(),
                "Check that the dataset is readable tabular data (CSV or Parquet), then retry.",
            )
            .detail("path", json!(shown))
        })
}

fn plan_error(tool: &'static str, error: &PlanEditError, action: String) -> ToolError {
    let mut out = ToolError::new(tool, &error.code, error.message.clone(), action);
    out.details = error.details.clone();
    out
}

// ---- draft_preprocessing_plan ------------------------------------------------

#[derive(serde::Deserialize, JsonSchema)]
pub struct DraftParams {
    /// Required. Dataset handle returned by ingest_dataset; an empty value returns MISSING_INPUT.
    pub dataset_id: String,
    /// Columns to hold out as evaluation outcomes (role='target', decided_by='agent', excluded from map features).
    #[serde(default)]
    pub target_columns: Option<Vec<String>>,
    /// Optional entity identifier column (role='entity', excluded from map features).
    #[serde(default)]
    pub entity_column: String,
    /// Optional longitudinal time column (role='time', excluded from map features).
    #[serde(default)]
    pub time_column: String,
    /// Compute k-NN neighborhood overlap stakes for open questions on a deterministic row sample (up to 2 * question_limit questions).
    #[serde(default = "yes")]
    pub stakes: bool,
    /// summary: capped question and evidence previews. full: uncapped lists plus every column entry in `entries`.
    #[serde(default)]
    pub detail: Detail,
    /// Max open questions returned in summary mode (>= 1).
    #[serde(default = "eight")]
    pub question_limit: i64,
    /// Max items per evidence list in summary mode (>= 1).
    #[serde(default = "twelve")]
    pub evidence_limit: i64,
    /// Text view: 'markdown' or 'json'. structured_content is identical.
    #[serde(default)]
    pub response_format: Format,
}

fn yes() -> bool {
    true
}

fn eight() -> i64 {
    8
}

fn twelve() -> i64 {
    12
}

pub fn draft_spec() -> ToolSpec {
    ToolSpec {
        name: DRAFT,
        title: "Draft Preprocessing Plan",
        description: "Draft a versioned preprocessing plan (`plan_id`, `revision=1`, `decided_by='default'`, questions ranked by k-NN `stakes`, evidence, `marker_legend`, `weak_hints.event_flag_columns`: 0/1 or count columns that explain other columns' blanks; only you can say if one is an outcome; `detail='full'` adds `how_to_use`). Resolve via `edit_preprocessing_plan` (`source='agent'` when inferred, `source='user'` after asking). Hold out outcomes via `edits=[{'column': '<col>', 'role': 'target'}]` (or `target_columns`); clear `TARGET_PROXY_COLUMNS` with `indicator='none'` or `role='drop'`; set `code_columns` to `encode='one_hot'` or `role='drop'`.",
        input_schema: input_schema::<DraftParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("plan_id", s::string()),
                ("dataset_id", s::string()),
                ("revision", s::integer()),
                ("plan_fingerprint", s::string()),
                ("detail", s::enumeration(&["summary", "full"])),
                ("counts", s::open_object()),
                ("processed_space", s::nullable(s::open_object())),
                ("marker_legend", s::array(s::open_object())),
                ("marker_legend_omitted", s::integer()),
                ("questions", s::array(s::open_object())),
                ("questions_omitted", s::integer()),
                ("evidence", s::open_object()),
                ("warnings", s::array(s::open_object())),
                ("weak_hints", s::open_object()),
                ("preprocessing_yaml_chars", s::integer()),
                ("cost", s::open_object()),
                ("how_to_use", s::string()),
                ("next_tools", s::array(s::string())),
                ("preprocessing_yaml", s::string()),
                ("entries", s::open_object()),
            ],
            &["status", "plan_id", "dataset_id", "revision", "plan_fingerprint", "detail", "counts", "questions", "evidence", "next_tools"],
            true,
        )),
        read_only: false,
        idempotent: false,
    }
}

fn limit(tool: &'static str, name: &str, value: i64, default: u32) -> Result<usize, ToolError> {
    if value < 1 {
        return Err(ToolError::invalid_argument(
            tool,
            format!("{name} must be >= 1, got '{value}'."),
            format!("Pass {name} >= 1 (default {default})."),
        ));
    }
    Ok(value as usize)
}

fn draft_error(error: DraftError) -> ToolError {
    match &error {
        DraftError::Plan(e) => plan_error(
            DRAFT,
            e,
            "Pass only column names present in the dataset for target_columns, entity_column, and time_column.".into(),
        ),
        DraftError::Evidence(e) => ToolError::new(
            DRAFT,
            e.code(),
            e.to_string(),
            "Verify the dataset loads and has rows and uniquely named columns, then retry.",
        ),
    }
}

pub async fn draft(state: &ServerState, p: DraftParams) -> Result<CallToolResult, ToolError> {
    if p.dataset_id.is_empty() {
        return Err(ToolError::new(
            DRAFT,
            "MISSING_INPUT",
            "dataset_id is required.",
            "Pass dataset_id from ingest_dataset.",
        ));
    }
    let question_limit = limit(DRAFT, "question_limit", p.question_limit, 8)?;
    let evidence_limit = limit(DRAFT, "evidence_limit", p.evidence_limit, 12)?;
    let (dataset_id, path) = resolve_dataset(state, DRAFT, &p.dataset_id)?
        .expect("a non-empty dataset_id always resolves or errors");
    let table = read_table(DRAFT, &path).await?;
    let (imported, sidecar_warnings) = ImportedDecisions::load(std::path::Path::new(&path));

    let opts = DraftOptions {
        target_columns: p.target_columns.clone().unwrap_or_default(),
        entity_column: p.entity_column.clone(),
        time_column: p.time_column.clone(),
        stakes: p.stakes,
        stakes_max_questions: 2 * question_limit,
        imported,
        phirewall_release: state.phirewall_release(),
        ..DraftOptions::default()
    };
    let detail = if p.detail == Detail::Full {
        PlanDetail::Full
    } else {
        PlanDetail::Summary
    };
    let limits = Limits {
        question_limit,
        evidence_limit,
        ..Limits::default()
    };
    let dataset = dataset_id.to_string();

    let built =
        tokio::task::spawn_blocking(move || -> Result<(PreprocessingPlan, Value), DraftError> {
            let mut plan = draft_plan(&table, &dataset, &opts)?;
            let (space, validation) = dry_run_processed_space(&table, &mut plan);
            if let (None, Some(error)) = (&space, &validation.error) {
                let notes = plan.cost.entry("stakes_notes").or_insert_with(|| json!([]));
                if let Some(list) = notes.as_array_mut() {
                    list.push(json!(format!("Processed-space dry-run failed: {error}")));
                }
            }
            let mut payload = build_draft_plan_payload(&plan, space.as_ref(), detail, limits);
            if !sidecar_warnings.is_empty() {
                if let Some(list) = payload.get_mut("warnings").and_then(Value::as_array_mut) {
                    for w in &sidecar_warnings {
                        list.push(w.to_plan_warning());
                    }
                }
            }
            attach_summary(&mut payload, &table, &plan, None);
            Ok((plan, payload))
        })
        .await
        .map_err(|e| ToolError::worker_stopped(DRAFT, "DRAFT_PLAN_FAILED", &e))?
        .map_err(draft_error)?;
    let (plan, mut payload) = built;

    let plan_id = PlanId::parse(&plan.plan_id).map_err(|e| ToolError::from_store(DRAFT, e))?;
    state
        .store
        .save_plan(&plan_id, &p.dataset_id, plan.revision, plan.to_value())
        .map_err(|e| ToolError::from_store(DRAFT, e))?;
    {
        let mut session = state.session();
        session.latest_plan_id = Some(plan_id);
        session.latest_dataset_id = Some(dataset_id);
    }

    only_registered(&mut payload);
    // The constant prose is in the tool description; `detail='full'` keeps it.
    if p.detail != Detail::Full {
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("how_to_use");
            if let Some(hints) = obj.get_mut("weak_hints").and_then(Value::as_object_mut) {
                hints.remove("description");
            }
        }
    }
    let text = match p.response_format {
        Format::Markdown => draft_plan_payload_to_markdown(&payload),
        Format::Json => compact(&payload),
    };
    Ok(ok_result(payload, Some(text)))
}

// ---- edit_preprocessing_plan -------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    #[default]
    Agent,
    User,
}

#[derive(serde::Deserialize, JsonSchema)]
pub struct EditParams {
    /// Plan handle returned by draft_preprocessing_plan.
    pub plan_id: String,
    /// Revision being edited; mismatch returns SESSION_REVISION_CONFLICT. Omit to edit the current revision.
    #[serde(default)]
    pub base_revision: Option<i64>,
    /// Answers as [{question_id, option_id}]; option_ids: [id, ...] ensembles compatible imputation treatments, preserving each option's indicator.
    #[serde(default)]
    pub answers: Option<Vec<Value>>,
    /// Direct column edits: [{'column', 'role'?, 'impute_method'?, 'impute_value'?, 'indicator'?, 'encode'?}]. 'impute' takes a method, value, seed and indicator, or a candidate list.
    #[serde(default)]
    pub edits: Option<Vec<Value>>,
    /// Columns to one-hot even with more levels than sqrt(n_rows); until named here, such a one-hot is held and listed in `warnings`.
    pub confirm_one_hot: Option<Vec<String>>,
    /// Decision provenance: 'agent' when inferred from context/schema; 'user' when relaying a user's answer.
    #[serde(default)]
    pub source: Source,
    /// Text view: 'markdown' or 'json'. structured_content is identical.
    #[serde(default)]
    pub response_format: Format,
}

pub fn edit_spec() -> ToolSpec {
    ToolSpec {
        name: EDIT,
        title: "Edit Preprocessing Plan",
        description: "Apply question `answers` and/or per-column `edits` to a stored plan. Checks `base_revision` (SESSION_REVISION_CONFLICT on mismatch), records `source='agent'|'user'`, dry-runs preprocessing (failures set `validation.valid=false`, not an error) and stores a new revision. One-hotting a column with more levels than sqrt(n_rows) is held in `warnings` (HIGH_CARDINALITY_ONE_HOT) until named in `confirm_one_hot`; `option_ids` ensembles compatible treatments that must agree on role, encoding and missingness.",
        input_schema: input_schema::<EditParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("plan_id", s::string()),
                ("dataset_id", s::string()),
                ("revision", s::integer()),
                ("plan_fingerprint", s::string()),
                ("changes", s::array(s::open_object())),
                ("changes_omitted", s::integer()),
                ("change_summary", s::array(s::open_object())),
                ("change_summary_omitted", s::integer()),
                ("counts", s::open_object()),
                ("processed_space", s::nullable(s::open_object())),
                ("marker_legend", s::array(s::open_object())),
                ("marker_legend_omitted", s::integer()),
                ("evidence", s::open_object()),
                ("warnings", s::array(s::open_object())),
                ("validation", s::object(&[("valid", s::boolean()), ("error", s::nullable(s::string()))], &["valid"], true)),
                ("open_questions_preview", s::array(s::open_object())),
                ("open_questions_omitted", s::integer()),
                ("preprocessing_yaml_chars", s::integer()),
                ("next_tools", s::array(s::string())),
            ],
            &["status", "plan_id", "revision", "plan_fingerprint", "changes", "counts", "validation", "next_tools"],
            true,
        )),
        read_only: false,
        idempotent: false,
    }
}

fn edit_error_action(error: &PlanEditError) -> String {
    let topic = error.details.get("topic").and_then(Value::as_str);
    if topic == Some("outcome_columns") {
        "To mark outcome columns, call edit_preprocessing_plan with edits=[{column, role: 'target'}].".into()
    } else if topic == Some("code_columns") {
        "To treat code columns as categories or drop them, call edit_preprocessing_plan with edits=[{column, encode: 'one_hot'}] or edits=[{column, role: 'drop'}].".into()
    } else if error.details.get("n_missing") == Some(&json!(0))
        || error.message.contains("has no blanks")
    {
        "Remove the impute/indicator edit for this column; it has no blanks.".into()
    } else {
        "Check question_id, option_id, column names, and edit fields in details, then retry.".into()
    }
}

fn revision_conflict(plan_id: &str, current: u32, base: i64) -> ToolError {
    ToolError::new(
        EDIT,
        "SESSION_REVISION_CONFLICT",
        format!("Plan '{plan_id}' is at revision {current}, but base_revision={base} was supplied."),
        format!("Inspect the current plan or retry edit_preprocessing_plan with base_revision={current}."),
    )
    .detail("plan_id", json!(plan_id))
    .detail("current_revision", json!(current))
    .detail("base_revision", json!(base))
}

/// The warning for one-hot encodings left out because the column has more levels
/// than the plan's categorical threshold. Facts, the alternatives, and how to confirm.
fn held_one_hot_warning(held: &[HeldOneHot], features_now: Option<usize>) -> Value {
    let added: usize = held.iter().map(|h| h.features_added).sum();
    let parts: Vec<String> = held
        .iter()
        .map(|h| {
            format!(
                "`{}` has {} levels (threshold sqrt(n_rows) = {}) and would add {} features",
                h.column, h.levels, h.threshold, h.features_added
            )
        })
        .collect();
    let total = match features_now {
        Some(n) => format!(" Processed features: {n} now, {} if applied.", n + added),
        None => String::new(),
    };
    let cols: Vec<&str> = held.iter().map(|h| h.column.as_str()).collect();
    let answered: Vec<&str> = held
        .iter()
        .filter_map(|h| h.question_id.as_deref())
        .collect();
    let scope = if answered.is_empty() {
        String::new()
    } else {
        format!(
            " An answer is held whole, so question(s) {} stay open.",
            answered.join(", ")
        )
    };
    json!({
        "code": "HIGH_CARDINALITY_ONE_HOT",
        "target": "plan",
        "message": format!(
            "Not applied, the plan keeps its current treatment: {}.{total}{scope} Alternatives are role 'drop', treating the column as numeric (omit encode), or one-hot anyway by resending the same answers/edits with confirm_one_hot={}.",
            parts.join("; "),
            json!(cols)
        ),
        "columns": cols,
    })
}

pub async fn edit(state: &ServerState, p: EditParams) -> Result<CallToolResult, ToolError> {
    if p.plan_id.is_empty() {
        return Err(ToolError::new(
            EDIT,
            "MISSING_INPUT",
            "plan_id is required.",
            "Pass plan_id from draft_preprocessing_plan.",
        ));
    }
    let plan_id = PlanId::parse(&p.plan_id).map_err(|e| ToolError::from_store(EDIT, e))?;
    let record = state
        .store
        .get_plan(&plan_id)
        .map_err(|e| ToolError::from_store(EDIT, e))?;
    let plan = PreprocessingPlan::from_value(record.plan).map_err(|reason| {
        ToolError::new(
            EDIT,
            "PLAN_RECORD_CORRUPT",
            format!("The stored plan cannot be read: {reason}"),
            "Draft the plan again with draft_preprocessing_plan.",
        )
        .detail("plan_id", json!(p.plan_id))
    })?;

    let expected = p.base_revision.unwrap_or(i64::from(plan.revision));
    if expected != i64::from(plan.revision) {
        return Err(revision_conflict(&p.plan_id, plan.revision, expected));
    }

    let (_, path) = resolve_dataset(state, EDIT, &plan.dataset_id)?.ok_or_else(|| {
        ToolError::from_store(
            EDIT,
            StoreError::InvalidArgument("the plan has no dataset".into()),
        )
    })?;
    let table = read_table(EDIT, &path).await?;

    let (answers, edits) = (
        p.answers.clone().unwrap_or_default(),
        p.edits.clone().unwrap_or_default(),
    );
    let source = if p.source == Source::User {
        DecidedBy::User
    } else {
        DecidedBy::Agent
    };
    let confirmed = p.confirm_one_hot.clone().unwrap_or_default();
    let phirewall_release = state.phirewall_release();
    let mut worker_plan = plan.clone();
    worker_plan.phirewall_release = phirewall_release;
    let (updated, mut payload) = tokio::task::spawn_blocking(
        move || -> Result<(PreprocessingPlan, Value), PlanEditError> {
            let (mut updated, held) = apply_answers_and_edits_guarded(
                &worker_plan,
                &answers,
                &edits,
                source,
                &confirmed,
            )?;
            // Whatever was not held is applied, so a new revision is stored for any
            // change to entries (settings, provenance) or question state. Only when
            // everything was held is the stored revision kept rather than minting an
            // unchanged one.
            if updated.entries == worker_plan.entries && updated.questions == worker_plan.questions
            {
                updated = worker_plan.clone();
            }
            let changes = diff_plan_entries(&worker_plan, &updated);
            let (space, validation) = dry_run_processed_space(&table, &mut updated);
            let mut payload = build_edit_plan_payload(
                &updated,
                &changes,
                space.as_ref(),
                &validation,
                Limits::default(),
            );
            if !held.is_empty() {
                let now = space.as_ref().map(|sp| sp.feature_count);
                payload["warnings"]
                    .as_array_mut()
                    .expect("warnings is an array")
                    .push(held_one_hot_warning(&held, now));
            }
            attach_summary(&mut payload, &table, &updated, Some(&worker_plan));
            Ok((updated, payload))
        },
    )
    .await
    .map_err(|e| ToolError::worker_stopped(EDIT, "EDIT_PLAN_FAILED", &e))?
    .map_err(|e| plan_error(EDIT, &e, edit_error_action(&e)))?;

    if updated.revision != plan.revision {
        state
            .store
            .update_plan(
                &plan_id,
                plan.revision,
                updated.revision,
                updated.to_value(),
            )
            .map_err(|e| match e {
                StoreError::RevisionConflict { current, .. } => {
                    revision_conflict(&p.plan_id, current, expected)
                }
                other => ToolError::from_store(EDIT, other),
            })?;
    }
    state.session().latest_plan_id = Some(plan_id);

    only_registered(&mut payload);
    let text = match p.response_format {
        Format::Markdown => edit_plan_payload_to_markdown(&payload),
        Format::Json => compact(&payload),
    };
    Ok(ok_result(payload, Some(text)))
}

// ---- validate_preprocessing_config -------------------------------------------

#[derive(serde::Deserialize, JsonSchema)]
pub struct ValidateParams {
    /// Full config as YAML text (a mapping with a `preprocessing` section); only preprocessing is executed.
    pub config_yaml: String,
    /// Dataset handle from ingest_dataset to validate against. Omit to use the dataset most recently registered or validated in this session.
    #[serde(default)]
    pub dataset_id: String,
}

pub fn validate_spec() -> ToolSpec {
    ToolSpec {
        name: VALIDATE,
        title: "Validate Preprocessing Config",
        description: "Dry-run only the preprocessing of a config against a dataset. Cheap (no sweep) and read-only. Returns row/feature counts and expansion diagnostics, or a CONFIG_INVALID error with guidance to fix it via `edit_preprocessing_plan` or by editing the YAML. validate_config does the full schema check.",
        input_schema: input_schema::<ValidateParams>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("valid", s::boolean()),
                ("input_rows", s::integer()),
                ("output_rows", s::integer()),
                ("output_feature_count", s::integer()),
                ("dummy_expansion_count", s::integer()),
                ("missingness_flag_count", s::integer()),
                ("high_cardinality_encoded_columns", s::array(s::string())),
                ("feature_names_preview", s::array(s::string())),
                ("nan_remaining", s::integer()),
            ],
            &["status", "valid", "input_rows", "output_rows", "output_feature_count", "nan_remaining"],
            false,
        )),
        read_only: true,
        idempotent: true,
    }
}

fn config_error(errors: &pulsar_config::ConfigErrors) -> ToolError {
    let syntax = errors
        .iter()
        .any(|e| e.code == pulsar_config::ErrorCode::YamlSyntax);
    let issues: Vec<Value> = errors
        .iter()
        .map(|e| serde_json::to_value(e).unwrap_or_else(|_| json!(e.to_string())))
        .collect();
    if syntax {
        ToolError::new(
            VALIDATE,
            "CONFIG_YAML_INVALID",
            format!("config_yaml is not valid YAML: {errors}"),
            "Pass raw YAML (no Markdown fences) and fix the syntax error, then retry.",
        )
        .detail("valid", json!(false))
        .detail("issues", Value::Array(issues))
    } else {
        ToolError::new(
            VALIDATE,
            "CONFIG_INVALID",
            errors.to_string(),
            "Update the preprocessing plan via edit_preprocessing_plan or edit the preprocessing block in config_yaml, then re-run validate_preprocessing_config.",
        )
        .detail("valid", json!(false))
        .detail("issues", Value::Array(issues))
    }
}

fn validation_payload(
    table: &Table,
    config: &pulsar_config::PulsarConfig,
    phirewall_release: bool,
) -> Result<Value, ToolError> {
    let invalid = |reason: String| {
        ToolError::new(
            VALIDATE,
            "CONFIG_INVALID",
            reason,
            "Update the preprocessing plan via edit_preprocessing_plan or edit the preprocessing block in config_yaml, then re-run validate_preprocessing_config.",
        )
        .detail("valid", json!(false))
    };
    // Every variant of a multi-imputation ensemble must be applicable.
    let variants = config.preprocessing_variants();
    let first = pulsar_data::preprocess(table, &variants[0].preprocessing, None, None)
        .map_err(|e| invalid(e.display_for(phirewall_release)))?;
    let mut nan_remaining = first.matrix.iter().filter(|v| v.is_nan()).count();
    for variant in &variants[1..] {
        let out = pulsar_data::preprocess(table, &variant.preprocessing, None, None)
            .map_err(|e| invalid(e.display_for(phirewall_release)))?;
        nan_remaining = nan_remaining.max(out.matrix.iter().filter(|v| v.is_nan()).count());
    }

    let names = first.feature_names();
    let inputs: HashSet<&str> = table.names.iter().map(String::as_str).collect();
    let encoded: Vec<&str> = config
        .preprocessing
        .encode
        .keys()
        .map(String::as_str)
        .collect();
    let is_marker = |n: &str| n.ends_with("_was_missing");
    let dummy_owner = |n: &str| -> Option<&str> {
        if is_marker(n) || inputs.contains(n) {
            return None;
        }
        encoded
            .iter()
            .copied()
            .filter(|c| n.len() > c.len() + 1 && n.starts_with(c) && n[c.len()..].starts_with('_'))
            .max_by_key(|c| c.len())
    };
    let dummies = names.iter().filter(|n| dummy_owner(n).is_some()).count();
    let mut level_idx: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut preview: Vec<String> = names
        .iter()
        .enumerate()
        .take(8)
        .map(|(j, name)| {
            if phirewall_release {
                if let Some(col) = dummy_owner(name) {
                    let idx = level_idx.entry(col).and_modify(|k| *k += 1).or_insert(1);
                    let rows = first.matrix.column(j).iter().filter(|&&v| v == 1.0).count();
                    if rows < phirewall::ROW_FLOOR {
                        return format!("{col}_level_{idx}");
                    }
                }
            }
            name.clone()
        })
        .collect();
    if names.len() > 8 {
        preview.push(format!("... +{} more", names.len() - 8));
    }
    let high_cardinality: Vec<&String> = first
        .layout
        .vocab
        .iter()
        .filter(|(_, cats)| cats.len() > 20)
        .map(|(c, _)| c)
        .collect();
    Ok(json!({
        "status": "ok",
        "valid": true,
        "input_rows": table.n_rows(),
        "output_rows": first.layout.n_rows,
        "output_feature_count": names.len(),
        "dummy_expansion_count": dummies,
        "missingness_flag_count": names.iter().filter(|n| is_marker(n)).count(),
        "high_cardinality_encoded_columns": high_cardinality,
        "feature_names_preview": preview,
        "nan_remaining": nan_remaining,
    }))
}

pub async fn validate(state: &ServerState, p: ValidateParams) -> Result<CallToolResult, ToolError> {
    let (dataset_id, path) = match resolve_dataset(state, VALIDATE, &p.dataset_id)? {
        Some(found) => found,
        None => {
            let latest = state.session().latest_dataset_id.clone();
            let id = latest.ok_or_else(|| {
                ToolError::new(
                    VALIDATE,
                    "SESSION_DATA_MISSING",
                    "No dataset_id given and no dataset has been used in this session.",
                    "Pass dataset_id (from ingest_dataset) to validate against that dataset.",
                )
            })?;
            let record = state
                .store
                .check_dataset(&id)
                .map_err(|e| ToolError::from_store(VALIDATE, e))?;
            (id, record.path.display().to_string())
        }
    };
    state.session().latest_dataset_id = Some(dataset_id);

    let text = super::config::strip_fences(&p.config_yaml).to_string();
    if text.is_empty() {
        return Err(ToolError::new(
            VALIDATE,
            "CONFIG_YAML_INVALID",
            "config_yaml must be a valid YAML mapping.",
            "Pass config_yaml as a YAML mapping with a preprocessing section.",
        ));
    }
    let config = pulsar_config::from_yaml_str(&text).map_err(|e| config_error(&e))?;
    let table = read_table(VALIDATE, &path).await?;
    let phirewall_release = state.phirewall_release();
    let payload =
        tokio::task::spawn_blocking(move || validation_payload(&table, &config, phirewall_release))
            .await
            .map_err(|e| {
                ToolError::worker_stopped(VALIDATE, "VALIDATE_PREPROCESSING_FAILED", &e)
            })??;
    Ok(ok_result(payload, None))
}
