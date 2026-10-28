//! `create_config`: a broad baseline sweep config for an ingested dataset
//! (`create_config` in `pulsar/mcp/tools/config.py`).
//!
//! Epsilon and projection dimensions are calibrated in the processed feature
//! space under a preprocessing plan: the one named by `plan_id`, else a fresh
//! default plan that is stored so it can be edited afterwards. The YAML
//! becomes the session's active config.

use pulsar_analysis::characterization::characterize_dataset;
use pulsar_analysis::config_build::{
    build_initial_config_yaml, calibration_curve, grid_ceiling, variance_elbow_dimension,
    CeilingSource, GridCeiling,
};
use pulsar_analysis::plan::{
    calibrate_processed_space, draft_plan, plan_fingerprint, summarize_plan_counts, DraftOptions,
    PreprocessingPlan, QuestionStatus,
};
use pulsar_analysis::pyround::round_to;
use pulsar_store::PlanId;
use rmcp::model::CallToolResult;
use schemars::JsonSchema;
use serde_json::{json, Value};

use super::characterize::{characterization_error, load, registered};
use super::config::{parse_config, resolve_dataset};
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const NAME: &str = "create_config";
/// Questions whose stakes are measured when a default plan is drafted here.
const STAKES_QUESTIONS: usize = 6;
/// Open questions previewed in the plan provenance block.
const PREVIEW: usize = 3;

#[derive(serde::Deserialize, JsonSchema, Default)]
#[serde(default)]
pub struct Params {
    /// Dataset handle returned by ingest_dataset.
    pub dataset_id: String,
    /// Optional short run name (e.g. 'baseline') stored as run.name in the config. Defaults to 'initial_sweep'.
    pub intent: String,
    /// Optional preprocessing plan handle from draft_preprocessing_plan. If omitted, baseline defaults are drafted and stored automatically.
    pub plan_id: String,
}

pub fn spec() -> ToolSpec {
    let warning = s::object(
        &[("code", s::string()), ("message", s::string())],
        &["code", "message"],
        false,
    );
    let sweep_strategy = s::object(
        &[
            ("projection_method", s::string()),
            ("projection_dimensions", s::array(s::integer())),
            ("projection_seeds", s::array(s::integer())),
            ("epsilon_steps", s::integer()),
            ("imputation_variants", s::integer()),
            ("estimated_ball_maps", s::integer()),
            ("n_rows", s::integer()),
            ("grid_ceiling", s::integer()),
            (
                "grid_ceiling_source",
                s::enumeration(&["variance_elbow", "probe_limit", "n_features", "fallback"]),
            ),
            ("variance_at_ceiling", s::nullable(s::number())),
            ("cost_note", s::string()),
        ],
        &[
            "projection_method",
            "projection_dimensions",
            "projection_seeds",
            "epsilon_steps",
            "imputation_variants",
            "estimated_ball_maps",
            "n_rows",
            "grid_ceiling",
            "grid_ceiling_source",
            "variance_at_ceiling",
            "cost_note",
        ],
        true,
    );
    let provenance = s::object(
        &[
            ("plan_id", s::string()),
            ("revision", s::integer()),
            ("plan_fingerprint", s::string()),
            (
                "source",
                s::enumeration(&["explicit_plan", "baseline_defaults"]),
            ),
            ("open_questions", s::integer()),
            ("open_questions_preview", s::array(s::open_object())),
            ("decided_by", s::open_object()),
        ],
        &[
            "plan_id",
            "revision",
            "plan_fingerprint",
            "source",
            "open_questions",
            "open_questions_preview",
            "decided_by",
        ],
        false,
    );
    ToolSpec {
        name: NAME,
        title: "Create Config",
        description: "Generate a broad baseline Pulsar sweep config (YAML) for an ingested dataset. Epsilon and dimensions are calibrated in the processed feature space under the preprocessing plan (or baseline defaults if plan_id is omitted, with a warning when open questions remain). Stores the YAML as the session's active config.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("config_yaml", s::string()),
                ("calibration_space", s::enumeration(&["processed", "raw"])),
                ("preprocessing_plan", provenance),
                ("warnings", s::array(warning)),
                ("sweep_strategy", sweep_strategy),
                ("processed_feature_count", s::integer()),
                ("raw_to_processed_expansion_ratio", s::number()),
                ("knn_distance_percentiles", s::open_object()),
                ("calibration_note", s::string()),
            ],
            &["status", "config_yaml", "calibration_space", "preprocessing_plan", "sweep_strategy", "calibration_note"],
            false,
        )),
        read_only: false,
        idempotent: false,
    }
}

fn failed(code: &str, reason: impl Into<String>, action: &str) -> ToolError {
    ToolError::new(NAME, code, reason, action)
}

/// The plan named by `plan_id`, checked against the dataset.
fn stored_plan(
    state: &ServerState,
    plan_id: &str,
    dataset_id: &str,
) -> Result<PreprocessingPlan, ToolError> {
    let id = PlanId::parse(plan_id).map_err(|e| ToolError::from_store(NAME, e))?;
    let record = state
        .store
        .get_plan(&id)
        .map_err(|e| ToolError::from_store(NAME, e))?;
    let plan = PreprocessingPlan::from_value(record.plan).map_err(|reason| {
        failed(
            "PLAN_RECORD_CORRUPT",
            format!("The stored plan cannot be read: {reason}"),
            "Draft the plan again with draft_preprocessing_plan.",
        )
        .detail("plan_id", json!(plan_id))
    })?;
    if plan.dataset_id != dataset_id {
        return Err(ToolError::invalid_argument(
            NAME,
            format!("Plan '{plan_id}' belongs to dataset '{}', not '{dataset_id}'.", plan.dataset_id),
            "Pass a plan_id created for this dataset_id, or call draft_preprocessing_plan(dataset_id).",
        )
        .detail("plan_id", json!(plan_id))
        .detail("plan_dataset_id", json!(plan.dataset_id))
        .detail("dataset_id", json!(dataset_id)));
    }
    Ok(plan)
}

/// An agent- or user-edited plan for the dataset that this call is about to
/// bypass, most recently worked on last.
fn bypassed_edited_plan(state: &ServerState, dataset_id: &str) -> Option<PreprocessingPlan> {
    state
        .store
        .list_plans(Some(dataset_id))
        .ok()?
        .into_iter()
        .filter(|r| r.revision > 1)
        .filter_map(|r| PreprocessingPlan::from_value(r.plan).ok())
        .next_back()
}

fn agent_guidance() -> String {
    let next = if registered("refine_config") {
        "use refine_config"
    } else {
        "edit the config YAML (validate_config checks it)"
    };
    let then = if registered("compare_sweeps") {
        " plus compare_sweeps"
    } else {
        ""
    };
    format!(
        "Run this broad baseline first, inspect diagnose_cosmic_graph, then {next}{then} to shift or concentrate the grid around the informative region."
    )
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    if p.dataset_id.is_empty() {
        return Err(failed(
            "MISSING_INPUT",
            "dataset_id is required.",
            "Pass dataset_id from ingest_dataset.",
        ));
    }
    let (dataset_id, path) = resolve_dataset(state, NAME, &p.dataset_id)?
        .expect("a non-empty dataset_id always resolves or errors");

    let (given, plan_source) = if p.plan_id.is_empty() {
        (None, "baseline_defaults")
    } else {
        (
            Some(stored_plan(state, &p.plan_id, &p.dataset_id)?),
            "explicit_plan",
        )
    };
    let edited_plan = if given.is_none() {
        bypassed_edited_plan(state, &p.dataset_id)
    } else {
        None
    };

    let table = load(NAME, &path).await?;
    let (dataset, data_path) = (p.dataset_id.clone(), path.clone());
    let run_name = Some(p.intent.trim())
        .filter(|n| !n.is_empty())
        .unwrap_or("initial_sweep")
        .to_owned();
    let built = tokio::task::spawn_blocking(move || -> Result<_, ToolError> {
        let plan = match given {
            Some(plan) => plan,
            None => {
                let opts = DraftOptions {
                    stakes: true,
                    stakes_max_questions: STAKES_QUESTIONS,
                    ..DraftOptions::default()
                };
                draft_plan(&table, &dataset, &opts).map_err(|e| {
                    failed(
                        "CONFIG_CREATION_FAILED",
                        e.to_string(),
                        "Check the dataset is valid and the data is readable, then retry.",
                    )
                })?
            }
        };
        let geo =
            characterize_dataset(&table, 1000, 42).map_err(|e| characterization_error(NAME, e))?;
        let processed = calibrate_processed_space(&table, &plan);
        let yaml = build_initial_config_yaml(
            &geo,
            &data_path,
            &run_name,
            processed.as_ref().ok(),
            &plan,
            None,
        );
        Ok((plan, geo, processed, yaml, table.n_rows()))
    })
    .await
    .map_err(|e| ToolError::worker_stopped(NAME, "CONFIG_CREATION_FAILED", &e))??;
    let (plan, geo, processed, config_yaml, n_rows) = built;
    let (cumulative, n_features) = calibration_curve(&geo, processed.as_ref().ok());
    let ceiling = grid_ceiling(cumulative, n_features);
    let ceiling_note = ceiling_description(&ceiling, cumulative, n_features);

    // A freshly drafted plan is stored so `edit_preprocessing_plan` can refine it.
    let plan_handle = PlanId::parse(&plan.plan_id).map_err(|e| ToolError::from_store(NAME, e))?;
    if plan_source == "baseline_defaults" {
        state
            .store
            .save_plan(&plan_handle, &p.dataset_id, plan.revision, plan.to_value())
            .map_err(|e| ToolError::from_store(NAME, e))?;
    }

    let config = parse_config(NAME, &config_yaml)?;
    let variants = config.preprocessing_variants().len();
    let (dims, seeds) = (&config.projection.dimensions, &config.projection.seeds);
    let counts = summarize_plan_counts(&plan);
    let open = counts["open_questions"].as_u64().unwrap_or(0);
    let preview: Vec<Value> = plan
        .questions
        .values()
        .filter(|q| q.status == QuestionStatus::Open)
        .take(PREVIEW)
        .map(|q| {
            json!({
                "question_id": q.question_id,
                "topic": q.topic,
                "rows_affected": q.rows_affected,
                "stakes": q.stakes,
            })
        })
        .collect();

    let mut payload = json!({
        "status": "ok",
        "config_yaml": config_yaml,
        "calibration_space": if processed.is_ok() { "processed" } else { "raw" },
        "preprocessing_plan": {
            "plan_id": plan.plan_id,
            "revision": plan.revision,
            "plan_fingerprint": plan_fingerprint(&plan),
            "source": plan_source,
            "open_questions": open,
            "open_questions_preview": preview,
            "decided_by": counts["decided_by"],
        },
        "sweep_strategy": {
            "projection_method": "jl",
            "projection_dimensions": dims,
            "projection_seeds": seeds,
            "epsilon_steps": config.ball_mapper.epsilons.len(),
            "imputation_variants": variants,
            "estimated_ball_maps": variants * dims.len() * seeds.len() * config.ball_mapper.epsilons.len(),
            "n_rows": n_rows,
            "grid_ceiling": ceiling.dim,
            "grid_ceiling_source": ceiling.source.as_str(),
            "variance_at_ceiling": ceiling.variance,
            "cost_note": "Runtime grows with ball maps times rows and with the ball count per map (a smaller epsilon or a higher dimension gives more balls); the radius-query backend changes with dimension, ball membership does not.",
            "compatibility_note": (!dims.is_empty()).then_some(
                "sweep.pca is a legacy mirror of sweep.projection for older configs; sweep.projection is authoritative."
            ),
            "agent_guidance": agent_guidance(),
            "parameter_reasoning": [
                "Projection dimensions are selected from the processed cumulative-variance structure to span multiple resolutions; the default projection method is JL.",
                "Epsilon is calibrated in processed feature space when possible, with k-NN percentiles returned as the safe domain for refinement.",
            ],
        },
    });

    let mut warnings = vec![];
    if let Some(edited) = &edited_plan {
        let decisions = counts_decisions(edited);
        warnings.push(json!({
            "code": "PREPROCESSING_PLAN_NOT_USED",
            "message": format!(
                "Session holds edited plan '{}' (revision {}, {decisions} agent/user decisions) for dataset '{}', but plan_id was not passed so a fresh default plan '{}' was drafted instead. Pass plan_id={} to use it.",
                edited.plan_id, edited.revision, p.dataset_id, plan.plan_id, edited.plan_id
            ),
        }));
    }
    if open > 0 {
        let preview_json =
            serde_json::to_string(&payload["preprocessing_plan"]["open_questions_preview"])
                .unwrap_or_default();
        warnings.push(json!({
            "code": "PREPROCESSING_QUESTIONS_OPEN",
            "message": format!(
                "{open} preprocessing questions are unanswered on plan '{}' (open_questions_preview={preview_json}); this config uses labelled defaults for them. call draft_preprocessing_plan(dataset_id) for the full cards, or edit_preprocessing_plan(plan_id, ...).",
                plan.plan_id
            ),
        }));
    }
    if !warnings.is_empty() {
        payload["warnings"] = Value::Array(warnings);
    }
    match &processed {
        Ok(m) => {
            payload["processed_feature_count"] = json!(m.n_features);
            payload["raw_to_processed_expansion_ratio"] = json!(round_to(
                m.n_features as f64 / geo.n_features.max(1) as f64,
                2
            ));
            payload["knn_distance_percentiles"] = json!({
                "p5": round_to(m.knn_p5, 4),
                "p25": round_to(m.knn_p25, 4),
                "p50": round_to(m.knn_p50, 4),
                "p75": round_to(m.knn_p75, 4),
                "p95": round_to(m.knn_p95, 4),
            });
            payload["calibration_note"] = json!(format!(
                "Geometry calibrated under the active preprocessing plan. Epsilon and projection dimensions reflect the processed feature space (after drop/impute/encode/scale), not raw columns. Dimensions are selected from the cumulative-variance structure: the 50/70/85% frontiers plus the elbow where cumulative variance first reaches 90%, bounded by the processed feature count. Variance is probed at those of 2, 3, 5, 10, 15 and 20 dimensions that fit within the feature and row counts. {ceiling_note} sweep_strategy.grid_ceiling_source names the bound that set the top. The default projection is JL. knn_distance_percentiles show the valid epsilon domain \u{2014} epsilon values outside [p5, p95] will produce degenerate graphs."
            ));
        }
        Err(fallback) => {
            payload["calibration_note"] = json!(format!(
                "Processed-space calibration fell back to raw numeric columns because {fallback}. Epsilon and projection dimensions in this config are therefore measured on the raw numeric columns, mean-imputed and z-scored, not on the plan's drop/impute/encode/scale output, and no knn_distance_percentiles are reported. {ceiling_note}"
            ));
        }
    }

    {
        let mut session = state.session();
        session.active_config_yaml = Some(config_yaml);
        session.active_config_dataset_id = Some(dataset_id.clone());
        session.latest_dataset_id = Some(dataset_id);
        session.latest_plan_id = Some(plan_handle);
    }
    Ok(ok_result(payload, None))
}

/// What set the top of the dimension grid, stated against the dimensions the
/// variance curve actually measured.
fn ceiling_description(
    ceiling: &GridCeiling,
    cumulative: &[(usize, f64)],
    n_features: usize,
) -> String {
    let dim = ceiling.dim;
    let probes = cumulative
        .iter()
        .map(|(d, _)| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    match ceiling.source {
        CeilingSource::VarianceElbow => format!(
            "Cumulative variance first reached 90% at dimension {dim}; the curve was probed at {probes} within {n_features} features, so the grid tops out at {dim}."
        ),
        CeilingSource::ProbeLimit => {
            let unmeasured = if dim < n_features {
                format!(" Dimensions between {dim} and {n_features} were not measured.")
            } else {
                String::new()
            };
            format!(
                "Cumulative variance had not reached 90% at the largest probed dimension ({dim}); the curve was probed at {probes} within {n_features} features.{unmeasured}"
            )
        }
        CeilingSource::NFeatures => format!(
            "The variance elbow ({}) lies above the feature count, so the grid tops out at the {n_features} available features.",
            variance_elbow_dimension(cumulative, n_features)
        ),
        CeilingSource::Fallback => format!(
            "No variance curve was measured; the top dimension ({dim}) comes from a feature-count heuristic over {n_features} features."
        ),
    }
}

/// Answers given by an agent or user on `plan`.
fn counts_decisions(plan: &PreprocessingPlan) -> u64 {
    let counts = summarize_plan_counts(plan);
    counts["decided_by"]["agent"].as_u64().unwrap_or(0)
        + counts["decided_by"]["user"].as_u64().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_limit_names_the_largest_probe_and_the_unmeasured_range() {
        // Eight features: the probe set filters to 2, 3, 5.
        let cumulative = [(2, 0.4), (3, 0.55), (5, 0.8)];
        let ceiling = grid_ceiling(&cumulative, 8);
        assert_eq!(ceiling.source, CeilingSource::ProbeLimit);
        let text = ceiling_description(&ceiling, &cumulative, 8);
        assert!(text.contains("largest probed dimension (5)"), "{text}");
        assert!(
            text.contains("probed at 2, 3, 5 within 8 features"),
            "{text}"
        );
        assert!(text.contains("between 5 and 8 were not measured"), "{text}");
        assert!(!text.contains("20"), "{text}");
    }
}
