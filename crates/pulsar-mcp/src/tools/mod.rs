//! The tool catalog: one [`ToolSpec`] per tool (name, description written
//! for an agent that has never seen Pulsar, input schema from the parameter
//! struct, output schema, annotations) and the dispatcher.

pub mod artifact;
pub mod characterize;
pub mod cluster_evidence;
pub mod clustering;
pub mod config;
pub mod create_config;
pub mod diagnostics;
pub mod export;
pub mod ingest;
pub mod longitudinal;
pub mod meta;
pub mod plan;
pub mod refine_config;
pub mod report;
pub mod sweep;

use std::sync::Arc;

use rmcp::model::{CallToolResult, JsonObject, Tool, ToolAnnotations};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::envelope::ToolError;
use crate::jobs::ProgressTarget;
use crate::state::ServerState;

/// Everything `tools/list` advertises for one tool.
#[derive(Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub input_schema: Arc<JsonObject>,
    pub output_schema: Arc<JsonObject>,
    pub read_only: bool,
    pub idempotent: bool,
}

impl ToolSpec {
    pub fn to_tool(&self) -> Tool {
        let annotations = ToolAnnotations::with_title(self.title)
            .read_only(self.read_only)
            .destructive(false)
            .idempotent(self.idempotent)
            .open_world(false);
        Tool::new(self.name, self.description, self.input_schema.clone())
            .with_title(self.title)
            .with_raw_output_schema(self.output_schema.clone())
            .with_annotations(annotations)
    }
}

/// Per-call context handed to a tool.
pub struct CallContext {
    pub progress: Option<ProgressTarget>,
}

/// The catalog in its advertised order.
pub fn catalog() -> Vec<ToolSpec> {
    vec![
        ingest::spec(),
        longitudinal::sample_spec(),
        characterize::characterize_spec(),
        characterize::probe_spec(),
        create_config::spec(),
        refine_config::spec(),
        config::spec(),
        plan::draft_spec(),
        plan::edit_spec(),
        plan::validate_spec(),
        sweep::history_spec(),
        sweep::compare_spec(),
        sweep::sweep_spec(),
        sweep::get_job_spec(),
        sweep::wait_job_spec(),
        diagnostics::diagnose_spec(),
        diagnostics::curve_spec(),
        diagnostics::skeleton_spec(),
        artifact::spec(),
        clustering::spec(),
        longitudinal::build_spec(),
        longitudinal::diagnose_spec(),
        longitudinal::archetypes_spec(),
        longitudinal::neighbors_spec(),
        longitudinal::classify_spec(),
        cluster_evidence::profile_spec(),
        cluster_evidence::signal_spec(),
        cluster_evidence::matrix_spec(),
        cluster_evidence::compare_spec(),
        export::labeled_spec(),
        export::bundle_spec(),
        report::spec(),
        meta::guide_spec(),
        meta::runtime_spec(),
    ]
}

/// Deserialize tool arguments, reporting a structured `INVALID_ARGUMENT`
/// instead of a protocol error so the agent can read what was wrong.
pub fn parse_params<T: DeserializeOwned>(
    tool: &'static str,
    args: JsonObject,
) -> Result<T, ToolError> {
    serde_json::from_value(Value::Object(args)).map_err(|e| {
        ToolError::invalid_argument(
            tool,
            format!("arguments do not match the tool's input schema: {e}"),
            "Check the parameter names and types in the tool's input schema, then retry.",
        )
    })
}

/// Route a call to its tool.
pub async fn dispatch(
    state: Arc<ServerState>,
    name: &str,
    args: JsonObject,
    ctx: CallContext,
) -> Result<CallToolResult, ToolError> {
    match name {
        ingest::NAME => ingest::call(&state, parse_params(ingest::NAME, args)?).await,
        characterize::CHARACTERIZE => {
            characterize::characterize(&state, parse_params(characterize::CHARACTERIZE, args)?)
                .await
        }
        characterize::PROBE => {
            characterize::probe(&state, parse_params(characterize::PROBE, args)?).await
        }
        create_config::NAME => {
            create_config::call(&state, parse_params(create_config::NAME, args)?).await
        }
        refine_config::NAME => {
            refine_config::call(&state, parse_params(refine_config::NAME, args)?).await
        }
        config::NAME => config::call(&state, parse_params(config::NAME, args)?).await,
        plan::DRAFT => plan::draft(&state, parse_params(plan::DRAFT, args)?).await,
        plan::EDIT => plan::edit(&state, parse_params(plan::EDIT, args)?).await,
        plan::VALIDATE => plan::validate(&state, parse_params(plan::VALIDATE, args)?).await,
        sweep::HISTORY => sweep::history(&state, parse_params(sweep::HISTORY, args)?).await,
        sweep::COMPARE => sweep::compare(&state, parse_params(sweep::COMPARE, args)?).await,
        sweep::SWEEP => sweep::run(state.clone(), parse_params(sweep::SWEEP, args)?, ctx).await,
        sweep::GET_JOB => sweep::get_job(&state, parse_params(sweep::GET_JOB, args)?).await,
        sweep::WAIT_JOB => sweep::wait_job(&state, parse_params(sweep::WAIT_JOB, args)?).await,
        diagnostics::DIAGNOSE => {
            diagnostics::diagnose(&state, parse_params(diagnostics::DIAGNOSE, args)?).await
        }
        diagnostics::CURVE => {
            diagnostics::curve(&state, parse_params(diagnostics::CURVE, args)?).await
        }
        diagnostics::SKELETON => {
            diagnostics::skeleton(&state, parse_params(diagnostics::SKELETON, args)?).await
        }
        clustering::NAME => clustering::call(&state, parse_params(clustering::NAME, args)?).await,
        artifact::NAME => artifact::call(&state, parse_params(artifact::NAME, args)?).await,
        cluster_evidence::PROFILE => {
            cluster_evidence::profile(&state, parse_params(cluster_evidence::PROFILE, args)?).await
        }
        cluster_evidence::SIGNAL => {
            cluster_evidence::signal(&state, parse_params(cluster_evidence::SIGNAL, args)?).await
        }
        cluster_evidence::MATRIX => {
            cluster_evidence::matrix(&state, parse_params(cluster_evidence::MATRIX, args)?).await
        }
        cluster_evidence::COMPARE => {
            cluster_evidence::compare(&state, parse_params(cluster_evidence::COMPARE, args)?).await
        }
        longitudinal::BUILD => {
            longitudinal::build(&state, parse_params(longitudinal::BUILD, args)?).await
        }
        longitudinal::DIAGNOSE => {
            longitudinal::diagnose(&state, parse_params(longitudinal::DIAGNOSE, args)?).await
        }
        longitudinal::ARCHETYPES => {
            longitudinal::archetypes(&state, parse_params(longitudinal::ARCHETYPES, args)?).await
        }
        longitudinal::NEIGHBORS => {
            longitudinal::neighbors(&state, parse_params(longitudinal::NEIGHBORS, args)?).await
        }
        longitudinal::CLASSIFY => {
            longitudinal::classify(&state, parse_params(longitudinal::CLASSIFY, args)?).await
        }
        longitudinal::SAMPLE => {
            longitudinal::sample(&state, parse_params(longitudinal::SAMPLE, args)?).await
        }
        export::LABELED => export::labeled(&state, parse_params(export::LABELED, args)?).await,
        export::BUNDLE => export::bundle(&state, parse_params(export::BUNDLE, args)?).await,
        report::NAME => report::call(&state, parse_params(report::NAME, args)?).await,
        meta::GUIDE => meta::guide().await,
        meta::RUNTIME => meta::runtime(&state).await,
        other => Err(ToolError::new(
            "unknown",
            "TOOL_UNKNOWN",
            format!("no tool named {other:?}"),
            "Call tools/list and use one of the advertised tool names.",
        )),
    }
}

/// Shared parameter enums.
pub mod params {
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, JsonSchema)]
    #[serde(rename_all = "snake_case")]
    pub enum Format {
        #[default]
        Markdown,
        Json,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, JsonSchema)]
    #[serde(rename_all = "snake_case")]
    pub enum Detail {
        #[default]
        Summary,
        Full,
    }

    impl Detail {
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Summary => "summary",
                Self::Full => "full",
            }
        }
    }
}
