//! Plan data model (`pulsar/mcp/preprocessing_plan/model.py`): column roles,
//! per-column entries and the structured edit error.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Feature,
    Drop,
    Id,
    Entity,
    Time,
    Target,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Feature => "feature",
            Self::Drop => "drop",
            Self::Id => "id",
            Self::Entity => "entity",
            Self::Time => "time",
            Self::Target => "target",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecidedBy {
    Default,
    Agent,
    User,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryStatus {
    Resolved,
    Open,
}

/// How blanks in a column are filled: a `method` and, for `fill_value`, the
/// value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImputeSpec {
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    /// Only set on multi-imputation candidates with a stochastic method.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Per-candidate override of the entry's indicator mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indicator: Option<bool>,
}

impl ImputeSpec {
    pub fn method(method: &str) -> Self {
        Self {
            method: method.into(),
            value: None,
            seed: None,
            indicator: None,
        }
    }

    pub fn fill_value(value: impl Into<Value>) -> Self {
        Self {
            method: "fill_value".into(),
            value: Some(value.into()),
            seed: None,
            indicator: None,
        }
    }
}

/// What the plan does with one column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanEntry {
    pub column: String,
    pub role: Role,
    pub impute: Option<ImputeSpec>,
    /// `"none"`, `"own"` or `"group:<gid>"`.
    pub indicator: String,
    /// `"one_hot"` or none.
    pub encode: Option<String>,
    pub decided_by: DecidedBy,
    pub status: EntryStatus,
    pub reason: String,
    pub question_id: Option<String>,
    /// Alternatives when the column carries a multi-imputation ensemble
    /// (serialized only when there is more than one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impute_candidates: Option<Vec<ImputeSpec>>,
}

impl PlanEntry {
    /// An entry with no imputation, indicator or encoding, decided by default.
    pub fn new(column: &str, role: Role, status: EntryStatus, reason: String) -> Self {
        Self {
            column: column.into(),
            role,
            impute: None,
            indicator: "none".into(),
            encode: None,
            decided_by: DecidedBy::Default,
            status,
            reason,
            question_id: None,
            impute_candidates: None,
        }
    }

    pub fn impute_method(&self) -> Option<&str> {
        self.impute.as_ref().map(|i| i.method.as_str())
    }
}

/// One requested change to a column, as a loose map (`column`, `role`,
/// `impute_method`, `impute_value`, `indicator`, `encode`). Loose on purpose:
/// a key present with `null` (clear it) differs from an absent key (leave it),
/// and the typed edit layer interprets both.
pub type Edit = Map<String, Value>;

/// Build an [`Edit`] from key/value pairs.
pub fn edit<const N: usize>(pairs: [(&str, Value); N]) -> Edit {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionTopic {
    BlankMeaning,
    CodeColumns,
    ConditionalColumn,
    DateColumns,
    DuplicateSignal,
    MixedValues,
    MostlyBlank,
    OutcomeColumns,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionStatus {
    Open,
    Answered,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuestionOption {
    pub option_id: String,
    pub label: String,
    pub effect: String,
    pub edits: Vec<Edit>,
}

impl QuestionOption {
    pub fn new(id: &str, label: &str, effect: impl Into<String>, edits: Vec<Edit>) -> Self {
        Self {
            option_id: id.into(),
            label: label.into(),
            effect: effect.into(),
            edits,
        }
    }
}

/// A decision the data cannot make: what a blank means, whether a code is a
/// label, and so on. Each option carries the edits that apply it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanQuestion {
    pub question_id: String,
    pub topic: QuestionTopic,
    pub text: String,
    pub applies_to: Vec<String>,
    pub rows_affected: usize,
    pub options: Vec<QuestionOption>,
    pub default_option_id: Option<String>,
    pub stakes: Option<Value>,
    pub status: QuestionStatus,
    pub answer: Option<Value>,
}

impl PlanQuestion {
    pub fn new(
        topic: QuestionTopic,
        text: String,
        applies_to: Vec<String>,
        rows_affected: usize,
        options: Vec<QuestionOption>,
        default_option_id: &str,
    ) -> Self {
        Self {
            question_id: String::new(),
            topic,
            text,
            applies_to,
            rows_affected,
            options,
            default_option_id: Some(default_option_id.into()),
            stakes: None,
            status: QuestionStatus::Open,
            answer: None,
        }
    }
}

/// A question before it has an id, with the columns whose neighbourhood
/// overlap the stakes step should measure.
pub type PendingQuestion = (PlanQuestion, Vec<String>);

/// Invalid plan parameters, answers or edits, as a code the error envelope
/// can carry plus details naming what was wrong.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct PlanEditError {
    pub code: String,
    pub message: String,
    pub details: Map<String, Value>,
}

impl PlanEditError {
    pub fn new(code: &str, message: impl Into<String>, details: Map<String, Value>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details,
        }
    }
}
