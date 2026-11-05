//! Importing Phirewall's data-meaning sidecar (`<csv>.decisions.json`,
//! contract `io.phirewall/data-decisions` v1) into the preprocessing plan.
//!
//! Settled decisions set the column's semantic treatment with
//! `decided_by: "phirewall"` and suppress its semantic questions through
//! [`ImportedDecisions::decides`]; open items become open questions marked as
//! needing the clinician; unknown meanings are ignored with a warning.

use std::path::{Path, PathBuf};

use pulsar_data::Table;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::defaults::DraftContext;
use super::model::{
    DecidedBy, EntryStatus, ImputeSpec, PendingQuestion, PlanQuestion, QuestionOption,
    QuestionTopic, Role,
};
use super::text::{plural, thousands};
use crate::evidence::{ColumnEvidence, ColumnKind};

pub const CONTRACT: &str = "io.phirewall/data-decisions";
pub const VERSION: u64 = 1;

const KNOWN_ROLES: &[&str] = &["time", "outcome", "group"];

const KNOWN_MEANINGS: &[&str] = &[
    // data-decisions-sidecar v1 meaningCode enum
    "code",
    "ordinal",
    "quantity",
    "count",
    "date",
    "flag",
    "identifier",
    "free_text",
    "not_recorded",
    "not_applicable",
    "unknown",
    "refused",
    "real_value",
    "bound",
    "shared_missingness",
    // role meanings and spec aliases
    "time",
    "outcome",
    "group",
    "blank_means_none",
    "lower_bound",
    "upper_bound",
    "amount",
];

const KNOWN_OPS: &[&str] = &[
    // data-decisions-sidecar v1 opCode enum
    "normalize_text",
    "decimal_separator",
    "value_map",
    "sentinel_to_blank",
    "text_to_number",
    "capped_to_bound",
    "date_format",
    "split",
    "code_normalize",
    "unit_convert",
    "range_flag",
    "drop_column",
    "combine_columns",
    // spec aliases
    "blank_to_value",
];

/// Sanitize a code token to single-line ASCII `[a-zA-Z0-9_-]`, capped at 40 chars.
pub fn sanitize_code(raw: &str) -> String {
    let mut out = String::new();
    for c in raw.chars() {
        if out.len() >= 40 {
            break;
        }
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        }
    }
    if out.is_empty() {
        "unknown".into()
    } else {
        out
    }
}

/// Sanitize free text from a sidecar into a single-line, control-character-free,
/// markdown-safe string capped at `max_chars` characters.
pub fn sanitize_sidecar_text(raw: &str, max_chars: usize) -> String {
    let mut cleaned = String::new();
    let mut prev_space = true;
    for c in raw.chars() {
        let mapped = if c.is_control() || matches!(c, '`' | '#' | '[' | ']' | '<' | '>' | '|') {
            ' '
        } else {
            c
        };
        if mapped.is_whitespace() {
            if !prev_space {
                cleaned.push(' ');
                prev_space = true;
            }
        } else {
            cleaned.push(mapped);
            prev_space = false;
        }
    }
    let trimmed = cleaned.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        trimmed
            .chars()
            .take(max_chars)
            .collect::<String>()
            .trim_end()
            .to_string()
    }
}

/// Compute Phirewall's 16-hex-character `dataset_version` for a CSV file and its
/// column names: the first 8 bytes of `SHA-256(hex(SHA-256(csv_bytes)) || (\0 || col)*)`.
pub fn compute_dataset_version(csv_bytes: &[u8], columns: &[String]) -> String {
    let file_hex = hex_sha256(csv_bytes);
    let mut h = Sha256::new();
    h.update(file_hex.as_bytes());
    for col in columns {
        h.update([0u8]);
        h.update(col.as_bytes());
    }
    h.finalize()
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn csv_header_columns(path: &Path) -> Vec<String> {
    pulsar_data::read_table(path)
        .map(|t| t.names)
        .unwrap_or_default()
}

fn version_matches_bytes(version: &str, csv_bytes: &[u8], columns: &[String]) -> bool {
    let expected = compute_dataset_version(csv_bytes, columns);
    if version == expected || (version.len() >= 8 && expected.starts_with(version)) {
        return true;
    }
    let file_hex = hex_sha256(csv_bytes);
    version.len() >= 8
        && version.chars().all(|c| c.is_ascii_hexdigit())
        && file_hex.starts_with(&version.to_ascii_lowercase())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedVersion {
    pub name: String,
    pub parent: String,
    pub recipe_ops: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedColumn {
    pub column: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meaning: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
    pub decided_by: String,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedOpen {
    pub column: String,
    pub question: String,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedWarning {
    pub code: String,
    pub target: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columns: Vec<String>,
}

impl ImportedWarning {
    pub fn invalid_sidecar(message: impl Into<String>) -> Self {
        Self {
            code: "PHIREWALL_SIDECAR_INVALID".into(),
            target: "sidecar".into(),
            message: message.into(),
            columns: Vec::new(),
        }
    }

    pub fn unknown_meaning(kind: &str, value: &str) -> Self {
        let safe_kind = sanitize_code(kind);
        let safe_code = sanitize_code(value);
        Self {
            code: "PHIREWALL_DECISION_UNKNOWN".into(),
            target: "sidecar".into(),
            message: format!("Ignored unknown Phirewall {safe_kind} code '{safe_code}'."),
            columns: Vec::new(),
        }
    }

    pub fn absent_columns(count: usize) -> Self {
        Self {
            code: "PHIREWALL_COLUMNS_ABSENT".into(),
            target: "sidecar".into(),
            message: format!(
                "Skipped {count} Phirewall decision(s) for column(s) absent from the dataset."
            ),
            columns: Vec::new(),
        }
    }

    pub fn to_plan_warning(&self) -> Value {
        json!({
            "code": self.code,
            "target": self.target,
            "message": self.message,
            "columns": self.columns,
        })
    }

    pub fn to_config_warning(&self) -> Value {
        json!({
            "code": self.code,
            "message": self.message,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedDecisions {
    pub dataset: String,
    pub dataset_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prepared: Option<PreparedVersion>,
    pub columns: Vec<ImportedColumn>,
    pub open: Vec<ImportedOpen>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<ImportedWarning>,
}

fn non_empty_str<'a>(
    obj: &'a serde_json::Map<String, Value>,
    key: &str,
    max_len: usize,
) -> Option<&'a str> {
    let s = obj.get(key)?.as_str()?;
    (!s.is_empty() && s.chars().count() <= max_len).then_some(s)
}

impl ImportedDecisions {
    /// Sidecar path for a dataset file (`<csv path>.decisions.json`).
    pub fn sidecar_path(data_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}.decisions.json", data_path.display()))
    }

    /// Load and validate `<data_path>.decisions.json` against the CSV file and `table`.
    pub fn load_for_table(data_path: &Path, table: &Table) -> (Option<Self>, Vec<ImportedWarning>) {
        Self::load_inner(data_path, Some(&table.names))
    }

    /// Load and validate `<data_path>.decisions.json` against the CSV file on disk.
    pub fn load_for_data_path(data_path: &Path) -> (Option<Self>, Vec<ImportedWarning>) {
        Self::load_inner(data_path, None)
    }

    fn load_inner(
        data_path: &Path,
        table_cols: Option<&[String]>,
    ) -> (Option<Self>, Vec<ImportedWarning>) {
        let path = Self::sidecar_path(data_path);
        let raw = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (None, Vec::new()),
            Err(_) => {
                return (
                    None,
                    vec![ImportedWarning::invalid_sidecar(
                        "Could not read Phirewall sidecar file.",
                    )],
                )
            }
        };
        let imp = match Self::parse_json(&raw) {
            Ok(imp) => imp,
            Err(w) => return (None, vec![w]),
        };
        if let Err(w) = imp.validate_against_csv(data_path, table_cols) {
            return (None, vec![w]);
        }
        (Some(imp), Vec::new())
    }

    /// Check `dataset` and `dataset_version` against the CSV file path and contents.
    pub fn validate_against_csv(
        &self,
        data_path: &Path,
        table_cols: Option<&[String]>,
    ) -> Result<(), ImportedWarning> {
        if let Some(p) = &self.prepared {
            if p.name != self.dataset {
                return Err(ImportedWarning::invalid_sidecar(
                    "Phirewall sidecar 'prepared.name' does not match 'dataset'.",
                ));
            }
        }

        let stem = data_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let parent_dir = data_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let matches_name = |name: &str| {
            stem == name
                || parent_dir == name
                || stem.strip_prefix(name).is_some_and(|rest| {
                    rest.starts_with('_') || rest.starts_with('-') || rest.starts_with('.')
                })
        };
        let dataset_matches = matches_name(&self.dataset)
            || self
                .prepared
                .as_ref()
                .is_some_and(|p| matches_name(&p.name) || matches_name(&p.parent));
        if !dataset_matches {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar 'dataset' does not match the CSV dataset.",
            ));
        }

        let Ok(csv_bytes) = std::fs::read(data_path) else {
            return Err(ImportedWarning::invalid_sidecar(
                "Could not read CSV file to verify Phirewall sidecar 'dataset_version'.",
            ));
        };
        let owned_cols;
        let cols: &[String] = match table_cols {
            Some(c) => c,
            None => {
                owned_cols = csv_header_columns(data_path);
                &owned_cols
            }
        };

        let mut version_ok = version_matches_bytes(&self.dataset_version, &csv_bytes, cols);
        if !version_ok {
            if let Some(p) = &self.prepared {
                let parent_csv = data_path.with_file_name(format!("{}.csv", p.parent));
                if let Ok(parent_bytes) = std::fs::read(&parent_csv) {
                    let parent_cols = csv_header_columns(&parent_csv);
                    version_ok =
                        version_matches_bytes(&self.dataset_version, &parent_bytes, &parent_cols);
                }
            }
        }
        if !version_ok {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar 'dataset_version' does not match the CSV file.",
            ));
        }
        Ok(())
    }

    /// Parse and validate a `io.phirewall/data-decisions` v1 sidecar JSON text.
    pub fn parse_json(raw: &str) -> Result<Self, ImportedWarning> {
        let val: Value = serde_json::from_str(raw)
            .map_err(|_| ImportedWarning::invalid_sidecar("Invalid JSON in Phirewall sidecar."))?;
        Self::from_sidecar_value(&val)
    }

    pub fn from_sidecar_value(val: &Value) -> Result<Self, ImportedWarning> {
        let Some(obj) = val.as_object() else {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar must be a JSON object.",
            ));
        };
        match obj.get("contract").and_then(Value::as_str) {
            Some(CONTRACT) => {}
            Some(other) => {
                let safe = sanitize_code(other);
                return Err(ImportedWarning::invalid_sidecar(format!(
                    "Unsupported Phirewall sidecar contract '{safe}' (expected '{CONTRACT}')."
                )));
            }
            None => {
                return Err(ImportedWarning::invalid_sidecar(
                    "Phirewall sidecar is missing required string 'contract'.",
                ))
            }
        }
        match obj.get("version").and_then(Value::as_u64) {
            Some(VERSION) if !obj["version"].is_boolean() => {}
            Some(other) => {
                return Err(ImportedWarning::invalid_sidecar(format!(
                    "Unsupported Phirewall sidecar version {other} (expected {VERSION})."
                )))
            }
            None => {
                return Err(ImportedWarning::invalid_sidecar(
                    "Phirewall sidecar is missing required integer 'version'.",
                ))
            }
        }
        let Some(dataset) = non_empty_str(obj, "dataset", 128) else {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar is missing valid 'dataset'.",
            ));
        };
        let Some(dataset_version) = non_empty_str(obj, "dataset_version", 128) else {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar is missing valid 'dataset_version'.",
            ));
        };
        if dataset.chars().any(char::is_control) || dataset_version.chars().any(char::is_control) {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar 'dataset' and 'dataset_version' must not contain control characters.",
            ));
        }

        let prepared = match obj.get("prepared") {
            None | Some(Value::Null) => None,
            Some(Value::Object(p)) => {
                let Some(name) = non_empty_str(p, "name", 128) else {
                    return Err(ImportedWarning::invalid_sidecar(
                        "Phirewall sidecar 'prepared.name' must be a non-empty string.",
                    ));
                };
                let Some(parent) = non_empty_str(p, "parent", 128) else {
                    return Err(ImportedWarning::invalid_sidecar(
                        "Phirewall sidecar 'prepared.parent' must be a non-empty string.",
                    ));
                };
                if name.chars().any(char::is_control) || parent.chars().any(char::is_control) {
                    return Err(ImportedWarning::invalid_sidecar(
                        "Phirewall sidecar 'prepared' names must not contain control characters.",
                    ));
                }
                let Some(recipe_ops) = p
                    .get("recipe_ops")
                    .and_then(Value::as_u64)
                    .filter(|_| !p["recipe_ops"].is_boolean())
                else {
                    return Err(ImportedWarning::invalid_sidecar(
                        "Phirewall sidecar 'prepared.recipe_ops' must be a non-negative integer.",
                    ));
                };
                Some(PreparedVersion {
                    name: name.to_string(),
                    parent: parent.to_string(),
                    recipe_ops,
                })
            }
            Some(_) => {
                return Err(ImportedWarning::invalid_sidecar(
                    "Phirewall sidecar 'prepared' must be an object.",
                ))
            }
        };

        let Some(raw_cols) = obj.get("columns").and_then(Value::as_array) else {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar is missing required array 'columns'.",
            ));
        };
        let Some(raw_open) = obj.get("open").and_then(Value::as_array) else {
            return Err(ImportedWarning::invalid_sidecar(
                "Phirewall sidecar is missing required array 'open'.",
            ));
        };

        let mut warnings = Vec::new();
        let mut columns = Vec::new();
        for item in raw_cols {
            let Some(c) = item.as_object() else {
                return Err(ImportedWarning::invalid_sidecar(
                    "Each entry in Phirewall sidecar 'columns' must be an object.",
                ));
            };
            let Some(col_name) = non_empty_str(c, "column", 128) else {
                return Err(ImportedWarning::invalid_sidecar(
                    "Column entry in Phirewall sidecar is missing valid 'column'.",
                ));
            };
            let Some(decided_by) = c
                .get("decided_by")
                .and_then(Value::as_str)
                .filter(|s| matches!(*s, "agent" | "clinician"))
            else {
                return Err(ImportedWarning::invalid_sidecar(
                    "Invalid 'decided_by' in Phirewall sidecar 'columns'.",
                ));
            };
            let Some(state) = c
                .get("state")
                .and_then(Value::as_str)
                .filter(|s| matches!(*s, "agent_decided" | "clinician_confirmed"))
            else {
                return Err(ImportedWarning::invalid_sidecar(
                    "Invalid 'state' in Phirewall sidecar 'columns'.",
                ));
            };

            let mut meaning = match c.get("meaning") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(_) => {
                    return Err(ImportedWarning::invalid_sidecar(
                        "Invalid 'meaning' type in Phirewall sidecar 'columns'.",
                    ))
                }
            };
            let mut role = match c.get("role") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(_) => {
                    return Err(ImportedWarning::invalid_sidecar(
                        "Invalid 'role' type in Phirewall sidecar 'columns'.",
                    ))
                }
            };
            let mut op = match c.get("op") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(_) => {
                    return Err(ImportedWarning::invalid_sidecar(
                        "Invalid 'op' type in Phirewall sidecar 'columns'.",
                    ))
                }
            };

            if let Some(m) = &meaning {
                if !KNOWN_MEANINGS.contains(&m.as_str()) {
                    warnings.push(ImportedWarning::unknown_meaning("meaning", m));
                    meaning = None;
                }
            }
            if let Some(r) = &role {
                if !KNOWN_ROLES.contains(&r.as_str()) {
                    warnings.push(ImportedWarning::unknown_meaning("role", r));
                    role = None;
                }
            }
            if let Some(o) = &op {
                if !KNOWN_OPS.contains(&o.as_str()) {
                    warnings.push(ImportedWarning::unknown_meaning("op", o));
                    op = None;
                }
            }

            if !is_importable_decision(meaning.as_deref(), role.as_deref(), op.as_deref()) {
                continue;
            }

            columns.push(ImportedColumn {
                column: col_name.to_string(),
                meaning,
                role,
                op,
                decided_by: decided_by.to_string(),
                state: state.to_string(),
            });
        }

        let mut open = Vec::new();
        for item in raw_open {
            let Some(o) = item.as_object() else {
                return Err(ImportedWarning::invalid_sidecar(
                    "Each entry in Phirewall sidecar 'open' must be an object.",
                ));
            };
            let Some(col_name) = non_empty_str(o, "column", 128) else {
                return Err(ImportedWarning::invalid_sidecar(
                    "Open entry in Phirewall sidecar is missing valid 'column'.",
                ));
            };
            let Some(raw_question) = non_empty_str(o, "question", 400) else {
                return Err(ImportedWarning::invalid_sidecar(
                    "Open entry in Phirewall sidecar is missing valid 'question'.",
                ));
            };
            if o.get("state").and_then(Value::as_str) != Some("open") {
                return Err(ImportedWarning::invalid_sidecar(
                    "Open entry in Phirewall sidecar must have state='open'.",
                ));
            }
            let question = sanitize_sidecar_text(raw_question, 160);
            if question.is_empty() {
                return Err(ImportedWarning::invalid_sidecar(
                    "Open entry 'question' in Phirewall sidecar is empty after sanitization.",
                ));
            }
            open.push(ImportedOpen {
                column: col_name.to_string(),
                question,
                state: "open".into(),
            });
        }

        Ok(Self {
            dataset: dataset.to_string(),
            dataset_version: dataset_version.to_string(),
            prepared,
            columns,
            open,
            warnings,
        })
    }

    /// The dataset version decisions belong to: the prepared version's name
    /// when prepared, else `dataset_version`.
    pub fn effective_version(&self) -> &str {
        self.prepared
            .as_ref()
            .map_or(self.dataset_version.as_str(), |p| p.name.as_str())
    }

    /// One-line fact summary for the draft plan payload, counting only applied decisions.
    pub fn fact_line(&self) -> String {
        let ver = sanitize_sidecar_text(self.effective_version(), 64);
        format!(
            "imported {} Phirewall decisions (version {ver})",
            self.columns.len()
        )
    }

    /// Look up the settled decision for `column` (last entry wins if repeated).
    pub fn column(&self, column: &str) -> Option<&ImportedColumn> {
        self.columns.iter().rev().find(|c| c.column == column)
    }

    /// Build the `io.phirewall/summary` v1 `$defs/decision` object for `column`.
    pub fn summary_decision(&self, column: &str) -> Option<Value> {
        let c = self.column(column)?;
        let mut out = json!({
            "source": "phirewall",
            "decided_by": c.decided_by,
            "version": self.effective_version(),
        });
        if let Some(meaning) = c
            .meaning
            .as_deref()
            .filter(|m| *m != "free_text")
            .or(c.role.as_deref())
        {
            out["meaning"] = json!(meaning);
        }
        if let Some(op) =
            c.op.as_deref()
                .filter(|o| !matches!(*o, "split" | "combine_columns"))
        {
            out["op"] = json!(op);
        }
        Some(out)
    }

    /// Whether Phirewall has already settled (or holds an open clinician
    /// question for) the semantic question `topic` on `column`.
    pub fn decides(&self, column: &str, topic: QuestionTopic) -> bool {
        if topic == QuestionTopic::OutcomeColumns {
            return self.columns.iter().any(|c| {
                c.role.as_deref() == Some("outcome") || c.meaning.as_deref() == Some("outcome")
            });
        }
        if self.open.iter().any(|o| o.column == column) {
            return matches!(
                topic,
                QuestionTopic::BlankMeaning
                    | QuestionTopic::CodeColumns
                    | QuestionTopic::ConditionalColumn
                    | QuestionTopic::MixedValues
            );
        }
        let Some(d) = self.column(column) else {
            return false;
        };
        let meaning = d.meaning.as_deref();
        let role = d.role.as_deref();
        let op = d.op.as_deref();

        // A column given a non-feature role or dropped via drop_column needs no feature questions.
        if matches!(role, Some("outcome" | "time" | "group"))
            || matches!(meaning, Some("outcome" | "time" | "group" | "identifier"))
            || op == Some("drop_column")
        {
            return true;
        }

        match topic {
            QuestionTopic::BlankMeaning => {
                matches!(
                    meaning,
                    Some(
                        "blank_means_none"
                            | "not_recorded"
                            | "not_applicable"
                            | "unknown"
                            | "refused"
                            | "shared_missingness"
                    )
                ) || matches!(op, Some("blank_to_value" | "sentinel_to_blank"))
            }
            QuestionTopic::ConditionalColumn => {
                // `not_applicable` settles only the meaning of a conditional column;
                // Pulsar still asks the method choice (leave_out / blank_as_zero / estimate_and_mark).
                matches!(
                    meaning,
                    Some("blank_means_none" | "not_recorded" | "unknown" | "refused")
                ) || matches!(op, Some("blank_to_value" | "sentinel_to_blank"))
            }
            QuestionTopic::MixedValues => {
                matches!(
                    meaning,
                    Some(
                        "not_recorded"
                            | "not_applicable"
                            | "unknown"
                            | "refused"
                            | "real_value"
                            | "bound"
                            | "lower_bound"
                            | "upper_bound"
                            | "code"
                            | "ordinal"
                            | "quantity"
                            | "count"
                            | "amount"
                            | "date"
                    )
                ) || matches!(
                    op,
                    Some(
                        "sentinel_to_blank"
                            | "text_to_number"
                            | "capped_to_bound"
                            | "value_map"
                            | "decimal_separator"
                            | "normalize_text"
                            | "code_normalize"
                            | "unit_convert"
                            | "date_format"
                    )
                )
            }
            QuestionTopic::CodeColumns => {
                matches!(
                    meaning,
                    Some(
                        "code"
                            | "ordinal"
                            | "quantity"
                            | "count"
                            | "amount"
                            | "flag"
                            | "bound"
                            | "lower_bound"
                            | "upper_bound"
                            | "real_value"
                            | "blank_means_none"
                            | "date"
                    )
                ) || matches!(
                    op,
                    Some(
                        "code_normalize"
                            | "value_map"
                            | "unit_convert"
                            | "capped_to_bound"
                            | "blank_to_value"
                            | "date_format"
                    )
                )
            }
            QuestionTopic::MostlyBlank => {
                matches!(meaning, Some("blank_means_none")) || matches!(op, Some("blank_to_value"))
            }
            QuestionTopic::DateColumns
            | QuestionTopic::DuplicateSignal
            | QuestionTopic::OutcomeColumns => false,
        }
    }

    /// Apply settled Phirewall decisions to `ctx.entries` and role sets,
    /// filtering `self.columns` to applied decisions on existing columns and
    /// `self.open` to existing columns.
    pub fn apply_to_context(&mut self, ctx: &mut DraftContext) {
        let is_prepared = self.prepared.is_some();
        let mut applied = Vec::new();
        let mut absent_count = 0usize;

        for d in self.columns.clone() {
            if !ctx.ev_by_col.contains_key(d.column.as_str()) {
                absent_count += 1;
                continue;
            }
            if apply_single_column(&d, is_prepared, ctx) {
                applied.push(d);
            }
        }
        self.columns = applied;
        self.open
            .retain(|o| ctx.ev_by_col.contains_key(o.column.as_str()));

        if absent_count > 0 {
            self.warnings
                .push(ImportedWarning::absent_columns(absent_count));
        }
    }

    /// Convert sidecar `open` items into open questions marked as needing the clinician.
    pub fn open_questions(&self, ctx: &mut DraftContext) -> Vec<PendingQuestion> {
        let mut out = Vec::new();
        for item in &self.open {
            let Some(&ev) = ctx.ev_by_col.get(item.column.as_str()) else {
                continue;
            };
            let rows = if ev.n_missing > 0 {
                ev.n_missing
            } else {
                ev.n_rows
            };
            if let Some(entry) = ctx.entries.get_mut(&item.column) {
                entry.status = EntryStatus::Open;
            }
            let safe_col = sanitize_sidecar_text(&item.column, 80);
            let safe_q = sanitize_sidecar_text(&item.question, 160);
            let question = PlanQuestion::new(
                QuestionTopic::CodeColumns,
                format!("Needs clinician confirmation (`{safe_col}`): {safe_q}"),
                vec![item.column.clone()],
                rows,
                vec![QuestionOption::new(
                    "keep_default",
                    "Keep current default until clinician confirms",
                    "Keeps the current default treatment until the clinician settles this question",
                    vec![],
                )],
                "keep_default",
            );
            out.push((question, vec![item.column.clone()]));
        }
        out
    }
}

/// Whether a `(meaning, role, op)` combination maps to an explicit Pulsar plan treatment.
/// `free_text`, `split`, and `combine_columns` on their own are not imported.
fn is_importable_decision(meaning: Option<&str>, role: Option<&str>, op: Option<&str>) -> bool {
    let actionable_role = matches!(role, Some("outcome" | "time" | "group"));
    let actionable_meaning = matches!(
        meaning,
        Some(
            "code"
                | "ordinal"
                | "quantity"
                | "count"
                | "date"
                | "flag"
                | "identifier"
                | "not_recorded"
                | "not_applicable"
                | "unknown"
                | "refused"
                | "real_value"
                | "bound"
                | "shared_missingness"
                | "time"
                | "outcome"
                | "group"
                | "blank_means_none"
                | "lower_bound"
                | "upper_bound"
                | "amount"
        )
    );
    let actionable_op = matches!(
        op,
        Some(
            "normalize_text"
                | "decimal_separator"
                | "value_map"
                | "sentinel_to_blank"
                | "text_to_number"
                | "capped_to_bound"
                | "date_format"
                | "code_normalize"
                | "unit_convert"
                | "range_flag"
                | "drop_column"
                | "blank_to_value"
        )
    );
    actionable_role || actionable_meaning || actionable_op
}

fn remove_from_lists(ctx: &mut DraftContext, col: &str, keep_date: bool, keep_conditional: bool) {
    if !keep_date {
        ctx.date_cols.retain(|c| c != col);
    }
    ctx.mixed_value_cols.retain(|c| c != col);
    ctx.rule7b_cols.retain(|c| c != col);
    if !keep_conditional {
        for cols in ctx.conditional_by_partner.values_mut() {
            cols.retain(|c| c != col);
        }
        ctx.conditional_by_partner
            .retain(|_, cols| !cols.is_empty());
    }
}

fn apply_single_column(d: &ImportedColumn, is_prepared: bool, ctx: &mut DraftContext) -> bool {
    let col = d.column.as_str();
    let Some(&ev) = ctx.ev_by_col.get(col) else {
        return false;
    };
    let meaning = d.meaning.as_deref();
    let role = d.role.as_deref();
    let op = d.op.as_deref();

    if role == Some("outcome") || meaning == Some("outcome") {
        ctx.target_set.insert(d.column.clone());
        remove_from_lists(ctx, col, false, false);
        if let Some(entry) = ctx.entries.get_mut(col) {
            set_non_feature(
                entry,
                Role::Target,
                "designated as target column (Phirewall)",
            );
        }
        return true;
    }
    if role == Some("time") || meaning == Some("time") {
        if ctx.time_col.is_empty() {
            ctx.time_col = d.column.clone();
        }
        remove_from_lists(ctx, col, false, false);
        if let Some(entry) = ctx.entries.get_mut(col) {
            set_non_feature(entry, Role::Time, "designated as time column (Phirewall)");
        }
        return true;
    }
    if role == Some("group") || meaning == Some("group") {
        if ctx.entity_col.is_empty() {
            ctx.entity_col = d.column.clone();
        }
        remove_from_lists(ctx, col, false, false);
        if let Some(entry) = ctx.entries.get_mut(col) {
            set_non_feature(
                entry,
                Role::Entity,
                "designated as group column (Phirewall)",
            );
        }
        return true;
    }
    if meaning == Some("identifier") {
        remove_from_lists(ctx, col, false, false);
        if let Some(entry) = ctx.entries.get_mut(col) {
            set_non_feature(entry, Role::Id, "identifier column (Phirewall)");
        }
        return true;
    }
    if op == Some("drop_column") {
        remove_from_lists(ctx, col, false, false);
        if let Some(entry) = ctx.entries.get_mut(col) {
            set_non_feature(entry, Role::Drop, "dropped by Phirewall decision");
        }
        return true;
    }

    if meaning == Some("date") || op == Some("date_format") {
        remove_from_lists(ctx, col, true, false);
        if !ctx.date_cols.iter().any(|c| c == col) {
            ctx.date_cols.push(d.column.clone());
        }
        if let Some(entry) = ctx.entries.get_mut(col) {
            entry.role = Role::Drop;
            entry.impute = None;
            entry.impute_candidates = None;
            entry.indicator = "none".into();
            entry.encode = None;
            entry.decided_by = DecidedBy::Phirewall;
            entry.status = EntryStatus::Open;
            entry.reason = format!(
                "date/time text ({} distinct values; {} {})",
                thousands(ev.n_unique),
                thousands(ev.n_missing),
                plural(ev.n_missing, "blank", "blanks")
            );
        }
        return true;
    }

    if meaning == Some("blank_means_none") || op == Some("blank_to_value") {
        remove_from_lists(ctx, col, false, false);
        if let Some(entry) = ctx.entries.get_mut(col) {
            entry.role = Role::Feature;
            entry.encode = None;
            if is_prepared && op == Some("blank_to_value") {
                // Prepared version already applied `blank_to_value` to the twin; do not re-fill.
                if ev.n_missing == 0 {
                    entry.impute = None;
                    entry.indicator = "none".into();
                }
                entry.reason =
                    "blank_to_value already applied in prepared version (Phirewall)".into();
            } else {
                entry.impute = (ev.n_missing > 0).then(|| ImputeSpec::fill_value(0));
                entry.indicator = "none".into();
                entry.reason = "blanks mean none (Phirewall); filled with 0".into();
            }
            entry.decided_by = DecidedBy::Phirewall;
            entry.status = EntryStatus::Resolved;
        }
        return true;
    }

    if meaning == Some("not_applicable") {
        let is_conditional = ctx
            .conditional_by_partner
            .values()
            .any(|cols| cols.iter().any(|c| c == col));
        if is_conditional {
            // Meaning is imported, but the method choice (leave_out / 0 / estimate) stays open.
            if let Some(entry) = ctx.entries.get_mut(col) {
                entry.decided_by = DecidedBy::Phirewall;
                entry.status = EntryStatus::Open;
                entry.reason = format!("{}; blanks are not applicable (Phirewall)", entry.reason);
            }
            return true;
        }
        ctx.mixed_value_cols.retain(|c| c != col);
        if let Some(entry) = ctx.entries.get_mut(col) {
            entry.role = Role::Feature;
            if ev.kind == ColumnKind::Text {
                entry.encode = Some("one_hot".into());
                entry.impute =
                    (ev.n_missing > 0).then(|| ImputeSpec::fill_value("(not_applicable)"));
                entry.indicator = "none".into();
            } else {
                let has_missing = ev.n_missing > 0 || ev.n_non_numeric.unwrap_or(0) > 0;
                entry.encode = None;
                entry.impute = has_missing.then(|| ImputeSpec::method("fill_median"));
                entry.indicator = if has_missing {
                    blank_indicator(ev)
                } else {
                    "none".into()
                };
            }
            entry.decided_by = DecidedBy::Phirewall;
            entry.status = EntryStatus::Resolved;
            entry.reason = "blanks are not applicable (Phirewall)".into();
        }
        return true;
    }

    if matches!(
        meaning,
        Some("unknown" | "not_recorded" | "refused" | "shared_missingness")
    ) || op == Some("sentinel_to_blank")
    {
        ctx.mixed_value_cols.retain(|c| c != col);
        for cols in ctx.conditional_by_partner.values_mut() {
            cols.retain(|c| c != col);
        }
        ctx.conditional_by_partner
            .retain(|_, cols| !cols.is_empty());
        if let Some(entry) = ctx.entries.get_mut(col) {
            entry.role = Role::Feature;
            if ev.kind == ColumnKind::Text {
                entry.impute = (ev.n_missing > 0).then(|| ImputeSpec::fill_value("(blank)"));
            } else {
                let has_missing = ev.n_missing > 0 || ev.n_non_numeric.unwrap_or(0) > 0;
                entry.encode = None;
                entry.impute = has_missing.then(|| ImputeSpec::method("fill_median"));
                entry.indicator = if has_missing {
                    blank_indicator(ev)
                } else {
                    "none".into()
                };
                if has_missing && !ctx.rule7b_cols.iter().any(|c| c == col) {
                    ctx.rule7b_cols.push(d.column.clone());
                }
            }
            entry.decided_by = DecidedBy::Phirewall;
            entry.status = EntryStatus::Resolved;
            entry.reason = "missingness meaning settled by Phirewall".into();
        }
        return true;
    }

    if matches!(meaning, Some("bound" | "lower_bound" | "upper_bound"))
        || op == Some("capped_to_bound")
    {
        ctx.mixed_value_cols.retain(|c| c != col);
        let has_missing = ev.n_missing > 0 || ev.n_non_numeric.unwrap_or(0) > 0;
        if let Some(entry) = ctx.entries.get_mut(col) {
            entry.role = Role::Feature;
            entry.encode = None;
            entry.impute = has_missing.then(|| ImputeSpec::method("fill_median"));
            entry.indicator = if has_missing {
                blank_indicator(ev)
            } else {
                "none".into()
            };
            entry.decided_by = DecidedBy::Phirewall;
            entry.status = EntryStatus::Resolved;
            entry.reason = "bounded numeric column (Phirewall)".into();
        }
        return true;
    }

    if matches!(meaning, Some("code" | "ordinal"))
        || matches!(op, Some("code_normalize" | "value_map"))
    {
        remove_from_lists(ctx, col, false, false);
        if let Some(entry) = ctx.entries.get_mut(col) {
            entry.role = Role::Feature;
            entry.encode = Some("one_hot".into());
            if ev.n_missing > 0 {
                if matches!(
                    ev.kind,
                    ColumnKind::Text | ColumnKind::NumericText | ColumnKind::DatetimeLike
                ) {
                    entry.impute = Some(ImputeSpec::fill_value("(blank)"));
                    entry.indicator = "none".into();
                } else {
                    entry.impute = Some(ImputeSpec::method("fill_median"));
                    entry.indicator = blank_indicator(ev);
                }
            } else {
                entry.impute = None;
                entry.indicator = "none".into();
            }
            entry.decided_by = DecidedBy::Phirewall;
            entry.status = EntryStatus::Resolved;
            entry.reason = "categorical code column (Phirewall)".into();
        }
        return true;
    }

    if matches!(
        meaning,
        Some("quantity" | "count" | "amount" | "real_value" | "flag")
    ) || matches!(
        op,
        Some(
            "text_to_number"
                | "decimal_separator"
                | "unit_convert"
                | "range_flag"
                | "normalize_text"
        )
    ) {
        ctx.mixed_value_cols.retain(|c| c != col);
        let has_missing = ev.n_missing > 0 || ev.n_non_numeric.unwrap_or(0) > 0;
        let mixed_text = ev.kind == ColumnKind::NumericText && ev.n_non_numeric.unwrap_or(0) > 0;
        if let Some(entry) = ctx.entries.get_mut(col) {
            entry.role = Role::Feature;
            entry.encode = None;
            if (has_missing && entry.impute.is_none()) || mixed_text {
                entry.impute = Some(ImputeSpec::method("fill_median"));
                entry.indicator = blank_indicator(ev);
                if !ctx.rule7b_cols.iter().any(|c| c == col) {
                    ctx.rule7b_cols.push(d.column.clone());
                }
            }
            entry.decided_by = DecidedBy::Phirewall;
            entry.status = EntryStatus::Resolved;
            entry.reason = "numeric/flag meaning settled by Phirewall".into();
        }
        return true;
    }

    false
}

pub trait Decides {
    fn decides(&self, column: &str, topic: QuestionTopic) -> bool;
}

impl Decides for Option<ImportedDecisions> {
    fn decides(&self, column: &str, topic: QuestionTopic) -> bool {
        self.as_ref().is_some_and(|imp| imp.decides(column, topic))
    }
}

fn blank_indicator(ev: &ColumnEvidence) -> String {
    match &ev.missing_group {
        Some(g) => format!("group:{g}"),
        None => "own".into(),
    }
}

fn set_non_feature(entry: &mut super::model::PlanEntry, role: Role, reason: &str) {
    entry.role = role;
    entry.impute = None;
    entry.impute_candidates = None;
    entry.indicator = "none".into();
    entry.encode = None;
    entry.decided_by = DecidedBy::Phirewall;
    entry.status = EntryStatus::Resolved;
    entry.reason = reason.into();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_sidecar_and_maps_decisions() {
        let raw = r#"{
            "contract": "io.phirewall/data-decisions",
            "version": 1,
            "dataset": "patients",
            "dataset_version": "a1b2c3d4e5f6",
            "prepared": {
                "name": "patients",
                "parent": "patients_raw",
                "recipe_ops": 3
            },
            "columns": [
                {
                    "column": "n_visits",
                    "meaning": "blank_means_none",
                    "op": "blank_to_value",
                    "decided_by": "clinician",
                    "state": "clinician_confirmed"
                },
                {
                    "column": "crp",
                    "meaning": "lower_bound",
                    "op": "capped_to_bound",
                    "decided_by": "agent",
                    "state": "agent_decided"
                },
                {
                    "column": "died",
                    "role": "outcome",
                    "decided_by": "agent",
                    "state": "agent_decided"
                }
            ],
            "open": [
                {
                    "column": "ward_code",
                    "question": "Is ward_code a code or a count?",
                    "state": "open"
                }
            ]
        }"#;
        let imp = ImportedDecisions::parse_json(raw).expect("valid sidecar");
        assert_eq!(imp.effective_version(), "patients");
        assert_eq!(
            imp.fact_line(),
            "imported 3 Phirewall decisions (version patients)"
        );
        assert!(imp.warnings.is_empty());
        assert!(imp.decides("n_visits", QuestionTopic::BlankMeaning));
        assert!(imp.decides("crp", QuestionTopic::MixedValues));
        assert!(imp.decides("", QuestionTopic::OutcomeColumns));
        assert!(imp.decides("ward_code", QuestionTopic::CodeColumns));
        assert!(!imp.decides("other_col", QuestionTopic::BlankMeaning));
    }

    #[test]
    fn unknown_meaning_is_ignored_with_sanitized_warning() {
        let raw = r#"{
            "contract": "io.phirewall/data-decisions",
            "version": 1,
            "dataset": "patients",
            "dataset_version": "v1",
            "columns": [
                {
                    "column": "mystery",
                    "meaning": "future_code\n## injected heading",
                    "decided_by": "agent",
                    "state": "agent_decided"
                }
            ],
            "open": []
        }"#;
        let imp = ImportedDecisions::parse_json(raw).expect("sidecar parses");
        assert!(imp.columns.is_empty());
        assert_eq!(imp.warnings.len(), 1);
        assert_eq!(imp.warnings[0].code, "PHIREWALL_DECISION_UNKNOWN");
        assert!(!imp.warnings[0].message.contains('\n'));
        assert!(!imp.warnings[0].message.contains("##"));
    }

    #[test]
    fn non_importable_codes_like_free_text_split_combine_are_not_imported() {
        let raw = r#"{
            "contract": "io.phirewall/data-decisions",
            "version": 1,
            "dataset": "patients",
            "dataset_version": "v1",
            "columns": [
                {
                    "column": "notes",
                    "meaning": "free_text",
                    "decided_by": "agent",
                    "state": "agent_decided"
                },
                {
                    "column": "bp",
                    "op": "split",
                    "decided_by": "agent",
                    "state": "agent_decided"
                },
                {
                    "column": "total",
                    "op": "combine_columns",
                    "decided_by": "agent",
                    "state": "agent_decided"
                }
            ],
            "open": []
        }"#;
        let imp = ImportedDecisions::parse_json(raw).expect("sidecar parses");
        assert!(imp.columns.is_empty());
        assert!(imp.warnings.is_empty());
    }

    #[test]
    fn count_does_not_suppress_blank_meaning_and_not_applicable_keeps_conditional_open() {
        let raw = r#"{
            "contract": "io.phirewall/data-decisions",
            "version": 1,
            "dataset": "patients",
            "dataset_version": "v1",
            "columns": [
                {
                    "column": "visits",
                    "meaning": "count",
                    "decided_by": "agent",
                    "state": "agent_decided"
                },
                {
                    "column": "dose",
                    "meaning": "not_applicable",
                    "decided_by": "agent",
                    "state": "agent_decided"
                }
            ],
            "open": []
        }"#;
        let imp = ImportedDecisions::parse_json(raw).expect("sidecar parses");
        assert!(!imp.decides("visits", QuestionTopic::BlankMeaning));
        assert!(imp.decides("visits", QuestionTopic::CodeColumns));
        assert!(!imp.decides("dose", QuestionTopic::ConditionalColumn));
        assert!(imp.decides("dose", QuestionTopic::BlankMeaning));
    }

    #[test]
    fn open_question_text_is_sanitized_to_single_line() {
        let raw = r#"{
            "contract": "io.phirewall/data-decisions",
            "version": 1,
            "dataset": "patients",
            "dataset_version": "v1",
            "columns": [],
            "open": [
                {
                    "column": "ward",
                    "question": "line 1\n## Heading `code` [link]<script>",
                    "state": "open"
                }
            ]
        }"#;
        let imp = ImportedDecisions::parse_json(raw).expect("sidecar parses");
        let q = &imp.open[0].question;
        assert!(!q.contains('\n'));
        assert!(!q.contains('#'));
        assert!(!q.contains('`'));
        assert!(!q.contains('<'));
    }

    #[test]
    fn invalid_contract_or_version_returns_warning() {
        let bad = r#"{"contract": "wrong", "version": 99}"#;
        let err = ImportedDecisions::parse_json(bad).unwrap_err();
        assert_eq!(err.code, "PHIREWALL_SIDECAR_INVALID");
    }
}
