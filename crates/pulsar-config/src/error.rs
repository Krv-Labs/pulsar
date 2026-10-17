//! Structured validation errors.
//!
//! Every problem in a config is reported as a [`ConfigError`] with a JSON
//! pointer into the document, a stable machine-readable [`ErrorCode`], a
//! human message, and (where it helps) the allowed values. Loading collects
//! every semantic error it can find instead of stopping at the first one, so
//! an agent can fix a config in one pass.

use std::fmt;

use schemars::JsonSchema;
use serde::Serialize;

/// Stable error codes. Serialized as `SCREAMING_SNAKE_CASE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    /// The document is not valid YAML.
    YamlSyntax,
    /// A key that the schema does not define.
    UnknownKey,
    /// A key that older Pulsar versions accepted and v1 does not.
    LegacyKey,
    /// A required key is absent.
    MissingField,
    /// A value has the wrong type (string where a number is required, ...).
    InvalidType,
    /// A value has the right type but is not acceptable.
    InvalidValue,
    /// A number is outside its allowed range.
    OutOfRange,
    /// A sweep parameter expands to no values.
    EmptyGrid,
    /// `version` names a config version this build does not understand.
    UnsupportedVersion,
    /// A column is listed in more than one place where it must be unique.
    DuplicateColumn,
    /// A generated column name collides with another column.
    NameCollision,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::YamlSyntax => "YAML_SYNTAX",
            Self::UnknownKey => "UNKNOWN_KEY",
            Self::LegacyKey => "LEGACY_KEY",
            Self::MissingField => "MISSING_FIELD",
            Self::InvalidType => "INVALID_TYPE",
            Self::InvalidValue => "INVALID_VALUE",
            Self::OutOfRange => "OUT_OF_RANGE",
            Self::EmptyGrid => "EMPTY_GRID",
            Self::UnsupportedVersion => "UNSUPPORTED_VERSION",
            Self::DuplicateColumn => "DUPLICATE_COLUMN",
            Self::NameCollision => "NAME_COLLISION",
        }
    }
}

/// One validation problem.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ConfigError {
    /// JSON pointer (RFC 6901) to the offending node; `""` is the whole document.
    pub path: String,
    pub code: ErrorCode,
    pub message: String,
    /// Accepted values or keys, when the set is small and known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed: Option<Vec<String>>,
}

impl ConfigError {
    pub fn new(path: impl Into<String>, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            code,
            message: message.into(),
            allowed: None,
        }
    }

    pub fn with_allowed<I, S>(mut self, allowed: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed = Some(allowed.into_iter().map(Into::into).collect());
        self
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = if self.path.is_empty() {
            "/"
        } else {
            &self.path
        };
        write!(f, "{} at {}: {}", self.code.as_str(), path, self.message)?;
        if let Some(allowed) = &self.allowed {
            write!(f, " (allowed: {})", allowed.join(", "))?;
        }
        Ok(())
    }
}

/// All problems found while loading one config.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ConfigErrors(pub Vec<ConfigError>);

impl ConfigErrors {
    pub fn single(err: ConfigError) -> Self {
        Self(vec![err])
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ConfigError> {
        self.0.iter()
    }
}

impl fmt::Display for ConfigErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, err) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{err}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ConfigErrors {}

/// Builds RFC 6901 JSON pointers.
#[derive(Debug, Clone, Default)]
pub(crate) struct Pointer(String);

impl Pointer {
    pub(crate) fn root() -> Self {
        Self(String::new())
    }

    pub(crate) fn key(&self, key: &str) -> Self {
        let escaped = key.replace('~', "~0").replace('/', "~1");
        Self(format!("{}/{}", self.0, escaped))
    }

    pub(crate) fn index(&self, i: usize) -> Self {
        Self(format!("{}/{}", self.0, i))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<Pointer> for String {
    fn from(p: Pointer) -> Self {
        p.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_escapes_per_rfc6901() {
        let p = Pointer::root().key("preprocessing").key("a/b~c").index(2);
        assert_eq!(p.as_str(), "/preprocessing/a~1b~0c/2");
    }

    #[test]
    fn display_includes_code_path_and_allowed() {
        let e = ConfigError::new("/x", ErrorCode::InvalidValue, "bad").with_allowed(["a", "b"]);
        assert_eq!(e.to_string(), "INVALID_VALUE at /x: bad (allowed: a, b)");
    }
}
