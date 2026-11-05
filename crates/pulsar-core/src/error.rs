use thiserror::Error;

/// All errors that can originate from the Pulsar numeric core. Each variant
/// has a stable [`code`](PulsarError::code) for the agent-facing envelopes.
#[derive(Debug, Error)]
pub enum PulsarError {
    /// Raised when an array passed to a function has the wrong number of rows
    /// or columns.  The `expected` and `got` fields are human-readable strings
    /// (e.g. `"3 columns"`) so the error message is self-explanatory.
    ///
    /// Produced by: `scale.rs` (`transform`, `inverse_transform`),
    ///              `pca.rs` (`transform`).
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// Raised when a slice or column contains no finite (non-NaN) values and
    /// therefore no statistics can be computed.
    ///
    /// Produced by: `impute.rs` (`impute_column_inplace`).
    #[error("Empty input: {msg}")]
    EmptyInput { msg: &'static str },

    /// Raised for any parameter that is out of range or logically inconsistent
    /// (e.g. `n_components > n_features`, unknown imputation method name).
    ///
    /// Produced by: `impute.rs`, `pca.rs`.
    #[error("Invalid parameter: {msg}")]
    InvalidParameter { msg: String },

    /// Raised when nalgebra's SVD decomposition fails to converge.  In
    /// practice this should not happen on well-conditioned covariance matrices,
    /// but the error is retained as a safety net.
    ///
    /// Produced by: `pca.rs`.
    #[error("SVD failed to converge")]
    SvdFailed,
}

impl PulsarError {
    /// Stable, machine-readable code for this error.
    ///
    /// Callers (the MCP error envelope, CLI `--json` output) branch on this
    /// code, never on the `Display` text. Codes are part of the public
    /// contract: renaming one is a breaking change.
    pub fn code(&self) -> &'static str {
        match self {
            PulsarError::ShapeMismatch { .. } => "SHAPE_MISMATCH",
            PulsarError::EmptyInput { .. } => "EMPTY_INPUT",
            PulsarError::InvalidParameter { .. } => "INVALID_PARAMETER",
            PulsarError::SvdFailed => "SVD_FAILED",
        }
    }
}
