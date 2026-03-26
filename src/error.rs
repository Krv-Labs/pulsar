use thiserror::Error;

/// All errors that can originate from the Pulsar Rust backend.
///
/// Every variant maps to a Python `ValueError` when propagated across the
/// PyO3 boundary (see the `From<PulsarError> for pyo3::PyErr` impl below).
#[derive(Debug, Error)]
pub enum PulsarError {
    /// Raised when an array passed to a function has the wrong number of rows
    /// or columns.  The `expected` and `got` fields are human-readable strings
    /// (e.g. `"3 columns"`) so the Python error message is self-explanatory.
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

/// Convert a `PulsarError` into a Python `ValueError`.
///
/// This impl is required by PyO3: any `Result<_, PulsarError>` returned from
/// a `#[pyfunction]` or `#[pymethods]` method is automatically converted to a
/// Python exception by this conversion.
impl From<PulsarError> for pyo3::PyErr {
    fn from(e: PulsarError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}
