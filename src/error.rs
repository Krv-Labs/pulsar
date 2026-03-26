use thiserror::Error;

#[derive(Debug, Error)]
pub enum PulsarError {
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },
    #[error("Empty input: {msg}")]
    EmptyInput { msg: &'static str },
    #[error("Invalid parameter: {msg}")]
    InvalidParameter { msg: String },
    #[error("SVD failed to converge")]
    SvdFailed,
}

impl From<PulsarError> for pyo3::PyErr {
    fn from(e: PulsarError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    }
}
