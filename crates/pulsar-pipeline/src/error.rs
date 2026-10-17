//! Pipeline errors: orchestration failures plus the data and kernel errors
//! they wrap, each with a stable code.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("no data path in config and no table provided")]
    NoData,

    #[error("datasets must be non-empty")]
    EmptyDatasets,

    #[error("dataset {index} has {got} rows after preprocessing; expected {expected} (same as dataset 0)")]
    RowCountMismatch {
        index: usize,
        got: usize,
        expected: usize,
    },

    #[error("{what} must be positive")]
    InvalidOption { what: &'static str },

    #[error("precomputed embeddings have {got} rows; the current data has {expected}")]
    EmbeddingRows { got: usize, expected: usize },

    #[error(transparent)]
    Data(#[from] pulsar_data::DataError),

    #[error(transparent)]
    Core(#[from] pulsar_core::PulsarError),
}

impl PipelineError {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoData => "NO_DATA",
            Self::EmptyDatasets => "EMPTY_DATASETS",
            Self::RowCountMismatch { .. } => "ROW_COUNT_MISMATCH",
            Self::InvalidOption { .. } => "INVALID_OPTION",
            Self::EmbeddingRows { .. } => "EMBEDDING_ROWS",
            Self::Data(e) => e.code(),
            Self::Core(e) => e.code(),
        }
    }
}
