use thiserror::Error;

#[derive(Debug, Error)]
pub enum AnalysisError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("unknown threshold candidate policy {0:?}; expected one of balanced, report_ready, detail_seeking, outlier_mining")]
    UnknownPolicy(String),
    #[error("unknown clustering method {0:?}; expected agglomerative or spectral")]
    UnknownMethod(String),
    #[error("empty graph: the model has no nodes")]
    EmptyGraph,
    #[error("eigensolver did not converge: {0}")]
    Convergence(String),
    #[error(transparent)]
    Core(#[from] pulsar_core::PulsarError),
}

impl AnalysisError {
    /// Stable code for error envelopes.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidArgument(_) => "INVALID_ARGUMENT",
            Self::UnknownPolicy(_) => "UNKNOWN_POLICY",
            Self::UnknownMethod(_) => "UNKNOWN_METHOD",
            Self::EmptyGraph => "EMPTY_GRAPH",
            Self::Convergence(_) => "SOLVER_ERROR",
            Self::Core(_) => "CORE",
        }
    }
}
