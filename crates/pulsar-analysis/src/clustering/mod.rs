//! Clustering of the constructed cosmic graph (`cosmic_clusters`).

pub mod agglomerative;
pub mod lobpcg;
pub mod spectral;

use pulsar_pipeline::Graph;

use crate::error::AnalysisError;

pub use agglomerative::cluster_agglomerative;
pub use spectral::{cluster_spectral, cluster_spectral_with, spectral_embedding};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Agglomerative,
    Spectral,
}

impl Method {
    pub fn parse(name: &str) -> Result<Self, AnalysisError> {
        match name {
            "agglomerative" => Ok(Self::Agglomerative),
            "spectral" => Ok(Self::Spectral),
            other => Err(AnalysisError::UnknownMethod(other.to_owned())),
        }
    }
}

/// `cosmic_clusters(graph, method, n_clusters)`: labels numbered `0..`.
pub fn cosmic_clusters(
    graph: &Graph,
    method: Method,
    n_clusters: usize,
    seed: u64,
) -> Result<Vec<usize>, AnalysisError> {
    match method {
        Method::Agglomerative => cluster_agglomerative(graph, n_clusters),
        Method::Spectral => cluster_spectral(graph, n_clusters, seed),
    }
}
