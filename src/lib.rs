//! # `pulsar._pulsar` — Rust extension module for large-scale topological data analysis
//!
//! Optimized for large EHR datasets. All algorithms avoid O(n²) memory where possible
//! and use parallel execution via rayon.
//!
//! ## Core functions
//!
//! | Function | Description |
//! |---|---|
//! | `pca_grid` | Randomized PCA across dimensions/seeds (parallel) |
//! | `ball_mapper_grid` | Ball Mapper across embeddings/epsilons (parallel) |
//! | `accumulate_pseudo_laplacians` | Fused Laplacian accumulation (parallel) |
//!
//! ## Classes
//!
//! | Class | Description |
//! |---|---|
//! | `PCA` | Randomized SVD-based dimensionality reduction |
//! | `StandardScaler` | Z-score normalisation |
//! | `BallMapper` | Topological Ball Mapper complex |
//! | `CosmicGraph` | Normalised adjacency from accumulated Laplacian |

use pyo3::prelude::*;

mod error;
mod impute;
mod scale;
mod pca;
mod ballmapper;
mod pseudolaplacian;
mod cosmic;

#[pymodule]
fn _pulsar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Imputation
    m.add_function(wrap_pyfunction!(impute::impute_column, m)?)?;
    
    // Scaling
    m.add_class::<scale::StandardScaler>()?;
    
    // PCA (randomized SVD only - optimized for large datasets)
    m.add_class::<pca::PCA>()?;
    m.add_function(wrap_pyfunction!(pca::pca_grid, m)?)?;
    
    // Ball Mapper
    m.add_class::<ballmapper::BallMapper>()?;
    m.add_function(wrap_pyfunction!(ballmapper::ball_mapper_grid, m)?)?;
    
    // Pseudo-Laplacian (fused accumulation only)
    m.add_function(wrap_pyfunction!(pseudolaplacian::accumulate_pseudo_laplacians, m)?)?;
    
    // Cosmic Graph
    m.add_class::<cosmic::CosmicGraph>()?;
    
    Ok(())
}
