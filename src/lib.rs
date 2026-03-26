//! # `pulsar._pulsar` — Rust extension module for the Pulsar pipeline
//!
//! This crate is compiled by [maturin](https://github.com/PyO3/maturin) into a
//! Python extension module named `pulsar._pulsar` (the leading underscore
//! signals that it is a private implementation detail, not the public API).
//!
//! The public Python API is re-exported through the `pulsar` package
//! (`pulsar/__init__.py`).
//!
//! ## Module contents
//!
//! | Python name | Rust source | Description |
//! |---|---|---|
//! | `impute_column` | `impute.rs` | Fill NaN values in a 1-D column |
//! | `StandardScaler` | `scale.rs` | Fit/transform z-score normalisation |
//! | `PCA` | `pca.rs` | Exact SVD-based dimensionality reduction |
//! | `BallMapper` | `ballmapper.rs` | Topological Ball Mapper complex |
//! | `ball_mapper_grid` | `ballmapper.rs` | Parallel sweep over (embedding, epsilon) pairs |
//! | `pseudo_laplacian` | `pseudolaplacian.rs` | Build pseudo-Laplacian from ball membership |
//! | `CosmicGraph` | `cosmic.rs` | Normalised adjacency from accumulated pseudo-Laplacian |
//!
//! ## Naming note
//!
//! Cargo cannot compile a `cdylib` whose crate name starts with `_`.  The
//! `[lib]` section in `Cargo.toml` therefore omits the `name` field, letting
//! Cargo use the package name `pulsar`.  Maturin then renames the compiled
//! `.so` file to `_pulsar.so` (or platform equivalent) according to the
//! `module-name = "pulsar._pulsar"` setting in `pyproject.toml`.

use pyo3::prelude::*;

mod error;
mod impute;
mod scale;
mod pca;
mod ballmapper;
mod pseudolaplacian;
mod cosmic;

/// Register all public symbols with the Python module.
///
/// The function name `_pulsar` must exactly match the last component of
/// `module-name` in `pyproject.toml`.
#[pymodule]
fn _pulsar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(impute::impute_column, m)?)?;
    m.add_class::<scale::StandardScaler>()?;
    m.add_class::<pca::PCA>()?;
    m.add_class::<ballmapper::BallMapper>()?;
    m.add_function(wrap_pyfunction!(ballmapper::ball_mapper_grid, m)?)?;
    m.add_function(wrap_pyfunction!(pseudolaplacian::pseudo_laplacian, m)?)?;
    m.add_class::<cosmic::CosmicGraph>()?;
    Ok(())
}
