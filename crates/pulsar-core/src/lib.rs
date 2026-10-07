//! # `pulsar-core` — pure-Rust engine for large-scale topological data analysis
//!
//! The algorithmic core of Pulsar, optimized for large EHR datasets. All
//! algorithms avoid O(n²) memory where possible and use parallel execution via
//! rayon. The Python extension (`thema-pulsar`, exposing `pulsar._pulsar`) is a
//! thin numpy wrapper over this crate; Rust services depend on this crate
//! directly.
//!
//! ## Pipeline building blocks
//!
//! | Item | Description |
//! |---|---|
//! | [`impute_column_inplace`] | Fill NaNs in one column (5 strategies, seeded) |
//! | [`StandardScalerInner`] | Z-score normalisation (population std) |
//! | [`jl_grid`] / [`JLProjectionInner`] | Johnson-Lindenstrauss projections across dimensions/seeds (parallel) |
//! | [`pca_grid`] / [`PCAInner`] | Randomized PCA across dimensions/seeds (parallel) |
//! | [`ball_mapper_grid`] / [`BallMapper`] | Ball Mapper across embeddings/epsilons (parallel) |
//! | [`accumulate_pseudo_laplacians`] / [`accumulate_pseudo_laplacians_sparse`] | Fused Laplacian accumulation (parallel) |
//! | [`cosmic_edges_minhash`] / [`MinHashSignatures`] | Approximate cosmic edges via MinHash + LSH |
//! | [`CosmicGraphInner`] | Normalised adjacency from accumulated Laplacian |
//! | [`find_stable_thresholds`] / [`find_stable_thresholds_from_edges`] | Approximate H₀ persistent homology for threshold selection |
//! | [`accumulate_temporal_pseudo_laplacians_inner`] / [`normalize_temporal_laplacian`] | Temporal (longitudinal) tensors |

pub mod ballmapper;
pub mod cosmic;
pub mod error;
pub mod impute;
pub mod jl;
pub mod minhash;
pub mod pca;
pub mod ph;
pub mod pseudolaplacian;
pub mod scale;
pub mod temporal;

pub use ballmapper::{ball_mapper_grid, fit_inner, BallMapper};
pub use cosmic::{connected_components, laplacian_diag, CosmicGraphInner, WeightedEdge};
pub use error::PulsarError;
pub use impute::impute_column_inplace;
pub use jl::{jl_grid, JLProjectionInner};
pub use minhash::{cosmic_edges_minhash, MinHashSignatures};
pub use pca::{pca_grid, PCAInner};
pub use ph::{
    find_stable_thresholds, find_stable_thresholds_from_edges, Plateau, StabilityResult,
    UnionFind, DEFAULT_NUM_BINS,
};
pub use pseudolaplacian::{
    accumulate_pseudo_laplacians, accumulate_pseudo_laplacians_sparse, pseudo_laplacian_inner,
    SparsePseudoLaplacian,
};
pub use scale::StandardScalerInner;
pub use temporal::{accumulate_temporal_pseudo_laplacians_inner, normalize_temporal_laplacian};
