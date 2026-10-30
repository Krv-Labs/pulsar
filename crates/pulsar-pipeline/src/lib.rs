//! Sweep orchestration: the Rust replacement for `ThemaRS` (#63).
//!
//! ```text
//! table ─▶ preprocess ─▶ scale ─▶ projection grid ─▶ ball-mapper grid
//!       ─▶ cosmic accumulation (MinHash | exact) ─▶ [sparsify] ─▶ finalize
//! ```
//!
//! [`fit`] runs that chain on one table; [`fit_multi`] runs the sweep on
//! several representations of the same points and folds every ball map into
//! one cosmic graph. Both stream ball maps in batches so peak memory stays
//! bounded, emit `(stage, cumulative_fraction)` progress events, and return a
//! [`Fitted`] model whose [`Graph`] is the thresholded, `[0, 1]`-weighted
//! cosmic graph.

pub mod error;
pub mod fingerprint;
pub mod fit;
pub mod graph;
pub mod progress;
pub mod representatives;
pub mod threads;
pub mod trajectory;

pub use error::PipelineError;
pub use fingerprint::projection_fingerprint;
pub use fit::{fit, fit_multi, fit_path, projection_grid, FitOptions, Fitted};
pub use graph::{Csr, Graph};
pub use progress::{ProgressEvent, ProgressFn, ProgressTracker};
pub use representatives::select_representatives;
pub use threads::{configure_threads, current_threads};
pub use trajectory::{Ball, CosmicTrajectory, EntityIds, Observation};
