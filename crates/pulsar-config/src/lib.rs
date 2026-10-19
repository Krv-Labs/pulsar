//! Config model, parameter expansion, and JSON Schema for `params.yaml` (#61).
//!
//! ```
//! let cfg = pulsar_config::from_yaml_str("sweep: {ball_mapper: {epsilon: [0.3, 0.5]}}").unwrap();
//! assert_eq!(cfg.ball_mapper.epsilons, vec![0.3, 0.5]);
//!
//! let err = pulsar_config::from_yaml_str("cosmic_graph: {threshold: 0.2}").unwrap_err();
//! assert_eq!(err.0[0].code, pulsar_config::ErrorCode::LegacyKey);
//! assert_eq!(err.0[0].path, "/cosmic_graph/threshold");
//! ```
//!
//! Differences from the Python v0.5 loader, all deliberate:
//! - Unknown keys are errors everywhere (Python ignored them outside
//!   `cosmic_graph`), and all problems are reported at once with JSON
//!   pointers and stable codes instead of the first `ValueError`.
//! - Values are not coerced: `seed: "42"` or `dimensions: [2.5]` are errors
//!   (Python silently ran `int()`), and a `range` over integers must land on
//!   whole numbers.
//! - Ranges are checked up front: empty grids, `epsilon <= 0`,
//!   `n_reps < 1`, and similar fail at load time instead of mid-run.
//! - The `preprocessing` block follows #57 (`fill_value`, per-column
//!   `indicator`, `missing_indicator_groups`).

mod emit;
mod error;
mod expand;
pub mod file;
mod load;
mod model;
mod schema;

pub use emit::{to_file, to_yaml};
pub use error::{ConfigError, ConfigErrors, ErrorCode};
pub use expand::{linspace, DEFAULT_RANGE_STEPS};
pub use load::{from_path, from_value, from_yaml_str, MISSING_SUFFIX};
pub use model::*;
pub use schema::json_schema;
