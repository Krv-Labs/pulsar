//! Tabular ingest, column typing, and preprocessing (#63 slice of #68/#69).
//!
//! This crate turns a CSV or Parquet file into the all-numeric, NaN-free
//! feature matrix the pipeline consumes, reproducing what Pulsar v0.5 did
//! with pandas:
//!
//! - [`read_table`] reads a file into a [`Table`] whose columns are typed the
//!   way `pandas.read_csv` would type them (int64, float64, bool, str), using
//!   the same set of [`NULL_TOKENS`].
//! - [`preprocess`] applies a resolved `preprocessing` block
//!   ([`pulsar_config::Preprocessing`]): drop, coerce, missing indicators,
//!   impute, one-hot encode, then validate. Every failure is a structured
//!   [`DataError`] with a stable [`DataError::code`].
//!
//! The declarative recipe (#68) and profiling tools (#69) build on these
//! types; nothing here decides *what* to do with a column.

pub mod csv_out;
pub mod error;
pub mod ingest;
pub mod key;
pub mod preprocess;
pub mod pyrandom;
pub mod table;

pub use csv_out::{to_csv_string, write_csv};
pub use error::DataError;
pub use ingest::{
    infer_column, is_null_token, read_csv, read_csv_path, read_parquet, read_table, write_parquet,
    NULL_TOKENS,
};
pub use key::Key;
pub use preprocess::{preprocess, shared_vocab, Layout, Preprocessed, Vocab};
pub use table::{Column, Table};
