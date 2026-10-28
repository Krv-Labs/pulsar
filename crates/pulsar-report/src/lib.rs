//! Exports: the Parquet dataset bundle (`pulsar/exports.py`) and the
//! offline HTML report (`pulsar/mcp/report/`).

pub mod bundle;
pub mod html;
pub mod layout;

pub use bundle::{export_dataset_bundle, BundleError, BundleOptions, Layout, BUNDLE_FILES};
pub use html::{dossier_to_html, ReportGraph};
