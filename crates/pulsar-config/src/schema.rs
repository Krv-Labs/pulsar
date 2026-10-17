//! JSON Schema for the config file, generated from [`crate::file`].

use serde_json::Value;

use crate::file::ConfigFile;

/// The JSON Schema (draft 2020-12) for `params.yaml`.
///
/// Exposed by `pulsar schema config` and, from #71, as an MCP resource.
/// Snapshot: `crates/pulsar-config/config.schema.json`.
pub fn json_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(ConfigFile)).expect("schema serializes")
}
