//! The Pulsar MCP server, built on `rmcp` (#71).
//!
//! A general-purpose topology workbench: tools expose measurements
//! (geometry, graph state, threshold surfaces, provenance), not conclusions.
//! Every tool has typed parameters with per-field descriptions, an output
//! schema, `read_only` / `idempotent` / `open_world = false` annotations,
//! and returns structured content plus a text view. Failures use the error
//! envelope (`error_code`, `tool`, `agent_action`, handle context) with the
//! MCP error flag set.
//!
//! Handles resolve through `pulsar-store`, so a `run_id` works on any
//! transport and across restarts. Long-running sweeps run as store jobs:
//! `execution='async'` returns the `run_id` at once for `wait_job` /
//! `get_job` polling, and clients that declare the MCP tasks extension
//! (SEP-2663) get a task handle instead of a blocking call. Progress events
//! from the pipeline are forwarded as MCP progress notifications.

// Tool errors are rare, agent-facing paths; boxing every `ToolError` would
// add noise for no measurable gain.
#![allow(clippy::result_large_err)]

pub mod envelope;
pub mod guide;
pub mod jobs;
pub mod markdown;
pub mod schema;
pub mod server;
pub mod state;
pub mod tools;

pub use envelope::ToolError;
pub use server::{run_stdio, PulsarServer};
pub use state::{ServerOptions, ServerState};
pub use tools::{catalog, ToolSpec};

/// Character budget of the advertised catalog (descriptions plus input
/// schemas): ~12k tokens. Raise deliberately, not by accident.
pub const CATALOG_BUDGET_CHARS: usize = 48_000;
