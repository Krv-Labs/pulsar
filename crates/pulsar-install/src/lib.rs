//! Register the Pulsar MCP server in agent harness configs.
//!
//! Port of the Python `pulsar install | uninstall | status` (#73), with the
//! same safety rules: only entries recognized as Pulsar's are ever rewritten
//! or removed, files with comments we would drop are refused, every rewrite
//! is atomic with a backup, and uninstall removes only what install created.
//! Launch modes are `binary` (absolute path to this executable) and `npx`
//! (`npx -y @krv-labs/pulsar mcp`).

/// The product name: the binary, the MCP server key written into agent
/// configs and the state directory (see also [`launch::PACKAGE`]).
pub const NAME: &str = "pulsar";

pub mod entry;
pub mod fsops;
pub mod harness;
pub mod launch;
pub mod ops;
pub mod state;

pub use harness::{detected, home_dir, ids, HARNESSES};
pub use launch::{default_mode, resolve, which, LaunchSpec, Mode};
pub use ops::{install, status, uninstall, uninstall_plan, Report, Status};

use std::path::Path;

/// A failure with a message written for the person running the command.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InstallError(String);

impl InstallError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    pub(crate) fn io(action: &str, path: &Path, err: &std::io::Error) -> Self {
        Self(format!("{action} {}: {err}", path.display()))
    }
}
