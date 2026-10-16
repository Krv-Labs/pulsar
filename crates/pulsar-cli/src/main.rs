//! The `pulsar` binary. Subcommands land with #73.

use clap::Parser;

/// Topological data analysis for tabular data.
#[derive(Parser)]
#[command(name = "pulsar", version)]
struct Cli {}

fn main() {
    Cli::parse();
}
