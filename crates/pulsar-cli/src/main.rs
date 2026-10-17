//! The `pulsar` binary. The full command set lands with #73.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

/// Topological data analysis for tabular data.
#[derive(Parser)]
#[command(name = "pulsar", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Print a JSON Schema.
    Schema {
        #[arg(value_enum)]
        kind: SchemaKind,
    },
    /// Work with run configs (params.yaml).
    #[command(subcommand)]
    Config(ConfigCommand),
}

#[derive(Clone, Copy, ValueEnum)]
enum SchemaKind {
    /// The params.yaml config format.
    Config,
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Validate a config. Exit 0 and print the resolved config as JSON, or
    /// exit 2 and print the structured errors as JSON.
    Validate { path: PathBuf },
    /// Normalize a config: every default explicit, written back as YAML.
    Fmt { path: PathBuf },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        None => ExitCode::SUCCESS,
        Some(Command::Schema {
            kind: SchemaKind::Config,
        }) => print_json(&pulsar_config::json_schema()),
        Some(Command::Config(ConfigCommand::Validate { path })) => {
            match pulsar_config::from_path(&path) {
                Ok(cfg) => print_json(&cfg),
                Err(errors) => {
                    print_json(&serde_json::json!({ "errors": errors }));
                    ExitCode::from(2)
                }
            }
        }
        Some(Command::Config(ConfigCommand::Fmt { path })) => {
            match pulsar_config::from_path(&path) {
                Ok(cfg) => {
                    print!("{}", pulsar_config::to_yaml(&cfg));
                    ExitCode::SUCCESS
                }
                Err(errors) => {
                    eprintln!("{errors}");
                    ExitCode::from(2)
                }
            }
        }
    }
}

fn print_json<T: serde::Serialize>(value: &T) -> ExitCode {
    match serde_json::to_string_pretty(value) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
