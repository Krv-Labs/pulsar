//! The `pulsar` binary. The full command set lands with #73.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use pulsar_pipeline::{fit, FitOptions, Fitted};
use serde_json::json;

/// Topological data analysis for tabular data.
#[derive(Parser)]
#[command(name = "pulsar", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a sweep from a params.yaml and print a JSON summary of the
    /// constructed cosmic graph.
    Fit {
        /// Path to params.yaml.
        config: PathBuf,
        /// Dataset (CSV or Parquet); overrides `run.data` in the config.
        #[arg(long)]
        data: Option<PathBuf>,
        /// Directory to write `summary.json` and `graph.json` into.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Do not draw a progress bar on stderr.
        #[arg(long)]
        quiet: bool,
        /// Worker threads for the parallel kernels (default: all cores, or
        /// RAYON_NUM_THREADS).
        #[arg(long)]
        threads: Option<usize>,
    },
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
        Some(Command::Fit {
            config,
            data,
            out,
            quiet,
            threads,
        }) => run_fit(&config, data.as_deref(), out.as_deref(), quiet, threads),
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

/// `pulsar fit`: exit 2 on a config error, 1 on a data or pipeline error
/// (a JSON error envelope on stderr), 0 with the summary on stdout.
fn run_fit(
    config: &Path,
    data: Option<&Path>,
    out: Option<&Path>,
    quiet: bool,
    threads: Option<usize>,
) -> ExitCode {
    if let Err(e) = pulsar_pipeline::configure_threads(threads) {
        return fail("THREAD_POOL", &e.to_string(), None);
    }
    let cfg = match pulsar_config::from_path(config) {
        Ok(cfg) => cfg,
        Err(errors) => {
            print_json(&json!({ "errors": errors }));
            return ExitCode::from(2);
        }
    };
    let data_path: PathBuf = match data {
        Some(p) => p.to_path_buf(),
        None if cfg.data.is_empty() => {
            return fail(
                "NO_DATA",
                "no `run.data` in the config and no --data given",
                None,
            );
        }
        None => PathBuf::from(&cfg.data),
    };
    let table = match pulsar_data::read_table(&data_path) {
        Ok(t) => t,
        Err(e) => return fail(e.code(), &e.to_string(), Some(&data_path)),
    };

    let bar = if quiet {
        ProgressBar::hidden()
    } else {
        let bar = ProgressBar::new(1000);
        bar.set_style(
            ProgressStyle::with_template("{msg:<48} [{bar:30}] {percent:>3}% {elapsed}")
                .expect("static template")
                .progress_chars("=> "),
        );
        bar
    };
    let mut on_progress = |stage: &str, fraction: f64| {
        bar.set_message(stage.to_string());
        bar.set_position((fraction * 1000.0).round() as u64);
    };
    let fitted = match fit(&cfg, &table, FitOptions::default(), Some(&mut on_progress)) {
        Ok(f) => f,
        Err(e) => {
            bar.finish_and_clear();
            return fail(e.code(), &e.to_string(), Some(&data_path));
        }
    };
    bar.finish_and_clear();

    let summary = summarize(&fitted, &data_path);
    if let Some(dir) = out {
        if let Err(e) = write_outputs(dir, &fitted, &summary) {
            return fail("IO", &e.to_string(), Some(dir));
        }
    }
    print_json(&summary)
}

fn summarize(fitted: &Fitted, data_path: &Path) -> serde_json::Value {
    let cfg = &fitted.config;
    let sizes = fitted.graph.component_sizes();
    let plateaus: Vec<serde_json::Value> = fitted
        .stability
        .as_ref()
        .map(|s| {
            s.plateaus
                .iter()
                .take(5)
                .map(|p| {
                    json!({
                        "start": p.start_threshold,
                        "end": p.end_threshold,
                        "component_count": p.component_count,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json!({
        "run_name": cfg.run_name,
        "data": data_path,
        "n_rows": fitted.n,
        "n_features": fitted.layout.feature_names.len(),
        "n_embeddings": fitted.embeddings.len(),
        "n_ball_maps": fitted.ball_maps.len(),
        "construction": {
            "method": cfg.cosmic_graph.construction,
            "threshold_policy": match cfg.cosmic_graph.construction_threshold {
                pulsar_config::Threshold::Auto => "auto",
                pulsar_config::Threshold::Value(_) => "fixed",
            },
            "construction_threshold": fitted.construction_threshold,
            "weight_scale": fitted.weight_scale,
            "sparsified": cfg.cosmic_graph.sparsify,
        },
        "full_affinity": {
            "n_edges": fitted.cosmic.weighted_edges().len(),
        },
        "graph": {
            "n_edges": fitted.graph.n_edges(),
            "n_components": sizes.len(),
            "largest_components": sizes.iter().take(10).collect::<Vec<_>>(),
        },
        "stability_plateaus": plateaus,
    })
}

fn write_outputs(dir: &Path, fitted: &Fitted, summary: &serde_json::Value) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let graph = json!({
        "n": fitted.graph.n,
        "construction_threshold": fitted.construction_threshold,
        "weight_scale": fitted.weight_scale,
        "edges": fitted.graph.edges,
    });
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_string_pretty(summary)? + "\n",
    )?;
    std::fs::write(
        dir.join("graph.json"),
        serde_json::to_string(&graph)? + "\n",
    )?;
    Ok(())
}

fn fail(code: &str, message: &str, path: Option<&Path>) -> ExitCode {
    let envelope = json!({ "error": { "code": code, "message": message, "path": path } });
    eprintln!("{envelope}");
    ExitCode::FAILURE
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
