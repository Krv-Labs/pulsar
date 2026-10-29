//! The `pulsar` binary.

mod install_cmd;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use pulsar_pipeline::{fit, FitOptions, Fitted};
use pulsar_store::{
    summarize_history, DatasetId, JobHandle, JobKind, RunArtifacts, RunId, RunRecord, Store,
    StoreError,
};
use serde_json::json;

/// Topological data analysis for tabular data.
#[derive(Parser)]
#[command(name = env!("CARGO_BIN_NAME"), version)]
struct Cli {
    /// Store directory for runs and datasets (default: $PULSAR_STORE_DIR or
    /// the XDG data dir).
    #[arg(long, global = true, value_name = "DIR")]
    store: Option<PathBuf>,
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
        /// Persist the run (record, graph and affinity) in the store and
        /// report its `run_id`.
        #[arg(long)]
        save: bool,
    },
    /// Inspect persisted runs.
    #[command(subcommand)]
    Runs(RunsCommand),
    /// Serve the MCP server on stdio (for agent harnesses).
    Mcp,
    /// Register the MCP server in agent configs (Claude Code, Claude
    /// Desktop, Codex, Gemini, Copilot, Cursor, VS Code, Antigravity).
    /// With no agent named, registers in every one detected on this machine.
    Install {
        /// Agents to configure (see `pulsar status` for the ids).
        harnesses: Vec<String>,
        /// Configure every supported agent, detected or not.
        #[arg(long)]
        all: bool,
        /// Show what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// How agents launch the server: `binary` (this executable's
        /// absolute path) or `npx` (`npx -y @krv-labs/pulsar mcp`). Default:
        /// `npx` when run from a node_modules cache, else `binary`.
        #[arg(long)]
        mode: Option<String>,
        /// With `--mode npx`, pin the package to this binary's version.
        #[arg(long)]
        pin_version: bool,
        /// With `--mode binary`, record this executable instead of the one
        /// running now (e.g. the stable shim of a global npm install).
        #[arg(long)]
        command: Option<PathBuf>,
    },
    /// Remove the MCP server registration from agent configs. Only entries
    /// Pulsar wrote are touched; files it created are removed again.
    Uninstall {
        /// Agents to remove from.
        harnesses: Vec<String>,
        /// Remove from every supported agent.
        #[arg(long)]
        all: bool,
        /// Show what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Do not ask for confirmation.
        #[arg(short, long)]
        yes: bool,
        /// Also delete the `.pulsar.backup` files install left behind.
        #[arg(long)]
        purge_backups: bool,
    },
    /// Show which agents have Pulsar registered and whether each is healthy.
    Status {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
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
enum RunsCommand {
    /// List completed runs, oldest first.
    List {
        /// Only runs of this dataset_id.
        #[arg(long)]
        dataset: Option<String>,
    },
    /// Show one run: its job state and, once complete, its record.
    Show { run_id: String },
    /// Sweep-history observations over the completed runs.
    History {
        /// Only runs of this dataset_id.
        #[arg(long)]
        dataset: Option<String>,
    },
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
            save,
        }) => {
            let store = if save {
                Some(cli.store.as_deref())
            } else {
                None
            };
            run_fit(
                &config,
                data.as_deref(),
                out.as_deref(),
                quiet,
                threads,
                store,
            )
        }
        Some(Command::Runs(cmd)) => run_runs(cli.store.as_deref(), cmd),
        Some(Command::Mcp) => run_mcp(cli.store.as_deref()),
        Some(Command::Install {
            harnesses,
            all,
            dry_run,
            mode,
            pin_version,
            command,
        }) => install_cmd::run_install(install_cmd::InstallArgs {
            harnesses,
            all,
            dry_run,
            mode,
            pin_version,
            command,
        }),
        Some(Command::Uninstall {
            harnesses,
            all,
            dry_run,
            yes,
            purge_backups,
        }) => install_cmd::run_uninstall(install_cmd::UninstallArgs {
            harnesses,
            all,
            dry_run,
            yes,
            purge_backups,
        }),
        Some(Command::Status { json }) => install_cmd::run_status(json),
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

/// `pulsar mcp`: serve on stdio until the client disconnects. Logs go to
/// stderr; stdout is the protocol channel.
fn run_mcp(store_dir: Option<&Path>) -> ExitCode {
    let store = match open_store(store_dir) {
        Ok(s) => s,
        Err(e) => return store_fail(&e),
    };
    let state = match pulsar_mcp::ServerState::new(store, pulsar_mcp::ServerOptions::default()) {
        Ok(s) => std::sync::Arc::new(s),
        Err(e) => return store_fail(&e),
    };
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => return fail("RUNTIME", &e.to_string(), None),
    };
    match runtime.block_on(pulsar_mcp::run_stdio(state)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail("MCP", &e.to_string(), None),
    }
}

fn open_store(dir: Option<&Path>) -> Result<Store, StoreError> {
    match dir {
        Some(dir) => Store::open_at(dir),
        None => Store::open(),
    }
}

fn store_fail(e: &StoreError) -> ExitCode {
    fail(e.code(), &e.to_string(), None)
}

/// `pulsar fit`: exit 2 on a config error, 1 on a data or pipeline error
/// (a JSON error envelope on stderr), 0 with the summary on stdout. With
/// `store` set the run is persisted through the job model, so a failure is
/// recorded against its `run_id` too.
fn run_fit(
    config: &Path,
    data: Option<&Path>,
    out: Option<&Path>,
    quiet: bool,
    threads: Option<usize>,
    store: Option<Option<&Path>>,
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

    // Persisting: register the dataset and queue the job before fitting, so
    // the run_id exists (and a failure is recorded) whatever happens next.
    let mut job: Option<(Store, DatasetId, JobHandle)> = None;
    if let Some(dir) = store {
        let opened = match open_store(dir) {
            Ok(s) => s,
            Err(e) => return store_fail(&e),
        };
        let dataset = match opened.register_dataset(&data_path) {
            Ok(d) => d,
            Err(e) => return store_fail(&e),
        };
        let handle = match opened
            .submit(JobKind::Sweep, Some(dataset.dataset_id.clone()))
            .and_then(|h| h.start().map(|()| h))
        {
            Ok(h) => h,
            Err(e) => return store_fail(&e),
        };
        job = Some((opened, dataset.dataset_id, handle));
    }

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
    let mut on_progress = |ev: pulsar_pipeline::ProgressEvent<'_>| {
        bar.set_message(ev.label.to_string());
        bar.set_position((ev.fraction * 1000.0).round() as u64);
        if let Some((_, _, handle)) = &job {
            let _ = handle.progress(ev.label, ev.fraction);
        }
    };
    let fitted = match fit(&cfg, &table, FitOptions::default(), Some(&mut on_progress)) {
        Ok(f) => f,
        Err(e) => {
            bar.finish_and_clear();
            if let Some((_, _, handle)) = &job {
                let _ = handle.fail(e.code(), &e.to_string());
            }
            return fail(e.code(), &e.to_string(), Some(&data_path));
        }
    };
    bar.finish_and_clear();

    let mut summary = summarize(&fitted, &data_path);
    if let Some((store, dataset_id, handle)) = &job {
        let record = RunRecord::from_fitted(
            handle.id(),
            &fitted,
            Some(dataset_id.clone()),
            Some(&data_path.to_string_lossy()),
        );
        if let Err(e) = handle.complete(&record, &RunArtifacts::from_fitted(&fitted)) {
            let _ = handle.fail(e.code(), &e.to_string());
            return store_fail(&e);
        }
        summary["run_id"] = json!(record.run_id);
        summary["dataset_id"] = json!(dataset_id);
        summary["store"] = json!(store.root());
    }
    if let Some(dir) = out {
        if let Err(e) = write_outputs(dir, &fitted, &summary) {
            return fail("IO", &e.to_string(), Some(dir));
        }
    }
    print_json(&summary)
}

/// `pulsar runs ...`: exit 1 with a JSON error envelope on a store or
/// handle error, 0 with JSON on stdout.
fn run_runs(store_dir: Option<&Path>, cmd: RunsCommand) -> ExitCode {
    let store = match open_store(store_dir) {
        Ok(s) => s,
        Err(e) => return store_fail(&e),
    };
    let parse_dataset = |raw: Option<String>| -> Result<Option<DatasetId>, StoreError> {
        raw.map(|s| DatasetId::parse(&s)).transpose()
    };
    match cmd {
        RunsCommand::List { dataset } => {
            let dataset = match parse_dataset(dataset) {
                Ok(d) => d,
                Err(e) => return store_fail(&e),
            };
            match store.list_runs(dataset.as_ref()) {
                Ok(listing) => print_json(&json!({
                    "store": store.root(),
                    "n_runs": listing.runs.len(),
                    "skipped": listing.skipped,
                    "runs": listing.runs.iter().map(|r| json!({
                        "run_id": r.run_id,
                        "dataset_id": r.dataset_id,
                        "run_name": r.config.run_name,
                        "created_at": r.created_at,
                        "n_rows": r.n_rows,
                        "construction_threshold": r.resolved_construction_threshold,
                        "n_edges": r.metrics.n_edges,
                        "component_count": r.metrics.component_count,
                        "health": r.metrics.health_label(),
                    })).collect::<Vec<_>>(),
                })),
                Err(e) => store_fail(&e),
            }
        }
        RunsCommand::Show { run_id } => {
            let run_id = match RunId::parse(&run_id) {
                Ok(id) => id,
                Err(e) => return store_fail(&e),
            };
            let job = match store.get_job(&run_id) {
                Ok(job) => job,
                Err(e) => return store_fail(&e),
            };
            let run = store.get_run(&run_id).ok();
            print_json(&json!({ "store": store.root(), "job": job, "run": run }))
        }
        RunsCommand::History { dataset } => {
            let dataset = match parse_dataset(dataset) {
                Ok(d) => d,
                Err(e) => return store_fail(&e),
            };
            match store.list_runs(dataset.as_ref()) {
                Ok(listing) => print_json(&summarize_history(&listing.runs)),
                Err(e) => store_fail(&e),
            }
        }
    }
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
