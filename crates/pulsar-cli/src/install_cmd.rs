//! `pulsar install | uninstall | status`: register the MCP server in agent
//! harness configs (see `pulsar-install`).

use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;
use std::process::ExitCode;

use pulsar_install::{
    default_mode, detected, home_dir, ids, install, resolve, status, uninstall, uninstall_plan,
    which, Mode,
};

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("{}: {message}", env!("CARGO_BIN_NAME"));
    ExitCode::FAILURE
}

fn outcome(ok: bool) -> ExitCode {
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

pub struct InstallArgs {
    pub harnesses: Vec<String>,
    pub all: bool,
    pub dry_run: bool,
    pub mode: Option<String>,
    pub pin_version: bool,
    pub command: Option<std::path::PathBuf>,
}

/// Explicit ids, `--all`, or (for install only) whatever agents are present.
fn select(
    home: &Path,
    harnesses: Vec<String>,
    all: bool,
    auto_detect: bool,
) -> Result<Vec<String>, String> {
    if all {
        return Ok(ids().into_iter().map(String::from).collect());
    }
    if !harnesses.is_empty() {
        return Ok(harnesses);
    }
    if auto_detect {
        let found: Vec<String> = detected(home).iter().map(|h| h.id.to_string()).collect();
        if found.is_empty() {
            return Err(format!(
                "no supported agent found on this machine; name one explicitly or use --all (known: {})",
                ids().join(", ")
            ));
        }
        eprintln!("{}: detected {}", env!("CARGO_BIN_NAME"), found.join(", "));
        return Ok(found);
    }
    Err(format!(
        "name the agents to remove or pass --all (known: {})",
        ids().join(", ")
    ))
}

pub fn run_install(args: InstallArgs) -> ExitCode {
    let home = match home_dir() {
        Ok(h) => h,
        Err(e) => return fail(e),
    };
    let exe = match args.command {
        Some(path) if path.is_file() => path,
        Some(path) => return fail(format!("--command {} is not a file", path.display())),
        None => match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => return fail(format!("cannot locate the running executable: {e}")),
        },
    };
    let mode = match args.mode.as_deref().map(Mode::parse) {
        Some(Ok(m)) => m,
        Some(Err(e)) => return fail(e),
        None => default_mode(&exe),
    };
    let pin = args.pin_version.then_some(env!("CARGO_PKG_VERSION"));
    let launch = match resolve(mode, pin, &exe, &which) {
        Ok(l) => l,
        Err(e) => return fail(e),
    };
    let selected = match select(&home, args.harnesses, args.all, true) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let report = install(&home, &launch, &selected, args.dry_run);
    println!("{}", report.render());
    outcome(report.ok)
}

pub struct UninstallArgs {
    pub harnesses: Vec<String>,
    pub all: bool,
    pub dry_run: bool,
    pub yes: bool,
    pub purge_backups: bool,
}

fn confirm(question: &str, plan: &[String]) -> bool {
    eprintln!("{question}");
    for line in plan {
        eprintln!("  - {line}");
    }
    eprint!("Proceed? [y/N] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer).is_ok()
        && matches!(answer.trim(), "y" | "Y" | "yes")
}

pub fn run_uninstall(args: UninstallArgs) -> ExitCode {
    // Removal never needs a working launcher: uninstalling must stay possible
    // after the binary or npx has gone.
    let home = match home_dir() {
        Ok(h) => h,
        Err(e) => return fail(e),
    };
    let selected = match select(&home, args.harnesses, args.all, false) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    if !args.dry_run && !args.yes {
        if !std::io::stdin().is_terminal() {
            return fail("there is no console to confirm on; re-run with --yes to apply, or --dry-run to preview");
        }
        let plan = uninstall_plan(&home, &selected, args.purge_backups);
        if !confirm("Uninstall Pulsar from these agents?", &plan) {
            eprintln!("{}: cancelled, nothing changed", env!("CARGO_BIN_NAME"));
            return ExitCode::SUCCESS;
        }
    }
    let report = uninstall(&home, &selected, args.dry_run, args.purge_backups);
    println!("{}", report.render());
    outcome(report.ok)
}

pub fn run_status(json: bool) -> ExitCode {
    let home = match home_dir() {
        Ok(h) => h,
        Err(e) => return fail(e),
    };
    let status = status(&home);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&status.to_json(&home)).expect("status serializes")
        );
    } else {
        println!("{}", status.render(&home));
    }
    ExitCode::SUCCESS
}
