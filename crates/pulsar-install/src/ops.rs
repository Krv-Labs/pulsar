//! install, uninstall and status over the selected harnesses. Each returns a
//! [`Report`] (lines for the terminal plus an overall success flag) so the
//! CLI prints and sets the exit code, and tests assert on the text.

use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use crate::entry::{self, Inspection, State};
use crate::fsops::{backup_path, is_empty_json_config, prune_dirs, resolve_symlink};
use crate::harness::{self, Harness, HARNESSES};
use crate::launch::LaunchSpec;
use crate::state;

const OK: &str = "✓";
const FAILED: &str = "✗";
const CONFLICT: &str = "!";
const ABSENT: &str = "○";
const PENDING: &str = "…";
const REMOVED: &str = "−";
const INCOMPLETE: &str = "▲";

#[derive(Debug, Default)]
pub struct Report {
    pub lines: Vec<String>,
    pub ok: bool,
}

impl Report {
    fn new(title: &str, dry_run: bool) -> Self {
        let suffix = if dry_run { " (dry run)" } else { "" };
        Self {
            lines: vec![String::new(), format!("{title}{suffix}"), "│".into()],
            ok: true,
        }
    }
    fn line(&mut self, text: impl Into<String>) {
        self.lines.push(text.into());
    }
    fn detail(&mut self, glyph: &str, msg: impl AsRef<str>) {
        self.lines.push(format!("│    {glyph} {}", msg.as_ref()));
    }
    fn harness(&mut self, name: &str) {
        self.lines.push(format!("│  {name}"));
    }
    fn note(&mut self, msg: &str) {
        self.lines.push(format!("│      note: {msg}"));
    }
    fn footer(&mut self, msg: &str) {
        self.lines.push(format!("└─ {msg}"));
        self.lines.push(String::new());
    }
    pub fn render(&self) -> String {
        self.lines.join("\n")
    }
}

fn glyph(state: State) -> &'static str {
    match state {
        State::Active => OK,
        State::Incomplete => INCOMPLETE,
        State::Conflict => CONFLICT,
        State::Absent => ABSENT,
    }
}

fn display_path(home: &Path, path: &Path) -> String {
    path.strip_prefix(home).map_or_else(
        |_| path.display().to_string(),
        |rel| format!("~/{}", rel.to_string_lossy().replace('\\', "/")),
    )
}

// ---- install -------------------------------------------------------------

pub fn install(home: &Path, launch: &LaunchSpec, selected: &[String], dry_run: bool) -> Report {
    let mut report = Report::new("Pulsar Harness Install", dry_run);
    report.line(format!("│  Using {}", launch.display()));
    report.line("│");
    for id in selected {
        let ok = configure_one(&mut report, home, launch, id, dry_run);
        report.ok &= ok;
    }
    if !report.ok {
        report.footer("Incomplete; existing files were preserved, review the entries above.");
    } else if dry_run {
        report.footer("Done. (dry run; re-run without --dry-run to apply)");
    } else {
        report.footer("Done. Restart any running agent for it to pick up the new server.");
    }
    report
}

fn configure_one(
    report: &mut Report,
    home: &Path,
    launch: &LaunchSpec,
    id: &str,
    dry_run: bool,
) -> bool {
    let Some(h) = harness::get(id) else {
        report.detail(
            FAILED,
            format!(
                "unknown harness: {id} (known: {})",
                harness::ids().join(", ")
            ),
        );
        return false;
    };
    report.harness(h.name);
    let ok = configure_harness(report, home, launch, h, dry_run);
    if let Some(note) = (h.note)(home) {
        report.note(&note);
    }
    report.line("│");
    ok
}

fn configure_harness(
    report: &mut Report,
    home: &Path,
    launch: &LaunchSpec,
    h: &Harness,
    dry_run: bool,
) -> bool {
    let path = (h.config_path)(home);
    let inspection = entry::inspect(h.kind, &path, launch);
    match inspection.state {
        State::Active => {
            report.detail(OK, h.active_msg);
            return true;
        }
        State::Conflict => {
            report.detail(
                CONFLICT,
                inspection
                    .detail
                    .unwrap_or_else(|| format!("{} needs manual attention", path.display())),
            );
            return false;
        }
        _ => {}
    }
    if dry_run {
        let msg = match &inspection.detail {
            Some(d) => format!("[dry run] would repair {}: {d}", path.display()),
            None => format!(
                "[dry run] would register the MCP server in {}",
                path.display()
            ),
        };
        report.detail(PENDING, msg);
        return true;
    }
    let repair = inspection.state == State::Incomplete;
    match entry::apply(h.kind, &path, launch) {
        Err(e) => {
            report.detail(FAILED, e.to_string());
            false
        }
        Ok(outcome) => {
            let wrote = outcome.is_some();
            if let Some(o) = outcome {
                let recorded = (if o.created_file {
                    state::record_created_file(home, h.id, &path)
                } else {
                    Ok(())
                })
                .and_then(|()| {
                    if o.created_dirs.is_empty() {
                        Ok(())
                    } else {
                        state::record_created_dirs(home, &o.created_dirs)
                    }
                });
                if let Err(e) = recorded {
                    // The registration is in place; only uninstall's cleanup is affected.
                    report.note(&format!("could not record install state ({e}); uninstall may leave empty files behind"));
                }
            }
            let msg = match (wrote, repair) {
                (false, _) => format!("{} (unchanged)", h.active_msg),
                (true, true) => format!("repaired: {}", h.active_msg),
                (true, false) => h.active_msg.to_string(),
            };
            report.detail(OK, msg);
            true
        }
    }
}

// ---- uninstall -----------------------------------------------------------

/// What `uninstall` would do, one line per harness, for confirmation prompts.
pub fn uninstall_plan(home: &Path, selected: &[String], purge_backups: bool) -> Vec<String> {
    let mut actions = Vec::new();
    for id in selected {
        let Some(h) = harness::get(id) else {
            actions.push(format!("unknown harness: {id}"));
            continue;
        };
        let path = (h.config_path)(home);
        let inspection = entry::inspect_loose(h.kind, &path);
        actions.push(match inspection.state {
            State::Absent => format!("{} - already clear", h.name),
            State::Conflict => format!(
                "{} - left untouched ({})",
                h.name,
                inspection.detail.unwrap_or_else(|| "conflict".into())
            ),
            _ => format!(
                "{} - remove MCP entry from {}",
                h.name,
                display_path(home, &path)
            ),
        });
        let backup = backup_path(&resolve_symlink(&path));
        if purge_backups && backup.is_file() {
            actions.push(format!(
                "{} - delete backup {}",
                h.name,
                display_path(home, &backup)
            ));
        }
    }
    actions
}

pub fn uninstall(home: &Path, selected: &[String], dry_run: bool, purge_backups: bool) -> Report {
    let mut report = Report::new("Pulsar Harness Uninstall", dry_run);
    for id in selected {
        let ok = remove_one(&mut report, home, id, dry_run);
        report.ok &= ok;
    }
    clean_up(home, selected, dry_run, purge_backups);
    if !report.ok {
        report.footer("Incomplete; review the entries above.");
    } else if dry_run {
        report.footer("Done. (dry run; re-run without --dry-run to apply)");
    } else {
        report.footer("Done. Restart any running agent for the change to take effect.");
    }
    report
}

fn remove_one(report: &mut Report, home: &Path, id: &str, dry_run: bool) -> bool {
    let Some(h) = harness::get(id) else {
        report.detail(
            FAILED,
            format!(
                "unknown harness: {id} (known: {})",
                harness::ids().join(", ")
            ),
        );
        return false;
    };
    report.harness(h.name);
    let path = (h.config_path)(home);
    let inspection = entry::inspect_loose(h.kind, &path);
    let ok = match inspection.state {
        State::Absent => {
            report.detail(ABSENT, h.absent_msg);
            true
        }
        State::Conflict => {
            report.detail(
                CONFLICT,
                inspection
                    .detail
                    .unwrap_or_else(|| format!("{} left untouched", path.display())),
            );
            true
        }
        _ if dry_run => {
            report.detail(
                REMOVED,
                format!(
                    "would remove the MCP server entry from {}",
                    display_path(home, &path)
                ),
            );
            true
        }
        _ => match entry::remove(h.kind, &path, false) {
            Err(e) => {
                report.detail(FAILED, e.to_string());
                false
            }
            Ok(_) => {
                delete_if_emptied_and_ours(home, h.id, &path);
                report.detail(
                    REMOVED,
                    format!("removed the MCP server entry from {}", path.display()),
                );
                true
            }
        },
    };
    report.line("│");
    ok
}

/// A config file we created and have now emptied is removed, not left blank.
fn delete_if_emptied_and_ours(home: &Path, harness_id: &str, path: &Path) {
    if !state::was_created_by_install(home, harness_id, path) {
        return;
    }
    let empty = if path.extension().is_some_and(|e| e == "toml") {
        fs::read_to_string(path).map_or(true, |t| t.trim().is_empty())
    } else {
        is_empty_json_config(path)
    };
    if empty {
        let _ = fs::remove_file(path);
    }
}

fn clean_up(home: &Path, selected: &[String], dry_run: bool, purge_backups: bool) {
    if dry_run {
        return;
    }
    let dirs = state::created_dirs(home);
    for id in selected {
        let _ = state::clear_created_files(home, id);
    }
    prune_dirs(&dirs);
    if purge_backups {
        for h in HARNESSES
            .iter()
            .filter(|h| selected.iter().any(|s| s == h.id))
        {
            let _ = fs::remove_file(backup_path(&resolve_symlink(&(h.config_path)(home))));
        }
    }
    prune_dirs(&[state::state_dir(home)]);
}

// ---- status --------------------------------------------------------------

pub struct Status {
    pub rows: Vec<(&'static Harness, Inspection)>,
}

pub fn status(home: &Path) -> Status {
    Status {
        rows: HARNESSES
            .iter()
            .map(|h| (h, entry::inspect_loose(h.kind, &(h.config_path)(home))))
            .collect(),
    }
}

impl Status {
    pub fn active(&self) -> usize {
        self.rows
            .iter()
            .filter(|(_, i)| i.state == State::Active)
            .count()
    }

    pub fn render(&self, home: &Path) -> String {
        let mut report = Report::new("Pulsar Harness Status", false);
        report.line("│  Launch: validated from each registration");
        report.line("│");
        let mut rows: Vec<_> = self.rows.iter().collect();
        rows.sort_by_key(|(_, i)| match i.state {
            State::Active => 0,
            State::Incomplete => 1,
            State::Conflict => 2,
            State::Absent => 3,
        });
        for (h, i) in rows {
            report.harness(h.name);
            let text = match i.state {
                State::Active => h.active_msg.to_string(),
                State::Absent => h.absent_msg.to_string(),
                State::Incomplete => format!(
                    "{} - run `{} install {}`",
                    i.detail.as_deref().unwrap_or("needs repair"),
                    crate::NAME,
                    h.id
                ),
                State::Conflict => i
                    .detail
                    .clone()
                    .unwrap_or_else(|| "needs manual attention".into()),
            };
            report.detail(glyph(i.state), text);
            if let Some(note) = (h.note)(home) {
                report.note(&note);
            }
            report.line("│");
        }
        report.footer(&format!(
            "{}/{} harness integrations active.",
            self.active(),
            self.rows.len()
        ));
        report.render()
    }

    pub fn to_json(&self, home: &Path) -> Value {
        json!({
            "active": self.active(),
            "total": self.rows.len(),
            "harnesses": self.rows.iter().map(|(h, i)| json!({
                "id": h.id,
                "name": h.name,
                "state": i.state.as_str(),
                "config": (h.config_path)(home).display().to_string(),
                "detail": i.detail,
                "note": (h.note)(home),
            })).collect::<Vec<_>>(),
        })
    }
}
