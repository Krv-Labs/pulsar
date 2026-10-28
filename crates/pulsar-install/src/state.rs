//! Install ownership ledger: which config files and directories *we* created,
//! so uninstall can leave no trace without touching anything the user owns.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::fsops::{read_json_object, write_json_object};
use crate::harness::app_data;
use crate::InstallError;

const HARNESSES_KEY: &str = "harnesses";
const CREATED_FILES_KEY: &str = "createdFiles";
const CREATED_DIRS_KEY: &str = "createdDirs";
const STATE_FILE: &str = "install.json";
/// Shared roots we never record for pruning, even if we had to create them.
const NEVER_PRUNE: [&str; 3] = [".local", ".config", "Library"];

pub fn state_dir(home: &Path) -> PathBuf {
    if cfg!(windows) {
        app_data(home).join(crate::NAME)
    } else {
        home.join(".local").join("state").join(crate::NAME)
    }
}

fn state_file(home: &Path) -> PathBuf {
    state_dir(home).join(STATE_FILE)
}

struct Ledger {
    harnesses: Map<String, Value>,
    dirs: Vec<String>,
}

fn load(home: &Path) -> Ledger {
    let data = read_json_object(&state_file(home)).unwrap_or_default();
    Ledger {
        harnesses: data
            .get(HARNESSES_KEY)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
        dirs: data
            .get(CREATED_DIRS_KEY)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn save(home: &Path, ledger: &Ledger) -> Result<(), InstallError> {
    let mut data = Map::new();
    data.insert(
        HARNESSES_KEY.into(),
        Value::Object(ledger.harnesses.clone()),
    );
    data.insert(CREATED_DIRS_KEY.into(), json!(ledger.dirs));
    let outcome = write_json_object(&state_file(home), &data, false)?;

    // Directories made just to hold this ledger are ours too. Record them
    // (the shared-root skip is for config parents, not these) so uninstall
    // can prune them; `remove_dir` only ever removes empty directories.
    let new: Vec<String> = outcome
        .created_dirs
        .iter()
        .filter(|d| d.as_path() != home)
        .map(|d| d.to_string_lossy().into_owned())
        .filter(|d| !ledger.dirs.contains(d))
        .collect();
    if new.is_empty() {
        return Ok(());
    }
    let mut dirs = ledger.dirs.clone();
    dirs.extend(new);
    data.insert(CREATED_DIRS_KEY.into(), json!(dirs));
    write_json_object(&state_file(home), &data, false).map(|_| ())
}

fn files_of(ledger: &Ledger, harness: &str) -> Vec<String> {
    ledger
        .harnesses
        .get(harness)
        .and_then(|r| r.get(CREATED_FILES_KEY))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

pub fn record_created_file(home: &Path, harness: &str, path: &Path) -> Result<(), InstallError> {
    let mut ledger = load(home);
    let mut files = files_of(&ledger, harness);
    let key = path.to_string_lossy().into_owned();
    if !files.contains(&key) {
        files.push(key);
    }
    ledger
        .harnesses
        .insert(harness.into(), json!({ CREATED_FILES_KEY: files }));
    save(home, &ledger)
}

pub fn was_created_by_install(home: &Path, harness: &str, path: &Path) -> bool {
    files_of(&load(home), harness).contains(&path.to_string_lossy().into_owned())
}

pub fn record_created_dirs(home: &Path, dirs: &[PathBuf]) -> Result<(), InstallError> {
    let mut ledger = load(home);
    for dir in dirs {
        let key = dir.to_string_lossy().into_owned();
        let shared = dir == home
            || dir
                .file_name()
                .is_some_and(|n| NEVER_PRUNE.contains(&n.to_string_lossy().as_ref()));
        if !shared && !ledger.dirs.contains(&key) {
            ledger.dirs.push(key);
        }
    }
    save(home, &ledger)
}

pub fn created_dirs(home: &Path) -> Vec<PathBuf> {
    load(home).dirs.into_iter().map(PathBuf::from).collect()
}

/// Forget a harness's created files; delete the ledger when none remain.
pub fn clear_created_files(home: &Path, harness: &str) -> Result<(), InstallError> {
    let mut ledger = load(home);
    ledger.harnesses.remove(harness);
    if !ledger.harnesses.is_empty() {
        return save(home, &ledger);
    }
    let _ = std::fs::remove_file(state_file(home));
    let _ = std::fs::remove_dir(state_dir(home));
    Ok(())
}
