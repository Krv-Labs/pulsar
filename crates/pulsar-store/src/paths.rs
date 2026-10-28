//! Where the store lives and how it is laid out.

use std::path::{Path, PathBuf};

use crate::error::StoreError;

/// Environment override for the store root.
pub const ENV_ROOT: &str = "PULSAR_STORE_DIR";
/// Directory name under the platform data dir.
pub const APP_DIR: &str = "pulsar";

/// The default store root.
///
/// - `$PULSAR_STORE_DIR` when set.
/// - Unix (including macOS): `$XDG_DATA_HOME/pulsar`, else
///   `$HOME/.local/share/pulsar`.
/// - Windows: `%LOCALAPPDATA%\pulsar`, else `%USERPROFILE%\AppData\Local\pulsar`.
pub fn default_root() -> Result<PathBuf, StoreError> {
    if let Some(dir) = env_path(ENV_ROOT) {
        return Ok(dir);
    }
    if cfg!(windows) {
        if let Some(dir) = env_path("LOCALAPPDATA") {
            return Ok(dir.join(APP_DIR));
        }
        return env_path("USERPROFILE")
            .map(|home| home.join("AppData").join("Local").join(APP_DIR))
            .ok_or(StoreError::NoHome);
    }
    if let Some(dir) = env_path("XDG_DATA_HOME").filter(|p| p.is_absolute()) {
        return Ok(dir.join(APP_DIR));
    }
    env_path("HOME")
        .map(|home| home.join(".local").join("share").join(APP_DIR))
        .ok_or(StoreError::NoHome)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// The directories under one root.
#[derive(Debug, Clone)]
pub(crate) struct Dirs {
    pub root: PathBuf,
    pub datasets: PathBuf,
    pub dataset_files: PathBuf,
    pub uploads: PathBuf,
    pub runs: PathBuf,
    pub jobs: PathBuf,
    pub cluster_assignments: PathBuf,
    pub longitudinal: PathBuf,
    pub plans: PathBuf,
}

impl Dirs {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            datasets: root.join("datasets"),
            dataset_files: root.join("datasets").join("files"),
            uploads: root.join("uploads"),
            runs: root.join("runs"),
            jobs: root.join("jobs"),
            cluster_assignments: root.join("cluster_assignments"),
            longitudinal: root.join("longitudinal"),
            plans: root.join("plans"),
        }
    }

    pub fn all(&self) -> [&Path; 8] {
        [
            &self.datasets,
            &self.dataset_files,
            &self.uploads,
            &self.runs,
            &self.jobs,
            &self.cluster_assignments,
            &self.longitudinal,
            &self.plans,
        ]
    }

    pub fn run_dir(&self, run_id: &str) -> PathBuf {
        self.runs.join(run_id)
    }
}
