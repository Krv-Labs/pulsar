//! Atomic file writes: a record is either entirely visible or absent.
//!
//! Every write goes to a dot-prefixed temp file in the target directory and
//! is then renamed over the final path. Readers that open the final path
//! never see a partial file, so concurrent readers need no lock. Directory
//! listings skip dot-prefixed names, so in-flight temps are invisible too.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::{io, StoreError};
use crate::handle::random_hex;
use crate::record::SCHEMA_VERSION;

const TEMP_SUFFIX: &str = ".tmp";

fn temp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(".{name}.{}{TEMP_SUFFIX}", random_hex()))
}

fn create_private(path: &Path) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

/// A file fully written under its temp name but not yet renamed into place.
/// [`StagedFile::commit`] is a single rename; dropping it uncommitted
/// removes the temp file, so a cancelled publish leaves nothing behind.
#[derive(Debug)]
pub(crate) struct StagedFile {
    tmp: PathBuf,
    path: PathBuf,
    committed: bool,
}

impl StagedFile {
    /// The final path the file becomes visible at.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn commit(mut self) -> Result<(), StoreError> {
        fs::rename(&self.tmp, &self.path).map_err(io("rename", &self.path))?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for StagedFile {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

/// Write `bytes` to a temp file next to `path` without publishing it. With
/// `durable`, fsync so the record survives power loss once committed, not
/// only a process crash.
pub(crate) fn stage_atomic(
    path: &Path,
    bytes: &[u8],
    durable: bool,
) -> Result<StagedFile, StoreError> {
    let staged = StagedFile {
        tmp: temp_path(path),
        path: path.to_path_buf(),
        committed: false,
    };
    let mut file = create_private(&staged.tmp).map_err(io("create", &staged.tmp))?;
    file.write_all(bytes).map_err(io("write", &staged.tmp))?;
    if durable {
        file.sync_all().map_err(io("fsync", &staged.tmp))?;
    }
    Ok(staged)
}

/// Write `bytes` to `path` atomically (stage, then commit).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8], durable: bool) -> Result<(), StoreError> {
    stage_atomic(path, bytes, durable)?.commit()
}

pub(crate) fn write_json<T: Serialize>(
    path: &Path,
    value: &T,
    durable: bool,
) -> Result<(), StoreError> {
    stage_json(path, value, durable)?.commit()
}

pub(crate) fn stage_json<T: Serialize>(
    path: &Path,
    value: &T,
    durable: bool,
) -> Result<StagedFile, StoreError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| StoreError::CorruptRecord {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    bytes.push(b'\n');
    stage_atomic(path, &bytes, durable)
}

/// Read a JSON record, checking its `schema` field. `Ok(None)` when the
/// file does not exist.
pub(crate) fn read_record<T: DeserializeOwned>(path: &Path) -> Result<Option<T>, StoreError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io("read", path)(e)),
    };
    let corrupt = |message: String| StoreError::CorruptRecord {
        path: path.to_path_buf(),
        message,
    };
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| corrupt(e.to_string()))?;
    let found = value
        .get("schema")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    if found != Some(SCHEMA_VERSION) {
        return Err(StoreError::StaleRecord {
            path: path.to_path_buf(),
            found,
            expected: SCHEMA_VERSION,
        });
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(|e| corrupt(e.to_string()))
}

/// Names of the visible `*.json` records in `dir` (without extension),
/// sorted. Dot-prefixed temp files are skipped.
pub(crate) fn list_json_stems(dir: &Path) -> Result<Vec<String>, StoreError> {
    let mut stems = Vec::new();
    for entry in fs::read_dir(dir).map_err(io("list", dir))? {
        let entry = entry.map_err(io("list", dir))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if let Some(stem) = name.strip_suffix(".json") {
            stems.push(stem.to_owned());
        }
    }
    stems.sort();
    Ok(stems)
}

/// Visible subdirectory names of `dir`, sorted.
pub(crate) fn list_subdirs(dir: &Path) -> Result<Vec<String>, StoreError> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir).map_err(io("list", dir))? {
        let entry = entry.map_err(io("list", dir))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// Remove temp files older than `max_age` that nobody will rename (a crash
/// between create and rename). Returns the count. Housekeeping only: such
/// files are invisible to every lookup.
pub(crate) fn sweep_orphans(dirs: &[&Path], max_age: Duration) -> usize {
    let now = SystemTime::now();
    let mut removed = 0;
    for dir in dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !(name.starts_with('.') && name.ends_with(TEMP_SUFFIX)) {
                continue;
            }
            let old = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|m| now.duration_since(m).ok())
                .is_some_and(|age| age > max_age);
            if old && fs::remove_file(entry.path()).is_ok() {
                removed += 1;
            }
        }
    }
    removed
}

/// Seconds since the Unix epoch, as the Python registry stored it.
pub(crate) fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempRoot;

    #[test]
    fn write_is_atomic_and_leaves_no_temp_files() {
        let root = TempRoot::new("atomic");
        let path = root.path().join("a.json");
        write_atomic(&path, b"{\"schema\":1}", true).unwrap();
        write_atomic(&path, b"{\"schema\":1,\"v\":2}", false).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "{\"schema\":1,\"v\":2}");
        let names: Vec<_> = fs::read_dir(root.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.json"]);
    }

    #[test]
    fn read_record_distinguishes_missing_stale_and_corrupt() {
        let root = TempRoot::new("read-record");
        let missing: Option<serde_json::Value> =
            read_record(&root.path().join("none.json")).unwrap();
        assert!(missing.is_none());

        let stale = root.path().join("stale.json");
        fs::write(&stale, "{\"resolved_threshold\": 0.1}").unwrap();
        let err = read_record::<serde_json::Value>(&stale).unwrap_err();
        assert_eq!(err.code(), "STALE_RECORD");

        let corrupt = root.path().join("corrupt.json");
        fs::write(&corrupt, "{\"schema\": 1, ").unwrap();
        let err = read_record::<serde_json::Value>(&corrupt).unwrap_err();
        assert_eq!(err.code(), "RECORD_CORRUPT");
    }

    #[test]
    fn orphan_sweep_only_touches_old_temp_files() {
        let root = TempRoot::new("orphans");
        let fresh = root.path().join(".x.json.abc.tmp");
        fs::write(&fresh, b"").unwrap();
        fs::write(root.path().join("keep.json"), b"{}").unwrap();
        assert_eq!(sweep_orphans(&[root.path()], Duration::from_secs(3600)), 0);
        assert_eq!(sweep_orphans(&[root.path()], Duration::ZERO), 1);
        assert!(!fresh.exists());
        assert!(root.path().join("keep.json").exists());
    }
}
