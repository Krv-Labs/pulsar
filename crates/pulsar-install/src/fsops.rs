//! Filesystem primitives: atomic writes with backup, JSON/JSONC reading,
//! created-directory tracking and pruning.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::InstallError;

/// Suffix of the one backup kept per edited config (never replaced once made).
pub const BACKUP_SUFFIX: &str = ".pulsar.backup";
const TMP_SUFFIX: &str = ".pulsar.tmp";

#[derive(Debug, Default)]
pub struct WriteOutcome {
    pub created_dirs: Vec<PathBuf>,
    pub created_file: bool,
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

pub fn backup_path(path: &Path) -> PathBuf {
    with_suffix(path, BACKUP_SUFFIX)
}

pub fn is_dangling_symlink(path: &Path) -> bool {
    path.symlink_metadata()
        .is_ok_and(|m| m.file_type().is_symlink())
        && !path.exists()
}

/// Follow a symlink so we edit the real file and leave the link in place.
pub fn resolve_symlink(path: &Path) -> PathBuf {
    if path
        .symlink_metadata()
        .is_ok_and(|m| m.file_type().is_symlink())
    {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    } else {
        path.to_path_buf()
    }
}

fn create_parents(path: &Path) -> Result<Vec<PathBuf>, InstallError> {
    let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return Ok(vec![]);
    };
    let mut missing = Vec::new();
    let mut current = parent;
    while !current.exists() {
        missing.push(current.to_path_buf());
        match current.parent() {
            Some(p) if p != current => current = p,
            _ => break,
        }
    }
    missing.reverse();
    for dir in &missing {
        if let Err(e) = fs::create_dir(dir) {
            if !dir.is_dir() {
                return Err(InstallError::io("creating", dir, &e));
            }
        }
    }
    Ok(missing)
}

/// Write `contents` to `path` through a temp file in the same directory so a
/// crash never leaves a half-written config. Optionally backs up the contents
/// from before the first rewrite (an existing backup is never replaced, so it
/// stays the user's original), and preserves file permissions.
pub fn atomic_write(
    path: &Path,
    contents: &str,
    backup: bool,
) -> Result<WriteOutcome, InstallError> {
    if is_dangling_symlink(path) {
        return Err(InstallError::new(format!(
            "{} is a dangling symlink; refusing to replace it",
            path.display()
        )));
    }
    let created_dirs = create_parents(path)?;
    let target = resolve_symlink(path);
    let created_file = !target.exists();
    let backup_file = backup_path(&target);
    if backup && !created_file && !backup_file.exists() {
        fs::copy(&target, &backup_file).map_err(|e| InstallError::io("backing up", &target, &e))?;
    }
    let tmp = with_suffix(&target, TMP_SUFFIX);
    fs::write(&tmp, contents).map_err(|e| InstallError::io("writing", &tmp, &e))?;
    if !created_file {
        if let Ok(meta) = fs::metadata(&target) {
            let _ = fs::set_permissions(&tmp, meta.permissions());
        }
    }
    fs::rename(&tmp, &target).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        InstallError::io("replacing", &target, &e)
    })?;
    Ok(WriteOutcome {
        created_dirs,
        created_file,
    })
}

fn read_optional(path: &Path) -> Result<Option<String>, InstallError> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|e| InstallError::io("reading", path, &e))?;
    Ok((!text.trim().is_empty()).then_some(text))
}

fn parse_object(path: &Path, text: &str) -> Result<Map<String, Value>, InstallError> {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(InstallError::new(format!(
            "{} top-level value must be an object",
            path.display()
        ))),
        Err(e) => Err(InstallError::new(format!(
            "parsing {}: {e}",
            path.display()
        ))),
    }
}

/// A JSON object from `path`; a missing or blank file is an empty object.
pub fn read_json_object(path: &Path) -> Result<Map<String, Value>, InstallError> {
    match read_optional(path)? {
        None => Ok(Map::new()),
        Some(text) => parse_object(path, &text),
    }
}

/// Like [`read_json_object`] for JSONC; also reports whether comments were
/// present (we refuse to rewrite those files, since rewriting drops them).
pub fn read_jsonc_object(path: &Path) -> Result<(Map<String, Value>, bool), InstallError> {
    match read_optional(path)? {
        None => Ok((Map::new(), false)),
        Some(text) => {
            let (stripped, had_comments) = strip_jsonc(&text);
            Ok((parse_object(path, &stripped)?, had_comments))
        }
    }
}

pub fn write_json_object(
    path: &Path,
    data: &Map<String, Value>,
    backup: bool,
) -> Result<WriteOutcome, InstallError> {
    let mut contents = serde_json::to_string_pretty(data).expect("a JSON object serializes");
    contents.push('\n');
    atomic_write(path, &contents, backup)
}

/// Strip comments and trailing commas without touching string contents.
/// Returns the cleaned text and whether any comment was present.
pub fn strip_jsonc(text: &str) -> (String, bool) {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut commas: Vec<usize> = Vec::new();
    let (mut i, mut had_comments, mut in_string, mut escape) = (0, false, false, false);
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match (c, chars.get(i + 1)) {
            ('"', _) => in_string = true,
            ('/', Some('/')) => {
                let end = chars[i..]
                    .iter()
                    .position(|&x| x == '\n')
                    .map_or(chars.len(), |p| i + p);
                out.extend(std::iter::repeat_n(' ', end - i));
                i = end;
                had_comments = true;
                continue;
            }
            ('/', Some('*')) => {
                let end = (i + 2..chars.len().saturating_sub(1))
                    .find(|&k| chars[k] == '*' && chars[k + 1] == '/')
                    .map_or(chars.len(), |k| k + 2);
                out.extend(
                    chars[i..end]
                        .iter()
                        .map(|&x| if x == '\n' { '\n' } else { ' ' }),
                );
                i = end;
                had_comments = true;
                continue;
            }
            (',', _) => commas.push(out.len()),
            _ => {}
        }
        out.push(c);
        i += 1;
    }
    // Blank commas that precede a closing brace/bracket. Only positions
    // collected outside string literals are considered.
    for start in commas {
        let next = out[start + 1..].iter().position(|c| !c.is_whitespace());
        if next.is_some_and(|n| matches!(out[start + 1 + n], '}' | ']')) {
            out[start] = ' ';
        }
    }
    (out.into_iter().collect(), had_comments)
}

/// Remove directories we created, deepest first; non-empty ones stay.
pub fn prune_dirs(dirs: &[PathBuf]) {
    for dir in dirs.iter().rev() {
        let _ = fs::remove_dir(dir);
    }
}

/// True when a JSON config holds nothing (empty object or blank file).
pub fn is_empty_json_config(path: &Path) -> bool {
    match read_json_object(path) {
        Ok(map) => map.is_empty(),
        Err(_) => fs::read_to_string(path).is_ok_and(|t| t.trim().is_empty()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonc_comments_and_trailing_commas_are_stripped_outside_strings_only() {
        let src =
            "{\n // line\n \"a\": \"x // not a comment, }\", /* block\n */ \"b\": [1, 2,],\n}\n";
        let (clean, had) = strip_jsonc(src);
        assert!(had);
        let v: Value = serde_json::from_str(&clean).unwrap();
        assert_eq!(v["a"], "x // not a comment, }");
        assert_eq!(v["b"], serde_json::json!([1, 2]));
        assert!(!strip_jsonc("{\"a\": 1}").1);
    }

    #[test]
    fn atomic_write_backs_up_creates_dirs_and_reports_what_it_made() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b/config.json");
        let first = atomic_write(&path, "one", true).unwrap();
        assert!(first.created_file);
        assert_eq!(
            first.created_dirs,
            [dir.path().join("a"), dir.path().join("a/b")]
        );
        assert!(!backup_path(&path).exists(), "nothing to back up on create");
        let second = atomic_write(&path, "two", true).unwrap();
        assert!(!second.created_file && second.created_dirs.is_empty());
        assert_eq!(fs::read_to_string(&path).unwrap(), "two");
        assert_eq!(fs::read_to_string(backup_path(&path)).unwrap(), "one");
        atomic_write(&path, "three", true).unwrap();
        assert_eq!(
            fs::read_to_string(backup_path(&path)).unwrap(),
            "one",
            "the first backup is kept"
        );
        assert!(
            !with_suffix(&path, TMP_SUFFIX).exists(),
            "no temp file left behind"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_configs_are_edited_in_place_and_dangling_ones_refused() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.json");
        fs::write(&real, "{}").unwrap();
        let link = dir.path().join("link.json");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        atomic_write(&link, "{\"x\":1}", false).unwrap();
        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
        assert_eq!(fs::read_to_string(&real).unwrap(), "{\"x\":1}");

        let dangling = dir.path().join("dangling.json");
        std::os::unix::fs::symlink(dir.path().join("missing"), &dangling).unwrap();
        let err = atomic_write(&dangling, "x", false).unwrap_err();
        assert!(err.to_string().contains("dangling symlink"), "{err}");
    }

    #[test]
    fn json_objects_keep_key_order_and_reject_non_objects() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.json");
        fs::write(&p, "{\"z\":1,\"a\":2}").unwrap();
        let map = read_json_object(&p).unwrap();
        write_json_object(&p, &map, false).unwrap();
        let text = fs::read_to_string(&p).unwrap();
        assert!(
            text.find("\"z\"").unwrap() < text.find("\"a\"").unwrap(),
            "{text}"
        );
        fs::write(&p, "[1]").unwrap();
        assert!(read_json_object(&p)
            .unwrap_err()
            .to_string()
            .contains("must be an object"));
        fs::write(&p, "  \n").unwrap();
        assert!(read_json_object(&p).unwrap().is_empty());
    }
}
