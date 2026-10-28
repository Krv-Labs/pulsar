//! Read, classify, write and remove the Pulsar entry in one config file.
//!
//! Three file shapes: plain JSON (`mcpServers`), JSON with comments (VS Code's
//! `servers`) and TOML (Codex's `mcp_servers`). We only ever touch an entry we
//! recognize as ours, and refuse to rewrite files whose comments we would drop.

use std::fs;
use std::path::Path;

use serde_json::{json, Map, Value};
use toml_edit::{value, Array, DocumentMut, Item, Table};

use crate::fsops::{
    atomic_write, is_dangling_symlink, read_json_object, read_jsonc_object, write_json_object,
    WriteOutcome,
};
use crate::launch::{drift, entry_args, owns_entry, recorded_drift, LaunchSpec};
use crate::InstallError;

pub const SERVER_KEY: &str = crate::NAME;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    McpJson,
    McpToml,
    VscodeJsonc,
}

impl Kind {
    pub fn container_key(self) -> &'static str {
        match self {
            Self::VscodeJsonc => "servers",
            Self::McpToml => "mcp_servers",
            Self::McpJson => "mcpServers",
        }
    }

    fn wants_stdio_type(self) -> bool {
        self == Self::VscodeJsonc
    }

    fn is_jsonc(self) -> bool {
        self == Self::VscodeJsonc
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Active,
    Incomplete,
    Conflict,
    Absent,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Incomplete => "incomplete",
            Self::Conflict => "conflict",
            Self::Absent => "absent",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inspection {
    pub state: State,
    pub detail: Option<String>,
}

impl Inspection {
    fn plain(state: State) -> Self {
        Self {
            state,
            detail: None,
        }
    }
    fn incomplete(detail: String) -> Self {
        Self {
            state: State::Incomplete,
            detail: Some(detail),
        }
    }
    fn conflict(detail: String) -> Self {
        Self {
            state: State::Conflict,
            detail: Some(detail),
        }
    }
}

fn foreign(path: &Path) -> String {
    format!(
        "`{SERVER_KEY}` in {} is an MCP entry {SERVER_KEY} did not write; inspect it by hand",
        path.display()
    )
}

// ---- JSON ----------------------------------------------------------------

fn read_json(kind: Kind, path: &Path) -> Result<(Map<String, Value>, bool), InstallError> {
    if kind.is_jsonc() {
        read_jsonc_object(path)
    } else {
        Ok((read_json_object(path)?, false))
    }
}

fn json_entry(kind: Kind, data: &Map<String, Value>) -> Option<&Map<String, Value>> {
    data.get(kind.container_key())?
        .as_object()?
        .get(SERVER_KEY)?
        .as_object()
}

fn command_of(entry: &Map<String, Value>) -> &str {
    entry.get("command").and_then(Value::as_str).unwrap_or("")
}

fn fresh_json_entry(kind: Kind, launch: &LaunchSpec) -> Map<String, Value> {
    let mut entry = Map::new();
    if kind.wants_stdio_type() {
        entry.insert("type".into(), json!("stdio"));
    }
    entry.insert("command".into(), json!(launch.command_str()));
    entry.insert("args".into(), json!(launch.args));
    entry
}

fn classify_json(
    kind: Kind,
    data: &Map<String, Value>,
    path: &Path,
    launch: Option<&LaunchSpec>,
) -> Inspection {
    let Some(entry) = json_entry(kind, data) else {
        return Inspection::plain(State::Absent);
    };
    let command = command_of(entry);
    let args = entry_args(entry.get("args"));
    if !owns_entry(command, args.as_deref()) {
        return Inspection::conflict(foreign(path));
    }
    let reason = match launch {
        Some(l) => drift(command, args.as_deref(), l),
        None => recorded_drift(command),
    };
    if let Some(reason) = reason {
        return Inspection::incomplete(reason);
    }
    if kind.wants_stdio_type() && entry.get("type").and_then(Value::as_str) != Some("stdio") {
        return Inspection::incomplete(format!(
            "`{SERVER_KEY}` in {} is missing \"type\": \"stdio\"",
            path.display()
        ));
    }
    Inspection::plain(State::Active)
}

fn inspect_json(kind: Kind, path: &Path, launch: Option<&LaunchSpec>) -> Inspection {
    let (data, comments) = match read_json(kind, path) {
        Ok(v) => v,
        Err(e) => return Inspection::conflict(e.to_string()),
    };
    let found = classify_json(kind, &data, path, launch);
    if found.state == State::Active || found.state == State::Conflict {
        return found;
    }
    if comments {
        let hint = match (found.state, launch) {
            (State::Absent, None) => return Inspection::plain(State::Absent),
            (_, Some(l)) => format!(
                "{} contains comments, which {SERVER_KEY} will not rewrite; add \"{SERVER_KEY}\": {} to its \"{}\" object by hand",
                path.display(),
                Value::Object(fresh_json_entry(kind, l)),
                kind.container_key()
            ),
            _ => format!("{} contains comments; remove its `{SERVER_KEY}` entry by hand", path.display()),
        };
        return Inspection::conflict(hint);
    }
    found
}

fn write_json(kind: Kind, path: &Path, launch: &LaunchSpec) -> Result<WriteOutcome, InstallError> {
    let (mut data, _) = read_json(kind, path)?;
    let key = kind.container_key();
    let container = data.entry(key).or_insert_with(|| json!({}));
    let container = container.as_object_mut().ok_or_else(|| {
        InstallError::new(format!("{} `{key}` must be an object", path.display()))
    })?;
    let entry = container.entry(SERVER_KEY).or_insert_with(|| json!({}));
    let entry = entry.as_object_mut().ok_or_else(|| {
        InstallError::new(format!(
            "{} `{SERVER_KEY}` must be an object",
            path.display()
        ))
    })?;
    for (k, v) in fresh_json_entry(kind, launch) {
        entry.insert(k, v);
    }
    write_json_object(path, &data, true)
}

fn remove_json(kind: Kind, path: &Path, dry_run: bool) -> Result<bool, InstallError> {
    if !path.is_file() {
        return Ok(false);
    }
    let Ok((mut data, comments)) = read_json(kind, path) else {
        return Ok(false);
    };
    let removable = json_entry(kind, &data)
        .is_some_and(|e| owns_entry(command_of(e), entry_args(e.get("args")).as_deref()));
    if !removable {
        return Ok(false);
    }
    if comments {
        return Err(InstallError::new(format!(
            "{} contains comments; remove its `{SERVER_KEY}` entry by hand",
            path.display()
        )));
    }
    let key = kind.container_key();
    let Some(container) = data.get_mut(key).and_then(Value::as_object_mut) else {
        return Ok(false);
    };
    container.shift_remove(SERVER_KEY);
    if container.is_empty() {
        data.shift_remove(key);
    }
    if !dry_run {
        write_json_object(path, &data, false)?;
    }
    Ok(true)
}

// ---- TOML ----------------------------------------------------------------

fn parse_toml(path: &Path) -> Result<Option<DocumentMut>, InstallError> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|e| InstallError::io("reading", path, &e))?;
    text.parse::<DocumentMut>()
        .map(Some)
        .map_err(|e| InstallError::new(format!("parsing {}: {e}", path.display())))
}

fn toml_entry(kind: Kind, doc: &DocumentMut) -> Option<&Table> {
    doc.get(kind.container_key())?
        .as_table_like()?
        .get(SERVER_KEY)?
        .as_table()
}

fn toml_command_args(entry: &Table) -> (String, Option<Vec<String>>) {
    let command = entry
        .get("command")
        .and_then(Item::as_str)
        .unwrap_or("")
        .to_string();
    let args = entry.get("args").and_then(Item::as_array).and_then(|a| {
        a.iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
    });
    (command, args)
}

fn inspect_toml(kind: Kind, path: &Path, launch: Option<&LaunchSpec>) -> Inspection {
    let doc = match parse_toml(path) {
        Ok(Some(d)) => d,
        Ok(None) => return Inspection::plain(State::Absent),
        Err(e) => return Inspection::conflict(e.to_string()),
    };
    let Some(entry) = toml_entry(kind, &doc) else {
        return Inspection::plain(State::Absent);
    };
    let (command, args) = toml_command_args(entry);
    if !owns_entry(&command, args.as_deref()) {
        return Inspection::conflict(format!(
            "[mcp_servers.{SERVER_KEY}] in {} is an entry {SERVER_KEY} did not write; inspect it by hand",
            path.display()
        ));
    }
    let reason = match launch {
        Some(l) => drift(&command, args.as_deref(), l),
        None => recorded_drift(&command),
    };
    reason.map_or(Inspection::plain(State::Active), Inspection::incomplete)
}

fn write_toml(kind: Kind, path: &Path, launch: &LaunchSpec) -> Result<WriteOutcome, InstallError> {
    let mut doc = parse_toml(path)?.unwrap_or_default();
    let key = kind.container_key();
    if doc.get(key).is_none() {
        let mut servers = Table::new();
        servers.set_implicit(true);
        doc.insert(key, Item::Table(servers));
    }
    let servers = doc[key]
        .as_table_mut()
        .ok_or_else(|| InstallError::new(format!("{} `{key}` must be a table", path.display())))?;
    let mut entry = Table::new();
    entry["command"] = value(launch.command_str());
    entry["args"] = value(launch.args.iter().collect::<Array>());
    servers.insert(SERVER_KEY, Item::Table(entry));
    let mut contents = doc.to_string();
    if !contents.ends_with('\n') {
        contents.push('\n');
    }
    atomic_write(path, &contents, true)
}

fn remove_toml(kind: Kind, path: &Path, dry_run: bool) -> Result<bool, InstallError> {
    let Ok(Some(mut doc)) = parse_toml(path) else {
        return Ok(false);
    };
    let key = kind.container_key();
    let owned = toml_entry(kind, &doc).is_some_and(|e| {
        let (command, args) = toml_command_args(e);
        owns_entry(&command, args.as_deref())
    });
    if !owned {
        return Ok(false);
    }
    let servers = doc[key].as_table_mut().expect("checked above");
    servers.remove(SERVER_KEY);
    if servers.is_empty() {
        doc.remove(key);
    }
    if !dry_run {
        let mut contents = doc.to_string();
        if !contents.is_empty() && !contents.ends_with('\n') {
            contents.push('\n');
        }
        atomic_write(path, &contents, false)?;
    }
    Ok(true)
}

// ---- public --------------------------------------------------------------

/// Classify the entry against the launch we would write now.
pub fn inspect(kind: Kind, path: &Path, launch: &LaunchSpec) -> Inspection {
    inspect_with(kind, path, Some(launch))
}

/// Classify the entry without an expected launch: is it one of ours and does
/// the recorded command still run? Used by status and uninstall.
pub fn inspect_loose(kind: Kind, path: &Path) -> Inspection {
    inspect_with(kind, path, None)
}

fn inspect_with(kind: Kind, path: &Path, launch: Option<&LaunchSpec>) -> Inspection {
    if is_dangling_symlink(path) {
        return Inspection::conflict(format!("{} is a dangling symlink", path.display()));
    }
    match kind {
        Kind::McpToml => inspect_toml(kind, path, launch),
        _ => inspect_json(kind, path, launch),
    }
}

/// Register or repair the entry. `None` means it was already correct.
pub fn apply(
    kind: Kind,
    path: &Path,
    launch: &LaunchSpec,
) -> Result<Option<WriteOutcome>, InstallError> {
    let inspection = inspect(kind, path, launch);
    match inspection.state {
        State::Active => Ok(None),
        State::Conflict => {
            Err(InstallError::new(inspection.detail.unwrap_or_else(|| {
                format!("{} cannot be updated safely", path.display())
            })))
        }
        // Back up on repair too: a drifted entry may have been hand-written.
        _ => match kind {
            Kind::McpToml => write_toml(kind, path, launch).map(Some),
            _ => write_json(kind, path, launch).map(Some),
        },
    }
}

/// Remove our entry. `Ok(false)` when there was nothing of ours to remove.
pub fn remove(kind: Kind, path: &Path, dry_run: bool) -> Result<bool, InstallError> {
    match kind {
        Kind::McpToml => remove_toml(kind, path, dry_run),
        _ => remove_json(kind, path, dry_run),
    }
}
