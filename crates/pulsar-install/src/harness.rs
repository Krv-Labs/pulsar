//! Supported agent harnesses and where each keeps its MCP registrations.

use std::path::{Path, PathBuf};

use crate::entry::Kind;

pub struct Harness {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: Kind,
    pub config_path: fn(&Path) -> PathBuf,
    pub active_msg: &'static str,
    pub absent_msg: &'static str,
    pub detect: fn(&Path) -> bool,
    pub note: fn(&Path) -> Option<String>,
}

/// `%APPDATA%`, falling back to `<home>/AppData/Roaming`.
pub fn app_data(home: &Path) -> PathBuf {
    std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join("AppData").join("Roaming"))
}

/// The user's home directory from the environment.
pub fn home_dir() -> Result<PathBuf, crate::InstallError> {
    let keys: [&str; 2] = if cfg!(windows) {
        ["USERPROFILE", "HOME"]
    } else {
        ["HOME", "USERPROFILE"]
    };
    keys.iter()
        .filter_map(std::env::var_os)
        .find(|v| !v.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            crate::InstallError::new(
                "cannot resolve the home directory (HOME and USERPROFILE are unset)",
            )
        })
}

fn claude(h: &Path) -> PathBuf {
    h.join(".claude.json")
}
fn codex(h: &Path) -> PathBuf {
    h.join(".codex").join("config.toml")
}
fn gemini(h: &Path) -> PathBuf {
    h.join(".gemini").join("settings.json")
}
fn copilot(h: &Path) -> PathBuf {
    h.join(".copilot").join("mcp-config.json")
}
fn cursor(h: &Path) -> PathBuf {
    h.join(".cursor").join("mcp.json")
}
fn antigravity(h: &Path) -> PathBuf {
    h.join(".gemini").join("config").join("mcp_config.json")
}
fn claude_desktop(h: &Path) -> PathBuf {
    if cfg!(windows) {
        app_data(h)
            .join("Claude")
            .join("claude_desktop_config.json")
    } else if cfg!(target_os = "macos") {
        h.join("Library/Application Support/Claude/claude_desktop_config.json")
    } else {
        h.join(".config/Claude/claude_desktop_config.json")
    }
}
fn vscode(h: &Path) -> PathBuf {
    if cfg!(windows) {
        app_data(h).join("Code").join("User").join("mcp.json")
    } else if cfg!(target_os = "macos") {
        h.join("Library/Application Support/Code/User/mcp.json")
    } else {
        h.join(".config/Code/User/mcp.json")
    }
}

fn parent_is_dir(path: PathBuf) -> bool {
    path.parent().is_some_and(Path::is_dir)
}

fn migration_marker(h: &Path) -> PathBuf {
    h.join(".gemini").join("config").join(".migrated")
}

fn antigravity_data_dirs(h: &Path) -> impl Iterator<Item = PathBuf> + '_ {
    ["antigravity", "antigravity-cli", "antigravity-ide"]
        .into_iter()
        .map(|n| h.join(".gemini").join(n))
}

fn no_note(_: &Path) -> Option<String> {
    None
}

fn antigravity_note(h: &Path) -> Option<String> {
    if migration_marker(h).exists() || !antigravity_data_dirs(h).any(|p| p.is_dir()) {
        return None;
    }
    Some(
        "Antigravity has not migrated MCP config to ~/.gemini/config/ yet; the entry may be overwritten on next Antigravity launch"
            .into(),
    )
}

pub static HARNESSES: [Harness; 8] = [
    Harness {
        id: "claude",
        name: "Claude Code",
        kind: Kind::McpJson,
        config_path: claude,
        active_msg: "MCP server registered in ~/.claude.json",
        absent_msg: "no MCP server entry in ~/.claude.json",
        detect: |h| h.join(".claude").is_dir(),
        note: no_note,
    },
    Harness {
        id: "claude-desktop",
        name: "Claude Desktop",
        kind: Kind::McpJson,
        config_path: claude_desktop,
        active_msg: "MCP server registered in the Claude Desktop config",
        absent_msg: "no MCP server entry in the Claude Desktop config",
        detect: |h| parent_is_dir(claude_desktop(h)),
        note: no_note,
    },
    Harness {
        id: "codex",
        name: "Codex CLI",
        kind: Kind::McpToml,
        config_path: codex,
        active_msg: "[mcp_servers.pulsar] present in ~/.codex/config.toml",
        absent_msg: "no [mcp_servers.pulsar] in ~/.codex/config.toml",
        detect: |h| h.join(".codex").is_dir(),
        note: no_note,
    },
    Harness {
        id: "gemini",
        name: "Gemini CLI",
        kind: Kind::McpJson,
        config_path: gemini,
        active_msg: "MCP server registered in ~/.gemini/settings.json",
        absent_msg: "no MCP server entry in ~/.gemini/settings.json",
        detect: |h| h.join(".gemini").is_dir(),
        note: no_note,
    },
    Harness {
        id: "copilot",
        name: "GitHub Copilot CLI",
        kind: Kind::McpJson,
        config_path: copilot,
        active_msg: "MCP server registered in ~/.copilot/mcp-config.json",
        absent_msg: "no MCP server entry in ~/.copilot/mcp-config.json",
        detect: |h| h.join(".copilot").is_dir(),
        note: no_note,
    },
    Harness {
        id: "cursor",
        name: "Cursor",
        kind: Kind::McpJson,
        config_path: cursor,
        active_msg: "MCP server registered in ~/.cursor/mcp.json",
        absent_msg: "no MCP server entry in ~/.cursor/mcp.json",
        detect: |h| h.join(".cursor").is_dir(),
        note: no_note,
    },
    Harness {
        id: "vscode",
        name: "VS Code",
        kind: Kind::VscodeJsonc,
        config_path: vscode,
        active_msg: "servers.pulsar present in the VS Code user mcp.json",
        absent_msg: "no servers.pulsar in the VS Code user mcp.json",
        detect: |h| parent_is_dir(vscode(h)),
        note: no_note,
    },
    Harness {
        id: "antigravity",
        name: "Google Antigravity",
        kind: Kind::McpJson,
        config_path: antigravity,
        active_msg: "MCP server registered in ~/.gemini/config/mcp_config.json",
        absent_msg: "no MCP server entry in ~/.gemini/config/mcp_config.json",
        detect: |h| migration_marker(h).exists() || antigravity_data_dirs(h).any(|p| p.is_dir()),
        note: antigravity_note,
    },
];

pub fn ids() -> Vec<&'static str> {
    HARNESSES.iter().map(|h| h.id).collect()
}

pub fn get(id: &str) -> Option<&'static Harness> {
    let lowered = id.to_lowercase();
    HARNESSES.iter().find(|h| h.id == lowered)
}

/// Harnesses whose config directory exists on this machine.
pub fn detected(home: &Path) -> Vec<&'static Harness> {
    HARNESSES.iter().filter(|h| (h.detect)(home)).collect()
}
