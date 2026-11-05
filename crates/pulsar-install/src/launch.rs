//! The command recorded in agent configs, and what counts as "ours".

use std::path::{Path, PathBuf};

use crate::InstallError;

/// The npm package recorded in `npx` mode (see [`crate::NAME`]).
pub const PACKAGE: &str = "@krv-labs/pulsar";
const BIN: &str = crate::NAME;
/// Launch shapes written by the Python v0.5/v0.6 installer; still recognized
/// as Pulsar's own so an upgrade repairs them instead of refusing to touch them.
const LEGACY_PACKAGE: &str = "thema-pulsar[mcp]";
const LEGACY_SERVER: &str = "pulsar-mcp";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Record the absolute path of a `pulsar` executable.
    Binary,
    /// Record `npx -y @krv-labs/pulsar mcp` (survives npx cache cleanups).
    Npx,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Self, InstallError> {
        match s {
            "binary" => Ok(Self::Binary),
            "npx" => Ok(Self::Npx),
            other => Err(InstallError::new(format!(
                "unknown launch mode {other:?} (supported: binary, npx)"
            ))),
        }
    }
}

/// Absolute command and args written into agent MCP configs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchSpec {
    pub command: PathBuf,
    pub args: Vec<String>,
    pub mode: Mode,
}

impl LaunchSpec {
    pub fn command_str(&self) -> String {
        self.command.to_string_lossy().into_owned()
    }

    pub fn display(&self) -> String {
        std::iter::once(self.command_str())
            .chain(self.args.iter().cloned())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Binary mode unless the executable lives in a `node_modules` tree (an npx
/// or package-manager cache), where its path is not stable.
pub fn default_mode(exe: &Path) -> Mode {
    if exe.components().any(|c| c.as_os_str() == "node_modules") {
        Mode::Npx
    } else {
        Mode::Binary
    }
}

/// Find an executable on `PATH` (honouring `PATHEXT` on Windows).
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT".into())
            .split(';')
            .map(str::to_string)
            .collect()
    } else {
        vec![String::new()]
    };
    std::env::split_paths(&path).find_map(|dir| {
        exts.iter().find_map(|ext| {
            let candidate = dir.join(format!("{name}{ext}"));
            candidate.is_file().then_some(candidate)
        })
    })
}

/// Absolute but not symlink-resolved: a shim such as `/usr/local/bin/pulsar`
/// must stay the recorded path so it keeps working across upgrades.
fn absolute(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir().map_or(path.clone(), |cwd| cwd.join(path))
    }
}

/// Resolve what to record. `exe` is the running `pulsar` executable and
/// `find` looks a command up on `PATH` (injected for tests).
pub fn resolve(
    mode: Mode,
    pin_version: Option<&str>,
    exe: &Path,
    find: &dyn Fn(&str) -> Option<PathBuf>,
) -> Result<LaunchSpec, InstallError> {
    match mode {
        Mode::Binary => Ok(LaunchSpec {
            command: absolute(exe.to_path_buf()),
            args: vec!["mcp".into()],
            mode,
        }),
        Mode::Npx => {
            let npx = find("npx").ok_or_else(|| {
                InstallError::new(
                    "npx is not on PATH; install Node.js (https://nodejs.org) or use --mode binary",
                )
            })?;
            let package = match pin_version {
                Some(v) => format!("{PACKAGE}@{v}"),
                None => PACKAGE.to_string(),
            };
            Ok(LaunchSpec {
                command: absolute(npx),
                args: vec!["-y".into(), package, "mcp".into()],
                mode,
            })
        }
    }
}

/// Last path component, splitting on both separators so a Windows path is
/// read the same on any host (synced or shared configs).
fn file_name(command: &str) -> &str {
    command.rsplit(['/', '\\']).next().unwrap_or("")
}

fn is_named(command: &str, name: &str) -> bool {
    let n = file_name(command);
    n == name || n.strip_suffix(".exe") == Some(name) || n.strip_suffix(".cmd") == Some(name)
}

/// Normalize a config `args` value to strings, or `None` when it is not a
/// list of strings.
pub fn entry_args(value: Option<&serde_json::Value>) -> Option<Vec<String>> {
    value?
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect()
}

/// True when an entry is recognizably a Pulsar MCP launch.
///
/// Deliberately loose: it matches the *shape* of a launch we wrote, not the
/// exact arguments. Exact matching would classify an entry from another mode,
/// a pinned entry or one from an older release as someone else's config and
/// refuse to touch it while uninstall removed it happily. Argument
/// differences are drift, and drift is repairable.
pub fn owns_entry(command: &str, args: Option<&[String]>) -> bool {
    let args = args.unwrap_or(&[]);
    if is_named(command, BIN) {
        return args == ["mcp"];
    }
    if is_named(command, "npx") {
        return matches!(args, [y, pkg, m]
            if y == "-y"
                && m == "mcp"
                && (pkg == PACKAGE || pkg.starts_with(&format!("{PACKAGE}@"))));
    }
    // Python-era shapes.
    if is_named(command, LEGACY_SERVER) {
        return args.is_empty();
    }
    if is_named(command, "uvx") {
        return matches!(args, [from, pkg, server]
            if from == "--from"
                && pkg.split("==").next() == Some(LEGACY_PACKAGE)
                && server == LEGACY_SERVER);
    }
    false
}

pub fn recorded_drift(recorded: &str) -> Option<String> {
    let path = Path::new(recorded);
    if !path.is_absolute() {
        return Some(format!("`{recorded}` is not an absolute path"));
    }
    if !path.exists() {
        return Some(format!("`{recorded}` no longer exists"));
    }
    if !path.is_file() {
        return Some(format!("`{recorded}` is not a file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if path
            .metadata()
            .is_ok_and(|m| m.permissions().mode() & 0o100 == 0)
        {
            return Some(format!("`{recorded}` is not executable"));
        }
    }
    None
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

pub fn drift(recorded: &str, args: Option<&[String]>, expected: &LaunchSpec) -> Option<String> {
    if let Some(reason) = recorded_drift(recorded) {
        return Some(reason);
    }
    if !same_file(Path::new(recorded), &expected.command) {
        return Some(format!(
            "`{recorded}` is a different binary than the expected {}",
            expected.command.display()
        ));
    }
    if args.unwrap_or(&[]) != expected.args.as_slice() {
        let want = if expected.args.is_empty() {
            "(none)".to_string()
        } else {
            expected.args.join(" ")
        };
        return Some(format!("recorded args differ from the expected `{want}`"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn owns_every_launch_shape_we_write_or_wrote() {
        let own = |c: &str, a: &[&str]| owns_entry(c, Some(&s(a)));
        assert!(own("/usr/local/bin/pulsar", &["mcp"]));
        assert!(own(r"C:\x\pulsar.exe", &["mcp"]));
        assert!(own("/usr/bin/npx", &["-y", "@krv-labs/pulsar", "mcp"]));
        assert!(own(
            "/usr/bin/npx",
            &["-y", "@krv-labs/pulsar@0.1.0", "mcp"]
        ));
        // Python-era entries are ours, so an upgrade repairs them.
        assert!(own(
            "/u/uvx",
            &["--from", "thema-pulsar[mcp]", "pulsar-mcp"]
        ));
        assert!(own(
            "/u/uvx",
            &["--from", "thema-pulsar[mcp]==0.5.0", "pulsar-mcp"]
        ));
        assert!(owns_entry("/u/pulsar-mcp", None));
    }

    #[test]
    fn does_not_own_other_peoples_entries() {
        let own = |c: &str, a: &[&str]| owns_entry(c, Some(&s(a)));
        assert!(!own("/usr/bin/pulsar", &["serve"]));
        assert!(!own("/usr/bin/pulsar", &[]));
        assert!(!own("/usr/bin/npx", &["-y", "@other/pulsar", "mcp"]));
        assert!(!own(
            "/usr/bin/npx",
            &["-y", "@krv-labs/pulsar-extra", "mcp"]
        ));
        assert!(!own("/usr/bin/node", &["server.js"]));
        assert!(!own("/u/uvx", &["--from", "other", "pulsar-mcp"]));
        assert!(!owns_entry("/usr/bin/pulsar", None));
    }

    #[test]
    fn mode_defaults_to_npx_inside_node_modules() {
        assert_eq!(
            default_mode(Path::new("/usr/local/bin/pulsar")),
            Mode::Binary
        );
        assert_eq!(
            default_mode(Path::new(
                "/h/.npm/_npx/x/node_modules/@krv-labs/pulsar-darwin-arm64/bin/pulsar"
            )),
            Mode::Npx
        );
    }

    #[test]
    fn resolve_reports_a_missing_npx_with_an_action() {
        let err = resolve(Mode::Npx, None, Path::new("/x/pulsar"), &|_| None).unwrap_err();
        assert!(err.to_string().contains("npx is not on PATH"), "{err}");
        assert!(Mode::parse("uvx")
            .unwrap_err()
            .to_string()
            .contains("supported: binary, npx"));
    }

    #[test]
    fn resolve_pins_the_version_only_when_asked() {
        let npx = || Some(PathBuf::from("/definitely/not/real/npx"));
        let pinned = resolve(Mode::Npx, Some("0.1.0"), Path::new("/x"), &|_| npx()).unwrap();
        assert_eq!(pinned.args, ["-y", "@krv-labs/pulsar@0.1.0", "mcp"]);
        let floating = resolve(Mode::Npx, None, Path::new("/x"), &|_| npx()).unwrap();
        assert_eq!(floating.args, ["-y", "@krv-labs/pulsar", "mcp"]);
    }
}
