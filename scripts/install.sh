#!/bin/sh
# Install Pulsar and register its MCP server with your agents.
#
#   scripts/install.sh                     build from this checkout, install, register detected agents
#   scripts/install.sh --binary ./pulsar  install a prebuilt binary
#   scripts/install.sh --from npm          install @krv-labs/pulsar globally from the registry
#   scripts/install.sh --uninstall         unregister from agents and remove the binary
#
# Idempotent and safe to re-run. It never edits shell profiles; it tells you
# what to add to PATH. Existing agent configs are backed up (*.pulsar.backup)
# and only Pulsar's own entries are ever changed (see `pulsar install --help`).
# Windows: use `npm install -g @krv-labs/pulsar` and `pulsar install`.
set -eu

NAME=pulsar
PACKAGE="@krv-labs/$NAME"

PROG=$(basename "$0")
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

FROM=auto          # auto | source | binary | npm
BINARY=""
PREFIX="${PULSAR_PREFIX:-${HOME:-}/.local}"
REGISTER=1
AGENTS=""          # comma-separated ids; empty = every agent detected
ALL_AGENTS=0
DRY_RUN=0
UNINSTALL=0
QUIET=0

usage() {
  cat <<EOF
Usage: $PROG [options]

Source (default: auto = this checkout if it has cargo, else npm):
  --from source|binary|npm   where the binary comes from
  --binary PATH              install this prebuilt binary (implies --from binary)

Where:
  --prefix DIR               install to DIR/bin/$NAME (default: \$HOME/.local; env PULSAR_PREFIX)

Agents:
  --agents a,b               register only these (claude, claude-desktop, codex, gemini,
                             copilot, cursor, vscode, antigravity)
  --all-agents               register every supported agent, detected or not
  --no-register              install the binary only

Other:
  --dry-run                  print what would happen; change nothing
  --uninstall                unregister from agents and remove the binary
  -q, --quiet                only print errors and the final summary
  -h, --help                 this help

Exit codes: 0 ok, 1 failed, 2 bad usage.
EOF
}

# ---- output ----------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_OK=$(printf '\033[32m'); C_WARN=$(printf '\033[33m'); C_ERR=$(printf '\033[31m'); C_OFF=$(printf '\033[0m')
else
  C_OK=""; C_WARN=""; C_ERR=""; C_OFF=""
fi
info() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }
step() { [ "$QUIET" = 1 ] || printf '%s==>%s %s\n' "$C_OK" "$C_OFF" "$*"; }
warn() { printf '%swarning:%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$C_ERR" "$C_OFF" "$1" >&2; [ -z "${2:-}" ] || printf '  fix: %s\n' "$2" >&2; exit 1; }
usage_err() { printf '%s: %s\n\n' "$PROG" "$1" >&2; usage >&2; exit 2; }

# ---- args --------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --from)        [ $# -ge 2 ] || usage_err "--from needs a value"; FROM=$2; shift 2 ;;
    --from=*)      FROM=${1#*=}; shift ;;
    --binary)      [ $# -ge 2 ] || usage_err "--binary needs a path"; BINARY=$2; FROM=binary; shift 2 ;;
    --binary=*)    BINARY=${1#*=}; FROM=binary; shift ;;
    --prefix)      [ $# -ge 2 ] || usage_err "--prefix needs a directory"; PREFIX=$2; shift 2 ;;
    --prefix=*)    PREFIX=${1#*=}; shift ;;
    --agents)      [ $# -ge 2 ] || usage_err "--agents needs a list"; AGENTS=$2; shift 2 ;;
    --agents=*)    AGENTS=${1#*=}; shift ;;
    --all-agents)  ALL_AGENTS=1; shift ;;
    --no-register) REGISTER=0; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --uninstall)   UNINSTALL=1; shift ;;
    -q|--quiet)    QUIET=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             usage_err "unknown option: $1" ;;
  esac
done

case "$FROM" in auto|source|binary|npm) ;; *) usage_err "--from must be one of: source, binary, npm" ;; esac
[ "$FROM" != binary ] || [ -n "$BINARY" ] || usage_err "--from binary needs --binary PATH"
[ -n "${HOME:-}" ] && [ -d "$HOME" ] || die "HOME is not set to an existing directory" "export HOME=/path/to/home"

case "$PREFIX" in /*) ;; *) PREFIX="$(pwd)/$PREFIX" ;; esac
BIN_DIR="$PREFIX/bin"
TARGET="$BIN_DIR/$NAME"

have() { command -v "$1" >/dev/null 2>&1; }

# Run the argument unless this is a dry run, in which case only print it.
run() {
  if [ "$DRY_RUN" = 1 ]; then printf '  [dry run] %s\n' "$*"; else "$@"; fi
}

# ---- uninstall ---------------------------------------------------------------
if [ "$UNINSTALL" = 1 ]; then
  step "Uninstalling Pulsar"
  if [ -x "$TARGET" ]; then
    if [ "$DRY_RUN" = 1 ]; then
      "$TARGET" uninstall --all --yes --dry-run || warn "could not preview agent removal"
    else
      "$TARGET" uninstall --all --yes || warn "agent removal reported problems; review the output above"
    fi
    run rm -f "$TARGET" "$TARGET.prev"
    info "Removed $TARGET"
  else
    warn "no binary at $TARGET; nothing to remove there (agents may still reference another install: run '$NAME status')"
  fi
  exit 0
fi

# ---- choose a source -----------------------------------------------------------
in_checkout() { [ -f "$REPO_ROOT/Cargo.toml" ] && [ -d "$REPO_ROOT/crates/pulsar-cli" ]; }

if [ "$FROM" = auto ]; then
  if in_checkout && have cargo; then FROM=source
  elif have npm; then FROM=npm
  else
    die "no way to get a $NAME binary: this is not a checkout with cargo, and npm is not installed" \
        "install Rust (https://rustup.rs) and re-run from a checkout, install Node.js for --from npm, or pass --binary PATH"
  fi
fi

# ---- preflight -------------------------------------------------------------------
OS=$(uname -s 2>/dev/null || echo unknown); ARCH=$(uname -m 2>/dev/null || echo unknown)
case "$OS" in Linux|Darwin) ;; *) die "unsupported OS: $OS" "on Windows use: npm install -g $PACKAGE && $NAME install" ;; esac
info "Pulsar installer ($OS/$ARCH, source: $FROM, prefix: $PREFIX)"

check_msrv() {
  want=$(sed -n 's/^rust-version *= *"\([0-9.]*\)".*/\1/p' "$REPO_ROOT/Cargo.toml" | head -1)
  have_v=$(cargo --version | sed -n 's/^cargo \([0-9][0-9.]*\).*/\1/p')
  [ -n "$want" ] && [ -n "$have_v" ] || return 0
  lowest=$(printf '%s\n%s\n' "$want" "$have_v" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)
  [ "$lowest" = "$want" ] || die "Rust $have_v is older than the required $want" "run: rustup update stable"
}

case "$FROM" in
  source)
    in_checkout || die "--from source must run from a Pulsar checkout" "run $REPO_ROOT/scripts/install.sh or use --binary / --from npm"
    have cargo || die "cargo is not installed" "install Rust from https://rustup.rs"
    check_msrv ;;
  binary)
    [ -f "$BINARY" ] || die "binary not found: $BINARY"
    [ -x "$BINARY" ] || die "binary is not executable: $BINARY" "chmod +x $BINARY" ;;
  npm)
    have npm || die "npm is not installed" "install Node.js from https://nodejs.org" ;;
esac

# ---- obtain the binary -------------------------------------------------------------
TMP=$(mktemp -d "${TMPDIR:-/tmp}/$NAME-install.XXXXXX")
STAGED=""
cleanup() { rm -rf "$TMP"; [ -z "$STAGED" ] || rm -f "$STAGED"; }
trap cleanup EXIT INT TERM HUP

NEW="$TMP/$NAME"
case "$FROM" in
  source)
    step "Building from source (cargo build --release; this takes a few minutes the first time)"
    if [ "$DRY_RUN" = 1 ]; then
      printf '  [dry run] cargo build --release --locked -p pulsar-cli\n'
    else
      ( cd "$REPO_ROOT" && cargo build --release --locked -p pulsar-cli ) \
        || die "build failed" "fix the compiler error above; 'rustup update' fixes most toolchain problems"
      CARGO_TARGET=$( cd "$REPO_ROOT" && cargo metadata --format-version 1 --no-deps 2>/dev/null \
        | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p' )
      BUILT="${CARGO_TARGET:-$REPO_ROOT/target}/release/$NAME"
      [ -x "$BUILT" ] || die "build succeeded but $BUILT is missing"
      cp "$BUILT" "$NEW"
    fi ;;
  binary)
    cp "$BINARY" "$NEW" 2>/dev/null || [ "$DRY_RUN" = 1 ] || die "could not read $BINARY" ;;
  npm)
    step "Installing $PACKAGE from the registry"
    run npm install -g "$PACKAGE" \
      || die "npm install failed" "check your network and npm registry settings (npm config get registry), then retry"
    if [ "$DRY_RUN" != 1 ]; then
      NPM_BIN="$(npm prefix -g)/bin/$NAME"
      [ -e "$NPM_BIN" ] || die "npm succeeded but $NPM_BIN is missing" "check 'npm prefix -g' and your PATH"
      TARGET="$NPM_BIN"
    fi ;;
esac

# ---- install it atomically, keep the previous one for rollback -----------------
if [ "$FROM" != npm ]; then
  step "Installing to $TARGET"
  if [ "$DRY_RUN" = 1 ]; then
    printf '  [dry run] install -m 0755 <new binary> %s\n' "$TARGET"
  else
    mkdir -p "$BIN_DIR" 2>/dev/null || die "cannot create $BIN_DIR" "pick another location with --prefix DIR"
    [ -w "$BIN_DIR" ] || die "$BIN_DIR is not writable" "pick another location with --prefix DIR"
    chmod 0755 "$NEW"
    # Verify the binary actually runs before it replaces anything.
    "$NEW" --version >/dev/null 2>&1 || die "the new binary does not run on this machine" "is it built for $OS/$ARCH?"
    STAGED="$BIN_DIR/.$NAME.new.$$"
    cp "$NEW" "$STAGED"
    [ ! -e "$TARGET" ] || cp -p "$TARGET" "$TARGET.prev"
    mv -f "$STAGED" "$TARGET"
    if ! "$TARGET" --version >/dev/null 2>&1; then
      [ ! -e "$TARGET.prev" ] || mv -f "$TARGET.prev" "$TARGET"
      die "the installed binary failed its self-check; the previous version was restored"
    fi
  fi
fi

if [ "$DRY_RUN" != 1 ]; then
  VERSION=$("$TARGET" --version 2>/dev/null || echo "$NAME (version unknown)")
  info "Installed $VERSION"
fi

# ---- PATH hint ---------------------------------------------------------------------
if [ "$DRY_RUN" != 1 ]; then
  TARGET_DIR=$(dirname "$TARGET")
  case ":$PATH:" in
    *":$TARGET_DIR:"*) ;;
    *) warn "$TARGET_DIR is not on your PATH. Add this to your shell profile:
    export PATH=\"$TARGET_DIR:\$PATH\"" ;;
  esac
fi

# ---- register with agents ------------------------------------------------------------
REGISTER_STATUS=ok
if [ "$REGISTER" = 1 ]; then
  step "Registering the MCP server with your agents"
  set -- install
  if [ "$FROM" = npm ]; then set -- "$@" --mode binary --command "$TARGET"; fi
  if [ "$ALL_AGENTS" = 1 ]; then set -- "$@" --all
  elif [ -n "$AGENTS" ]; then
    OLD_IFS=$IFS; IFS=,; for a in $AGENTS; do set -- "$@" "$a"; done; IFS=$OLD_IFS
  fi
  if [ "$DRY_RUN" = 1 ]; then
    if [ -x "$TARGET" ]; then "$TARGET" "$@" --dry-run || REGISTER_STATUS=failed
    else printf '  [dry run] %s %s --dry-run\n' "$NAME" "$*"; fi
  else
    "$TARGET" "$@" || REGISTER_STATUS=failed
  fi
fi

# ---- summary -------------------------------------------------------------------------
echo
if [ "$DRY_RUN" = 1 ]; then
  info "Dry run complete; nothing was changed."
elif [ "$REGISTER_STATUS" = failed ]; then
  printf '%sPulsar is installed, but some agents need attention%s (see above).\n' "$C_WARN" "$C_OFF"
  info "Check with: $TARGET status    Retry with: $TARGET install"
  exit 1
else
  printf '%sPulsar is ready.%s Restart your agents to pick up the server. Check with: %s status\n' "$C_OK" "$C_OFF" "$TARGET"
fi
