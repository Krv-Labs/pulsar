#!/bin/sh
# End-to-end test of scripts/install.sh and `pulsar install|uninstall|status`
# against a throwaway HOME. Needs a built binary:
#   cargo build --release -p pulsar-cli && sh scripts/test-install.sh
# Override the binary with PULSAR_TEST_BINARY.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NAME=pulsar
SRC_BIN="${PULSAR_TEST_BINARY:-$ROOT/target/release/$NAME}"
[ -x "$SRC_BIN" ] || { echo "build first: cargo build --release -p pulsar-cli (missing $SRC_BIN)" >&2; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/$NAME-install-test.XXXXXX")
trap 'rm -rf "$WORK"' EXIT INT TERM HUP
export HOME="$WORK/home" NO_COLOR=1
PREFIX="$WORK/prefix"
TARGET="$PREFIX/bin/$NAME"
mkdir -p "$HOME"
INSTALL="sh $ROOT/scripts/install.sh --binary $SRC_BIN --prefix $PREFIX"

PASS=0
ok()   { PASS=$((PASS + 1)); printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; [ -z "${2:-}" ] || printf '%s\n' "$2" >&2; exit 1; }
contains() { printf '%s' "$1" | grep -F -q -- "$2"; }
# expect_status <expected> <description> <command...>: run, capture output in $OUT.
expect_status() {
  want=$1; desc=$2; shift 2
  set +e; OUT=$("$@" 2>&1); got=$?; set -e
  [ "$got" = "$want" ] || fail "$desc (exit $got, wanted $want)" "$OUT"
}

echo "install.sh"

# 1. dry run changes nothing
mkdir -p "$HOME/.cursor"
expect_status 0 "dry run exits 0" $INSTALL --dry-run
[ ! -e "$TARGET" ] && [ ! -e "$HOME/.cursor/mcp.json" ] || fail "dry run wrote something"
contains "$OUT" "Dry run complete" || fail "dry run summary missing" "$OUT"
ok "dry run touches nothing"

# 2. install registers the agents it detects, with the installed path
mkdir -p "$HOME/.codex"
expect_status 0 "install" $INSTALL
[ -x "$TARGET" ] || fail "binary not installed at $TARGET"
"$TARGET" --version | grep -q "^$NAME " || fail "installed binary does not report its version"
grep -F -q "$TARGET" "$HOME/.cursor/mcp.json" || fail "cursor entry does not point at $TARGET" "$(cat "$HOME/.cursor/mcp.json")"
grep -F -q "[mcp_servers.$NAME]" "$HOME/.codex/config.toml" || fail "codex entry missing"
[ ! -e "$HOME/.claude.json" ] || fail "registered an agent that was not detected"
contains "$OUT" "Pulsar is ready" || fail "ready summary missing" "$OUT"
ok "installs the binary and registers detected agents"

# 3. status agrees
expect_status 0 "status --json" "$TARGET" status --json
contains "$OUT" '"active": 2' || fail "status should report 2 active" "$OUT"
ok "status reports the registrations as active"

# 4. re-running is idempotent
before=$(cat "$HOME/.cursor/mcp.json")
expect_status 0 "second install" $INSTALL
[ "$(cat "$HOME/.cursor/mcp.json")" = "$before" ] || fail "second install rewrote the config"
ok "re-install is idempotent"

# 5. a binary that does not run is rejected and the working one is kept
printf '#!/bin/sh\nexit 1\n' > "$WORK/broken"; chmod +x "$WORK/broken"
expect_status 1 "broken binary is rejected" sh "$ROOT/scripts/install.sh" --binary "$WORK/broken" --prefix "$PREFIX"
contains "$OUT" "does not run on this machine" || fail "no explanation for the broken binary" "$OUT"
"$TARGET" --version >/dev/null || fail "the previously installed binary was damaged"
ok "a broken binary is rejected and the old install survives"

# 6. usage and input errors
expect_status 1 "missing binary" sh "$ROOT/scripts/install.sh" --binary "$WORK/nope" --prefix "$PREFIX"
contains "$OUT" "binary not found" || fail "missing-binary message" "$OUT"
expect_status 2 "unknown option" sh "$ROOT/scripts/install.sh" --bogus
expect_status 2 "bad --from" sh "$ROOT/scripts/install.sh" --from rpm
expect_status 2 "--from binary without --binary" sh "$ROOT/scripts/install.sh" --from binary
ok "bad input fails clearly (1 for runtime errors, 2 for usage)"

# 7. a foreign entry is left alone and reported, but the binary still installs
cat > "$HOME/.cursor/mcp.json" <<'EOF'
{"mcpServers": {"pulsar": {"command": "/opt/other-tool", "args": ["--serve"]}}}
EOF
foreign=$(cat "$HOME/.cursor/mcp.json")
expect_status 1 "foreign entry is a failure" $INSTALL
[ "$(cat "$HOME/.cursor/mcp.json")" = "$foreign" ] || fail "overwrote someone else's entry"
contains "$OUT" "did not write" && contains "$OUT" "some agents need attention" || fail "conflict not explained" "$OUT"
ok "someone else's '$NAME' entry is never overwritten"
printf '{}\n' > "$HOME/.cursor/mcp.json"

# 8. --agents restricts, --no-register skips
rm -f "$HOME/.codex/config.toml"
expect_status 0 "--agents cursor" $INSTALL --agents cursor
[ ! -e "$HOME/.codex/config.toml" ] || fail "--agents registered an extra agent"
grep -F -q "$TARGET" "$HOME/.cursor/mcp.json" || fail "--agents cursor did not register cursor"
expect_status 0 "--no-register" $INSTALL --no-register
contains "$OUT" "Registering" && fail "--no-register still registered" "$OUT"
ok "--agents and --no-register are honoured"

# 9. uninstall removes the registrations and the binary
expect_status 0 "uninstall" sh "$ROOT/scripts/install.sh" --uninstall --prefix "$PREFIX"
[ ! -e "$TARGET" ] || fail "binary still present after uninstall"
grep -q "$NAME" "$HOME/.cursor/mcp.json" 2>/dev/null && fail "cursor entry still present after uninstall"
ok "uninstall unregisters and removes the binary"

echo "$NAME install | uninstall | status (leave no trace)"
rm -rf "$HOME" "$PREFIX"; mkdir -p "$HOME/.cursor"
expect_status 0 "reinstall into a clean home" $INSTALL --agents cursor,codex
expect_status 0 "uninstall --all" sh "$ROOT/scripts/install.sh" --uninstall --prefix "$PREFIX"
left=$(cd "$HOME" && find . -type f | sort)
[ -z "$left" ] || fail "files left behind in HOME after uninstall" "$left"
ok "uninstall leaves no files behind"

echo "all $PASS checks passed"
