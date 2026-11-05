# AGENTS.md

## Style & Spelling
- **Writing Style**: Always use **American English spelling** ("optimize", "analyze", "modeling").

## Project Architecture
**Pulsar** is a pure-Rust topology workbench for tabular data, served to agents over MCP and shipped as the npm package `@krv-labs/pulsar` (npmjs.com). There is no Python in this repository. `.claude/CLAUDE.md` is the full contributor guide.

### Layout
- **`crates/`**: the workspace (`pulsar-core` numerics, `pulsar-config`, `pulsar-data`, `pulsar-pipeline`, `pulsar-store`, `pulsar-analysis`, `pulsar-report`, `pulsar-mcp`, `pulsar-install`, `pulsar-cli`, `pulsar-parity`)
- **`npm/`**: the `pulsar` launcher and the platform-package builder
- **`tests/fixtures/golden/`**: frozen parity fixtures (never regenerated; see its README)
- **`docs/`**: ADRs and design notes

## Dev Commands
```bash
cargo build --release -p pulsar-cli
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all
node npm/scripts/smoke.mjs   # pack + clean install + MCP handshake (needs a release build)
```

## Versioning & Releases
`[workspace.package].version` in the root `Cargo.toml` is the single source of truth; the npm builder reads it. A `v*` tag matching that version triggers `.github/workflows/release.yml`, which builds every platform package, publishes to npmjs.com and verifies the published packages.

## MCP Server
Exposes workflow-aware tools for agent-driven topological analysis. Agents follow ingest → characterize → create_config → sweep → diagnose → dossier, and call `get_workflow_guide` once at session start for the procedure map.
