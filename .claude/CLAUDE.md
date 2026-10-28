# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## Project Overview

Pulsar is a **pure-Rust** topology workbench for tabular data, served to
agents over **MCP** and shipped on npm as **`@krv-labs/pulsar`** (public,
npmjs.com). It implements the Thema pipeline:

```
CSV/Parquet → Impute → Scale → Projection grid (PCA/JL) → Ball Mapper grid → Pseudo-Laplacian → Cosmic Graph
```

There is no Python anywhere in the repository, build, CI or distribution
(ADR 0002). The Python implementation was the porting reference until the port
was complete, then removed; `tests/fixtures/golden/README.md` records where the
frozen fixtures came from (tag `python-reference`, generators in git history).

## Product name

`pulsar` is the binary, the MCP server key the installer writes, the store
directory and the backup suffix (`*.pulsar.backup`); `@krv-labs/pulsar` (+
`pulsar-<platform>`) on npm. It lives in: the `pulsar-cli` `[[bin]]` name,
`pulsar_install::NAME` and `launch::PACKAGE`, `fsops::BACKUP_SUFFIX`,
`pulsar_store::APP_DIR`, the MCP `serverInfo` name, `NAME` in
`npm/scripts/build-packages.mjs` (stamped into the launcher), `NAME` in
`.github/workflows/release.yml`, `scripts/install.sh` and `scripts/test-install.sh`.

## Commands

```bash
cargo build --release -p pulsar-cli     # target/release/pulsar
cargo test --workspace                  # unit, integration, golden-fixture parity
cargo test -p pulsar-parity             # parity against frozen Python fixtures
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
node npm/scripts/smoke.mjs              # pack + clean install + MCP handshake (needs a release build)
```

## Crates (`crates/`)

- `pulsar-core`: numeric engine (impute, scale, PCA/JL, Ball Mapper, pseudo-Laplacian, cosmic graph, H0 `ph.rs`, temporal). `PulsarError` via `thiserror`.
- `pulsar-config`: `params.yaml` loader (`values`, `range`, `distribution`), JSON Schema, structured errors.
- `pulsar-data`: CSV/Parquet ingest (pandas-compatible typing), preprocessing executor.
- `pulsar-pipeline`: `fit` / `fit_multi`, projection fingerprint, progress schedule.
- `pulsar-store`: XDG store (`PULSAR_STORE_DIR`), typed handles, job model, `ModelCache`, Arrow IPC sidecars.
- `pulsar-analysis`: thresholds, clustering, diagnostics, `evidence/` and `plan/` (preprocessing plan port).
- `pulsar-mcp`: the `rmcp` server (typed tools, error envelope, async sweep jobs, tasks).
- `pulsar-cli`: the `pulsar` binary. `pulsar-parity`: golden-fixture harness.
- `npm/`: launcher + platform-package builder (`npm/scripts/build-packages.mjs`). Release = `v*` tag; see `.github/workflows/release.yml`.

## Parity rules

- `tests/fixtures/golden/` is the frozen reference (`manifest.json` declares tolerances). **Never regenerate or edit a fixture to silence a failure**: a parity failure is a bug or an undocumented divergence.
- The Python reference is gone, so there are no new fixtures. New behaviour gets Rust known-answer tests and edge-case tests that assert the exact error message.
- Document divergences from the Python reference in the module header (see `pulsar-analysis/src/evidence/mod.rs`).
- Do not add Python files (scripts, notebooks, fixture helpers) to this repository.

## Technical Standards

- **Memory**: avoid O(n²) in Rust. Use `ndarray`, parallelize with `rayon`.
- **Types**: `f64` for numerics.
- **Safety**: library code is panic-free; return `PulsarError` / typed errors. Validate every input at the boundary (0 rows, 0 cols, n < 2, parameter bounds) before computing.
- **ndarray ↔ nalgebra**: row-major vs column-major. Never `DMatrix::from_iterator` over `ndarray::iter()`; use `from_row_slice`.
- **No `.unwrap()` on `partial_cmp` for `f64`**: use `total_cmp` or guard NaN upstream with a documented `expect`.
- **Profile before optimizing.** Real targets are redundant projection recomputation and O(n²) allocations; k-NN preprocessing, components, scale and impute are not bottlenecks. Reuse embeddings by fingerprint (`projection_fingerprint`).
- Randomized SVD is approximate (Halko et al.); do not treat variance ratios as ground truth.
- `construction_threshold: "auto"` (H0 stability) is the safe generic default; do not default to a fixed value for unknown-domain data.

## MCP Surface Standards

The server is a general-purpose topology workbench, not domain-specific. The agent and user are the domain experts; Pulsar provides calibrated topology, evidence and provenance.

- **Measurements, not conclusions.** No domain interpretation in generic tools; domain naming belongs in agent synthesis or explicit inputs such as `cluster_names`.
- **Summary-first, bounded payloads.** Never put raw O(rows/columns/edges/thresholds/clusters) lists in default responses: previews, omitted counts, truncation flags, `detail="full"` for audits. `PULSAR_MCP_MAX_RESPONSE_BYTES` (default 100 KB) is a backstop, not a design target. `get_threshold_stability_curve` is the model of a decision-shaped summary.
- **Rich but bounded decision packets.** Include shape, calibration, cost (estimated ball maps, memory tier), provenance, gates and compatible `next_tools`.
- **Preserve agent autonomy.** Baselines are not answers; diagnostics are measurements; threshold policies are lenses, not scores to maximize. Do not reject valid one-component or giant-plus-tail topologies.
- **Label graph surfaces precisely:** construction vs interpretation threshold, constructed graph vs full weighted affinity, live model vs persisted run.
- **Scale costs are visible; heavy modes are opt-in.**
- **Errors unblock agents:** `ToolError` envelope with `error_code`, tool, handle/path context and a concrete `agent_action`; never opaque traces or benign defaults.
- **Tool contract:** typed params with per-field descriptions written for an agent that has never seen Pulsar; output schema matching the success payload exactly; annotations (`read_only`, `idempotent`, `open_world = false`); structured content plus text. Catalog stays under `CATALOG_BUDGET_CHARS`; raise it deliberately.
- **Handles:** `run_id` etc. resolve through `pulsar-store`, so they work across transports and restarts. Read-only means neither session state nor the registry is mutated.
- **Tests and docs move with the surface:** contract tests in `crates/pulsar-mcp/tests/contracts.rs` (in-memory client, schema validation), `guide.rs` workflow text, and `docs/design/next-mcp-roadmap.md` when direction changes.

## Rust conventions

- Never block the async runtime: run `fit` and other blocking work via `tokio::task::spawn_blocking`.
- Progress callbacks: `(stage, cumulative_fraction)`, monotonically increasing, ending at exactly 1.0.
- Keep error `code()`s stable; they are part of the agent-facing contract.
