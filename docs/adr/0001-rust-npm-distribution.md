# ADR 0001: Pure-Rust Pulsar distributed through npm

- **Status:** Accepted
- **Date:** 2026-10-16
- **Issue:** #58 (milestone *v1.0: Rust-native Pulsar on npm*, tracking #77)

## Context

Pulsar v0.5 is a Python package (`thema-pulsar` on PyPI) with Rust numeric
kernels behind PyO3. Agents reach it through a FastMCP server that harness
configs launch with `uvx --from thema-pulsar[mcp] pulsar-mcp`. This has two
costs:

- **Install fragility.** Agent hosts need a working `uv`/Python toolchain on
  the right path, inside whatever sandbox they run in (#10). Cold starts
  resolve and install a scientific Python stack.
- **Two-language logic.** About 29k lines of Python orchestration, statistics
  and MCP code sit on top of about 4k lines of Rust. Most of that Python
  exists to glue numpy, pandas, networkx, scikit-learn and scipy together.

Agent hosts already install MCP servers with `npx` / `pnpm dlx`, so npm is the
distribution channel they support best.

## Decision

1. **One Rust binary, `pulsar`.** Subcommands:
   - `fit`, `profile` and `schema` for analysis;
   - `mcp` for the stdio MCP server;
   - `install`, `uninstall` and `status` for registering the server in agent
     harness configs.

   There is no Python at runtime or build time once the port completes (#76).
2. **npm distribution follows the esbuild/biome platform-package pattern.**
   - The root package `@krv-labs/pulsar` holds only a small JS launcher (`bin`)
     plus `optionalDependencies` on one package per target. Each target package
     declares `os`, `cpu` and, on Linux, `libc`, and contains the prebuilt
     binary.
   - The package manager installs only the matching target. There is **no
     `postinstall` download**: it would break under pnpm's default
     `ignore-scripts`, offline mirrors and sandboxed agent hosts.
   - Users run `npx -y @krv-labs/pulsar mcp`, `pnpm dlx @krv-labs/pulsar mcp`
     or `pnpm add -g @krv-labs/pulsar`.
   - v1.0 targets:

     | Package suffix     | Rust target                   |
     |--------------------|-------------------------------|
     | `darwin-arm64`     | `aarch64-apple-darwin`        |
     | `darwin-x64`       | `x86_64-apple-darwin`         |
     | `linux-x64-gnu`    | `x86_64-unknown-linux-gnu`    |
     | `linux-arm64-gnu`  | `aarch64-unknown-linux-gnu`   |
     | `linux-x64-musl`   | `x86_64-unknown-linux-musl`   |
     | `linux-arm64-musl` | `aarch64-unknown-linux-musl`  |
     | `win32-x64-msvc`   | `x86_64-pc-windows-msvc`      |
3. **The MCP server uses `rmcp`** (official Rust SDK, pinned at `3.5.x`).
4. **Tabular IO uses arrow-rs** (`arrow`, `parquet`, CSV via `arrow-csv`), not
   Polars. Preprocessing recipe steps (#68) are implemented over Arrow arrays
   and `ndarray`. The `derive` escape hatch becomes a small, sandboxed
   expression language, or is deferred, rather than full SQL.
5. **Toolchain.**
   - Edition 2021 for existing crates; new crates may use 2024.
   - Workspace MSRV is **1.88**, set by `rmcp`. `arrow`/`parquet` need 1.85.
6. **Not in v1.0:**
   - Publishing library crates to crates.io. All crates stay `publish = false`
     until there is an external Rust consumer.
   - napi bindings for embedding in Node.
7. **PyPI sunset.**
   - One final `thema-pulsar` release whose README and an import-time
     `DeprecationWarning` point to `@krv-labs/pulsar`.
   - Earlier versions are **not** yanked.
   - The golden-fixture generator (#60) keeps a pinned Python environment so
     the reference outputs stay reproducible.

## Spike evidence

All measurements were taken on 2026-10-16 on macOS arm64 with Rust 1.97.1,
release profile, `lto = true`, `codegen-units = 1`.

### rmcp 3.5.1 covers the target protocol

A minimal server (`#[tool_router(server_handler)]`, one tool, stdio
transport) was driven over raw JSON-RPC on stdio:

- **2025-11-25 client.**
  - `initialize`, `tools/list` and `tools/call` all succeed.
  - `inputSchema` and `outputSchema` are generated from `schemars` types,
    including field doc comments as descriptions.
  - Results carry both `structuredContent` and text content.
- **2026-07-28 client.**
  - Works with no `initialize`.
  - `server/discover` returns
    `supportedVersions: [2024-11-05 … 2026-07-28]` and `resultType: complete`.
  - `tools/call` with per-request `_meta` succeeds.
- **Source review of 3.5.1** shows:
  - `ProtocolVersion::LATEST = 2026-07-28`;
  - the `io.modelcontextprotocol/tasks` extension model (get, update, cancel,
    status notifications) and capability flags;
  - `InputRequiredResult` (MRTR);
  - progress notifications.

  These cover what #71 needs: dual-era serving, async jobs and tasks, and
  progress. One binary serves old and new clients, so the TypeScript fallback
  is not needed.
- **Size:** the stripped server binary is 1.8 MB, or 0.8 MB gzipped.

### arrow-rs vs Polars

Both prototypes read `demos/penguins/penguins.csv`. The Polars prototype also
ran a SQL aggregation.

| Stack | Stripped binary | gzip (≈ npm tarball) | Cold release build |
|---|---|---|---|
| Polars 0.53 (`lazy, csv, parquet, sql, strings, temporal, …`) | 50.9 MB | 18.7 MB | 552 s |
| arrow-rs 57 (`csv`) + parquet 57 (`arrow, snap, zstd`) | 2.5 MB | 1.0 MB | 43 s |

Polars costs about 20× the download for every `npx` cold start and about 13×
the clean CI build time. Its main benefit for Pulsar was a SQL/expression
escape hatch for preprocessing. The recipe design (#68) is a closed, typed
step vocabulary, so that benefit does not justify the cost.

**Finding:** Polars' default CSV reader read the literal `NA` in the penguins
data as a string, so the numeric columns came back as `str`. Whatever the
library, ingest must **name its null tokens explicitly and report** which
tokens it treated as missing. The profiling tools (#69) should surface
null-token candidates such as `NA`, `N/A`, `?`, `-` and empty strings, rather
than guess silently.

### npm

- The `@krv-labs` scope is **not registered** on npm. `npm org ls krv-labs`
  returns *Scope not found*. **Action, owner: maintainers:** create the npm org
  `krv-labs`, then add an automation token or configure trusted publishing for
  GitHub Actions before #74.

## Alternatives considered

| Option | Why not |
|---|---|
| napi-rs addon plus a TypeScript MCP server | Logic splits across two languages again. Unnecessary, since `rmcp` already serves both protocol eras. |
| cargo-dist npm installer (downloads the binary in `postinstall`) | Fails under `ignore-scripts`, offline and in sandboxes. |
| WASM (wasm-pack) | No rayon threads without extra setup; slower dense linear algebra; awkward file and mmap IO. |
| Polars for IO and transforms | About 20× the binary size and 13× the build time (see above). |
| Keep PyPI alongside npm | Two packaging pipelines and two ABIs for a product whose users are agents launched through npm. |

## Consequences

- The workspace is restructured into product crates (#59). `pulsar-python`
  stays buildable only until #76, so parity tests can call both
  implementations.
- Every behaviour ported from Python is checked against golden fixtures (#60)
  before the Python is deleted.
- Python-only dependencies need Rust replacements, recorded per issue:
  - networkx → CSR/edge-list structs plus `petgraph`;
  - scikit-learn → in-house or `linfa`-family code;
  - scipy.stats → `statrs` plus in-house functions;
  - pyarrow → `parquet`.
