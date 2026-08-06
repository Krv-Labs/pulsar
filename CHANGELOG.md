# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`import pulsar` is now lazy, cutting CLI startup ~20x.** The package eagerly
  imported scikit-learn, SciPy, pandas, pyarrow, and NetworkX — about 1.6s — and
  because `pulsar.cli` is a subpackage, every CLI invocation paid it to write a
  few JSON config files. `pulsar --help` took 1659&nbsp;ms; the Rust extension
  itself accounts for ~1&nbsp;ms of that. Public names now resolve through a
  PEP 562 `__getattr__`, so `from pulsar import ThemaRS`, `pulsar.PCA`, and
  `from pulsar import *` behave exactly as before — only the timing moves.
  `pulsar --help` 1659→82&nbsp;ms, `pulsar status` 1722→100&nbsp;ms,
  `import pulsar` 1679→38&nbsp;ms.
- **`pulsar.analysis.characterization` defers scikit-learn** into the two
  functions that use it. It sits on `pulsar.pipeline`'s import path, so every
  consumer paid ~1s for `SimpleImputer`, `NearestNeighbors`, and a scaler that
  most never called. `import pulsar` plus `ThemaRS` is now 520&nbsp;ms, down
  from 1763&nbsp;ms.

### Fixed

- **`pulsar install` no longer corrupts the terminal it draws in.** The prompt
  advanced eight rows per frame and then moved the cursor up nine, so the anchor
  crept upward on every keypress and `\033[J` erased a line of scrollback each
  time. The title was printed outside the redrawn block and so could never be
  counted; on exit the cursor was parked back inside the list, and the install
  report printed on top of its own option rows (`Cancelled. Code (detected)`),
  leaving the remainder on screen. Escape sequences were also parsed with a
  blocking two-byte read, which consumed the *next* keypress and hung forever on
  a bare <kbd>Esc</kbd>. The renderer now holds one invariant — the cursor-up
  count equals the rows the frame advanced, and every line the prompt owns lives
  inside the redrawn frame — measured in soft-wrapped rows rather than logical
  lines, and collapses to a one-line summary so the report starts on a clean row.

### Added

- **Interactive install/uninstall on Windows.** Console access is now a capability
  ladder: a full redrawn TUI (`/dev/tty` + termios on POSIX, `CONIN$`/`CONOUT$`
  with `msvcrt` and `ENABLE_VIRTUAL_TERMINAL_PROCESSING` on Windows), a numbered
  ASCII prompt where ANSI will not render (`TERM=dumb`, or a console whose VT mode
  cannot be set), and the existing error where there is no console at all. Key
  reading is normalized to abstract names, because Windows reports arrow keys as a
  scancode pair rather than as CSI sequences.
- **Windows CI job** running the CLI test suite and a console-detection probe.
  Windows wheels ship from `release.yml`, but CI ran only on Linux, so this code
  path was previously shipped unverified.

### Changed

- **`pulsar install` and `pulsar uninstall` now prompt when stdin is redirected.**
  They talk to the controlling console rather than to `sys.stdin`, so
  `pulsar install | tee install.log` gets a working menu instead of
  `error: non-interactive shells must pass explicit harness names or --all`.
  This reverses the piped-stdin behavior added in "Stop crashing on Windows and
  no-opping on piped stdin"; that change's intent — never silently no-op — is
  better served by prompting than by refusing. A genuinely consoleless
  environment still errors, now with `no console available`.
- **Ctrl-C now exits 130 everywhere.** It previously exited 0 in the interactive
  menu (where raw mode delivers it as an ordinary key) and printed a traceback
  in line mode (where SIGINT is still live). `q` and <kbd>Esc</kbd> remain a
  clean cancel with exit 0, so scripts that check `$?` can tell the two apart.

## [0.3.0] - 2026-08-05

### Added

- **CosmicTrajectory**: observation-centric longitudinal representation that pools every `(entity, t)` observation into a single geometry, enabling cross-time BallMapper covers and similarity edges by construction. Sparse matrix storage (`obs`, `balls`, `similarity`, `incidence`) with hypergraph views via matrix products; `TemporalCosmicGraph` is unchanged. ([#35](https://github.com/Krv-Labs/pulsar/pull/35))
- **Longitudinal MCP tools**: `build_longitudinal_graph`, `diagnose_longitudinal_graph`, `get_trajectory_archetypes`, and `get_cross_time_neighbors` — pivot long-format panels into snapshot lists, build temporal and/or trajectory representations, and expose cross-time lookalike queries with explicit alignment policies and cost guards. ([#36](https://github.com/Krv-Labs/pulsar/pull/36))
- **macOS Intel CI smoke** for Pulsar MCP: PR CI now exercises the `macosx_*_x86_64` wheel path on `macos-15-intel`, matching the release matrix. ([#34](https://github.com/Krv-Labs/pulsar/pull/34))

### Fixed

- **Longitudinal panel safety**: preserve per-snapshot entity IDs for ragged trajectories, require identity alignment for temporal graphs, budget temporal peak tensor memory, remove redundant tensor copies, resolve original neighbor time labels, reject negative member limits, and omit gap-spanning trajectory edges. ([#39](https://github.com/Krv-Labs/pulsar/pull/39))
- **MCP setup documentation now describes a working no-clone install.** `pip install pulsar` installed an unrelated third-party package from PyPI; the correct distribution is `thema-pulsar`. The Claude Desktop JSON config specified `"command": "uv tool run"`, which can never resolve — MCP clients exec `command` directly with no shell, so a value containing a space is not an executable. Client commands now use `uvx`, and the Gemini registration documents workspace trust (stdio MCP servers stay dormant in an untrusted folder) plus `--timeout 60000` for the first-boot download. ([#33](https://github.com/Krv-Labs/pulsar/pull/33))
- **Ruff rule selection is now explicit and the engine is version-bounded.** The project had no `[tool.ruff]` configuration, so CI enforced whatever the installed ruff defaulted to. Ruff 0.16 widened that default (adding `I`, `UP`, `SIM`, `BLE`, `RUF`, and more), and because `uv.lock` is not committed, CI resolves dependencies fresh on every run — so the unbounded `ruff>=0.15.7` picked up 0.16.1 and failed lint on unrelated pull requests with 231 pre-existing findings. `[tool.ruff.lint] select` now pins the pre-0.16 rule set and the dev group is bounded to `>=0.16.1,<0.17`. ([#33](https://github.com/Krv-Labs/pulsar/pull/33))

## [0.2.5] - 2026-07-11

### Added

- **Structured dataset export bundle** (`pulsar/exports.py`): composable helpers for `cosmic_edges`, `snapshot_edges`, `node_table`, `group_table`, `clean_data`, `clean_embedding_at_center`, and `export_dataset_bundle` orchestrating the `{slug}/tabular+graph` Parquet layout plus `export_manifest.json`. Thin `ThemaRS` delegators keep the pipeline class free of export logic. ([#29](https://github.com/Krv-Labs/pulsar/pull/29), closes [#26](https://github.com/Krv-Labs/pulsar/issues/26))
- **MinHash/LSH cosmic-graph construction** in the Rust core: approximate edge weights as unbiased Jaccard estimates of each point's ball-set via seeded MinHash signatures and LSH banding — sub-quadratic, constant-memory, and deterministic for a given `(balls, n, d, seed)`. Exposed to Python as `MinHashAccumulator`.
- `cosmic_graph.construction` toggle (`"minhash"` default | `"exact"`): `"minhash"` uses the sketch path; `"exact"` preserves the bit-identical sparse pseudo-Laplacian backbone for reproducible co-occurrence weights.
- `cosmic_graph.minhash_d` and `cosmic_graph.minhash_seed` config knobs for signature depth and reproducibility (defaults `256` / `42`).
- Sparse cosmic-graph backbone in the Rust core: `accumulate_pseudo_laplacians_sparse` (COO co-membership accumulation, no n×n allocation), `CosmicGraph.from_pseudo_laplacian_sparse`, and `find_stable_thresholds_sparse` (edge-list threshold selection). Existing dense APIs are unchanged.
- MCP MinHash signature-depth advisory (`pulsar.mcp.minhash_advisor`): Hoeffding/CLT error bounds, memory estimates, and proactive `minhash_d` suggestions for massive datasets via `characterize_dataset`.
- `minhash_profile` in `diagnose_cosmic_graph` payloads when construction is `"minhash"`.
- **Release tooling**: `pulsar/_version.py`, `scripts/check_versions.py`, CI version check, and contributor/agent release docs (`docs/source/releases.rst`, `.agents/AGENTS.md`) mirroring the Topos release workflow.

### Changed

- **Default cosmic-graph construction is now MinHash** (`cosmic_graph.construction: minhash`). The pipeline routes both `fit` and `fit_multi` through a shared `_CosmicBuilder` that never allocates an n×n matrix on either path.
- **Spectral sparsification is now opt-in (`cosmic_graph.sparsify: false` by default).** It runs after the cosmic graph is already built, so as a default it was pure additional cost on the construction path (and its only downstream consumer re-densified the graph anyway). It remains available as a hook (`ThemaRS.spectral_sparsify`) — a leverage-aware, epsilon-controlled graph that preserves spectrum/effective-resistance (distances), not topology, for downstream spectral analysis.
- `weighted_adjacency` is now materialized lazily on first access; the cosmic-graph backbone is kept sparse end-to-end so the hot path (`fit` → threshold → networkx) never allocates a dense n×n matrix.
- Sphinx docs now read `release` from `pulsar.__version__` instead of a hard-coded string.

## [0.2.4]

### Added

- Implemented Johnson-Lindenstrauss (JL) projection and graph accelerations in the Rust core.

### Changed

- Configured `pyproject.toml` to dynamically inherit the package version from `Cargo.toml` using Maturin, establishing a single source of truth for versioning.
- Refactored project structure by moving benchmarks into the `tests` directory.

### Fixed

- Normalized the Cosmic graph onto a `[0, 1]` weight scale (`1 / max(1, max_weight)`) before threshold selection. Spectral sparsification can reweight edges above 1.0, which collapsed all such edges into a single bin in `find_stable_thresholds` (it quantizes over `[0, 1]`), destroying threshold resolution. `weighted_adjacency` / `weighted_edges` / `cosmic_graph` and the resolved construction threshold now share this scale; the dense / `sparsify: false` path (weights ≤ 1) is unchanged, and `cosmic_rust` still exposes the raw weights.
- Addressed Clippy `too_many_arguments` warnings in `pcg_component` by grouping options into `PcgOptions`.
- Applied `ruff` formatting and resolved linting warnings.
- Removed the unused `sprs` dependency and a dead `sparse_laplacian` allocation in the spectral sparsifier; hoisted invariant centering out of the `jl_grid` inner loop.

## [0.2.3]

### Changed

- Updated the MCP server invocation command to use `uvx` instead of standard module execution.
- Enhanced installation instructions across the `README.md` and MCP user guides.

## [0.2.2]

### Fixed

- Resolved test dependency mismatches and added `rich` to dev dependencies to prevent skipped tests.

## [0.2.1]

### Added

- **Topological Interpretation Engine:** Introduced a new FastMCP server for topological data analysis (`pulsar-mcp`).
- **Documentation:** Built a Sphinx documentation workflow including custom styles, a user guide, and an API reference.
- Added a `demos` dependency group for environment creation.

### Changed

- Updated the Python package name to `thema-pulsar`.
- Consolidated mixed PEP standards into regimented `--group` classes under `[dependency-groups]`.

## [0.2.0]

### Added

- Initial v0.2.0 release marking the transition of the core architecture to Rust with Python bindings.

