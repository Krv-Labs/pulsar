# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

