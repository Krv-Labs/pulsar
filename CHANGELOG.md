# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Changed
- **Spectral sparsification is now opt-in (`cosmic_graph.sparsify: false` by default).** It runs after the cosmic graph is already built, so as a default it was pure additional cost on the construction path (and its only downstream consumer re-densified the graph anyway). It remains available as a hook (`ThemaRS.spectral_sparsify`) — a leverage-aware, epsilon-free O(n)-edge graph that preserves spectrum/effective-resistance (distances), not topology, for downstream spectral analysis.
- `weighted_adjacency` is now materialized lazily on first access; the cosmic-graph backbone is kept sparse end-to-end so the hot path (`fit` → threshold → networkx) never allocates a dense n×n matrix.
### Added
- Sparse cosmic-graph backbone in the Rust core: `accumulate_pseudo_laplacians_sparse` (COO co-membership accumulation, no n×n allocation), `CosmicGraph.from_pseudo_laplacian_sparse`, and `find_stable_thresholds_sparse` (edge-list threshold selection). Existing dense APIs are unchanged.

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
