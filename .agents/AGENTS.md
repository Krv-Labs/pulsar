# AGENTS.md

## Style & Spelling
- **Writing Style**: Always use **American English spelling** ("optimize", "analyze", "modeling").

## Project Architecture
**Pulsar** is a Rust-backed Python library for topological data analysis. Performance-critical algorithms live in `src/` (PyO3/maturin) and are exposed as `pulsar._pulsar`. Python orchestrates the pipeline in `pulsar/pipeline.py`, MCP tooling in `pulsar/mcp/`, and exports in `pulsar/exports.py`.

### Layout
- **`src/`**: Rust core (Ball Mapper, cosmic graph, MinHash, pseudo-Laplacian, etc.)
- **`pulsar/`**: Python package (pipeline, config, MCP server, exports)
- **`tests/`**: Python and Rust tests; `tests/correctness/` for parity checks
- **`docs/source/`**: Sphinx documentation
- **`demos/`**: Domain-specific examples (penguins, MMLU, EHR, energy)

## Dev Commands
```bash
uv sync --group dev --group mcp
uv run maturin develop --release
uv run pytest -v
cargo test --lib
uv run ruff check pulsar tests demos
```

## Versioning & Releases
`Cargo.toml` `[package].version` is the **single source of truth**. Maturin injects it into wheels; `pulsar/_version.py` reads the same file for editable checkouts. CI runs `python scripts/check_versions.py` to enforce consistency.

### Patch release checklist (e.g. `v0.2.5`)
1. Branch from `main`: `git checkout -b release/v0.2.5`
2. Bump `[package].version` in `Cargo.toml` (pyproject.toml uses `dynamic = ["version"]` — no separate bump).
3. Update `CHANGELOG.md`:
   - Move `[Unreleased]` notes into a dated `## [X.Y.Z] - YYYY-MM-DD` section.
   - Leave an empty `## [Unreleased]` heading at the top.
4. Docs pick up the version automatically via `from pulsar import __version__` in `docs/source/conf.py`.
5. Run `python scripts/check_versions.py` and `uv run pytest tests/test_version.py -v`.
6. Open a PR to `main`; wait for CI.
7. After merge, tag and push to trigger PyPI + GitHub Release:
   ```bash
   git checkout main && git pull
   git tag v0.2.5
   git push origin v0.2.5
   ```

> Tag version must match `Cargo.toml` (prefix `v`, e.g. `v0.2.5`). The release workflow in `.github/workflows/release.yml` publishes wheels/sdist to PyPI and creates a GitHub Release on tag push.

See also: `docs/source/releases.rst` for the full contributor guide.

## MCP Server (`pulsar-mcp`)
Exposes thick, workflow-aware tools for agent-driven topological analysis. Agents should follow the ingest → characterize → create_config → sweep → diagnose → dossier loop documented in the README and `docs/source/userGuides/mcp.rst`.

Key tools: `characterize_dataset`, `create_config`, `run_topological_sweep`, `diagnose_cosmic_graph`, `generate_cluster_dossier`, `compare_clusters`, `export_labeled_data`, `export_dataset_bundle`.

Call `get_workflow_guide` once at session start for the opinionated procedure map.
