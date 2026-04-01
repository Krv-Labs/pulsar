# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pulsar is a **Rust-backed Python library** implementing the **Thema pipeline** for topological data analysis (TDA) of large datasets (primarily EHR). Performance-critical algorithms are in Rust (via PyO3/maturin); Python orchestrates the pipeline.

## Commands

```bash
# Install dependencies and build the Rust extension
uv sync
uv run maturin develop --release   # optimized build
uv run maturin develop             # debug build (faster to compile)

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_pca.py    # single test file

# Lint
uv run ruff check .
uv run ruff format .

# Demos
uv run python demos/energy/coal.py                           # US Coal Plants (downloads dataset automatically)
uv run python demos/ehr/physionet.py --synthetic             # EHR demo with synthetic data
uv run python demos/ehr/physionet.py --data path/to/eicu.csv # EHR demo with real eICU data
uv run python demos/ehr/ecg_arrhythmia.py                    # ECG arrhythmia classification demo
```

## Architecture

### Pipeline Stages

```
CSV/Parquet → Impute → Scale → PCA Grid → Ball Mapper Grid → Pseudo-Laplacian → Cosmic Graph
```

The entry point is `ThemaRS("params.yaml").fit()` (in `pulsar/pipeline.py`). It chains Rust components and returns a `networkx.Graph` at `.cosmic_graph`.

### Layer Separation

**Rust (`src/`)** — all computation:
- `impute.rs` — per-column missing value imputation (normal, categorical, mean, median, mode)
- `scale.rs` — StandardScaler (z-score)
- `pca.rs` — Randomized PCA (Halko et al. 2011) using parallel SVD via rayon
- `ballmapper.rs` — Ball Mapper: greedy center selection + point membership
- `pseudolaplacian.rs` — accumulates Laplacian matrices across all ball maps
- `cosmic.rs` — CosmicGraph: normalized adjacency from accumulated Laplacian
- `ph.rs` — approximate H₀ persistent homology for automatic threshold selection; exposes `PyStabilityResult` / `PyPlateau`
- `temporal.rs` — 3D pseudo-Laplacian accumulation across time steps; exposes `accumulate_temporal_pseudo_laplacians` and `py_normalize_temporal_laplacian`
- `error.rs` — shared `PulsarError` type via `thiserror`
- `lib.rs` — PyO3 module definition, all Python-exposed classes/functions

**Python (`pulsar/`)** — orchestration and post-processing:
- `pipeline.py` — `ThemaRS` class; loads data, drives the grid sweeps, calls Rust
- `config.py` — `PulsarConfig`; YAML loader supporting `values`, `range`, and `distribution` sweep specs
- `hooks.py` — post-processing utilities: `label_points`, `membership_matrix`, `cosmic_clusters`, `graph_to_dataframe`, `unclustered_points`
- `temporal.py` — `TemporalCosmicGraph`; wraps the `(n, n, T)` tensor from `temporal.rs` and provides six aggregation methods: `persistence_graph`, `mean_graph`, `recency_graph`, `volatility_graph`, `trend_graph`, `change_point_graph`

### Grid Search Design

The pipeline runs **parallel grid searches** over PCA (dimensions × seeds) and Ball Mapper (embeddings × epsilons), producing thousands of ball maps that are fused into a single Pseudo-Laplacian before computing the Cosmic Graph. The sweep spec in the YAML config follows a W&B-style format.

### Rust↔Python Interface

All Rust structs/functions exposed to Python are defined at the bottom of `src/lib.rs` via `#[pyclass]` / `#[pyfunction]`. When adding new Rust functionality that Python needs, register it there.

### Tests

Two-tier structure:
- `tests/test_*.py` — integration-level, fast feedback
- `tests/correctness/test_*.py` — numerical validation against reference implementations and edge cases

## Configuration (params.yaml)

```yaml
run:
  name: my_experiment
  data: path/to/data.csv
preprocessing:
  drop_columns: [id]
  impute:
    age: {method: sample_normal, seed: 42}
sweep:
  pca:
    dimensions: {values: [2, 3, 5]}
    seed: {values: [42, 7, 13]}
  ball_mapper:
    epsilon: {range: {min: 0.1, max: 1.5, steps: 8}}
cosmic_graph:
  threshold: 0.0   # or "auto" to use persistent homology for threshold selection
output:
  n_reps: 4
```