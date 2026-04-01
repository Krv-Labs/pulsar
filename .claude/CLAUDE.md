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
uv run pytest tests/correctness/   # numerical validation tests

# Lint / Format
uv run ruff check .
uv run ruff format .

# Documentation
cd docs && make html               # build Sphinx docs

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

### Advanced Orchestration

- **`fit_multi(datasets: list[pd.DataFrame])`**: Fuses multiple representations (e.g., different embedding models) of the same points. All resulting ball maps are accumulated into a single "Galactic" Pseudo-Laplacian.
- **Threshold Stability**: If `threshold: "auto"`, Pulsar uses approximate H₀ persistent homology (`src/ph.rs`) to find stable plateaus in the component-vs-threshold curve.
- **Representative Selection**: `model.select_representatives(n_reps)` clusters the thousands of generated ball maps (using node/edge/epsilon features) to find a diverse subset of the most descriptive maps.

### Layer Separation

**Rust (`src/`)** — all computation:
- `impute.rs` — missing value imputation; supports `sample_normal`, `sample_categorical`, `fill_mean`, `fill_median`, `fill_mode`.
- `scale.rs` — `StandardScaler` (z-score).
- `pca.rs` — Randomized PCA (Halko et al. 2011) using parallel SVD via `rayon`.
- `ballmapper.rs` — Ball Mapper: greedy center selection + point membership.
- `pseudolaplacian.rs` — accumulates Laplacian matrices across all ball maps.
- `cosmic.rs` — `CosmicGraph`: normalized adjacency from accumulated Laplacian.
- `ph.rs` — approximate H₀ persistent homology for automatic threshold selection.
- `temporal.rs` — 3D pseudo-Laplacian accumulation `(n, n, T)` across time steps.
- `error.rs` — shared `PulsarError` type via `thiserror`.
- `lib.rs` — PyO3 module definition.

**Python (`pulsar/`)** — orchestration and post-processing:
- `pipeline.py` — `ThemaRS` class; drives grid sweeps and handles imputation flags (`_was_missing`).
- `config.py` — `PulsarConfig`; YAML loader supporting `values`, `range`, and `distribution`.
- `temporal.py` — `TemporalCosmicGraph`; provides aggregation methods: `persistence`, `mean`, `recency`, `volatility`, `trend`, `change_point`.
- `hooks.py` — Maps graph nodes back to data: `label_points`, `membership_matrix`, `cosmic_clusters`, `graph_to_dataframe`.

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
  threshold: "auto"   # uses persistent homology stability analysis
output:
  n_reps: 4
```

## Technical Standards

- **Memory**: Avoid $O(n^2)$ memory in Rust. Large matrices should be handled via `ndarray` and parallelized with `rayon`.
- **Types**: Use `f64` for all numerical Rust code. In Python, use `np.float64` for compatibility.
- **Safety**: Rust code must be `panic`-free for Python; use `PulsarError` and `PyResult`.
- **Tests**: Every new Rust function should have a corresponding Python test in `tests/` and, if numerical, a validation test in `tests/correctness/`.