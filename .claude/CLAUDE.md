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

## Rust Pitfalls (Known Footguns)

- **ndarray ↔ nalgebra layout mismatch**: ndarray is row-major; nalgebra is column-major. Never use `DMatrix::from_iterator` with `ndarray::iter()` — it silently transposes. Use `DMatrix::from_row_slice` with contiguous row-major data instead.
- **No `.unwrap()` on `partial_cmp` for `f64`**: NaN makes `partial_cmp` return `None`. Either guard against NaN upstream and use `.expect()` with a rationale, or return a `PulsarError`.
- **Validate all inputs at the boundary**: Every `#[pyfunction]` and `#[pymethods]` entry point must validate dimensions, shapes, and edge cases (0 rows, 0 cols, n < 2 for statistics, parameter bounds) *before* any computation. A panic in Rust kills the Python process.

## Testing Standards

- **Correctness tests must compare against a reference** (e.g., sklearn, scipy, known analytic solution). A test that only checks shapes, signs, or ordering can pass despite silently wrong numerical results.
- **Include at least one "known-answer" test** per numerical function: construct input where the correct output is analytically obvious (e.g., PCA on axis-aligned variance should recover that axis).
- **Edge-case tests are mandatory**: 0 rows, 1 row, 0 columns, NaN input, dimension exceeding feature count. These must assert the correct error is raised, not just that *some* error occurs — match on the error message.

## Performance Review Standards

Before optimizing, profile and **prove** the bottleneck. Common false leads in this codebase:

- **Wrong target**: k-NN preprocessing, graph component computation, and scale/impute are NOT bottlenecks — Rust handles them efficiently. Do not move them.
- **Real targets**: Python loops over graph edges (`cosmic_to_networkx`), redundant PCA recomputation across retries, and unnecessary O(n²) allocations in Python post-processing.
- **Vectorize at Python/Rust boundaries**: Replace Python for-loops over arrays returned from Rust with `np.where`, `np.fromiter`, or `nx.from_numpy_array`. Example — weight assignment in `cosmic_to_networkx`:
  ```python
  # WRONG (O(E) Python loop):
  for u, v in G.edges():
      G[u][v]["weight"] = wadj[u, v]
  # CORRECT (vectorized):
  return nx.from_numpy_array(np.where(adj > 0, wadj, 0.0))
  ```
  Note: `wadj * adj` is semantically wrong here — if `adj[i,j] == 0` but `wadj[i,j] > 0`, you get spurious edges. Use `np.where`.

- **Cache expensive invariant computations**: PCA embeddings are invariant to epsilon changes. Cache them with a fingerprint (SHA256 of `{data_path, dimensions, seeds, n_rows}`) and reuse across retries. See `pulsar/mcp/server.py::_pca_fingerprint`.

## Async / MCP Patterns

When integrating `async def` MCP tools with blocking Python code:

- **Never call blocking fit() directly in `async def`** — it starves the event loop for 2–30s. Use `asyncio.to_thread()`.
- **Never use `asyncio.get_event_loop()`** (deprecated in 3.10+). Use `asyncio.get_running_loop()` inside `async def`.
- **Calling async from sync thread**: Use `asyncio.run_coroutine_threadsafe(coro, loop)` where `loop` is captured *before* entering the thread.
- **FastMCP progress**: `ctx.report_progress(progress, total, message)` is the correct API (FastMCP 3.2.0+). It silently no-ops if the client sends no progress token — safe to call unconditionally.

```python
# Canonical MCP + asyncio pattern for blocking fit():
loop = asyncio.get_running_loop()

def progress_callback(stage: str, fraction: float) -> None:
    asyncio.run_coroutine_threadsafe(
        ctx.report_progress(progress=fraction, total=1.0, message=stage), loop,
    )

await asyncio.to_thread(model.fit, progress_callback=progress_callback)
```

## Progress Callback Pattern

`ThemaRS.fit()` and `fit_multi()` accept `progress_callback: Callable[[str, float], None] | None`. Convention:

- Callback signature: `(stage_name: str, cumulative_fraction: float) -> None`
- Fractions are cumulative [0, 1], monotonically increasing, ending at exactly 1.0
- Stage weights live in `_STAGE_WEIGHTS` list of tuples (module-level, ordered). `_build_cumulative_fractions()` converts to a schedule with the final entry pinned to 1.0.
- Cursor-based `_notify()`: each call advances a cursor through the pre-built schedule. Use `_notify("label override")` to change the display name without affecting the fraction (e.g., `_notify("pca (cached)")`).
- Callback exceptions propagate with their original type — no wrapping.
- For notebooks: use `pulsar.progress.fit_with_progress(model, data)` or `fit_multi_with_progress(model, datasets)` — renders a transient rich progress bar.

## Key Architectural Decisions

- **`CosmicGraph.from_pseudo_laplacian` requires `int64`**: The pseudo-Laplacian accumulator expects integer counts. If constructing test inputs in Python, use `np.ascontiguousarray(L, dtype=np.int64)`.
- **`threshold: "auto"` is always correct for EHR/high-dim data**: Do not default to `"0.0"` or a fixed value for high-dimensional data — it produces a maximally connected, structureless graph. The H₀ persistent homology stability analysis exists precisely for this case.
- **Randomized SVD introduces small variance approximation error**: PCA via Halko et al. 2011 gives approximate, not exact, singular values. Document this in characterization code; do not treat approximate variance ratios as ground truth.
- **MCP session state lives in `_PulsarSession`**: Embeddings, fingerprint, and model are stored per-session. Read/write session state in the `async` context (before/after `to_thread`), not inside the thread.