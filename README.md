# Pulsar

Rust-backed Python library for topological data analysis. Implements the Thema pipeline: imputation → scaling → projection → Ball Mapper → Cosmic Graph.

Performance-critical algorithms are written in Rust (PyO3/maturin) and exposed as `pulsar._pulsar`. Python orchestrates the pipeline.



#### MCP!
<details> 
<summary>Let the Agent Do the Math</summary>

Until now, topological data analysis required a Ph.D. in algebraic topology and a masochistic tolerance for parameter tuning. 

By default, Pulsar exposes a rich Python API. This is great when you *actually* want to build custom pipelines, but when you just want to find out why your dataset is acting weird, writing boilerplate is tedious and unintuitive.

We get lots of complaints about it actually, with people asking things like:

> Why is my cosmic graph a hairball? What the hell should my epsilon range be? Why did my PCA just drop 90% of the variance?

We hear you, but we're not convinced that writing a 50-line hyperparameter grid search is what you really want. You don't want to have to manually calculate k-NN distances every time you load a CSV. And I doubt you really want to stare at a raw NetworkX adjacency matrix either — you want answers. You want to point an LLM at your data and say, "Find the natural clusters and tell me why they exist."

The Pulsar MCP (Model Context Protocol) Server is our attempt to give you what you *actually* want, without any of the downsides of doing something stupid like guessing topological parameters.

### Setup

Don't overcomplicate this. Add the server to your Claude Desktop config (or Gemini CLI, or whatever you're using):

```json
{
  "mcpServers": {
    "pulsar": {
      "command": "uvx",
      "args": ["--from", "thema-pulsar[mcp]", "pulsar-mcp"],
      "env": {}
    }
  }
}
```

This pulls `thema-pulsar` straight from PyPI — no clone, no `uv sync` required. If you prefer a persistent install, `pipx install "thema-pulsar[mcp]"` and use `"command": "pulsar-mcp"` instead.

Restart your client. Done.

### General Overview

What follows from here is the exact workflow we designed to dogfood the pipeline. It covers every sensible step of a topological analysis, from geometry probing to statistical dossiers. 

It's important you let the agent follow this exact sequence for a few reasons:
1. We want the graph to actually have signal out of the box.
2. Really just the first reason, that's the whole point of these tools.

Here is the exact loop the agent should run:

1. **Ingest the dataset** to get a stable `dataset_id` handle.
2. **Create a calibrated config** via `create_config(dataset_id)` — calibrates epsilon and projection dimensions against the processed feature space.
3. **Sweep the topology** using that config.
4. **Diagnose the graph** to see if it's a giant useless blob or actually balanced. Use the metrics to decide what to adjust, then iterate via `refine_config`.
5. **Generate the dossier** to explain the clusters in plain English.
6. **Compare clusters** for academic-grade p-values.
7. **Export** the labeled data.

### Tool Fly-By

We didn't just wrap our Python functions in JSON schemas. We built *Thick Tools*—stateful, workflow-aware engines that pass configuration directly between each other so you don't have to watch the agent screw up file I/O.

*   `create_config(dataset_id)`: The primary config generation tool. Analyzes k-NN distances and projection dimensions in the *processed* feature space (after preprocessing + scaling) to produce a calibrated YAML config. Never let the agent guess parameters.
*   `run_topological_sweep`: Runs the heavy Rust pipeline. Takes inline YAML and returns structured JSON with metrics and experiment diff. Config persistence is opt-in via `save_config=True`.
*   `diagnose_cosmic_graph`: Returns current graph-state observables: scale, component morphology, weight distribution, sweep support, observed patterns, and risk factors. The agent interprets those measurements against the user's objective.
*   `generate_cluster_dossier`: Returns structured JSON with per-cluster profiles (Z-scores, homogeneity, concentration) plus a Markdown summary. Includes clustering method metadata (method used, silhouette score).
*   `compare_clusters`: Runs Welch's T-tests, KS-tests, and Cohen's d between two specific clusters. Because sometimes your boss wants a p-value.
*   `export_labeled_data`: Maps semantic names to a `cluster_assignment_id` from `generate_cluster_dossier` and dumps that exact clustering to a CSV.

### Pitfalls & Annoyances

We try to make things foolproof, but some of you goofballs are going to try to break it anyway. Here is what to avoid:

*   **Don't let the agent write YAML files manually.** 
    The tools pass YAML strings directly in memory (`suggested_params_yaml` -> `config_yaml`). If you watch the agent try to use `write_file` to save a `params.yaml` before running the sweep, stop it. If you make the agent do unnecessary file I/O you belong in prison.
*   **Don't skip the diagnosis step.**
    If the graph is a giant hairball, your clusters will be garbage. Use `diagnose_cosmic_graph` to inspect graph-state measurements, then decide whether the user's objective calls for threshold inspection, config refinement, or a different interpretation surface.
*   **Handle non-numeric data appropriately.**
    Pulsar is a geometric engine. It needs floats. `characterize_dataset` will automatically tell the agent which low-cardinality strings to one-hot encode and which high-cardinality strings to drop. Don't fight it.

</details>

> [!NOTE]  
> For more guides, workflows, and an end-to-end MCP example, see [`demos/penguins/README.md`](demos/penguins/README.md).

## Citation

If you use this package in your research, please cite:

```bibtex
@article{Gathrid2025,
  author  = {Gathrid, Sidney and Wayland, Jeremy and Wayland, Stuart and Deshmukh, Ranjit and Wu, Grace C.},
  title   = {Strategies to accelerate US coal power phase-out using contextual retirement vulnerabilities},
  journal = {Nature Energy},
  year    = {2025},
  volume  = {10},
  number  = {10},
  pages   = {1274--1288},
  month   = {October},
  doi     = {10.1038/s41560-025-01871-0},
  url     = {https://doi.org/10.1038/s41560-025-01871-0},
  issn    = {2058-7546}
}
```
Which introduced the original [`Thema`](https://github.com/Krv-Labs/Thema) algorithm.

## Installation

Requires Rust and Python 3.10+.

```bash
uv sync
uv run maturin develop --release
```

## Quick start

```python
from pulsar import ThemaRS

model = ThemaRS("params.yaml").fit()

graph = model.cosmic_graph        # sparse networkx.Graph with 'weight' edge attributes
adj   = model.weighted_adjacency  # dense weighted adjacency, materialized lazily on access
edges = model.weighted_edges()    # thresholded sparse edge list
reps  = model.select_representatives()  # uses the configured default

# Opt-in spectral sparsification hook: a leverage-aware, epsilon-controlled graph
# that preserves spectrum/distances (not topology). Useful for spectral analysis;
# it is NOT a construction-time speedup. update=True refreshes model.cosmic_graph.
model.spectral_sparsify(epsilon=0.8, seed=7, update=True)
```

Copy `params.yaml.sample` to `params.yaml` and edit it for your dataset.

## Progress reporting

- Stage weight constants for `progress_callback` live in `pulsar.runtime.utils._STAGE_WEIGHTS`; `pulsar.runtime.utils._build_cumulative_fractions` turns them into cumulative fractions (used by `ThemaRS.fit`).
- `_rayon_thread_override` in `pulsar.runtime.utils` caps `RAYON_NUM_THREADS` for Rust-heavy stages when notebooks need stricter thread control.
- For notebooks: use `pulsar.runtime.progress.fit_with_progress(model, data)` or `fit_multi_with_progress(model, datasets)` — renders a transient rich progress bar.

## Demos

Demo scripts organized by domain under `demos/`:

**Energy domain:**
```bash
uv run python demos/energy/coal.py  # US Coal Plants (downloads dataset automatically)
```

**EHR domain:**
```bash
uv run python demos/ehr/physionet.py --synthetic             # Synthetic data mode
uv run python demos/ehr/physionet.py --data path/to/eicu.csv # Real eICU CSV data
uv run python demos/ehr/ecg_arrhythmia.py                    # ECG arrhythmia classification
```

**LLM/MMLU domain:**
```bash
jupyter notebook demos/mmlu/mmlu_topology_demo.ipynb
```

## Configuration

Cosmic graph thresholding is automatic by default, and representative selection has a sensible default. Most users only need to configure data, preprocessing, and sweeps.

```yaml
run:
  name: my_experiment
  data: path/to/data.csv # CSV or parquet

preprocessing:
  drop_columns: [id, timestamp]
  impute:
    age:
      method: sample_normal # fill_mean | fill_median | fill_mode |
      seed: 42 # sample_normal | sample_categorical
    category:
      method: sample_categorical
      seed: 7

sweep:
  projection:
    method: jl # default; set to pca for legacy randomized PCA
    dimensions:
      values: [2, 3, 5]
    seed:
      values: [42, 7, 13]
    center: true
  ball_mapper:
    epsilon:
      range: { min: 0.1, max: 1.5, steps: 8 } # or: values: [0.3, 0.5, 0.8]
cosmic_graph:
  construction: minhash
  minhash_d: 256
  minhash_seed: 42
  construction_threshold: auto
  sparsify: false # opt-in spectral sparsification hook (off by default)
  sparsify_epsilon: 1.0
  sparsify_seed: 42
```

## Development

```bash
uv run maturin develop        # debug build
uv run maturin develop --release  # optimised build
uv run pytest tests/ -v
uv run pytest tests/test_benchmark_accelerations.py -s -v
```
