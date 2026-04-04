# Pulsar

Rust-backed Python library for topological data analysis. Implements the Thema pipeline: imputation → scaling → PCA → Ball Mapper → Cosmic Graph.

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

> [!NOTE]
> For more guides, workflows, and an end-to-end MCP example, see [`demos/penguins/README.md`](demos/penguins/README.md).

### Setup

Don't overcomplicate this. Add the server to your Claude Desktop config (or Gemini CLI, or whatever you're using):

```json
{
  "mcpServers": {
    "pulsar": {
      "command": "uv",
      "args": ["run", "pulsar-mcp"],
      "env": {}
    }
  }
}
```

Restart your client. Done.

### General Overview

What follows from here is the exact workflow we designed to dogfood the pipeline. It covers every sensible step of a topological analysis, from geometry probing to statistical dossiers. 

It's important you let the agent follow this exact sequence for a few reasons:
1. We want the graph to actually have signal out of the box.
2. Really just the first reason, that's the whole point of these tools.

Here is the exact loop the agent should run:

1. **Probe the geometry** to get a baseline YAML config.
2. **Sweep the topology** using that config.
3. **Diagnose the graph** to see if it's a giant useless blob or actually balanced. (If it's a blob, it gets a corrected YAML and retries step 2).
4. **Generate the dossier** to explain the clusters in plain English.
5. **Compare clusters** for academic-grade p-values.
6. **Export** the labeled data.

### Tool Fly-By

We didn't just wrap our Python functions in JSON schemas. We built *Thick Tools*—stateful, workflow-aware engines that pass configuration directly between each other so you don't have to watch the agent screw up file I/O.

*   `characterize_dataset`: Analyzes k-NN distances and PCA variance to suggest a grounded YAML config. Never let the agent guess parameters. 
*   `run_topological_sweep`: Runs the heavy Rust pipeline. It takes the inline YAML from the previous step. No writing to disk required.
*   `diagnose_cosmic_graph`: Evaluates the fitted graph. If it sees a "hairball" or "singletons," it literally hands the agent the corrected YAML to try again.
*   `generate_cluster_dossier`: Spits out a massive Markdown report of what makes each cluster unique (Z-scores, homogeneity, concentration). 
*   `compare_clusters_tool`: Runs Welch's T-tests, KS-tests, and Cohen's d between two specific clusters. Because sometimes your boss wants a p-value.
*   `export_labeled_data`: Maps semantic names to the cluster IDs and dumps it to a CSV.

### Pitfalls & Annoyances

We try to make things foolproof, but some of you goofballs are going to try to break it anyway. Here is what to avoid:

*   **Don't let the agent write YAML files manually.** 
    The tools pass YAML strings directly in memory (`suggested_params_yaml` -> `config_yaml`). If you watch the agent try to use `write_file` to save a `params.yaml` before running the sweep, stop it. If you make the agent do unnecessary file I/O you belong in prison.
*   **Don't skip the diagnosis step.** 
    If the graph is a giant hairball, your clusters will be garbage. The diagnosis tool uses binary search over previous sweep history to fix the epsilon range automatically. Let it correct itself.
*   **Handle non-numeric data appropriately.**
    Pulsar is a geometric engine. It needs floats. `characterize_dataset` will automatically tell the agent which low-cardinality strings to one-hot encode and which high-cardinality strings to drop. Don't fight it.

</details>

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

graph = model.cosmic_graph        # networkx.Graph with 'weight' edge attributes
adj   = model.weighted_adjacency  # np.ndarray, shape (n, n)
reps  = model.select_representatives()  # uses the configured default
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
  pca:
    dimensions:
      values: [2, 3, 5]
    seed:
      values: [42, 7, 13]
  ball_mapper:
    epsilon:
      range: { min: 0.1, max: 1.5, steps: 8 } # or: values: [0.3, 0.5, 0.8]
```

## Development

```bash
uv run maturin develop        # debug build
uv run maturin develop --release  # optimised build
uv run pytest tests/ -v
```

