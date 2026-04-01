# Pulsar

Rust-backed Python library for topological data analysis. Implements the Thema pipeline: imputation → scaling → PCA → Ball Mapper → Cosmic Graph.

Performance-critical algorithms are written in Rust (PyO3/maturin) and exposed as `pulsar._pulsar`. Python orchestrates the pipeline.

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

