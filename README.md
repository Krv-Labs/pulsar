# Pulsar

Rust-backed Python library for topological data analysis. Implements the Thema pipeline: imputation → scaling → PCA → Ball Mapper → Cosmic Graph.

Performance-critical algorithms are written in Rust (PyO3/maturin) and exposed as `pulsar._pulsar`. Python orchestrates the pipeline.

## Installation

Requires Rust and Python 3.13+.

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

Demo scripts and their parameter files live under `demos/`:

- `demos/coal.py` with `demos/coal_params.yaml`
- `demos/physionet.py` with `demos/physionet_params.yaml`

Run the coal demo (US Coal Plants):

```bash
uv run python demos/coal.py
```

Run the PhysioNet demo (synthetic mode):

```bash
uv run python demos/physionet.py --synthetic
```

Run the PhysioNet demo with a real eICU CSV export:

```bash
uv run python demos/physionet.py --data path/to/eicu_patient_static.csv
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
