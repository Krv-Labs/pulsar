# Pulsar

Rust-backed Python library for topological data analysis. Implements the Thema pipeline: imputation → scaling → PCA → Ball Mapper → Cosmic Graph.

The performance-critical algorithms are written in Rust (via PyO3/maturin) and exposed as `pulsar._pulsar`. A Python layer (`pulsar`) orchestrates the pipeline and provides analysis utilities.

---

## Installation

Requires the Rust toolchain and Python 3.13+.

```bash
# Install Rust if necessary
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

uv sync
uv run maturin develop --release
```

For development (includes pytest):

```bash
uv sync --group dev
```

---

## Quick start

### Low-level API

```python
import numpy as np
from pulsar._pulsar import (
    impute_column, StandardScaler, PCA,
    BallMapper, ball_mapper_grid, pseudo_laplacian, CosmicGraph,
)

# 1. Impute missing values
arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
filled = impute_column(arr, "fill_mean")

# 2. Scale features (population std, ddof=0)
X = np.random.standard_normal((100, 4))
scaler = StandardScaler()
X_scaled = np.array(scaler.fit_transform(X))

# 3. Reduce dimensionality
pca = PCA(n_components=2, seed=42)
X_2d = np.array(pca.fit_transform(X_scaled))

# 4. Build a topological complex
bm = BallMapper(eps=0.8)
bm.fit(X_2d)
print(bm.nodes)   # list[list[int]] — ball membership
print(bm.edges)   # list[tuple[int,int]] — shared-member connections

# 5. Accumulate pseudo-Laplacians across a parameter sweep
n = len(X_2d)
galactic_L = np.zeros((n, n), dtype=np.int64)
for bm in ball_mapper_grid([X_2d], [0.5, 0.8, 1.2]):
    galactic_L += np.array(pseudo_laplacian(bm.nodes, n))

# 6. Build the Cosmic Graph
cg = CosmicGraph.from_pseudo_laplacian(galactic_L, threshold=0.0)
print(cg.weighted_adj)  # n×n float64 weights in [0, 1]
print(cg.adj)           # n×n uint8 binary adjacency
```

### Full pipeline

```python
from pulsar import ThemaRS

model = ThemaRS("params.yaml").fit()

graph = model.cosmic_graph        # networkx.Graph with 'weight' edge attributes
adj   = model.weighted_adjacency  # np.ndarray, shape (n, n)
maps  = model.ball_maps           # list[BallMapper] — full grid

# Pick diverse representatives
reps = model.select_representatives(n_reps=4)
```

---

## Configuration

Copy `params.yaml.sample` to `params.yaml` and edit:

```yaml
run:
  name: my_experiment
  data: path/to/data.csv # CSV or parquet

preprocessing:
  drop_columns: [id, timestamp]

  impute: # one block per column with missing values
    age:
      method: sample_normal # see imputation methods below
      seed: 42
    category:
      method: sample_categorical
      seed: 7

sweep:
  pca:
    dimensions:
      values: [2, 3, 5] # or: range: {min: 2, max: 8, steps: 4}
    seed:
      values: [42, 7, 13]

  ball_mapper:
    epsilon:
      values: [0.3, 0.5, 0.8] # or: range: {min: 0.1, max: 1.5, steps: 8}

cosmic_graph:
  threshold: 0.0 # minimum weight to include in binary adjacency

output:
  n_reps: 4 # number of representative ball maps to select
```

You can also pass a dict directly instead of a file path:

```python
model = ThemaRS({
    "run": {"data": "data.csv"},
    "preprocessing": {"drop_columns": [], "impute": {}},
    "sweep": {
        "pca": {"dimensions": {"values": [2, 3]}, "seed": {"values": [42]}},
        "ball_mapper": {"epsilon": {"values": [0.5, 1.0]}},
    },
    "cosmic_graph": {"threshold": 0.0},
    "output": {"n_reps": 4},
}).fit()
```

---

## API reference

### `impute_column(values, method, seed=0)`

Fill NaN values in a 1-D float64 array.

| Method                 | Behaviour                                     |
| ---------------------- | --------------------------------------------- |
| `"fill_mean"`          | Replace with column mean                      |
| `"fill_median"`        | Replace with column median                    |
| `"fill_mode"`          | Replace with most frequent value              |
| `"sample_normal"`      | Draw from `N(μ, σ)` fitted to observed values |
| `"sample_categorical"` | Weighted sampling from empirical distribution |

`seed` is used by the two sampling methods and ignored otherwise.

---

### `StandardScaler`

Z-score normalisation using population std (ddof=0), equivalent to sklearn's default.

```python
scaler = StandardScaler()
X_scaled  = np.array(scaler.fit_transform(X_train))   # fits and transforms
X_test_sc = np.array(scaler.transform(X_test))         # reuses stored stats
X_orig    = np.array(scaler.inverse_transform(X_scaled))
```

Constant columns (std < 1e-10) are treated as std=1 so they become all-zero rather than NaN.

---

### `PCA(n_components, seed=0)`

Exact SVD-based PCA. `seed` is present for API consistency but unused.

```python
pca = PCA(n_components=3, seed=42)
X_reduced = np.array(pca.fit_transform(X))    # centres, computes covariance, SVD
X_new     = np.array(pca.transform(X_new))    # project new data (same centering)
ev        = np.array(pca.explained_variance)  # singular values, descending
```

**Algorithm**: centre → covariance `XᵀX / (n−1)` → SVD via nalgebra → sign flip (largest-abs-value element of each component is positive, matching sklearn convention).

---

### `BallMapper(eps)`

Greedy topological covering. Deterministic: no randomness.

```python
bm = BallMapper(eps=0.8)
bm.fit(points)             # np.ndarray, shape (n_points, n_dims)

bm.nodes    # list[list[int]] — bm.nodes[k] = point indices in ball k
bm.edges    # list[tuple[int,int]] — (a, b) with a < b if balls share a point
bm.eps      # float
bm.n_nodes()
bm.n_edges()
```

**Algorithm**: (1) greedy centre selection — point becomes a centre if no existing centre is within `eps`; (2) membership — all points within `eps` of each centre; (3) edges — pairs of balls sharing ≥1 member.

---

### `ball_mapper_grid(embeddings, epsilons)`

Parallel sweep over all `(embedding, epsilon)` pairs using rayon.

```python
results = ball_mapper_grid(embeddings, epsilons)
# len(results) == len(embeddings) * len(epsilons)
# Order: row-major — embeddings outer, epsilons inner
```

---

### `pseudo_laplacian(nodes, n)`

Discrete Laplacian from ball membership. Integer arithmetic, result is `np.int64`.

```python
L = np.array(pseudo_laplacian(bm.nodes, n))
# L[i, i]  = number of balls containing point i
# L[i, j]  = -(number of balls containing both i and j)   for i ≠ j
```

Accumulate across multiple Ball Mappers before building the Cosmic Graph:

```python
galactic_L = np.zeros((n, n), dtype=np.int64)
for bm in all_ball_mappers:
    galactic_L += np.array(pseudo_laplacian(bm.nodes, n))
```

---

### `CosmicGraph.from_pseudo_laplacian(L, threshold)`

Normalised co-membership graph. Port of Thema's `normalize_cosmicGraph`.

```python
cg = CosmicGraph.from_pseudo_laplacian(galactic_L, threshold=0.0)

cg.weighted_adj  # np.ndarray[float64, (n, n)] — weights in [0, 1]
cg.adj           # np.ndarray[uint8, (n, n)]   — binary (W[i,j] > threshold)
cg.n             # int
```

**Weight formula** for off-diagonal entry `(i, j)`:

```
denom   = L[i,i] + L[j,j] + L[i,j]
W[i,j]  = -L[i,j] / denom    if denom > 0
         = 0                   otherwise
```

Maximum weight 1.0 when two points always appear together in every ball.

---

### Analysis utilities (`pulsar.hooks`)

```python
from pulsar import label_points, membership_matrix, cosmic_clusters
from pulsar import graph_to_dataframe, unclustered_points, cosmic_to_networkx

# Assign each point to its first ball (-1 if uncovered)
labels = label_points(bm, n)                          # (n,) int64

# Dense binary membership matrix
M = membership_matrix(bm, n)                          # (n, n_balls) uint8

# Cluster points using the cosmic graph
cluster_labels = cosmic_clusters(
    model.cosmic_graph,
    method="spectral",     # "agglomerative" | "spectral" | "hdbscan"
    n_clusters=5,
)

# Per-ball statistics (mean, std of original features)
node_df = graph_to_dataframe(bm, original_dataframe)

# Points not covered by any ball
uncovered = unclustered_points(bm, n)

# Convert to NetworkX (edges have 'weight' attributes)
g = cosmic_to_networkx(cg)
```

---

## Pipeline steps (ThemaRS.fit)

1. Load data (CSV or parquet)
2. Drop configured columns
3. Add `{col}_was_missing` indicator flags for imputed columns
4. Impute each configured column (Rust)
5. Drop remaining NaN rows
6. StandardScale the full feature matrix (Rust)
7. PCA grid — all `(dimension, seed)` combinations (Rust)
8. BallMapper grid — all `(embedding, epsilon)` pairs, parallel (Rust + rayon)
9. Accumulate pseudo-Laplacians across all Ball Mappers (Rust + numpy)
10. Build CosmicGraph from accumulated Laplacian (Rust)
11. Convert to NetworkX graph (Python)

---

## Development

```bash
# Build (debug, faster compilation)
uv run maturin develop

# Build (release, with LTO)
uv run maturin develop --release

# Run all tests
uv run pytest tests/ -v

# Run only correctness tests (Python reference vs Rust)
uv run pytest tests/correctness/ -v
```

The `tests/correctness/` submodule contains transparent Python reference implementations of every algorithm. Each test file defines the formula in plain Python and asserts that the Rust output matches numerically. These are the right place to look if you want to verify what a function is supposed to compute.

---

## Errors

All Rust functions raise `ValueError` on bad input:

| Situation                                                       | Raised by               |
| --------------------------------------------------------------- | ----------------------- |
| Shape mismatch (e.g. `transform` with wrong number of features) | `StandardScaler`, `PCA` |
| All-NaN column                                                  | `impute_column`         |
| Unknown imputation method                                       | `impute_column`         |
| `n_components > n_features`                                     | `PCA`                   |
| SVD failed to converge (rare)                                   | `PCA`                   |
| `transform` called before `fit_transform`                       | `StandardScaler`, `PCA` |
