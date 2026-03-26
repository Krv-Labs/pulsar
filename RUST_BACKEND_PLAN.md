# Thema-RS: Rust-Backend Implementation Plan

## Overview

`thema-rs` is a streamlined, production-grade sister repository to Thema. It retains the core algorithmic pipeline — imputation → PCA → Ball Mapper → Cosmic Graph — but reimplements the compute-heavy stages in Rust, exposed to Python via [PyO3](https://pyo3.rs/) + [maturin](https://www.maturin.rs/).

The goal is a library that is blazing fast, minimal in dependencies, and ergonomic to use from Python for data analysis, clustering, and production inference.

---

## Scope

### What is implemented in Rust

| Stage | Python (Thema) equivalent | Rust implementation |
|---|---|---|
| Imputation | `Moon.fit()` via `inner_utils` | Sampling from normal/categorical, mean/median/mode fill |
| Standard scaling | `StandardScaler` inside `Moon.fit()` | Z-score normalization, column-wise |
| PCA | `pcaProj.fit()` via sklearn | Covariance-based SVD using `nalgebra` or `ndarray-linalg` |
| Ball Mapper | `pyballStar.fit()` via pyBallMapper | Native epsilon-ball graph construction |
| Pseudo-Laplacian | `mapper_pseudo_laplacian()` in `starHelpers.py` | n×n sparse matrix accumulation |
| Cosmic Graph | `normalize_cosmicGraph()` in `starHelpers.py` | Normalization + thresholding into weighted adjacency |

### What stays in Python

- Orchestration: iterating over parameter grids (imputation methods × PCA seeds/dims × BallMapper epsilons)
- Downstream hooks: clustering on the cosmic graph, visualization, labelling, export
- Config loading (YAML or dict)
- File I/O (saving/loading intermediate results as numpy arrays or parquet)

---

## Repo Structure

```
thema-rs/
├── Cargo.toml                  # Rust workspace root
├── pyproject.toml              # maturin build config + Python metadata
├── src/
│   ├── lib.rs                  # PyO3 module root — re-exports all Python-facing structs
│   ├── impute.rs               # Imputation methods
│   ├── scale.rs                # Standard scaler
│   ├── pca.rs                  # PCA via SVD
│   ├── ballmapper.rs           # Ball Mapper graph construction
│   ├── pseudolaplacian.rs      # Pseudo-Laplacian accumulation
│   └── cosmic.rs               # Cosmic graph normalization
├── thema_rs/                   # Python package (pure Python layer)
│   ├── __init__.py
│   ├── pipeline.py             # Orchestration: ThemaRS class
│   ├── hooks.py                # Python hooks and analysis utilities
│   └── config.py               # Config dataclasses / YAML loader
├── tests/
│   ├── test_impute.py
│   ├── test_pca.py
│   ├── test_ballmapper.py
│   └── test_cosmic.py
├── notebooks/
│   └── quickstart.ipynb
└── params.yaml.sample
```

---

## Rust Crate Design

### Build tooling

Use **maturin** to build a mixed Python/Rust package. The Rust code compiles to a native extension (`thema_rs._thema_rs`) and is re-exported from the Python package.

```toml
# Cargo.toml
[package]
name = "thema-rs"
version = "0.1.0"
edition = "2021"

[lib]
name = "_thema_rs"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"          # pyo3-numpy: zero-copy ndarray <-> numpy
ndarray = "0.16"
nalgebra = "0.33"       # SVD for PCA
rand = "0.8"
rand_distr = "0.4"      # Normal distribution sampling
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "thema_rs"
module-name = "thema_rs._thema_rs"
```

---

## Rust Module Specifications

### `impute.rs`

Implements the imputation methods that Thema uses in `inner_utils.py`. All methods operate on a 1D f64 array and a boolean mask of which indices are missing.

**Methods to implement:**
- `sample_normal(values, missing_mask, seed)` — fill NaNs by sampling `N(μ, σ)` from observed values. Mirrors `sampleNormal()`.
- `sample_categorical(values, missing_mask, seed)` — fill NaNs by sampling from the empirical categorical distribution. Mirrors `sampleCategorical()`.
- `fill_mean(values, missing_mask)` — fill with column mean.
- `fill_median(values, missing_mask)` — fill with column median.
- `fill_mode(values, missing_mask)` — fill with most frequent value.

**Python binding:**
```python
from thema_rs._thema_rs import impute_column
# impute_column(values: np.ndarray, missing_mask: np.ndarray[bool], method: str, seed: int) -> np.ndarray
```

**Note:** The `add_imputed_flags` behavior (adding a binary indicator column per imputed column) should be handled in Python before calling into Rust, since it is a dataframe-level operation.

---

### `scale.rs`

Standard (Z-score) scaler operating on a 2D f64 matrix column-wise. Stores `mean` and `std` per column to allow transform/inverse-transform.

```rust
pub struct StandardScaler {
    means: Vec<f64>,
    stds: Vec<f64>,
}
impl StandardScaler {
    pub fn fit_transform(data: &Array2<f64>) -> (Array2<f64>, StandardScaler);
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64>;
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64>;
}
```

**Python binding:**
```python
from thema_rs._thema_rs import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(np_array)   # returns np.ndarray
```

---

### `pca.rs`

PCA via truncated SVD of the mean-centered data matrix. Uses `nalgebra`'s `SVD` decomposition.

**Parameters:** `n_components: usize`, `seed: u64` (used to seed random state for any stochastic initialization; deterministic for exact SVD).

```rust
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,   // eigenvectors (n_components x n_features)
    explained_variance: Option<Vec<f64>>,
}
impl PCA {
    pub fn fit_transform(data: &Array2<f64>) -> Array2<f64>;
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64>;
}
```

**Python binding:**
```python
from thema_rs._thema_rs import PCA
pca = PCA(n_components=2, seed=42)
embedding = pca.fit_transform(scaled_array)   # returns np.ndarray shape (n, n_components)
```

---

### `ballmapper.rs`

This is the core performance-critical component. Implements the Ball Mapper algorithm from [pyBallMapper](https://github.com/dioscuri-tda/pyBallMapper/blob/main/pyballmapper/ballmapper.py).

**Algorithm:**
1. For each point (in order), if it is not yet covered by any existing ball center, create a new ball centered on it.
2. A point is "covered" if its distance to any existing ball center is ≤ ε.
3. After all centers are chosen, assign every point to all balls whose center is within ε of it (membership).
4. Build the graph: nodes are balls; an edge exists between two nodes if they share at least one member point.

**Key output:** `nodes: HashMap<usize, Vec<usize>>` — maps ball ID to list of member point indices (matching pyBallMapper's `Graph` node attribute `"membership"`).

```rust
pub struct BallMapper {
    pub eps: f64,
    pub nodes: Vec<Vec<usize>>,       // nodes[ball_id] = [point indices]
    pub edges: Vec<(usize, usize)>,   // pairs of ball IDs
}
impl BallMapper {
    pub fn fit(points: &Array2<f64>, eps: f64) -> BallMapper;
}
```

**Python binding:**
```python
from thema_rs._thema_rs import BallMapper
bm = BallMapper(eps=0.5)
bm.fit(embedding_array)
# bm.nodes -> list[list[int]]    (ball_id -> member point indices)
# bm.edges -> list[tuple[int,int]]
# bm.to_networkx() -> nx.Graph   (implemented in Python wrapper)
```

**Performance notes:**
- Use a KD-tree or brute-force with early-exit for center selection. For datasets up to ~50k points and low-dimensional PCA embeddings (2-10D), brute-force with SIMD-friendly f64 distance is fast enough.
- Consider adding a `kd_tree` feature flag that switches to a spatial index (`kiddo` crate) for larger datasets.

---

### `pseudolaplacian.rs`

Computes the n×n pseudo-Laplacian matrix from a Ball Mapper result. This matches `mapper_pseudo_laplacian()` in `starHelpers.py`.

**Definition:**
- `L[i,i] += 1` for each ball that point i belongs to (diagonal = number of memberships)
- `L[i,j] -= 1` for each ball that both i and j belong to (off-diagonal = negative shared memberships)

This is called once per fitted `BallMapper` and accumulated across all graph instances to form the galactic pseudo-Laplacian.

```rust
pub fn pseudo_laplacian(nodes: &Vec<Vec<usize>>, n: usize) -> Array2<i64>;
```

**Sparsity note:** For large n, this matrix is sparse. Consider returning a sparse representation (COO triplets) and letting the Python side use `scipy.sparse` for summation. For moderate n (< 10k), a dense ndarray is fine.

**Python binding:**
```python
from thema_rs._thema_rs import pseudo_laplacian
L = pseudo_laplacian(bm.nodes, n=len(data))   # returns np.ndarray (n, n) int64
```

---

### `cosmic.rs`

Normalizes the summed galactic pseudo-Laplacian into a weighted adjacency matrix and builds the Cosmic Graph. Mirrors `normalize_cosmicGraph()` in `starHelpers.py`.

**Formula** (from existing code):
```
weight[i,j] = -(L[i,j]) / (L[i,i] + L[j,j] + L[i,j])   for i ≠ j
edge exists if weight[i,j] > threshold
```

```rust
pub struct CosmicGraph {
    pub weighted_adj: Array2<f64>,
    pub adj: Array2<u8>,
    pub n: usize,
}
impl CosmicGraph {
    pub fn from_pseudo_laplacian(L: &Array2<i64>, threshold: f64) -> CosmicGraph;
}
```

**Python binding:**
```python
from thema_rs._thema_rs import CosmicGraph
cg = CosmicGraph.from_pseudo_laplacian(galactic_L, threshold=0.0)
# cg.weighted_adj -> np.ndarray (n, n) float64
# cg.adj           -> np.ndarray (n, n) uint8
# cg.to_networkx() -> nx.Graph   (Python wrapper)
```

---

## Python Layer Design

### `thema_rs/pipeline.py` — `ThemaRS` class

The orchestrator. Does not touch Rust directly; calls the bindings through the module. Mirrors the `Thema.genesis()` flow but simplified and opinionated.

```python
class ThemaRS:
    def __init__(self, config: dict | str):
        """
        config: dict or path to YAML.
        Required keys:
          data: path to CSV or parquet
          impute: {column_name: {method: str, seed: int}, ...}
          drop_columns: [...]
          pca: {dimensions: [int, ...], seeds: [int, ...]}
          ball_mapper: {epsilons: [float, ...]}
          cosmic_graph: {threshold: float, neighborhood: str}
          n_reps: int   # number of representative models to select
        """

    def fit(self) -> "ThemaRS":
        """
        Runs the full pipeline:
        1. Impute + scale (Rust)
        2. PCA grid (Rust) — one embedding per (dim, seed) combo
        3. Ball Mapper grid (Rust) — one graph per (embedding, epsilon) combo
        4. Accumulate pseudo-Laplacians (Rust)
        5. Compute cosmic graph (Rust)
        Returns self for chaining.
        """

    @property
    def cosmic_graph(self) -> nx.Graph:
        """Cosmic graph as a NetworkX graph with 'weight' attributes."""

    @property
    def weighted_adjacency(self) -> np.ndarray:
        """n×n float64 weighted adjacency matrix."""

    @property
    def ball_maps(self) -> list[BallMapper]:
        """All fitted BallMapper objects across the grid."""

    def select_representatives(self, n_reps: int) -> list[BallMapper]:
        """
        Clusters the ball maps by graph edit distance or pseudo-Laplacian
        distance and selects n_reps diverse representatives.
        Thin Python wrapper; distance matrix computed in Python using
        the weighted_adjacency matrices.
        """
```

### `thema_rs/hooks.py` — Analysis hooks

Clean Python utilities that work on the outputs. These are the "production hooks".

```python
def label_points(ball_mapper: BallMapper, n: int) -> np.ndarray:
    """
    Returns an (n,) int array: for each data point, the ID of its
    primary ball assignment (-1 if unclustered).
    Useful as soft cluster labels.
    """

def membership_matrix(ball_mapper: BallMapper, n: int) -> np.ndarray:
    """
    Returns a dense (n, n_balls) binary matrix.
    Useful for downstream soft clustering analysis.
    """

def cosmic_clusters(
    cosmic_graph: nx.Graph,
    method: str = "agglomerative",
    n_clusters: int = 5,
) -> np.ndarray:
    """
    Runs clustering on the cosmic graph adjacency matrix.
    Returns (n,) int array of cluster labels.
    method: "agglomerative" | "spectral" | "hdbscan"
    """

def graph_to_dataframe(ball_mapper: BallMapper, data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with one row per ball node, including:
    - node_id, size (member count), centroid coordinates,
    - mean/std of each original feature for members in that node.
    Useful for interpretability.
    """

def unclustered_points(ball_mapper: BallMapper, n: int) -> list[int]:
    """Returns list of point indices not covered by any ball."""
```

---

## Parameter Grid Execution

Unlike Thema's file-based pipeline, `thema-rs` keeps all intermediate results in memory (as numpy arrays) and only writes to disk on explicit request. This avoids pickle I/O overhead that currently dominates Thema's runtime for large grids.

**Grid expansion:**
```
N_imputed_versions  (1 per imputation seed combination)
× N_pca             (len(dimensions) × len(seeds))
× N_epsilon         (len(epsilons))
= total ball maps
```

All BallMapper fits are independent and should be run in parallel. Use Rust's `rayon` crate for data-parallel execution across the grid without Python GIL involvement.

```rust
// In ballmapper.rs or pipeline.rs
use rayon::prelude::*;

pub fn fit_grid(embeddings: &[Array2<f64>], epsilons: &[f64]) -> Vec<BallMapper> {
    embeddings.par_iter()
        .flat_map(|emb| {
            epsilons.par_iter().map(|&eps| BallMapper::fit(emb, eps))
        })
        .collect()
}
```

---

## Reference Implementation Guide

When building the Rust components, use the following files from Thema as the reference specification:

| Rust module | Reference Python file | Key logic to port |
|---|---|---|
| `impute.rs` | `thema/multiverse/system/inner/inner_utils.py` | `sampleNormal`, `sampleCategorical`, `mean`, `median`, `mode` |
| `scale.rs` | `thema/multiverse/system/inner/moon.py` lines 202–210 | `StandardScaler.fit_transform` |
| `pca.rs` | `thema/multiverse/system/outer/projectiles/pcaProj.py` | `PCA(n_components, random_state).fit_transform(data)` |
| `ballmapper.rs` | [pyBallMapper source](https://github.com/dioscuri-tda/pyBallMapper/blob/main/pyballmapper/ballmapper.py) | Ball center selection loop + membership assignment + edge construction |
| `pseudolaplacian.rs` | `thema/multiverse/universe/utils/starHelpers.py` lines 55–118 | `mapper_pseudo_laplacian()` with `neighborhood="node"` |
| `cosmic.rs` | `thema/multiverse/universe/utils/starHelpers.py` lines 9–52 | `normalize_cosmicGraph()` |

---

## Key BallMapper Algorithm (for the implementer)

This is the heart of the Rust port. From pyBallMapper's source:

```
Input: point cloud X (n × d array), epsilon ε

Centers = []
For i in 0..n:
    if no center c in Centers satisfies dist(X[i], c) <= ε:
        Centers.append(i)   # point i becomes a new ball center

Nodes = {}   # maps center_index -> [member_indices]
For center_idx in Centers:
    Nodes[center_idx] = [i for i in 0..n if dist(X[i], X[center_idx]) <= ε]

Edges = {}
For each pair (a, b) of center indices:
    if Nodes[a] ∩ Nodes[b] is non-empty:
        add edge (a, b)
```

Distance metric: Euclidean (L2). The Rust implementation should use SIMD-friendly `f64` distance.

For the Python-facing API, node IDs should be 0-indexed integers (not re-labelled to alphabet as Thema does).

---

## Development Sequence

Build and verify each stage in isolation with a round-trip test against the Python/Thema implementation on the same input data:

1. **`impute.rs`** — verify output matches `inner_utils.sampleNormal/sampleCategorical` for the same seed
2. **`scale.rs`** — verify scaled output matches `sklearn.preprocessing.StandardScaler`
3. **`pca.rs`** — verify that `PCA(n_components=k).fit_transform(X)` produces the same subspace (up to sign flip) as sklearn's PCA
4. **`ballmapper.rs`** — verify that `BallMapper(eps=ε).fit(X)` produces the same node membership sets as `pyballmapper.BallMapper(X, ε)`
5. **`pseudolaplacian.rs`** — verify matrix equals `mapper_pseudo_laplacian()` for the same nodes dict
6. **`cosmic.rs`** — verify weighted adjacency equals `normalize_cosmicGraph()` for the same summed Laplacian
7. **`pipeline.py`** — end-to-end: same data, same config → verify cosmic graph edges match Thema output

---

## Build and Test Commands

```bash
# Install maturin
pip install maturin

# Build and install in dev mode (rebuilds Rust on changes)
maturin develop

# Run Python tests
pytest tests/ -v

# Build release wheel
maturin build --release

# Run a single test
pytest tests/test_ballmapper.py -v
```

---

## Recommended Rust Crates

| Crate | Purpose |
|---|---|
| `pyo3` | Python/Rust FFI |
| `numpy` (pyo3-numpy) | Zero-copy ndarray ↔ numpy array bridging |
| `ndarray` | N-dimensional array operations |
| `nalgebra` | SVD for PCA |
| `rand` + `rand_distr` | Seeded normal/categorical sampling |
| `rayon` | Data-parallel BallMapper grid execution |
| `thiserror` | Ergonomic error types |
