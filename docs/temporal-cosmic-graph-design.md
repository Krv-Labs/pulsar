# Temporal Cosmic Graph Design

This document outlines the design for extending Pulsar to support time series data with a fixed node set.

## Motivation

The current Cosmic Graph summarizes topological proximity across a **spatial sweep** of hyperparameters (PCA dimensions, seeds, epsilon values). Each representation in the sweep produces a pseudo-Laplacian, and these are accumulated to compute edge weights representing the *likelihood* of connectivity across representations.

For time series data (EHR, video, clinical notes), we need an analogous structure where representations vary **over time** rather than across hyperparameters.

## Core Data Structure: 3D Temporal Tensor

The natural generalization is a 3D tensor:

```
W[i, j, t] ∈ [0, 1]
```

Where:
- `i, j` are node indices (patients, frames, documents)
- `t` is the time index
- `W[i,j,t]` is the normalized co-membership weight at time `t`

This tensor can be:
- Computed by running the standard Pulsar pipeline independently at each time step
- Stored densely for small/medium datasets or sparsely for large ones
- Aggregated into summary graphs using various temporal statistics

## Target Data Modalities

| Modality | Node Type | Time Unit | Example |
|----------|-----------|-----------|---------|
| EHR (longitudinal) | Patients | Days / Admissions | ICU stays over hospitalization |
| Imaging (video) | Frames or regions | Seconds / Frames | Echocardiogram sequences |
| Clinical notes | Patients or documents | Visits / Documents | Progress notes over treatment |

## Aggregation Strategies

Given the 3D tensor `W[i, j, t]`, we define several useful graph summaries:

### 1. Persistence Graph (stable structure)

Edge weight = fraction of time steps where edge exists above threshold.

```
W_persist[i,j] = mean_t(W[i,j,t] > τ)
```

**Clinical meaning:** Identifies patients who are *always* similar — stable phenotype clusters that persist across the observation window.

**Use case:** Finding robust patient subgroups for stratified analysis.

### 2. Mean Graph (average connectivity)

Edge weight = mean weight across all time steps.

```
W_mean[i,j] = mean_t(W[i,j,t])
```

**Clinical meaning:** Overall similarity accounting for all observations equally.

**Use case:** General-purpose summary when no temporal weighting is needed.

### 3. Recency-Weighted Graph (current state emphasis)

Edge weight = exponentially decayed sum favoring recent observations.

```
W_recent[i,j] = Σ_t λ^(T-t) · W[i,j,t] / Σ_t λ^(T-t)
```

Where `λ ∈ (0, 1)` is the decay factor (e.g., 0.9).

**Clinical meaning:** Current patient similarity for real-time decision support, where recent observations matter more than distant history.

**Use case:** Clinical decision support, patient matching for treatment recommendations.

### 4. Volatility Graph (relationship instability)

Edge weight = temporal variance of the edge.

```
W_volatile[i,j] = var_t(W[i,j,t])
```

**Clinical meaning:** Identifies patient pairs whose similarity is *unstable* — one or both may be on a trajectory (deteriorating, responding to treatment, transitioning states).

**Use case:** Early warning systems, identifying patients requiring closer monitoring.

### 5. Trend Graph (directionality)

Edge weight = slope of linear regression over time.

```
W_trend[i,j] = slope of linear fit to W[i,j,:]
```

**Clinical meaning:** Positive values indicate converging patients (becoming more similar), negative values indicate diverging patients.

**Use case:** Trajectory analysis, treatment response monitoring.

### 6. Change-Point Graph (regime detection)

Edge weight = maximum change between consecutive time steps.

```
W_change[i,j] = max_t |W[i,j,t+1] - W[i,j,t]|
```

**Clinical meaning:** Identifies sudden state transitions — acute events, medication changes, procedure effects.

**Use case:** Event detection, anomaly identification in ICU data.

## Recommended Aggregations by Modality

| Modality | Primary Aggregation | Secondary Aggregation | Rationale |
|----------|---------------------|----------------------|-----------|
| EHR (longitudinal) | Persistence | Volatility | Distinguish stable vs unstable patient phenotypes |
| Video (imaging) | Recency | Change-point | Current state + motion/event detection |
| Clinical notes | Persistence | Trend | Stable themes + evolving narrative |

## API Design: `TemporalCosmicGraph`

```python
from pulsar import TemporalCosmicGraph

# Build from time-indexed data
tcg = TemporalCosmicGraph.from_snapshots(snapshots, threshold=0.0)

# Access the raw 3D tensor
tensor = tcg.tensor  # shape (n, n, T)

# Compute aggregated graphs
G_persist = tcg.persistence_graph(threshold=0.1)
G_mean = tcg.mean_graph()
G_recent = tcg.recency_graph(decay=0.9)
G_volatile = tcg.volatility_graph()
G_trend = tcg.trend_graph()
G_change = tcg.change_point_graph()

# Default summary (persistence)
G = tcg.to_networkx()

# Time-range queries
tcg_subset = tcg.slice(start=10, end=20)
```

## Implementation Considerations

### Storage Format

- **Dense (default):** `np.ndarray` of shape `(n, n, T)` — suitable for n < 10k, T < 1000
- **Sparse:** Store only non-zero entries as `(i, j, t, weight)` tuples — for large, sparse graphs
- **Chunked:** Memory-map or stream time slices — for very large datasets

### Rust Acceleration

The following operations benefit from Rust implementation:

1. `accumulate_temporal_pseudo_laplacians(ball_maps_per_time, n, T)` — parallel across time
2. Aggregation kernels (mean, variance, trend fitting) — SIMD-friendly
3. Sparse tensor operations if needed

### Time Labels

Two options:

1. **Index-only:** Time steps are 0..T, user maintains external metadata
2. **Labeled:** Store timestamps as `Vec<DateTime>` or similar, enable time-range queries

Recommend starting with index-only for simplicity, adding labels as a future enhancement.

## Open Questions

1. **Default aggregation:** Should `to_networkx()` return persistence, mean, or recency-weighted?
2. **Threshold handling:** Should each aggregation method accept its own threshold, or use a global one?
3. **Memory constraints:** At what scale should we switch to sparse/chunked representations?
4. **Streaming updates:** Should `TemporalCosmicGraph` support incremental addition of new time steps?

## Next Steps

1. Implement `TemporalCosmicGraph` Python class with dense tensor storage
2. Add Rust kernel for parallel pseudo-Laplacian accumulation across time steps
3. Implement core aggregation methods (persistence, mean, recency)
4. Add secondary aggregations (volatility, trend, change-point)
5. Write tests with synthetic temporal data
6. Create demo with longitudinal clinical data
