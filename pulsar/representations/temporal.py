"""
TemporalCosmicGraph — Cosmic Graph analysis for longitudinal time-series data.

This module extends Pulsar to handle data where the same set of nodes (e.g., patients)
are observed across multiple time steps. Instead of a single 2D weighted adjacency
matrix, we work with a 3D tensor W[i, j, t] representing edge weights at each time step.

## Core Data Structure

The temporal weighted adjacency tensor has shape (n, n, T) where:
- n is the number of nodes (fixed across time)
- T is the number of time steps
- W[i, j, t] ∈ [0, 1] is the normalized co-membership weight at time t

## Aggregation Strategies

Given the 3D tensor, we provide several methods to collapse into summary 2D graphs:

| Method | Formula | Clinical Meaning |
|--------|---------|------------------|
| persistence | mean_t(W > τ) | Stable relationships across time |
| mean | mean_t(W) | Average similarity |
| recency | Σ λ^(T-1-t) · W / Σ λ^(T-1-t) | Current state emphasis |
| volatility | var_t(W) | Relationship instability |
| trend | slope of linear fit | Converging/diverging trajectories |
| change_point | max |W[t+1] - W[t]| | Sudden state transitions |

## Example Usage

```python
from pulsar.representations import TemporalCosmicGraph

# Build from time-indexed snapshots
tcg = TemporalCosmicGraph.from_snapshots(
    snapshots=[X_t0, X_t1, X_t2, ...],  # List of (n, features) arrays
    config=config,
)

# Access raw 3D tensor
tensor = tcg.tensor  # shape (n, n, T)

# Compute aggregated graphs
G_persist = tcg.persistence_graph(threshold=0.1)
G_mean = tcg.mean_graph()
G_recent = tcg.recency_graph(decay=0.9)
G_volatile = tcg.volatility_graph()
G_trend = tcg.trend_graph()
G_change = tcg.change_point_graph()

# Convert to NetworkX
G = tcg.to_networkx(aggregation="persistence")
```
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from pulsar._pulsar import (
    StandardScaler,
    accumulate_temporal_pseudo_laplacians,
    ball_mapper_grid,
    pca_grid,
    py_normalize_temporal_laplacian,
)
from pulsar.config import PulsarConfig


class TemporalCosmicGraph:
    """
    Cosmic Graph for longitudinal time-series data.

    Stores a 3D tensor W[i, j, t] of edge weights and provides methods
    to aggregate into summary 2D graphs.
    """

    def __init__(self, tensor: np.ndarray, threshold: float = 0.0):
        """
        Initialize from a pre-computed 3D weighted adjacency tensor.

        Parameters
        ----------
        tensor : np.ndarray
            3D array of shape (n, n, T) with values in [0, 1].
        threshold : float
            Default threshold for binary adjacency operations.
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {tensor.shape}")
        if tensor.shape[0] != tensor.shape[1]:
            raise ValueError(f"Expected square slices, got shape {tensor.shape}")

        self._tensor = tensor.astype(np.float64)
        self._threshold = threshold
        self._n = tensor.shape[0]
        self._t = tensor.shape[2]

    @classmethod
    def from_snapshots(
        cls,
        snapshots: list[np.ndarray],
        config: PulsarConfig,
        threshold: float = 0.0,
    ) -> "TemporalCosmicGraph":
        """
        Build a TemporalCosmicGraph from time-indexed data snapshots.

        Runs the standard Pulsar pipeline (scale → PCA → BallMapper → pseudo-Laplacian)
        independently at each time step, then stacks results into a 3D tensor.

        Parameters
        ----------
        snapshots : list[np.ndarray]
            List of T arrays, each of shape (n, features_t). The number of rows n
            must be consistent across all snapshots (same node set over time).
        config : PulsarConfig
            Pulsar configuration specifying PCA dimensions, seeds, epsilon values, etc.
        threshold : float
            Default threshold for binary adjacency operations.

        Returns
        -------
        TemporalCosmicGraph
            Instance with 3D tensor of shape (n, n, T).
        """
        if not snapshots:
            raise ValueError("snapshots list cannot be empty")

        n = snapshots[0].shape[0]
        for t, snap in enumerate(snapshots):
            if snap.shape[0] != n:
                raise ValueError(
                    f"Inconsistent node count: snapshot 0 has {n} nodes, "
                    f"snapshot {t} has {snap.shape[0]} nodes"
                )

        # Process each time step
        ball_maps_per_time = []
        scaler = StandardScaler()

        for snap in snapshots:
            X = snap.astype(np.float64)
            X_scaled = np.array(scaler.fit_transform(X))

            embeddings = [
                np.ascontiguousarray(emb)
                for emb in pca_grid(X_scaled, config.pca.dimensions, config.pca.seeds)
            ]

            ball_maps = ball_mapper_grid(embeddings, config.ball_mapper.epsilons)
            ball_maps_per_time.append(ball_maps)

        # Accumulate into 3D pseudo-Laplacian tensor
        L_tensor = np.array(
            accumulate_temporal_pseudo_laplacians(ball_maps_per_time, n)
        )

        # Normalize to weighted adjacency
        W_tensor = np.array(py_normalize_temporal_laplacian(L_tensor))

        return cls(W_tensor, threshold=threshold)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def tensor(self) -> np.ndarray:
        """3D weighted adjacency tensor of shape (n, n, T)."""
        return self._tensor

    @property
    def n(self) -> int:
        """Number of nodes."""
        return self._n

    @property
    def T(self) -> int:
        """Number of time steps."""
        return self._t

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the tensor (n, n, T)."""
        return (self._n, self._n, self._t)

    # -------------------------------------------------------------------------
    # Aggregation Methods
    # -------------------------------------------------------------------------

    def persistence_graph(self, threshold: float | None = None) -> np.ndarray:
        """
        Compute persistence graph: fraction of time steps where edge exceeds threshold.

        W_persist[i,j] = mean_t(W[i,j,t] > τ)

        Clinical meaning: Identifies node pairs that are *always* similar —
        stable relationships that persist across the observation window.

        Parameters
        ----------
        threshold : float, optional
            Edge weight threshold. Defaults to instance threshold.

        Returns
        -------
        np.ndarray
            2D array of shape (n, n) with values in [0, 1].
        """
        tau = threshold if threshold is not None else self._threshold
        binary = (self._tensor > tau).astype(np.float64)
        return binary.mean(axis=2)

    def mean_graph(self) -> np.ndarray:
        """
        Compute mean graph: average edge weight across all time steps.

        W_mean[i,j] = mean_t(W[i,j,t])

        Clinical meaning: Overall similarity accounting for all observations equally.

        Returns
        -------
        np.ndarray
            2D array of shape (n, n) with values in [0, 1].
        """
        return self._tensor.mean(axis=2)

    def recency_graph(self, decay: float = 0.9) -> np.ndarray:
        """
        Compute recency-weighted graph: exponentially decayed sum favoring recent observations.

        W_recent[i,j] = Σ_t λ^(T-1-t) · W[i,j,t] / Σ_t λ^(T-1-t)

        where λ ∈ (0, 1) is the decay factor.

        Clinical meaning: Current similarity for real-time decision support,
        where recent observations matter more than distant history.

        Parameters
        ----------
        decay : float
            Decay factor λ in (0, 1). Values closer to 1 make the weights more uniform
            across time (less recency emphasis), while smaller values place more weight
            on the most recent steps. Default 0.9 means each step back is weighted
            0.9x the previous.

        Returns
        -------
        np.ndarray
            2D array of shape (n, n) with values in [0, 1].
        """
        if not 0 < decay < 1:
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        # Compute weights: λ^(T-1-t) for t in 0..T-1
        # At t=T-1 (most recent): weight = λ^0 = 1
        # At t=0 (oldest): weight = λ^(T-1)
        weights = np.array([decay ** (self._t - 1 - t) for t in range(self._t)])
        weights = weights / weights.sum()

        # Weighted sum along time axis
        return np.tensordot(self._tensor, weights, axes=([2], [0]))

    def volatility_graph(self) -> np.ndarray:
        """
        Compute volatility graph: temporal variance of each edge.

        W_volatile[i,j] = var_t(W[i,j,t])

        Clinical meaning: Identifies node pairs whose similarity is *unstable* —
        one or both may be on a trajectory (deteriorating, responding to treatment).

        Returns
        -------
        np.ndarray
            2D array of shape (n, n) with non-negative values.
        """
        return self._tensor.var(axis=2)

    def trend_graph(self) -> np.ndarray:
        """
        Compute trend graph: slope of linear regression over time for each edge.

        W_trend[i,j] = slope of linear fit to W[i,j,:]

        Clinical meaning: Positive values indicate converging nodes (becoming more
        similar over time), negative values indicate diverging nodes.

        Returns
        -------
        np.ndarray
            2D array of shape (n, n). Values can be positive or negative.
        """
        # Time indices centered for numerical stability
        t_vals = np.arange(self._t, dtype=np.float64)
        t_mean = t_vals.mean()
        t_centered = t_vals - t_mean
        t_var = (t_centered**2).sum()

        if t_var == 0:
            return np.zeros((self._n, self._n))

        # Compute slope for each (i, j) pair
        # slope = Σ(t - t_mean)(W - W_mean) / Σ(t - t_mean)²
        W_mean = self._tensor.mean(axis=2, keepdims=True)
        W_centered = self._tensor - W_mean

        # Dot product with centered time
        numerator = np.tensordot(W_centered, t_centered, axes=([2], [0]))
        slopes = numerator / t_var

        return slopes

    def change_point_graph(self) -> np.ndarray:
        """
        Compute change-point graph: maximum absolute change between consecutive time steps.

        W_change[i,j] = max_t |W[i,j,t+1] - W[i,j,t]|

        Clinical meaning: Identifies sudden state transitions — acute events,
        medication changes, procedure effects.

        Returns
        -------
        np.ndarray
            2D array of shape (n, n) with non-negative values.
        """
        if self._t < 2:
            return np.zeros((self._n, self._n))

        diffs = np.abs(np.diff(self._tensor, axis=2))
        return diffs.max(axis=2)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def slice(self, start: int = 0, end: int | None = None) -> "TemporalCosmicGraph":
        """
        Extract a time-range subset of the tensor.

        Parameters
        ----------
        start : int
            Start time index (inclusive).
        end : int, optional
            End time index (exclusive). Defaults to T.

        Returns
        -------
        TemporalCosmicGraph
            New instance with sliced tensor.
        """
        if end is None:
            end = self._t
        return TemporalCosmicGraph(
            self._tensor[:, :, start:end].copy(),
            threshold=self._threshold,
        )

    def to_networkx(
        self,
        aggregation: Literal[
            "persistence", "mean", "recency", "volatility", "trend", "change_point"
        ] = "persistence",
        threshold: float | None = None,
        **kwargs,
    ) -> nx.Graph:
        """
        Convert an aggregated graph to NetworkX format.

        Parameters
        ----------
        aggregation : str
            Which aggregation method to use. One of:
            "persistence", "mean", "recency", "volatility", "trend", "change_point".
        threshold : float, optional
            Edge weight threshold for including edges. Defaults to instance threshold.
            For aggregation="persistence", this value is also used as the persistence
            threshold passed to `persistence_graph`.
        **kwargs
            Additional arguments passed through to the selected aggregation method
            (e.g., decay=0.9 for recency_graph). Unsupported arguments raise TypeError.

        Returns
        -------
        nx.Graph
            NetworkX graph with 'weight' edge attributes.
        """
        aggregators = {
            "persistence": self.persistence_graph,
            "mean": self.mean_graph,
            "recency": self.recency_graph,
            "volatility": self.volatility_graph,
            "trend": self.trend_graph,
            "change_point": self.change_point_graph,
        }

        if aggregation not in aggregators:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from: {list(aggregators.keys())}"
            )

        # Get aggregated 2D matrix
        if aggregation == "persistence" and threshold is not None:
            adj = aggregators[aggregation](threshold=threshold, **kwargs)
        else:
            adj = aggregators[aggregation](**kwargs)

        # Apply threshold for edge inclusion
        edge_threshold = threshold if threshold is not None else self._threshold

        G = nx.Graph()
        G.add_nodes_from(range(self._n))

        for i in range(self._n):
            for j in range(i + 1, self._n):
                w = adj[i, j]
                if w > edge_threshold:
                    G.add_edge(i, j, weight=float(w))

        return G

    def __repr__(self) -> str:
        return (
            f"TemporalCosmicGraph(n={self._n}, T={self._t}, "
            f"threshold={self._threshold})"
        )
