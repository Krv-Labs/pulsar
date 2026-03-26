"""
ThemaRS — orchestrates the full Pulsar pipeline.
"""

from __future__ import annotations

from typing import Union

import networkx as nx
import numpy as np
import pandas as pd

from pulsar._pulsar import (
    BallMapper,
    CosmicGraph,
    StandardScaler,
    accumulate_pseudo_laplacians,
    ball_mapper_grid,
    impute_column,
    pca_grid,
)
from pulsar.config import PulsarConfig, load_config
from pulsar.hooks import cosmic_to_networkx


class ThemaRS:
    """
    End-to-end Pulsar pipeline orchestrator.

    Usage::

        model = ThemaRS("params.yaml").fit()
        graph = model.cosmic_graph          # networkx.Graph
        adj   = model.weighted_adjacency    # np.ndarray (n, n)
    """

    def __init__(self, config: Union[str, dict, PulsarConfig]):
        if isinstance(config, PulsarConfig):
            self.config = config
        else:
            self.config = load_config(config)

        self._ball_maps: list[BallMapper] = []
        self._cosmic_graph: nx.Graph | None = None
        self._weighted_adjacency: np.ndarray | None = None
        self._cosmic_rust: CosmicGraph | None = None
        self._data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame | None = None) -> "ThemaRS":
        """
        Run the full pipeline:
        1. Load data (if not provided)
        2. Impute columns (Rust)
        3. Add imputation indicator flags (Python)
        4. Standard-scale (Rust)
        5. PCA grid (Rust)
        6. BallMapper grid (Rust, rayon-parallel)
        7. Accumulate pseudo-Laplacians (Rust + numpy)
        8. Build CosmicGraph (Rust)

        Returns self for method chaining.
        """
        cfg = self.config

        # 1. Load data
        if data is None:
            if not cfg.data:
                raise ValueError("No data path in config and no DataFrame provided")
            if cfg.data.endswith(".parquet"):
                data = pd.read_parquet(cfg.data)
            else:
                data = pd.read_csv(cfg.data)

        self._data = data.copy()

        # 2. Drop unwanted columns
        drop = [c for c in cfg.drop_columns if c in data.columns]
        df = data.drop(columns=drop)

        # 3a. Add imputation indicator flags (pure Python, before imputing)
        for col in cfg.impute:
            if col in df.columns:
                flag_col = f"{col}_was_missing"
                df[flag_col] = df[col].isna().astype(np.float64)

        # 3b. Impute columns (Rust)
        for col, spec in cfg.impute.items():
            if col not in df.columns:
                continue
            arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            imputed = impute_column(arr, spec.method, spec.seed)
            df[col] = imputed

        # Drop any remaining NaN rows
        df = df.dropna(axis=0)

        # 4. Convert to float64 matrix and scale
        X = df.to_numpy(dtype=np.float64)
        n = X.shape[0]

        scaler = StandardScaler()
        X_scaled = np.array(scaler.fit_transform(X))

        # 5. PCA grid (randomized SVD, parallelised across seeds)
        embeddings = [
            np.ascontiguousarray(emb)
            for emb in pca_grid(X_scaled, cfg.pca.dimensions, cfg.pca.seeds)
        ]

        # 6. BallMapper grid (Rust parallel)
        self._ball_maps = ball_mapper_grid(embeddings, cfg.ball_mapper.epsilons)

        # 7. Accumulate pseudo-Laplacians (Rust parallel)
        galactic_L = np.array(accumulate_pseudo_laplacians(self._ball_maps, n))

        # 8. CosmicGraph
        self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
            galactic_L, cfg.cosmic_graph.threshold
        )
        self._weighted_adjacency = np.array(self._cosmic_rust.weighted_adj)
        self._cosmic_graph = cosmic_to_networkx(self._cosmic_rust)

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cosmic_graph(self) -> nx.Graph:
        """Cosmic graph as a NetworkX graph with 'weight' edge attributes."""
        if self._cosmic_graph is None:
            raise RuntimeError("Call fit() first")
        return self._cosmic_graph

    @property
    def weighted_adjacency(self) -> np.ndarray:
        """n×n float64 weighted adjacency matrix."""
        if self._weighted_adjacency is None:
            raise RuntimeError("Call fit() first")
        return self._weighted_adjacency

    @property
    def ball_maps(self) -> list[BallMapper]:
        """All fitted BallMapper objects across the parameter grid."""
        return self._ball_maps

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def select_representatives(self, n_reps: int | None = None) -> list[BallMapper]:
        """
        Select n_reps diverse representative BallMapper instances by clustering
        them based on structural similarity (node count and coverage overlap).

        Returns a list of n_reps BallMapper objects.
        """
        if not self._ball_maps:
            raise RuntimeError("Call fit() first")

        n_reps = n_reps or self.config.n_reps
        n_maps = len(self._ball_maps)

        if n_reps >= n_maps:
            return list(self._ball_maps)

        # Use lightweight features for clustering: (n_nodes, n_edges, eps)
        # This avoids O(n²) Laplacian computation per ball map
        features = np.array([
            [bm.n_nodes(), bm.n_edges(), bm.eps]
            for bm in self._ball_maps
        ])
        
        # Normalise features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        from sklearn.cluster import KMeans
        
        labels = KMeans(n_clusters=n_reps, random_state=42, n_init=10).fit_predict(features)

        # Pick representative closest to cluster centroid
        reps = []
        for cluster_id in range(n_reps):
            members = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            cluster_features = features[members]
            centroid = cluster_features.mean(axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest = members[int(distances.argmin())]
            reps.append(self._ball_maps[closest])

        return reps
