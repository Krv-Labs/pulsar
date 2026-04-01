"""
ThemaRS — orchestrates the full Pulsar pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd

from pulsar._pulsar import (
    BallMapper,
    CosmicGraph,
    StandardScaler,
    StabilityResult,
    accumulate_pseudo_laplacians,
    ball_mapper_grid,
    find_stable_thresholds,
    impute_column,
    pca_grid,
)
from pulsar.config import PulsarConfig, load_config
from pulsar.hooks import cosmic_to_networkx


# Approximate wall-clock fraction per pipeline stage (estimates, not guarantees).
# PCA weight is zeroed when _precomputed_embeddings is used; fractions are renormalized.
_STAGE_WEIGHTS: dict[str, float] = {
    "load":        0.03,
    "impute":      0.08,
    "scale":       0.01,
    "pca":         0.25,
    "ball_mapper": 0.42,
    "laplacian":   0.15,
    "cosmic":      0.04,
    "stability":   0.02,
}


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
        self._stability_result: StabilityResult | None = None
        self._resolved_threshold: float | None = None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame | None = None,
        *,
        progress_callback: Callable[[str, float], None] | None = None,
        _precomputed_embeddings: list | None = None,
    ) -> "ThemaRS":
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

        Args:
            data: Input DataFrame. If None, loaded from config data path.
            progress_callback: Optional ``(stage: str, fraction: float) -> None``.
                Called at the end of each pipeline stage with the stage name and
                cumulative progress in [0.0, 1.0]. Exceptions in the callback
                propagate and abort fit(). Pass None to disable (default).
            _precomputed_embeddings: Internal — cached PCA embeddings from a prior
                fit() call. Skips pca_grid() when provided.

        Returns self for method chaining.
        """
        cfg = self.config

        # Build cumulative progress fractions from stage weights.
        # PCA weight is zeroed when embeddings are pre-computed; fractions renormalize.
        use_cached_pca = _precomputed_embeddings is not None
        _weights = {
            k: (0.0 if k == "pca" and use_cached_pca else v)
            for k, v in _STAGE_WEIGHTS.items()
        }
        _total_w = sum(_weights.values())
        _cum: dict[str, float] = {}
        _c = 0.0
        for _s, _w in _weights.items():
            _c += _w / _total_w
            _cum[_s] = round(_c, 4)

        def _notify(stage: str, key: str | None = None) -> None:
            """Fire progress_callback for a stage.

            Args:
                stage: Human-readable stage name passed to the callback.
                key: Key into _cum for fraction lookup. Defaults to stage.
            """
            if progress_callback is None:
                return
            try:
                progress_callback(stage, _cum[key if key is not None else stage])
            except Exception as exc:
                raise RuntimeError(
                    f"progress_callback raised during '{stage}'"
                ) from exc

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
        _notify("load")

        # 4. Convert to float64 matrix and scale
        X = df.to_numpy(dtype=np.float64)
        n = X.shape[0]

        scaler = StandardScaler()
        X_scaled = np.array(scaler.fit_transform(X))
        _notify("impute")
        _notify("scale")

        # 5. PCA grid (randomized SVD, parallelised across seeds)
        if _precomputed_embeddings is not None:
            assert all(e.shape[0] == X_scaled.shape[0] for e in _precomputed_embeddings), (
                "Precomputed embedding row count does not match current data"
            )
            embeddings = _precomputed_embeddings
        else:
            embeddings = [
                np.ascontiguousarray(emb)
                for emb in pca_grid(X_scaled, cfg.pca.dimensions, cfg.pca.seeds)
            ]
        # Cache embeddings for MCP session reuse
        self._embeddings = embeddings
        _notify("pca (cached)" if use_cached_pca else "pca", key="pca")

        # 6. BallMapper grid (Rust parallel)
        self._ball_maps = ball_mapper_grid(embeddings, cfg.ball_mapper.epsilons)
        _notify("ball_mapper")

        # 7. Accumulate pseudo-Laplacians (Rust parallel)
        galactic_L = np.array(accumulate_pseudo_laplacians(self._ball_maps, n))
        _notify("laplacian")

        # 8. CosmicGraph
        threshold = cfg.cosmic_graph.threshold
        if threshold == "auto":
            # Weighted adjacency is independent of threshold; compute once at 0.0
            cg_temp = CosmicGraph.from_pseudo_laplacian(galactic_L, 0.0)
            weighted_adj = np.array(cg_temp.weighted_adj)
            self._stability_result = find_stable_thresholds(weighted_adj)
            self._resolved_threshold = float(self._stability_result.optimal_threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
            _notify("stability")
        else:
            self._resolved_threshold = float(threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
            weighted_adj = np.array(self._cosmic_rust.weighted_adj)
        self._weighted_adjacency = weighted_adj
        self._cosmic_graph = cosmic_to_networkx(self._cosmic_rust)
        # Always end at 1.0 — use stability key regardless of whether auto-threshold ran
        _notify("cosmic", key="stability")

        return self

    def fit_multi(
        self,
        datasets: list[pd.DataFrame],
        *,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> "ThemaRS":
        """
        Run the pipeline over multiple data versions (e.g. different embedding
        models) and fuse them via pseudo-Laplacian accumulation.

        Each DataFrame must have the same number of rows (same points, different
        representations). The sweep is run independently on each version and all
        resulting ball maps are accumulated into a single CosmicGraph — so a high
        edge weight means two points are topological neighbours across *all*
        representations, not just one.

        Imputation and column-dropping are applied per-dataset if configured.
        All datasets must yield the same n after preprocessing.

        Args:
            datasets: List of DataFrames (same points, different representations).
            progress_callback: Optional ``(stage: str, fraction: float) -> None``.
                Same semantics as in fit(). Stages are prefixed with dataset index
                (e.g. "Dataset 1/3: pca").

        Returns self for method chaining.
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")

        N = len(datasets)
        cfg = self.config

        # Compute cumulative fractions for fit_multi.
        # Per-dataset stages are spread across N iterations; final stages are shared.
        _per_ds_w = sum(_STAGE_WEIGHTS[s] for s in ("impute", "scale", "pca", "ball_mapper"))
        _final_w = sum(_STAGE_WEIGHTS[s] for s in ("laplacian", "cosmic", "stability"))
        _total_w = _per_ds_w * N + _final_w

        def _notify_multi(stage: str, frac: float) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(stage, round(frac, 4))
            except Exception as exc:
                raise RuntimeError(
                    f"progress_callback raised during '{stage}'"
                ) from exc

        all_ball_maps: list[BallMapper] = []
        n: int | None = None

        for i, data in enumerate(datasets):
            df = data.copy()

            drop = [c for c in cfg.drop_columns if c in df.columns]
            df = df.drop(columns=drop)

            for col in cfg.impute:
                if col in df.columns:
                    df[f"{col}_was_missing"] = df[col].isna().astype(np.float64)

            for col, spec in cfg.impute.items():
                if col not in df.columns:
                    continue
                arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                df[col] = impute_column(arr, spec.method, spec.seed)

            df = df.dropna(axis=0)

            X = df.to_numpy(dtype=np.float64)
            if n is None:
                n = X.shape[0]
            elif X.shape[0] != n:
                raise ValueError(
                    f"Dataset {i} has {X.shape[0]} rows after preprocessing; "
                    f"expected {n} (same as dataset 0)"
                )

            scaler = StandardScaler()
            X_scaled = np.array(scaler.fit_transform(X))

            # Cumulative fraction at each sub-stage for dataset i
            _ds_base = i * _per_ds_w / _total_w
            _notify_multi(
                f"Dataset {i + 1}/{N}: imputing",
                _ds_base + _STAGE_WEIGHTS["impute"] / _total_w,
            )
            _notify_multi(
                f"Dataset {i + 1}/{N}: scaling",
                _ds_base + (_STAGE_WEIGHTS["impute"] + _STAGE_WEIGHTS["scale"]) / _total_w,
            )

            embeddings = [
                np.ascontiguousarray(emb)
                for emb in pca_grid(X_scaled, cfg.pca.dimensions, cfg.pca.seeds)
            ]
            _notify_multi(
                f"Dataset {i + 1}/{N}: pca",
                _ds_base + (_STAGE_WEIGHTS["impute"] + _STAGE_WEIGHTS["scale"] + _STAGE_WEIGHTS["pca"]) / _total_w,
            )

            all_ball_maps.extend(ball_mapper_grid(embeddings, cfg.ball_mapper.epsilons))
            _notify_multi(
                f"Dataset {i + 1}/{N}: ball mapper",
                (i + 1) * _per_ds_w / _total_w,
            )

        self._ball_maps = all_ball_maps

        galactic_L = np.array(accumulate_pseudo_laplacians(self._ball_maps, n))
        _notify_multi("laplacian", (N * _per_ds_w + _STAGE_WEIGHTS["laplacian"]) / _total_w)

        threshold = cfg.cosmic_graph.threshold
        if threshold == "auto":
            cg_temp = CosmicGraph.from_pseudo_laplacian(galactic_L, 0.0)
            weighted_adj = np.array(cg_temp.weighted_adj)
            self._stability_result = find_stable_thresholds(weighted_adj)
            self._resolved_threshold = float(self._stability_result.optimal_threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
            _notify_multi("stability", 1.0)
        else:
            self._resolved_threshold = float(threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
            weighted_adj = np.array(self._cosmic_rust.weighted_adj)

        self._weighted_adjacency = weighted_adj
        self._cosmic_graph = cosmic_to_networkx(self._cosmic_rust)
        _notify_multi("cosmic", 1.0)

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

    @property
    def stability_result(self) -> StabilityResult | None:
        """Stability analysis result (only available if threshold='auto')."""
        return self._stability_result

    @property
    def resolved_threshold(self) -> float:
        """The actual threshold used (resolved from 'auto' or the manual value)."""
        if self._resolved_threshold is None:
            raise RuntimeError("Call fit() first")
        return self._resolved_threshold

    @property
    def data(self) -> pd.DataFrame:
        """The original DataFrame passed to fit() (before preprocessing)."""
        if self._data is None:
            raise RuntimeError("Call fit() first")
        return self._data

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
        features = np.array(
            [[bm.n_nodes(), bm.n_edges(), bm.eps] for bm in self._ball_maps]
        )

        # Normalise features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=n_reps, random_state=42, n_init=10).fit_predict(
            features
        )

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
