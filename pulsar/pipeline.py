"""
ThemaRS — orchestrates the full Pulsar pipeline.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import contextmanager
import os
from typing import Any, Union

import networkx as nx
import numpy as np
import pandas as pd

from pulsar._pulsar import (
    BallMapper,
    CosmicGraph,
    StabilityResult,
    StandardScaler,
    accumulate_pseudo_laplacians,
    ball_mapper_grid,
    find_stable_thresholds,
    impute_column,
    pca_grid,
)
from pulsar.config import ImputeSpec, PulsarConfig, load_config
from pulsar.analysis.hooks import cosmic_to_networkx


def _impute_string_column(df: pd.DataFrame, col: str, spec: ImputeSpec) -> None:
    """Impute a string/object column in Python (Rust impute_column only handles f64).

    Only fill_mode and sample_categorical are meaningful for string data.
    Raises ValueError for numeric-only methods (fill_mean, fill_median, sample_normal).
    """
    missing_mask = df[col].isna()
    if not missing_mask.any():
        return
    observed = df.loc[~missing_mask, col]

    if spec.method == "fill_mode":
        mode = observed.mode()
        if not mode.empty:
            df.loc[missing_mask, col] = mode.iloc[0]

    elif spec.method == "sample_categorical":
        import random

        counts = observed.value_counts()
        categories = counts.index.tolist()
        weights = counts.values.tolist()
        rng = random.Random(spec.seed)
        n_missing = int(missing_mask.sum())
        fills = rng.choices(categories, weights=weights, k=n_missing)
        df.loc[missing_mask, col] = fills

    else:
        raise ValueError(
            f"Column '{col}' is string/object dtype; method '{spec.method}' requires "
            f"numeric data. Use 'fill_mode' or 'sample_categorical' for string columns."
        )


# Approximate wall-clock fraction per pipeline stage (estimates, not guarantees).
# Order matters — stages are consumed sequentially by a cursor.
# PCA weight is zeroed when _precomputed_embeddings is used; fractions are renormalized.
_STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("load", 0.03),
    ("impute", 0.08),
    ("scale", 0.01),
    ("pca", 0.25),
    ("ball_mapper", 0.42),
    ("laplacian", 0.15),
    ("cosmic", 0.06),  # includes stability analysis when threshold="auto"
]


def _build_cumulative_fractions(
    stages: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Return [(label, cumulative_fraction), ...] with final entry pinned to 1.0."""
    total = sum(w for _, w in stages)
    result: list[tuple[str, float]] = []
    cumulative = 0.0
    for label, weight in stages:
        cumulative += weight / total
        result.append((label, round(cumulative, 6)))
    if result:
        result[-1] = (result[-1][0], 1.0)
    return result


@contextmanager
def _rayon_thread_override(workers: int | None):
    """
    Temporarily override Rayon worker count for Rust ops that respect
    RAYON_NUM_THREADS. Restores the previous value on exit.
    """
    if workers is None:
        yield
        return
    prev = os.environ.get("RAYON_NUM_THREADS")
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = prev


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
        self._preprocessed_data: pd.DataFrame | None = None
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

        # Build cumulative progress schedule from stage weights.
        # PCA weight is zeroed when embeddings are pre-computed; fractions renormalize.
        use_cached_pca = _precomputed_embeddings is not None
        stages = [
            (name, 0.0 if name == "pca" and use_cached_pca else weight)
            for name, weight in _STAGE_WEIGHTS
        ]
        schedule = _build_cumulative_fractions(stages)
        _cursor = 0

        def _notify(label_override: str | None = None) -> None:
            nonlocal _cursor
            if progress_callback is None or _cursor >= len(schedule):
                return
            label, frac = schedule[_cursor]
            _cursor += 1
            progress_callback(label_override or label, frac)

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
        _notify()  # load

        # 3a. Add imputation indicator flags (pure Python, before imputing)
        for col in cfg.impute:
            if col in df.columns:
                flag_col = f"{col}_was_missing"
                df[flag_col] = df[col].isna().astype(np.float64)

        # 3b. Impute columns (Rust for numeric, Python for string/object)
        for col, spec in cfg.impute.items():
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                _impute_string_column(df, col, spec)
            else:
                arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                imputed = impute_column(arr, spec.method, spec.seed)
                df[col] = imputed

        # 3c. Encode categorical columns (pure Python)
        for col, spec in cfg.encode.items():
            if col not in df.columns:
                continue
            if spec.method == "one_hot":
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype=np.float64)

        # Drop any remaining NaN rows
        df = df.dropna(axis=0)
        self._preprocessed_data = df.reset_index(drop=True)
        _notify()  # impute

        # 4. Convert to float64 matrix and scale
        X = df.to_numpy(dtype=np.float64)
        n = X.shape[0]

        scaler = StandardScaler()
        X_scaled = np.array(scaler.fit_transform(X))
        _notify()  # scale

        # 5. PCA grid (randomized SVD, parallelised across seeds)
        if _precomputed_embeddings is not None:
            assert all(
                e.shape[0] == X_scaled.shape[0] for e in _precomputed_embeddings
            ), "Precomputed embedding row count does not match current data"
            embeddings = _precomputed_embeddings
        else:
            embeddings = [
                np.ascontiguousarray(emb)
                for emb in pca_grid(X_scaled, cfg.pca.dimensions, cfg.pca.seeds)
            ]
        # Cache embeddings for MCP session reuse
        self._embeddings = embeddings
        _notify("pca (cached)" if use_cached_pca else None)  # pca

        # 6. BallMapper grid (Rust parallel)
        self._ball_maps = ball_mapper_grid(embeddings, cfg.ball_mapper.epsilons)
        _notify()  # ball_mapper

        # 7. Accumulate pseudo-Laplacians (Rust parallel)
        galactic_L = np.array(accumulate_pseudo_laplacians(self._ball_maps, n))
        _notify()  # laplacian

        # 8. CosmicGraph (+ optional stability analysis)
        threshold = cfg.cosmic_graph.threshold
        if threshold == "auto":
            cg_temp = CosmicGraph.from_pseudo_laplacian(galactic_L, 0.0)
            weighted_adj = np.array(cg_temp.weighted_adj)
            self._stability_result = find_stable_thresholds(weighted_adj)
            self._resolved_threshold = float(self._stability_result.optimal_threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
        else:
            self._resolved_threshold = float(threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
            weighted_adj = np.array(self._cosmic_rust.weighted_adj)
        self._weighted_adjacency = weighted_adj
        self._cosmic_graph = cosmic_to_networkx(self._cosmic_rust)
        _notify()  # cosmic (always last, always 1.0)

        return self

    def fit_multi(
        self,
        datasets: list[pd.DataFrame],
        *,
        progress_callback: Callable[[str, float], None] | None = None,
        store_ball_maps: bool = False,
        ballmap_batch_size: int | None = None,
        rayon_workers: int | None = None,
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
            store_ball_maps: If True, retain fitted BallMapper objects on self.
                Defaults to False to lower memory; when False, BallMappers are
                freed after their Laplacian contributions are accumulated.
            ballmap_batch_size: Optional cap on how many PCA embeddings to process
                per BallMapper batch. Smaller batches reduce peak RAM at the cost
                of more Rust crossings. None processes all embeddings together.
            rayon_workers: Optional cap for Rayon worker threads used inside Rust
                ops (PCA grid, BallMapper grid, Laplacian accumulation). Defaults
                to the library setting when None.

        Returns self for method chaining.
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")
        if ballmap_batch_size is not None and ballmap_batch_size <= 0:
            raise ValueError("ballmap_batch_size must be positive when provided")
        if rayon_workers is not None and rayon_workers <= 0:
            raise ValueError("rayon_workers must be positive when provided")

        N = len(datasets)
        cfg = self.config

        # Build stage schedule: per-dataset stages repeated N times, then shared stages.
        _per_ds_weights = [
            ("load", 0.03),
            ("impute", 0.08),
            ("scale", 0.01),
            ("pca", 0.25),
            ("ball_mapper", 0.42),
        ]
        stages: list[tuple[str, float]] = []
        for i in range(N):
            prefix = f"Dataset {i + 1}/{N}: "
            for name, weight in _per_ds_weights:
                stages.append((f"{prefix}{name}", weight))
        stages.append(("laplacian", 0.15))
        stages.append(("cosmic", 0.06))

        schedule = _build_cumulative_fractions(stages)
        _cursor = 0

        def _notify(label_override: str | None = None) -> None:
            nonlocal _cursor
            if progress_callback is None or _cursor >= len(schedule):
                return
            label, frac = schedule[_cursor]
            _cursor += 1
            progress_callback(label_override or label, frac)

        all_ball_maps: list[BallMapper] = []
        galactic_L_accum: np.ndarray | None = None
        n: int | None = None

        # Collect global vocabulary for categorical encoding (if any)
        # Ensures that pd.get_dummies produces identical columns across datasets
        vocab: dict[str, list[Any]] = {}
        for col in cfg.encode:
            categories = set()
            for ds in datasets:
                if col in ds.columns:
                    categories.update(ds[col].dropna().unique())
            vocab[col] = sorted(list(categories))

        with _rayon_thread_override(rayon_workers):
            for i, data in enumerate(datasets):
                df = data.copy()

                drop = [c for c in cfg.drop_columns if c in df.columns]
                df = df.drop(columns=drop)
                _notify()  # Dataset i: load

                for col in cfg.impute:
                    if col in df.columns:
                        df[f"{col}_was_missing"] = df[col].isna().astype(np.float64)

                for col, spec in cfg.impute.items():
                    if col not in df.columns:
                        continue
                    if df[col].dtype == object:
                        _impute_string_column(df, col, spec)
                    else:
                        arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                        df[col] = impute_column(arr, spec.method, spec.seed)

                # Encode categorical columns with shared vocabulary
                for col, spec in cfg.encode.items():
                    if col in df.columns:
                        if spec.method == "one_hot":
                            df[col] = pd.Categorical(df[col], categories=vocab[col])
                            df = pd.get_dummies(
                                df, columns=[col], prefix=col, dtype=np.float64
                            )

                df = df.dropna(axis=0)
                _notify()  # Dataset i: impute

                X = df.to_numpy(dtype=np.float64)
                if n is None:
                    n = X.shape[0]
                    galactic_L_accum = np.zeros((n, n), dtype=np.int64)
                elif X.shape[0] != n:
                    raise ValueError(
                        f"Dataset {i} has {X.shape[0]} rows after preprocessing; "
                        f"expected {n} (same as dataset 0)"
                    )

                scaler = StandardScaler()
                X_scaled = np.array(scaler.fit_transform(X))
                _notify()  # Dataset i: scale

                embeddings = [
                    np.ascontiguousarray(emb)
                    for emb in pca_grid(X_scaled, cfg.pca.dimensions, cfg.pca.seeds)
                ]
                _notify()  # Dataset i: pca

                batches = (
                    [embeddings]
                    if ballmap_batch_size is None
                    else [
                        embeddings[j : j + ballmap_batch_size]
                        for j in range(0, len(embeddings), ballmap_batch_size)
                    ]
                )

                for batch in batches:
                    batch_ball_maps = ball_mapper_grid(batch, cfg.ball_mapper.epsilons)
                    if store_ball_maps:
                        all_ball_maps.extend(batch_ball_maps)
                    if galactic_L_accum is None:
                        raise RuntimeError("galactic_L_accum not initialized")
                    galactic_L_accum += np.array(
                        accumulate_pseudo_laplacians(batch_ball_maps, n)
                    )
                    # Release batch memory aggressively in notebook contexts
                    del batch_ball_maps
                    gc.collect()

                # Drop per-dataset intermediates promptly
                del embeddings, X_scaled, X, df
                gc.collect()
                _notify()  # Dataset i: ball_mapper

        self._ball_maps = all_ball_maps if store_ball_maps else []

        if galactic_L_accum is None or n is None:
            raise RuntimeError("No datasets were processed")

        galactic_L = galactic_L_accum
        _notify()  # laplacian

        threshold = cfg.cosmic_graph.threshold
        if threshold == "auto":
            cg_temp = CosmicGraph.from_pseudo_laplacian(galactic_L, 0.0)
            weighted_adj = np.array(cg_temp.weighted_adj)
            self._stability_result = find_stable_thresholds(weighted_adj)
            self._resolved_threshold = float(self._stability_result.optimal_threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
        else:
            self._resolved_threshold = float(threshold)
            self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
                galactic_L, self._resolved_threshold
            )
            weighted_adj = np.array(self._cosmic_rust.weighted_adj)

        self._weighted_adjacency = weighted_adj
        self._cosmic_graph = cosmic_to_networkx(self._cosmic_rust)
        _notify()  # cosmic (always last, always 1.0)

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

    @property
    def preprocessed_data(self) -> pd.DataFrame:
        """DataFrame after drop/impute/encode/dropna — row-aligned with graph nodes."""
        if self._preprocessed_data is None:
            raise RuntimeError("Call fit() first")
        return self._preprocessed_data

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
