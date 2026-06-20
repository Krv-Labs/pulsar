"""
ThemaRS — orchestrates the full Pulsar pipeline.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any, Union

import networkx as nx
import numpy as np
import pandas as pd

from pulsar._pulsar import (
    BallMapper,
    CosmicGraph,
    StabilityResult,
    StandardScaler,
    accumulate_pseudo_laplacians_sparse,
    ball_mapper_grid,
    find_stable_thresholds_sparse,
    jl_grid,
    pca_grid,
)
from pulsar.analysis import cosmic_to_networkx
from pulsar.config import PulsarConfig, load_config
from pulsar.preprocessing import preprocess_dataframe
from pulsar.runtime.utils import (
    ProgressTracker,
    rayon_thread_override,
)


def projection_grid(X_scaled: np.ndarray, cfg: PulsarConfig) -> list[np.ndarray]:
    projection = getattr(cfg, "projection", None)
    if projection is None:
        dimensions = cfg.pca.dimensions
        seeds = cfg.pca.seeds
        method = "jl"
        center = True
    else:
        dimensions = projection.dimensions
        seeds = projection.seeds
        method = projection.method
        center = projection.center

    if method == "pca":
        raw = pca_grid(X_scaled, dimensions, seeds)
    elif method == "jl":
        raw = jl_grid(X_scaled, dimensions, seeds, center=center)
    else:
        raise ValueError(f"Unsupported projection method: {method!r}")
    return [np.ascontiguousarray(emb) for emb in raw]


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
        self._dense_cosmic_rust: CosmicGraph | None = None
        self._weight_scale: float = 1.0
        self._data: pd.DataFrame | None = None
        self._preprocessed_data: pd.DataFrame | None = None
        self._stability_result: StabilityResult | None = None
        self._resolved_construction_threshold: float | None = None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame | None = None,
        *,
        progress_callback: Callable[[str, float], None] | None = None,
        _precomputed_embeddings: list | None = None,
        ballmap_batch_size: int | None = 1,
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
            ballmap_batch_size: Optional cap on how many embeddings to process per
                BallMapper batch. Defaults to 1 to bound sparse accumulator peak RAM;
                pass None to process the full grid in one Rust call.

        Returns self for method chaining.
        """
        cfg = self.config
        if ballmap_batch_size is not None and ballmap_batch_size <= 0:
            raise ValueError("ballmap_batch_size must be positive when provided")

        use_cached_pca = _precomputed_embeddings is not None

        # 1. Load data
        if data is None:
            if not cfg.data:
                raise ValueError("No data path in config and no DataFrame provided")
            if cfg.data.endswith(".parquet"):
                data = pd.read_parquet(cfg.data)
            else:
                data = pd.read_csv(cfg.data)

        projection_count = (
            len(_precomputed_embeddings)
            if _precomputed_embeddings is not None
            else max(len(cfg.projection.dimensions) * len(cfg.projection.seeds), 1)
        )
        batch_count = (
            1
            if ballmap_batch_size is None
            else max(
                (projection_count + ballmap_batch_size - 1) // ballmap_batch_size, 1
            )
        )
        progress = ProgressTracker(
            [
                ("load", 1.0),
                ("preprocess", 1.0),
                ("scale", 1.0),
                ("projection", 0.1 if use_cached_pca else projection_count),
                ("sweep_batches", batch_count * max(len(cfg.ball_mapper.epsilons), 1)),
                ("cosmic_graph", 1.0 + (2.0 if cfg.cosmic_graph.sparsify else 0.0)),
                (
                    "stability",
                    2.0 if cfg.cosmic_graph.construction_threshold == "auto" else 0.0,
                ),
                ("networkx_graph", 1.0),
            ],
            progress_callback,
        )
        progress.complete("load")

        self._data = data.copy()

        # 2–3. Preprocessing (drop, coerce, impute, encode, validate)
        df, _layout = preprocess_dataframe(data, cfg)
        self._preprocessed_data = df
        progress.complete("preprocess")

        # 4. Convert to float64 matrix and scale
        X = df.to_numpy(dtype=np.float64)
        n = X.shape[0]

        scaler = StandardScaler()
        X_scaled = np.array(scaler.fit_transform(X))
        progress.complete("scale")

        # 5. Projection grid (JL by default; PCA retained as explicit legacy mode)
        if _precomputed_embeddings is not None:
            assert all(
                e.shape[0] == X_scaled.shape[0] for e in _precomputed_embeddings
            ), "Precomputed embedding row count does not match current data"
            embeddings = _precomputed_embeddings
        else:
            embeddings = projection_grid(X_scaled, cfg)
        # Cache embeddings for MCP session reuse
        self._embeddings = embeddings
        progress.complete(
            "projection", "projection (cached)" if use_cached_pca else None
        )

        # 6–7. BallMapper grid + sparse accumulation.
        # Batch by embedding so large sweeps do not hold every raw pair buffer at once.
        batches = (
            [embeddings]
            if ballmap_batch_size is None
            else [
                embeddings[j : j + ballmap_batch_size]
                for j in range(0, len(embeddings), ballmap_batch_size)
            ]
        )
        self._ball_maps = []
        galactic_L: Any | None = None
        total_batches = len(batches)
        for batch_index, batch in enumerate(batches, start=1):
            batch_ball_maps = ball_mapper_grid(batch, cfg.ball_mapper.epsilons)
            self._ball_maps.extend(batch_ball_maps)
            batch_spl = accumulate_pseudo_laplacians_sparse(batch_ball_maps, n)
            if galactic_L is None:
                galactic_L = batch_spl
            else:
                galactic_L.merge_in_place(batch_spl)
            del batch_spl
            gc.collect()
            progress.update(
                "sweep_batches",
                batch_index / total_batches,
                f"sweep batch {batch_index}/{total_batches}: ball_mapper + laplacian",
            )

        if galactic_L is None:
            raise RuntimeError("No BallMapper batches were processed")

        # 8. CosmicGraph: build the (sparse-backed) graph, sparsify if opted-in, threshold.
        self._dense_cosmic_rust = CosmicGraph.from_pseudo_laplacian_sparse(
            galactic_L, 0.0
        )
        sparse = self._apply_default_sparsification(self._dense_cosmic_rust)
        progress.complete("cosmic_graph")
        self._finalize_cosmic_graph(
            sparse,
            cfg.cosmic_graph.construction_threshold,
            progress=progress,
        )

        return self

    def _finalize_cosmic_graph(
        self,
        cosmic_rust: CosmicGraph,
        threshold_cfg: float | str,
        *,
        progress: ProgressTracker | None = None,
    ) -> None:
        """Normalize the exposed CosmicGraph onto a [0, 1] weight scale, resolve
        the construction threshold, and build the NetworkX graph.

        Operates entirely on the sparse edge list — the dense ``weighted_adjacency``
        is never materialized here (it is built lazily on first property access).
        Spectral sparsification can reweight edges above 1.0, which would both
        collapse those edges into one bin in ``find_stable_thresholds_sparse`` (it
        quantizes over [0, 1]) and break the [0, 1] threshold contract that
        clustering/validation rely on. We rescale by ``max(1.0, max_weight)`` so
        weights never exceed 1.0: a no-op for the un-sparsified path (max ≤ 1) and a
        fraction-of-max mapping otherwise.
        """
        self._cosmic_rust = cosmic_rust
        edges = cosmic_rust.weighted_edges()
        max_w = max((w for _, _, w in edges), default=0.0)
        self._weight_scale = max(1.0, max_w)
        # Invalidate the lazy dense adjacency; rebuilt on demand by the property.
        self._weighted_adjacency = None
        if threshold_cfg == "auto":
            norm_edges = [(i, j, w / self._weight_scale) for i, j, w in edges]
            self._stability_result = find_stable_thresholds_sparse(
                cosmic_rust.n, norm_edges
            )
            self._resolved_construction_threshold = float(
                self._stability_result.optimal_threshold
            )
            if progress is not None:
                progress.complete("stability")
        else:
            self._resolved_construction_threshold = float(threshold_cfg)
        self._cosmic_graph = cosmic_to_networkx(
            cosmic_rust,
            threshold=self._resolved_construction_threshold,
            scale=self._weight_scale,
        )
        if progress is not None:
            progress.complete("networkx_graph")

    def _apply_default_sparsification(self, cosmic: CosmicGraph) -> CosmicGraph:
        spec = self.config.cosmic_graph
        if not spec.sparsify:
            return cosmic
        return cosmic.spectral_sparsify(
            spec.sparsify_epsilon,
            seed=spec.sparsify_seed,
            sketch_dim=spec.sparsify_sketch_dim,
            sample_count=spec.sparsify_sample_count,
            pcg_tol=spec.sparsify_pcg_tol,
            max_iter=spec.sparsify_max_iter,
        )

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

        projection_count = max(
            len(cfg.projection.dimensions) * len(cfg.projection.seeds), 1
        )
        batch_count = (
            1
            if ballmap_batch_size is None
            else max(
                (projection_count + ballmap_batch_size - 1) // ballmap_batch_size, 1
            )
        )
        stages: list[tuple[str, float]] = []
        for i in range(N):
            prefix = f"Dataset {i + 1}/{N}: "
            stages.extend(
                [
                    (f"{prefix}load", 1.0),
                    (f"{prefix}preprocess", 1.0),
                    (f"{prefix}scale", 1.0),
                    (f"{prefix}projection", projection_count),
                    (
                        f"{prefix}sweep_batches",
                        batch_count * max(len(cfg.ball_mapper.epsilons), 1),
                    ),
                ]
            )
        stages.extend(
            [
                ("cosmic_graph", 1.0 + (2.0 if cfg.cosmic_graph.sparsify else 0.0)),
                (
                    "stability",
                    2.0 if cfg.cosmic_graph.construction_threshold == "auto" else 0.0,
                ),
                ("networkx_graph", 1.0),
            ]
        )
        progress = ProgressTracker(stages, progress_callback)

        all_ball_maps: list[BallMapper] = []
        # Sparse co-membership accumulator (SparsePseudoLaplacian), fused across
        # datasets via merge_in_place — never allocates an n×n matrix.
        galactic_L_accum: Any | None = None
        n: int | None = None
        ref_layout = None

        # Collect global vocabulary for categorical encoding (if any)
        # Ensures that pd.get_dummies produces identical columns across datasets
        vocab: dict[str, list[Any]] = {}
        for col in cfg.encode:
            categories: set[Any] = set()
            for ds in datasets:
                if col in ds.columns:
                    categories.update(ds[col].dropna().unique())
            vocab[col] = sorted(list(categories))

        with rayon_thread_override(rayon_workers):
            for i, data in enumerate(datasets):
                prefix = f"Dataset {i + 1}/{N}: "
                progress.complete(f"{prefix}load")
                df, layout = preprocess_dataframe(
                    data,
                    cfg,
                    vocab=vocab if vocab else None,
                    expected_layout=ref_layout,
                )
                if ref_layout is None:
                    ref_layout = layout
                progress.complete(f"{prefix}preprocess")

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
                progress.complete(f"{prefix}scale")

                embeddings = projection_grid(X_scaled, cfg)
                progress.complete(f"{prefix}projection")

                batches = (
                    [embeddings]
                    if ballmap_batch_size is None
                    else [
                        embeddings[j : j + ballmap_batch_size]
                        for j in range(0, len(embeddings), ballmap_batch_size)
                    ]
                )

                total_batches = len(batches)
                for batch_index, batch in enumerate(batches, start=1):
                    batch_ball_maps = ball_mapper_grid(batch, cfg.ball_mapper.epsilons)
                    if store_ball_maps:
                        all_ball_maps.extend(batch_ball_maps)
                    batch_spl = accumulate_pseudo_laplacians_sparse(batch_ball_maps, n)
                    if galactic_L_accum is None:
                        galactic_L_accum = batch_spl
                    else:
                        galactic_L_accum.merge_in_place(batch_spl)
                    # Release batch memory aggressively in notebook contexts
                    del batch_ball_maps, batch_spl
                    gc.collect()
                    progress.update(
                        f"{prefix}sweep_batches",
                        batch_index / total_batches,
                        f"{prefix}sweep batch {batch_index}/{total_batches}: "
                        "ball_mapper + laplacian",
                    )

                # Drop per-dataset intermediates promptly
                del embeddings, X_scaled, X, df
                gc.collect()

        self._ball_maps = all_ball_maps if store_ball_maps else []

        if galactic_L_accum is None or n is None:
            raise RuntimeError("No datasets were processed")

        galactic_L = galactic_L_accum

        self._dense_cosmic_rust = CosmicGraph.from_pseudo_laplacian_sparse(
            galactic_L, 0.0
        )
        sparse = self._apply_default_sparsification(self._dense_cosmic_rust)
        progress.complete("cosmic_graph")
        self._finalize_cosmic_graph(
            sparse,
            cfg.cosmic_graph.construction_threshold,
            progress=progress,
        )

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
        """Dense n×n float64 weighted adjacency, normalized to [0, 1], pre-threshold.

        Weights are scaled by ``1 / max(1, max_weight)`` so the matrix stays
        bounded by 1.0 (raw cosmic weights are already ≤ 1, but spectral
        sparsification can reweight edges above 1.0). This is the matrix threshold
        selection and clustering operate on. The cosmic graph backbone is kept
        sparse end-to-end; this dense view is materialized lazily on first access
        as a compatibility surface for diagnostics — prefer ``cosmic_rust`` /
        ``weighted_edges()`` on the hot path. ``cosmic_rust`` exposes the raw,
        un-normalized Rust backing.
        """
        if self._cosmic_rust is None:
            raise RuntimeError("Call fit() first")
        if self._weighted_adjacency is None:
            raw_adj = np.array(self._cosmic_rust.weighted_adj)
            self._weighted_adjacency = raw_adj / self._weight_scale
        return self._weighted_adjacency

    @property
    def cosmic_rust(self) -> CosmicGraph:
        """Rust CosmicGraph backing ``cosmic_graph``.

        Sparse-backed (edge-list) by default; spectral sparsification is opt-in
        (see :meth:`spectral_sparsify`). Holds raw (un-normalized) weights; see
        :attr:`weighted_adjacency` for the [0, 1]-normalized view used by
        thresholding and clustering.
        """
        if self._cosmic_rust is None:
            raise RuntimeError("Call fit() first")
        return self._cosmic_rust

    @property
    def dense_cosmic_rust(self) -> CosmicGraph:
        """The un-sparsified base CosmicGraph that :meth:`spectral_sparsify` consumes.

        Named ``dense`` for historical reasons; it is now the sparse-backed cosmic
        graph built directly from co-membership counts (no n×n materialization).
        Identical to ``cosmic_rust`` unless spectral sparsification has been applied.
        """
        if self._dense_cosmic_rust is None:
            raise RuntimeError("Call fit() first")
        return self._dense_cosmic_rust

    def weighted_edges(
        self, threshold: float | None = None
    ) -> list[tuple[int, int, float]]:
        """Weighted edge list from the exposed Cosmic graph, weights in [0, 1].

        Weights are normalized by the same ``max(1, max_weight)`` scale as
        :attr:`weighted_adjacency`. Defaults to the model's resolved construction
        threshold.
        """
        cutoff = (
            self.resolved_construction_threshold
            if threshold is None
            else float(threshold)
        )
        scale = self._weight_scale
        return [
            (i, j, w / scale)
            for i, j, w in self.cosmic_rust.weighted_edges()
            if w / scale > cutoff
        ]

    def spectral_sparsify(
        self,
        epsilon: float = 1.0,
        *,
        seed: int = 42,
        sketch_dim: int | None = None,
        sample_count: int | None = None,
        pcg_tol: float = 1e-6,
        max_iter: int = 1000,
        update: bool = False,
    ) -> CosmicGraph:
        """Spectrally sparsify the cosmic graph (opt-in hook).

        This is NOT a construction-time speedup — it runs after the graph is
        already built and is pure additional cost on that path. Its value is a
        leverage-aware, epsilon-controlled sparsifier that produces a compact graph
        **preserving the spectrum / effective resistances (distances)**, not the
        topology. Use it when you want a compact graph to run spectral algorithms
        on (spectral embeddings/clustering) or to hand to downstream analysis,
        where it is smarter than a naive low-weight edge filter.

        When ``update=True``, this also recomputes threshold selection on the
        sparsified graph and refreshes ``cosmic_graph`` / ``weighted_adjacency``.
        """
        sparse = self.dense_cosmic_rust.spectral_sparsify(
            epsilon,
            seed=seed,
            sketch_dim=sketch_dim,
            sample_count=sample_count,
            pcg_tol=pcg_tol,
            max_iter=max_iter,
        )
        if update:
            self._finalize_cosmic_graph(
                sparse, self.config.cosmic_graph.construction_threshold
            )
        return sparse

    @property
    def ball_maps(self) -> list[BallMapper]:
        """All fitted BallMapper objects across the parameter grid."""
        return self._ball_maps

    @property
    def stability_result(self) -> StabilityResult | None:
        """Stability analysis result (only available if threshold='auto')."""
        return self._stability_result

    @property
    def resolved_construction_threshold(self) -> float:
        """The actual construction threshold used (resolved from 'auto' or the manual value)."""
        if self._resolved_construction_threshold is None:
            raise RuntimeError("Call fit() first")
        return self._resolved_construction_threshold

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
