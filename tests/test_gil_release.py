"""Tests for PyO3 zero-copy borrowing and GIL release across Rust primitives."""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

from pulsar._pulsar import (
    PCA,
    BallMapper,
    CosmicGraph,
    JLProjection,
    MinHashAccumulator,
    StandardScaler,
    accumulate_pseudo_laplacians,
    accumulate_pseudo_laplacians_sparse,
    accumulate_temporal_pseudo_laplacians,
    ball_mapper_grid,
    find_stable_thresholds,
    find_stable_thresholds_sparse,
    impute_column,
    jl_grid,
    pca_grid,
    py_normalize_temporal_laplacian,
)
from pulsar.config import load_config
from pulsar.pipeline import ThemaRS, projection_grid

T = TypeVar("T")

_TICK_SLEEP_S = 0.005
_MAX_MEDIAN_TICK_S = 0.050
_MAX_PEAK_TICK_S = 0.150


async def _measure_ticks_during(fn: Callable[[], T]) -> tuple[T, list[float]]:
    """Run ``fn`` in ``asyncio.to_thread`` while recording event-loop tick deltas."""
    deltas: list[float] = []
    running = True

    async def _ticker() -> None:
        last = time.perf_counter()
        while running:
            await asyncio.sleep(_TICK_SLEEP_S)
            now = time.perf_counter()
            deltas.append(now - last)
            last = now

    ticker_task = asyncio.create_task(_ticker())
    try:
        result = await asyncio.to_thread(fn)
    finally:
        running = False
        await ticker_task
    return result, deltas


def _assert_responsive_ticks(deltas: list[float], min_ticks: int = 5) -> None:
    """Verify the asyncio loop ticked frequently without GIL starvation."""
    np.testing.assert_array_less(min_ticks - 1, len(deltas))
    np.testing.assert_array_less(float(np.median(deltas)), _MAX_MEDIAN_TICK_S)
    np.testing.assert_array_less(float(np.max(deltas)), _MAX_PEAK_TICK_S)


def _make_sweep_dataframe(n: int = 1800, d: int = 16, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame sized for ~150-400ms of Rust sweep work."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float64)
    return pd.DataFrame(data)


def _make_sweep_config(method: str = "pca"):
    """Build a multi-seed, multi-epsilon config that exercises Rust compute."""
    return load_config(
        {
            "preprocessing": {},
            "sweep": {
                "projection": {
                    "method": method,
                    "dimensions": {"values": [4, 6, 8]},
                    "seed": {"values": [11, 22, 33, 44]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [0.9, 1.2, 1.5, 1.8]},
                },
            },
            "cosmic_graph": {
                "construction": "minhash",
                "minhash_d": 128,
                "construction_threshold": "auto",
                "sparsify": True,
                "sparsify_epsilon": 1.0,
                "sparsify_seed": 42,
                "sparsify_sketch_dim": 8,
                "sparsify_sample_count": 200,
            },
        }
    )


def _run_primitives_workload(x_raw: np.ndarray, cfg) -> CosmicGraph:
    """Execute direct ``_pulsar`` primitives back-to-back on ``x_raw``."""
    n = x_raw.shape[0]
    x_scaled = np.asarray(StandardScaler().fit_transform(x_raw))
    embeddings = projection_grid(x_scaled, cfg)
    bms = ball_mapper_grid(embeddings, [0.9, 1.2, 1.5, 1.8])
    spl = accumulate_pseudo_laplacians_sparse(bms, n)
    _ = CosmicGraph.from_pseudo_laplacian_sparse(spl, 0.0)
    cg = CosmicGraph.from_ball_maps_minhash(bms, n, d=128, seed=42)
    return cg.spectral_sparsify(1.0, seed=42, sketch_dim=8, sample_count=200)


def test_event_loop_responsive_during_rust_sweep() -> None:
    """``ThemaRS.fit`` inside ``asyncio.to_thread`` must not block the event loop."""
    df = _make_sweep_dataframe(n=1800, d=16, seed=42)
    cfg = _make_sweep_config(method="pca")

    model, deltas = asyncio.run(_measure_ticks_during(lambda: ThemaRS(cfg).fit(df)))

    np.testing.assert_equal(model.cosmic_rust.n, 1800)
    _assert_responsive_ticks(deltas)


def test_fit_never_forces_gc_collect(monkeypatch) -> None:
    """``fit``/``fit_multi`` must not call ``gc.collect()`` from the worker thread.

    A full collection holds the GIL for its whole duration (tens to hundreds of
    ms in a long-lived MCP process), which stalls the event loop; the
    responsiveness test above only catches that on large heaps. Ball maps hold
    no reference cycles, so refcounting already frees each batch.
    """

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("gc.collect() called inside the sweep")

    monkeypatch.setattr(gc, "collect", _forbidden)
    df = _make_sweep_dataframe(n=300, d=8, seed=7)
    cfg = _make_sweep_config(method="jl")

    ThemaRS(cfg).fit(df)
    ThemaRS(cfg).fit_multi([df, df.copy()], ballmap_batch_size=2)


def test_event_loop_responsive_during_primitives() -> None:
    """Direct ``_pulsar`` primitives inside ``asyncio.to_thread`` release the GIL."""
    rng = np.random.default_rng(99)
    x_raw = rng.standard_normal((1800, 16)).astype(np.float64)
    cfg = _make_sweep_config(method="pca")

    cg, deltas = asyncio.run(
        _measure_ticks_during(lambda: _run_primitives_workload(x_raw, cfg))
    )

    np.testing.assert_equal(cg.n, 1800)
    _assert_responsive_ticks(deltas)


def _verify_scaling_and_single_projections(x: np.ndarray) -> np.ndarray:
    """Check StandardScaler, PCA, and JLProjection round-trip invariants."""
    scaler = StandardScaler()
    x_scaled = np.asarray(scaler.fit_transform(x))
    np.testing.assert_equal(x_scaled.shape, x.shape)
    np.testing.assert_allclose(np.asarray(scaler.transform(x)), x_scaled, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(scaler.inverse_transform(x_scaled)), x, atol=1e-10
    )

    pca = PCA(2, 42)
    x_pca = np.asarray(pca.fit_transform(x_scaled))
    np.testing.assert_equal(x_pca.shape, (x.shape[0], 2))
    np.testing.assert_allclose(np.asarray(pca.transform(x_scaled)), x_pca, atol=1e-12)

    jl = JLProjection(2, 42)
    x_jl = np.asarray(jl.fit_transform(x_scaled))
    np.testing.assert_equal(x_jl.shape, (x.shape[0], 2))
    np.testing.assert_allclose(np.asarray(jl.transform(x_scaled)), x_jl, atol=1e-12)
    return x_scaled


def _verify_projection_grids(x_scaled: np.ndarray) -> list[np.ndarray]:
    """Check pca_grid, jl_grid, and pipeline projection_grid parity."""
    n = x_scaled.shape[0]
    grid_pca = pca_grid(x_scaled, [2, 3], [42])
    cfg_pca = load_config(
        {
            "sweep": {
                "projection": {
                    "method": "pca",
                    "dimensions": {"values": [2, 3]},
                    "seed": {"values": [42]},
                }
            }
        }
    )
    proj_pca = projection_grid(x_scaled, cfg_pca)
    np.testing.assert_allclose(np.asarray(grid_pca[0]), proj_pca[0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(grid_pca[1]), proj_pca[1], atol=1e-12)

    grid_jl = jl_grid(x_scaled, [2, 3], [42], center=True)
    np.testing.assert_equal(np.asarray(grid_jl[0]).shape, (n, 2))
    np.testing.assert_equal(np.asarray(grid_jl[1]).shape, (n, 3))
    return proj_pca


def _verify_ball_mapper_and_laplacians(
    x_emb: np.ndarray,
) -> tuple[list[BallMapper], np.ndarray]:
    """Verify BallMapper vs ball_mapper_grid and dense/sparse/temporal Laplacians."""
    n = x_emb.shape[0]
    bm = BallMapper(0.9)
    bm.fit(x_emb)
    grid_bms = ball_mapper_grid([x_emb], [0.9])
    np.testing.assert_equal(bm.nodes, grid_bms[0].nodes)
    np.testing.assert_equal(bm.edges, grid_bms[0].edges)

    bms = ball_mapper_grid([x_emb], [0.7, 1.1])
    dense_l = np.asarray(accumulate_pseudo_laplacians(bms, n))
    spl = accumulate_pseudo_laplacians_sparse(bms, n)
    np.testing.assert_equal(dense_l.shape, (n, n))
    np.testing.assert_equal(spl.n, n)
    np.testing.assert_array_equal(np.diag(dense_l), np.asarray(spl.diag))

    reconstructed = np.diag(np.asarray(spl.diag)).astype(np.int64)
    for i, j, count in spl.offdiag:
        reconstructed[i, j] -= count
        reconstructed[j, i] -= count
    np.testing.assert_array_equal(reconstructed, dense_l)

    bm1, bm2 = bms[0], bms[1]
    spl_merged = accumulate_pseudo_laplacians_sparse([bm1], n)
    spl_merged.merge_in_place(accumulate_pseudo_laplacians_sparse([bm2], n))
    np.testing.assert_array_equal(np.asarray(spl_merged.diag), np.asarray(spl.diag))
    np.testing.assert_equal(spl_merged.offdiag, spl.offdiag)

    temporal_l = np.asarray(
        accumulate_temporal_pseudo_laplacians([[bm1, bm2], [bm2, bm1]], n)
    )
    np.testing.assert_equal(temporal_l.shape, (n, n, 2))
    np.testing.assert_array_equal(temporal_l[:, :, 0], dense_l)
    np.testing.assert_array_equal(temporal_l[:, :, 1], dense_l)

    temporal_w = np.asarray(py_normalize_temporal_laplacian(temporal_l))
    np.testing.assert_allclose(temporal_w[:, :, 0], temporal_w[:, :, 1], atol=1e-12)
    return bms, dense_l


def _verify_cosmic_and_thresholds(bms: list[BallMapper], dense_l: np.ndarray) -> None:
    """Verify MinHashAccumulator, CosmicGraph, sparsification, and PH thresholds."""
    n = dense_l.shape[0]
    spl = accumulate_pseudo_laplacians_sparse(bms, n)

    cg_dense = CosmicGraph.from_pseudo_laplacian(dense_l, 0.0)
    cg_sparse = CosmicGraph.from_pseudo_laplacian_sparse(spl, 0.0)
    np.testing.assert_allclose(
        np.asarray(cg_dense.weighted_adj),
        np.asarray(cg_sparse.weighted_adj),
        atol=1e-12,
    )

    acc = MinHashAccumulator(n, 64, 42)
    acc.accumulate(bms)
    cg_acc = acc.to_cosmic_graph()
    cg_direct = CosmicGraph.from_ball_maps_minhash(bms, n, 64, 42)
    np.testing.assert_equal(cg_acc.weighted_edges(), cg_direct.weighted_edges())

    sp1 = cg_sparse.spectral_sparsify(1.2, seed=42, sketch_dim=6, sample_count=40)
    sp2 = cg_sparse.spectral_sparsify(1.2, seed=42, sketch_dim=6, sample_count=40)
    np.testing.assert_equal(sp1.weighted_edges(), sp2.weighted_edges())

    st_dense = find_stable_thresholds(np.asarray(cg_dense.weighted_adj))
    st_sparse = find_stable_thresholds_sparse(n, cg_sparse.weighted_edges())
    np.testing.assert_allclose(
        st_dense.optimal_threshold, st_sparse.optimal_threshold, atol=1e-12
    )
    np.testing.assert_array_equal(
        np.asarray(st_dense.component_counts),
        np.asarray(st_sparse.component_counts),
    )


def _verify_strided_zero_copy_inputs(x: np.ndarray) -> None:
    """Ensure non-contiguous strided views produce identical results to copies."""
    strided = x[::2, :]
    np.testing.assert_equal(strided.flags["C_CONTIGUOUS"], False)
    contiguous = np.ascontiguousarray(strided)

    np.testing.assert_allclose(
        np.asarray(StandardScaler().fit_transform(strided)),
        np.asarray(StandardScaler().fit_transform(contiguous)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(PCA(2, 42).fit_transform(strided)),
        np.asarray(PCA(2, 42).fit_transform(contiguous)),
        atol=1e-12,
    )
    bm_s = BallMapper(1.1)
    bm_s.fit(strided)
    bm_c = BallMapper(1.1)
    bm_c.fit(contiguous)
    np.testing.assert_equal(bm_s.nodes, bm_c.nodes)

    f_arr = np.asfortranarray(x)
    np.testing.assert_equal(f_arr.flags["C_CONTIGUOUS"], False)
    bm_f = BallMapper(1.1)
    bm_f.fit(f_arr)
    bm_x = BallMapper(1.1)
    bm_x.fit(x)
    np.testing.assert_equal(bm_f.nodes, bm_x.nodes)
    np.testing.assert_equal(
        ball_mapper_grid([f_arr], [1.1])[0].nodes,
        ball_mapper_grid([x], [1.1])[0].nodes,
    )

    col = x[:, 0].copy()
    col[3] = np.nan
    np.testing.assert_allclose(
        np.asarray(impute_column(col[::2], "fill_mean", 0)),
        np.asarray(impute_column(np.ascontiguousarray(col[::2]), "fill_mean", 0)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(impute_column(col[::-1], "fill_mean", 0)),
        np.asarray(impute_column(np.ascontiguousarray(col[::-1]), "fill_mean", 0)),
        atol=1e-12,
    )


def test_gil_release_primitives_numerical_parity() -> None:
    """Verify deterministic numerical and signature parity across ``_pulsar`` APIs."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal((64, 6)).astype(np.float64)

    x_scaled = _verify_scaling_and_single_projections(x)
    embeddings = _verify_projection_grids(x_scaled)
    bms, dense_l = _verify_ball_mapper_and_laplacians(embeddings[0])
    _verify_cosmic_and_thresholds(bms, dense_l)
    _verify_strided_zero_copy_inputs(x)
