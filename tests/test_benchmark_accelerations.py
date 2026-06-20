"""Benchmark JL, KD-tree Ball Mapper, and CosmicGraph sparsification as a pytest-based benchmark."""

from __future__ import annotations

import statistics
import time

import numpy as np

from pulsar._pulsar import (
    BallMapper,
    CosmicGraph,
    jl_grid,
    pca_grid,
)


def bench(label: str, fn, repeats: int = 5):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    print(
        f"{label:34s} median={statistics.median(times):.4f}s "
        f"min={min(times):.4f}s max={max(times):.4f}s"
    )
    return result


def complete_laplacian(n: int) -> np.ndarray:
    lap = -np.ones((n, n), dtype=np.int64)
    np.fill_diagonal(lap, n - 1)
    return lap


def fit_ball_mapper(points: np.ndarray, eps: float) -> BallMapper:
    bm = BallMapper(eps)
    bm.fit(points)
    return bm


def test_benchmark_accelerations() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1_500, 128)).astype(np.float64)
    dims = [8, 16]
    seeds = [42, 7, 13]

    print()  # Ensure clean starting line for pytest with -s
    jl_res = bench("jl_grid 1500x128 dims=8,16", lambda: jl_grid(X, dims, seeds))
    pca_res = bench(
        "pca_grid 1500x128 dims=8,16", lambda: pca_grid(X, dims, seeds), repeats=3
    )

    assert len(jl_res) == len(dims) * len(seeds)
    assert len(pca_res) == len(dims) * len(seeds)

    bm16 = rng.standard_normal((5_000, 16)).astype(np.float64)
    bm17 = np.ascontiguousarray(
        np.concatenate([bm16, rng.standard_normal((5_000, 1))], axis=1)
    )
    bm16_res = bench(
        "BallMapper KD-eligible 16D", lambda: fit_ball_mapper(bm16, 3.0), repeats=5
    )
    bm17_res = bench(
        "BallMapper fallback 17D", lambda: fit_ball_mapper(bm17, 3.0), repeats=5
    )

    assert isinstance(bm16_res, BallMapper)
    assert isinstance(bm17_res, BallMapper)

    cg = CosmicGraph.from_pseudo_laplacian(complete_laplacian(180), 0.0)
    sparse = bench(
        "CosmicGraph spectral_sparsify",
        lambda: cg.spectral_sparsify(1.3, seed=42, sketch_dim=12, sample_count=600),
        repeats=3,
    )
    print(f"CosmicGraph edges dense={cg.n_edges} sparse={sparse.n_edges}")

    assert sparse.n_edges < cg.n_edges
