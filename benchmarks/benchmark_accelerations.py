"""Benchmark JL, KD-tree Ball Mapper, and CosmicGraph sparsification.

Run from the repository root after building the extension:

    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv run maturin develop --release
    uv run python benchmarks/benchmark_accelerations.py
"""

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


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1_500, 128)).astype(np.float64)
    dims = [8, 16]
    seeds = [42, 7, 13]

    jl_embeddings = bench("jl_grid 1500x128 dims=8,16", lambda: jl_grid(X, dims, seeds))
    bench("pca_grid 1500x128 dims=8,16", lambda: pca_grid(X, dims, seeds), repeats=3)

    bm16 = rng.standard_normal((5_000, 16)).astype(np.float64)
    bm17 = np.ascontiguousarray(
        np.concatenate([bm16, rng.standard_normal((5_000, 1))], axis=1)
    )
    bench("BallMapper KD-eligible 16D", lambda: fit_ball_mapper(bm16, 3.0), repeats=5)
    bench("BallMapper fallback 17D", lambda: fit_ball_mapper(bm17, 3.0), repeats=5)

    cg = CosmicGraph.from_pseudo_laplacian(complete_laplacian(180), 0.0)
    sparse = bench(
        "CosmicGraph spectral_sparsify",
        lambda: cg.spectral_sparsify(1.3, seed=42, sketch_dim=12, sample_count=600),
        repeats=3,
    )
    print(f"CosmicGraph edges dense={cg.n_edges} sparse={sparse.n_edges}")


def fit_ball_mapper(points: np.ndarray, eps: float) -> BallMapper:
    bm = BallMapper(eps)
    bm.fit(points)
    return bm


if __name__ == "__main__":
    main()
