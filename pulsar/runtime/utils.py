"""Utility functions for Pulsar."""

from __future__ import annotations

import os
from contextlib import contextmanager

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
