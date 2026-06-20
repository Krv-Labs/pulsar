"""Utility functions for Pulsar."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np

# Approximate wall-clock fraction per pipeline stage (estimates, not guarantees).
# Order matters — stages are consumed sequentially by a cursor.
# PCA weight is zeroed when _precomputed_embeddings is used; fractions are renormalized.
STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("load", 0.03),
    ("impute", 0.08),
    ("scale", 0.01),
    ("pca", 0.25),
    ("ball_mapper", 0.42),
    ("laplacian", 0.15),
    ("cosmic", 0.06),  # includes stability analysis when threshold="auto"
]


def build_cumulative_fractions(
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
def rayon_thread_override(workers: int | None):
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


# ---------------------------------------------------------------------------
# Terminal Graphics Utilities
# ---------------------------------------------------------------------------

_SPARK_CHARS = "  ▂▃▄▅▆▇█"
_BAR_FILLED = "█"
_BAR_EMPTY = "░"


def generate_distribution_sparkline(
    data: list[float] | np.ndarray, bins: int = 10
) -> str:
    """Creates a Unicode sparkline histogram for a numeric distribution."""
    if len(data) == 0:
        return ""
    counts, _ = np.histogram(data, bins=bins)
    if counts.max() == 0:
        return _SPARK_CHARS[0] * bins
    scaled = (counts / counts.max() * (len(_SPARK_CHARS) - 1)).astype(int)
    return "".join(_SPARK_CHARS[i] for i in scaled)


def generate_proportion_bar(value: float, max_value: float, length: int = 10) -> str:
    """Creates a horizontal progress-bar style graphic for a proportion."""
    if max_value <= 0:
        return _BAR_EMPTY * length
    frac = max(0.0, min(1.0, value / max_value))
    filled_len = int(frac * length)
    empty_len = length - filled_len
    return (_BAR_FILLED * filled_len) + (_BAR_EMPTY * empty_len)
