"""Shared pure-Python reference implementations for tests."""

from __future__ import annotations

import numpy as np


def pseudo_laplacian_py(nodes, n):
    """Pure Python pseudo-Laplacian for testing.

    This replaces the removed Rust single-ball-map function.
    The Rust accumulate_pseudo_laplacians is the production API.
    """
    L = np.zeros((n, n), dtype=np.int64)
    for members in nodes:
        for i in members:
            for j in members:
                if i == j:
                    L[i, j] += 1
                else:
                    L[i, j] -= 1
    return L
