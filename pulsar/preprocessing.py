"""
Python-side preprocessing utilities for Pulsar.

Complements the Rust impute_column (f64 only) with Python implementations
for data types Rust cannot handle directly (string/object columns).
"""

from __future__ import annotations

import random

import pandas as pd

from pulsar.config import ImputeSpec


def impute_string_column(df: pd.DataFrame, col: str, spec: ImputeSpec) -> None:
    """Impute a string/object column in-place.

    The Rust ``impute_column`` only handles f64 arrays, so string/object
    columns require a Python implementation.  Only ``fill_mode`` and
    ``sample_categorical`` are meaningful for string data.

    Raises:
        ValueError: If ``spec.method`` is numeric-only
            (``fill_mean``, ``fill_median``, ``sample_normal``).
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
