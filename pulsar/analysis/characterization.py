"""
Dataset characterization for geometry-aware parameter suggestions.

Probes raw data geometry (k-NN distances, PCA variance) to provide raw facts
to the agent. The agent must reason about these facts to build a configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.neighbors import NearestNeighbors

from pulsar._pulsar import pca_grid

logger = logging.getLogger(__name__)

@dataclass
class ColumnProfile:
    """Per-column metadata for LLM preprocessing decisions."""

    name: str
    dtype: str
    is_numeric: bool
    n_unique: int
    n_missing: int
    missing_pct: float
    sample_values: list[str]
    mean: float | None
    std: float | None
    min_val: float | None
    max_val: float | None
    top_values: list[tuple[str, int]] | None


@dataclass
class DatasetProfile:
    """Raw measurements only — no derived decisions."""

    n_samples: int
    n_features: int
    n_columns_total: int
    missingness_pct: float
    knn_k5_mean: float
    knn_k10_mean: float
    knn_k20_mean: float
    pca_cumulative_variance: list[tuple[int, float]]
    column_profiles: list[ColumnProfile]


def characterize_dataset(
    csv_path: str,
    subsample: int = 1000,
    seed: int = 42,
) -> DatasetProfile:
    """
    Probes dataset geometry before fitting to return raw geometric facts.

    Args:
        csv_path: Path to CSV file (must have >=2 numeric columns)
        subsample: Max rows to analyze (for speed on large datasets)
        seed: Random seed for reproducibility

    Returns:
        DatasetProfile containing pure empirical facts.

    Raises:
        ValueError: If CSV has fewer than 2 numeric columns
        FileNotFoundError: If CSV file not found
    """
    df = pd.read_csv(csv_path)

    column_profiles = _profile_columns(df)
    n_columns_total = len(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        raise ValueError(
            f"Need at least 2 numeric columns for geometry analysis, "
            f"found {len(numeric_cols)}."
        )

    n_samples = len(df)
    n_features = len(numeric_cols)
    missingness_pct = float(df[numeric_cols].isna().mean().mean() * 100)

    X = SimpleImputer(strategy="mean").fit_transform(
        df[numeric_cols].to_numpy(dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    n_sub = min(subsample, len(X))
    indices = rng.choice(len(X), n_sub, replace=False)
    X_sub = SkScaler().fit_transform(X[indices])

    k_max = min(21, n_sub - 1)
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto", metric="euclidean")
    distances, _ = nn.fit(X_sub).kneighbors(X_sub)

    knn_k5_mean = float(distances[:, 1:6].mean())
    knn_k10_mean = float(distances[:, 1:11].mean())
    knn_k20_mean = float(distances[:, 1 : min(21, distances.shape[1])].mean())

    dims_to_probe = [d for d in [2, 3, 5, 10, 15, 20] if d <= X_sub.shape[1]]
    if not dims_to_probe:
        dims_to_probe = [min(2, X_sub.shape[1])]
    embeddings = pca_grid(np.ascontiguousarray(X_sub), dims_to_probe, [seed])

    total_variance = float(np.var(X_sub, axis=0).sum())
    pca_cumulative_variance: list[tuple[int, float]] = []
    for dim, emb in zip(dims_to_probe, embeddings):
        ev = float(np.var(emb, axis=0).sum())
        cumvar = round(min(ev / max(total_variance, 1e-12), 1.0), 4)
        pca_cumulative_variance.append((dim, cumvar))

    return DatasetProfile(
        n_samples=n_samples,
        n_features=n_features,
        n_columns_total=n_columns_total,
        missingness_pct=missingness_pct,
        knn_k5_mean=knn_k5_mean,
        knn_k10_mean=knn_k10_mean,
        knn_k20_mean=knn_k20_mean,
        pca_cumulative_variance=pca_cumulative_variance,
        column_profiles=column_profiles,
    )


def _profile_columns(df: pd.DataFrame, max_sample: int = 5) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        non_null = series.dropna()
        n_missing = int(series.isna().sum())
        n_unique = int(non_null.nunique())
        sample_vals = [str(v) for v in non_null.unique()[:max_sample]]

        mean = std = min_val = max_val = None
        top_values = None

        if is_numeric and len(non_null) > 0:
            mean = round(float(non_null.mean()), 4)
            std = round(float(non_null.std()), 4) if len(non_null) > 1 else None
            min_val = round(float(non_null.min()), 4)
            max_val = round(float(non_null.max()), 4)
        elif not is_numeric:
            vc = series.value_counts().head(5)
            top_values = [(str(k), int(v)) for k, v in vc.items()]

        profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                is_numeric=is_numeric,
                n_unique=n_unique,
                n_missing=n_missing,
                missing_pct=round(float(n_missing / n_rows * 100), 2)
                if n_rows > 0
                else 0.0,
                sample_values=sample_vals,
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                top_values=top_values,
            )
        )
    return profiles
