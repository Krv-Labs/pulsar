"""
Dataset characterization for geometry-aware parameter suggestions.

Probes raw data geometry (k-NN distances, PCA variance) to provide raw facts
to the agent. The agent must reason about these facts to build a configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler as SkScaler

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
class NumericProfile:
    """k-NN and PCA geometry of an arbitrary numeric matrix.

    Shared math core used by both raw characterization and processed-space
    calibration.  No policy decisions — pure measurement.
    """

    knn_k5_mean: float
    knn_k10_mean: float
    knn_k20_mean: float
    knn_p5: float
    knn_p25: float
    knn_p50: float
    knn_p75: float
    knn_p95: float
    pca_cumulative_variance: list[tuple[int, float]]
    n_features: int
    n_samples_profiled: int


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


def profile_numeric_matrix(
    X: np.ndarray,
    subsample: int = 1000,
    seed: int = 42,
    dims_to_probe: list[int] | None = None,
) -> NumericProfile:
    """Compute k-NN distances and PCA variance on an arbitrary numeric matrix.

    This is the shared math core used by both :func:`characterize_dataset`
    (raw space) and processed-space calibration inside ``create_config``.

    Args:
        X: 2-D float64 array, already imputed (no NaN) and scaled.
        subsample: Max rows to analyze.
        seed: Random seed for reproducibility.
        dims_to_probe: PCA dimensions to test.  Defaults to
            ``[2, 3, 5, 10, 15, 20]`` clipped to feature count.

    Returns:
        NumericProfile with k-NN means and PCA cumulative variance.
    """
    rng = np.random.default_rng(seed)
    n_sub = min(subsample, len(X))
    indices = rng.choice(len(X), n_sub, replace=False)
    X_sub = X[indices]

    # k-NN
    k_max = min(21, n_sub - 1)
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto", metric="euclidean")
    distances, _ = nn.fit(X_sub).kneighbors(X_sub)

    knn_k5_mean = _bounded_knn_mean(distances, 5)
    knn_k10_mean = _bounded_knn_mean(distances, 10)
    knn_k20_mean = _bounded_knn_mean(distances, 20)

    # k-NN distance percentiles (k=5 neighbor distances across all points).
    # These define the valid epsilon domain for BallMapper.
    k5_upper = min(5, distances.shape[1] - 1)
    if k5_upper > 0:
        k5_dists = distances[:, k5_upper].ravel()
        knn_p5 = float(np.percentile(k5_dists, 5))
        knn_p25 = float(np.percentile(k5_dists, 25))
        knn_p50 = float(np.percentile(k5_dists, 50))
        knn_p75 = float(np.percentile(k5_dists, 75))
        knn_p95 = float(np.percentile(k5_dists, 95))
    else:
        knn_p5 = knn_p25 = knn_p50 = knn_p75 = knn_p95 = 0.0

    # PCA variance
    if dims_to_probe is None:
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

    return NumericProfile(
        knn_k5_mean=knn_k5_mean,
        knn_k10_mean=knn_k10_mean,
        knn_k20_mean=knn_k20_mean,
        knn_p5=knn_p5,
        knn_p25=knn_p25,
        knn_p50=knn_p50,
        knn_p75=knn_p75,
        knn_p95=knn_p95,
        pca_cumulative_variance=pca_cumulative_variance,
        n_features=X_sub.shape[1],
        n_samples_profiled=n_sub,
    )


def characterize_dataset(
    csv_path: str,
    subsample: int = 1000,
    seed: int = 42,
    *,
    dataframe: pd.DataFrame | None = None,
) -> DatasetProfile:
    """
    Probes dataset geometry before fitting to return raw geometric facts.

    Args:
        csv_path: Path to CSV file (must have >=2 numeric columns)
        subsample: Max rows to analyze (for speed on large datasets)
        seed: Random seed for reproducibility
        dataframe: Optional pre-loaded DataFrame. When provided, *csv_path*
            is ignored for reading (but still used for logging/identification).

    Returns:
        DatasetProfile containing pure empirical facts.

    Raises:
        ValueError: If CSV has fewer than 2 numeric columns
        FileNotFoundError: If CSV file not found
    """
    df = dataframe if dataframe is not None else pd.read_csv(csv_path)

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
    X_scaled = SkScaler().fit_transform(X)

    geo = profile_numeric_matrix(X_scaled, subsample=subsample, seed=seed)

    return DatasetProfile(
        n_samples=n_samples,
        n_features=n_features,
        n_columns_total=n_columns_total,
        missingness_pct=missingness_pct,
        knn_k5_mean=geo.knn_k5_mean,
        knn_k10_mean=geo.knn_k10_mean,
        knn_k20_mean=geo.knn_k20_mean,
        pca_cumulative_variance=geo.pca_cumulative_variance,
        column_profiles=column_profiles,
    )


def _profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    """Generates sparse profiles for ALL columns (Map View)."""
    profiles: list[ColumnProfile] = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                is_numeric=pd.api.types.is_numeric_dtype(series),
                n_unique=int(series.nunique()),
                n_missing=int(series.isna().sum()),
                missing_pct=round(float(series.isna().mean() * 100), 2)
                if n_rows > 0
                else 0.0,
                # Explicitly skip expensive fields in the global map
                sample_values=[],
                mean=None,
                std=None,
                min_val=None,
                max_val=None,
                top_values=None,
            )
        )
    return profiles


def profile_column_details(df: pd.DataFrame, col: str, max_sample: int = 10) -> ColumnProfile:
    """Generates a rich, detailed profile for a single column (Magnifying Glass)."""
    series = df[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    non_null = series.dropna()
    n_rows = len(df)
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
        vc = series.value_counts().head(10)
        top_values = [(str(k), int(v)) for k, v in vc.items()]

    return ColumnProfile(
        name=col,
        dtype=str(series.dtype),
        is_numeric=is_numeric,
        n_unique=n_unique,
        n_missing=n_missing,
        missing_pct=round(float(n_missing / n_rows * 100), 2) if n_rows > 0 else 0.0,
        sample_values=sample_vals,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        top_values=top_values,
    )


def _bounded_knn_mean(distances: np.ndarray, k: int) -> float:
    """Return the mean k-NN distance using whatever neighbors are available."""
    available = distances.shape[1] - 1
    if available <= 0:
        return 0.0
    upper = min(k, available)
    return float(distances[:, 1 : upper + 1].mean())
