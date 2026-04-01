"""
Dataset characterization for geometry-aware parameter suggestions.

Probes raw data geometry (k-NN distances, PCA variance) to suggest grounded
hyperparameters (PCA dimensions, epsilon range, threshold strategy) before
fitting the topological pipeline.

Note: This probe applies simplified preprocessing (mean impute + StandardScaler)
which is an approximation of the full pipeline. Use recommendations as starting
bounds, not exact targets.
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
    dtype: str                                          # pandas dtype as string
    is_numeric: bool
    n_unique: int                                       # unique non-null values
    n_missing: int
    missing_pct: float                                  # [0, 100]
    sample_values: list[str]                            # up to 5, cast to str for JSON safety
    mean: float | None                                  # numeric only
    std: float | None                                   # numeric only
    min_val: float | None                               # numeric only
    max_val: float | None                               # numeric only
    top_values: list[tuple[str, int]] | None            # non-numeric only: [(value, count)] top 5


@dataclass
class DatasetProfile:
    """Raw measurements only — no derived decisions."""

    n_samples: int
    n_features: int                                     # numeric feature count
    n_columns_total: int                                # all columns including non-numeric
    missingness_pct: float
    knn_k5_mean: float
    knn_k10_mean: float
    knn_k20_mean: float
    pca_cumulative_variance: list[tuple[int, float]]    # [(dims, cumulative_ratio), ...]
    column_profiles: list[ColumnProfile]                # one per original column


@dataclass
class GeometryRecommendations:
    """Derived outputs and advisory messages."""

    pca_dims: list[int]
    epsilon_min: float
    epsilon_max: float
    epsilon_steps: int
    threshold_strategy: Literal["auto", "0.0"]
    suggested_dims_at_80pct: int
    rationale: str
    warnings: list[str]
    suggested_params_yaml: str


@dataclass
class CharacterizationResult:
    """Container for profile and recommendations."""

    profile: DatasetProfile
    recommendations: GeometryRecommendations


def characterize_dataset(
    csv_path: str,
    subsample: int = 1000,
    seed: int = 42,
) -> CharacterizationResult:
    """
    Probes dataset geometry before fitting to suggest PCA dims and epsilon range.

    Loads a CSV, analyzes numeric columns via k-NN distances and PCA variance,
    and returns concrete hyperparameter suggestions.

    Args:
        csv_path: Path to CSV file (must have >=2 numeric columns)
        subsample: Max rows to analyze (for speed on large datasets)
        seed: Random seed for reproducibility

    Returns:
        CharacterizationResult with profile, recommendations, and YAML template

    Raises:
        ValueError: If CSV has fewer than 2 numeric columns
        FileNotFoundError: If CSV file not found
    """
    # Step 1: Load and validate
    df = pd.read_csv(csv_path)

    # Profile ALL columns before filtering to numeric
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

    # Step 2: Impute, scale, subsample
    X = SimpleImputer(strategy="mean").fit_transform(
        df[numeric_cols].to_numpy(dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    n_sub = min(subsample, len(X))
    indices = rng.choice(len(X), n_sub, replace=False)
    X_sub = SkScaler().fit_transform(X[indices])

    # Step 3: k-NN distances (one fit at k=20, slice for k=5 and k=10)
    k_max = min(21, n_sub - 1)
    nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto", metric="euclidean")
    distances, _ = nn.fit(X_sub).kneighbors(X_sub)  # shape (n, k_max), col 0 is self (0.0)

    knn_k5_mean = float(distances[:, 1:6].mean())
    knn_k10_mean = float(distances[:, 1:11].mean())
    knn_k20_mean = float(distances[:, 1:min(21, distances.shape[1])].mean())

    # Step 4: PCA variance using pca_grid (parallel, consistent with pipeline)
    # Probe up to 20 dims or the number of features, whichever is smaller
    dims_to_probe = [d for d in [2, 3, 5, 10, 15, 20] if d <= X_sub.shape[1]]
    # Fallback: if data has fewer than 2 features, we can't do PCA, but we validated >= 2 numeric cols
    if not dims_to_probe:
        dims_to_probe = [min(2, X_sub.shape[1])]
    embeddings = pca_grid(
        np.ascontiguousarray(X_sub), dims_to_probe, [seed]
    )

    # Explained variance: divide by total variance of scaled input
    # Note: np.var on randomized PCA output is an approximation of explained variance,
    # not exact (randomized SVD introduces small approximation error). Suitable for
    # heuristic parameter suggestions only.
    total_variance = float(np.var(X_sub, axis=0).sum())
    pca_cumulative_variance: list[tuple[int, float]] = []
    for dim, emb in zip(dims_to_probe, embeddings):
        ev = float(np.var(emb, axis=0).sum())
        cumvar = round(min(ev / max(total_variance, 1e-12), 1.0), 4)
        pca_cumulative_variance.append((dim, cumvar))

    # Step 5: Suggested dims at 80% variance threshold
    suggested_dims_at_80pct = dims_to_probe[-1]  # fallback: max probed
    for dim, cumvar in pca_cumulative_variance:
        if cumvar >= 0.80:
            suggested_dims_at_80pct = dim
            break

    pca_dims = _build_pca_dims_recommendation(
        suggested_dims_at_80pct, dims_to_probe, n_features, n_samples,
    )

    # Step 6: Epsilon suggestion anchored to k-NN distances
    eps_min = round(knn_k10_mean * 0.8, 2)
    eps_max = round(knn_k20_mean * 1.2, 2)
    eps_steps = 15

    # Step 7: Threshold recommendation
    # Default to "auto" (persistent homology) for threshold selection.
    # PH is designed to find stable plateaus in connected components across weights.
    top2_var = next(
        (v for d, v in pca_cumulative_variance if d == 2), 0.0
    )
    threshold_strategy: Literal["auto", "0.0"] = "auto"

    # Step 8: Build warnings list
    warnings = []
    # Report non-numeric columns as facts — the agent decides what to do.
    non_numeric_summaries = [
        f"{cp.name} ({cp.n_unique} unique"
        + (f": {', '.join(str(v) for v, _ in (cp.top_values or [])[:3])}" if cp.top_values else "")
        + ")"
        for cp in column_profiles
        if not cp.is_numeric
    ]
    if non_numeric_summaries:
        warnings.append(
            f"Non-numeric columns found: {', '.join(non_numeric_summaries)}. "
            f"The pipeline requires float64 input. Low-cardinality columns (<=10 unique) "
            f"have been added to the suggested encode block; others are in drop_columns. "
            f"Review and adjust as needed."
        )
    numeric_with_nan = [
        f"{cp.name} ({cp.missing_pct:.1f}%)"
        for cp in column_profiles
        if cp.is_numeric and cp.missing_pct > 0
    ]
    if numeric_with_nan:
        warnings.append(
            f"Numeric columns with missing values (fill_mean default added): "
            f"{', '.join(numeric_with_nan)}. Review imputation methods."
        )
    if n_samples > subsample:
        warnings.append(f"Subsampled to {subsample} of {n_samples} rows for speed.")
    if missingness_pct > 30:
        warnings.append(
            f"High missingness ({missingness_pct:.1f}%): review imputation strategy."
        )
    if suggested_dims_at_80pct <= 2:
        warnings.append("Very low intrinsic dimensionality — try dims [2, 3] only.")
    if suggested_dims_at_80pct >= 15:
        warnings.append(
            "High intrinsic dimensionality (≥15): Ball Mapper may be less effective."
        )
    if top2_var < 0.30:
        warnings.append(
            "High intrinsic dimensionality (top-2 PCA < 30% variance): "
            "'auto' threshold may not find a stable plateau. Review the threshold after fitting."
        )

    # Step 9: Build YAML template
    suggested_params_yaml = _build_yaml_template(
        csv_path, pca_dims, eps_min, eps_max, eps_steps, threshold_strategy,
        column_profiles,
    )

    rationale = (
        f"Based on {n_sub} samples: k-NN distances (k5={knn_k5_mean:.3f}, "
        f"k10={knn_k10_mean:.3f}, k20={knn_k20_mean:.3f}) suggest epsilon in "
        f"[{eps_min}, {eps_max}]. PCA analysis shows {suggested_dims_at_80pct} dims "
        f"capture 80% variance. High intrinsic dimensionality ({top2_var:.1%} in top 2) "
        f"→ threshold='{threshold_strategy}'."
    )

    profile = DatasetProfile(
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

    recommendations = GeometryRecommendations(
        pca_dims=pca_dims,
        epsilon_min=eps_min,
        epsilon_max=eps_max,
        epsilon_steps=eps_steps,
        threshold_strategy=threshold_strategy,
        suggested_dims_at_80pct=suggested_dims_at_80pct,
        rationale=rationale,
        warnings=warnings,
        suggested_params_yaml=suggested_params_yaml,
    )

    return CharacterizationResult(
        profile=profile,
        recommendations=recommendations,
    )


def _build_pca_dims_recommendation(
    knee: int, probed: list[int], n_features: int, n_samples: int,
) -> list[int]:
    """Return [knee-1, knee, knee+2] capped to valid range and sample-size limit.

    Ball Mapper uses Euclidean distance; in high dimensions with few samples,
    pairwise distances converge (curse of dimensionality), producing a razor-thin
    epsilon window. sqrt(n_samples) is a practical ceiling.
    """
    sample_cap = max(2, int(n_samples ** 0.5))
    dim_cap = min(n_features - 1, sample_cap)

    candidates = [
        max(2, knee - 1),
        knee,
        min(n_features - 1, knee + 2),
    ]
    capped = [min(d, dim_cap) for d in candidates]
    return sorted(set(capped))


def _profile_columns(df: pd.DataFrame, max_sample: int = 5) -> list[ColumnProfile]:
    """Build per-column metadata from raw DataFrame (before any filtering)."""
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

        profiles.append(ColumnProfile(
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
        ))
    return profiles


def _build_yaml_template(
    csv_path: str,
    pca_dims: list[int],
    eps_min: float,
    eps_max: float,
    eps_steps: int,
    threshold: Literal["auto", "0.0"],
    column_profiles: list[ColumnProfile],
) -> str:
    """Generate YAML config template with drop_columns, encode, and impute blocks."""
    # Columns to encode: non-numeric with low cardinality (<= 10 unique values)
    # This preserves valuable categorical context for the topological analysis.
    to_encode = [
        cp.name for cp in column_profiles if not cp.is_numeric and cp.n_unique <= 10
    ]

    # Columns to drop: everything else non-numeric
    # (except ID-like columns which characterization doesn't know yet)
    to_drop = [
        cp.name
        for cp in column_profiles
        if not cp.is_numeric and cp.n_unique > 10
    ]

    drop_line = str(to_drop) if to_drop else "[]"

    encode_block = ""
    if to_encode:
        encode_block = "\n  encode:"
        for col_name in to_encode:
            encode_block += f"\n    {col_name}: {{method: one_hot}}"

    numeric_missing = [cp.name for cp in column_profiles if cp.is_numeric and cp.missing_pct > 0]

    impute_block = ""
    if numeric_missing:
        impute_block = "\n  impute:"
        for col_name in numeric_missing:
            impute_block += f"\n    {col_name}: {{method: fill_mean, seed: 42}}"

    return f"""run:
  name: experiment
  data: {csv_path}
preprocessing:
  drop_columns: {drop_line}{encode_block}{impute_block}
sweep:
  pca:
    dimensions:
      values: {pca_dims}
    seed:
      values: [42]
  ball_mapper:
    epsilon:
      range:
        min: {eps_min}
        max: {eps_max}
        steps: {eps_steps}
cosmic_graph:
  threshold: {threshold}
output:
  n_reps: 4
"""
