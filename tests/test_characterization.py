"""
Tests for pulsar.characterization module.

Tests dataset geometry probing, k-NN analysis, PCA variance estimation,
and missingness characterization.
"""

import numpy as np
import pandas as pd
import pytest

from pulsar.analysis.characterization import (
    ColumnProfile,
    DatasetProfile,
    characterize_dataset,
)


@pytest.fixture
def numeric_csv(tmp_path):
    """Basic numeric CSV: 200 samples, 5 features, no missing values."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((200, 5)),
        columns=[f"f{i}" for i in range(5)],
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_with_missing(tmp_path):
    """CSV with 20% missing values in first column."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((200, 5)), columns=[f"f{i}" for i in range(5)]
    )
    na_indices = rng.choice(len(df), size=40, replace=False)
    df.loc[na_indices, "f0"] = np.nan
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def categorical_csv(tmp_path):
    """CSV with only categorical columns (no numeric columns)."""
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 67, "dog": ["x", "y", "z"] * 67})
    path = tmp_path / "categorical.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def mixed_csv(tmp_path):
    """CSV with numeric, string, and missing columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(200)],
            "age": rng.integers(18, 90, size=200).astype(float),
            "gender": rng.choice(["M", "F", "NB"], size=200),
            "weight": rng.normal(70, 15, size=200),
            "score1": rng.standard_normal(200),
            "score2": rng.standard_normal(200),
        }
    )
    na_idx = rng.choice(200, size=20, replace=False)
    df.loc[na_idx, "age"] = np.nan
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Core DatasetProfile tests
# ---------------------------------------------------------------------------


def test_returns_dataset_profile(numeric_csv):
    """Assert function returns DatasetProfile."""
    profile = characterize_dataset(numeric_csv)
    assert isinstance(profile, DatasetProfile)


def test_profile_fields(numeric_csv):
    """Check profile basic fields."""
    profile = characterize_dataset(numeric_csv)
    assert profile.n_samples == 200
    assert profile.n_features == 5
    assert profile.missingness_pct == pytest.approx(0.0, abs=0.1)


def test_knn_distances_monotone(numeric_csv):
    """Assert k-NN distances are monotone: k5 <= k10 <= k20."""
    profile = characterize_dataset(numeric_csv)
    assert profile.knn_k5_mean <= profile.knn_k10_mean
    assert profile.knn_k10_mean <= profile.knn_k20_mean


def test_missing_values_handled(csv_with_missing):
    """Assert mean imputation works and missingness is tracked."""
    profile = characterize_dataset(csv_with_missing)
    assert profile.missingness_pct == pytest.approx(4.0, abs=0.1)  # 40/1000 total cells
    f0_prof = next(p for p in profile.column_profiles if p.name == "f0")
    assert f0_prof.missing_pct == pytest.approx(20.0, abs=0.1)


def test_non_numeric_handled(mixed_csv):
    """Assert non-numeric columns are excluded from geometry but profiled."""
    profile = characterize_dataset(mixed_csv)
    assert profile.n_features == 4
    assert profile.n_columns_total == 6

    gender_prof = next(p for p in profile.column_profiles if p.name == "gender")
    assert gender_prof.is_numeric is False
    assert gender_prof.n_unique == 3


def test_error_fewer_than_two_numeric_cols(categorical_csv):
    """ValueError if CSV lacks numeric columns."""
    with pytest.raises(ValueError, match="Need at least 2 numeric columns"):
        characterize_dataset(categorical_csv)


def test_pca_cumulative_variance_structure(numeric_csv):
    """Assert pca_cumulative_variance is a list of (dim, fraction) tuples in [0,1]."""
    profile = characterize_dataset(numeric_csv)
    assert isinstance(profile.pca_cumulative_variance, list)
    assert len(profile.pca_cumulative_variance) > 0
    for dim, frac in profile.pca_cumulative_variance:
        assert isinstance(dim, int)
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# Column profile tests
# ---------------------------------------------------------------------------


def test_column_profiles_present(mixed_csv):
    """Assert column_profiles covers ALL original columns."""
    profile = characterize_dataset(mixed_csv)
    assert profile.n_columns_total == 6
    assert len(profile.column_profiles) == 6
    assert all(isinstance(cp, ColumnProfile) for cp in profile.column_profiles)


def test_column_profiles_non_numeric_detected(mixed_csv):
    """Assert non-numeric columns are correctly flagged."""
    profile = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in profile.column_profiles}
    assert not by_name["patient_id"].is_numeric
    assert by_name["patient_id"].dtype in ("object", "str")
    assert not by_name["gender"].is_numeric
    assert by_name["age"].is_numeric
    assert by_name["weight"].is_numeric


def test_column_profiles_cardinality(mixed_csv):
    """Assert cardinality distinguishes IDs from categoricals."""
    profile = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in profile.column_profiles}
    assert by_name["patient_id"].n_unique == 200
    assert by_name["gender"].n_unique == 3


def test_column_profiles_missing_detected(mixed_csv):
    """Assert per-column missingness is reported."""
    profile = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in profile.column_profiles}
    assert by_name["age"].n_missing == 20
    assert by_name["age"].missing_pct == pytest.approx(10.0, abs=0.1)
    assert by_name["weight"].n_missing == 0
    assert by_name["weight"].missing_pct == 0.0


def test_column_profiles_top_values_for_strings(mixed_csv):
    """Assert top_values populated for non-numeric columns."""
    profile = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in profile.column_profiles}
    assert by_name["gender"].top_values is not None
    assert len(by_name["gender"].top_values) == 3  # M, F, NB
    for val, count in by_name["gender"].top_values:
        assert isinstance(val, str)
        assert isinstance(count, int)
    assert by_name["weight"].top_values is None


def test_column_profiles_numeric_stats(mixed_csv):
    """Assert numeric stats populated for numeric columns."""
    profile = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in profile.column_profiles}
    assert by_name["weight"].mean is not None
    assert by_name["weight"].std is not None
    assert by_name["weight"].min_val is not None
    assert by_name["weight"].max_val is not None
    assert by_name["gender"].mean is None
    assert by_name["gender"].std is None


def test_sample_values_are_strings(mixed_csv):
    """Assert sample_values are always strings (JSON-safe)."""
    profile = characterize_dataset(mixed_csv)
    for cp in profile.column_profiles:
        for sv in cp.sample_values:
            assert isinstance(sv, str)
