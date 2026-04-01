"""
Tests for pulsar.characterization module.

Tests dataset geometry probing, k-NN analysis, PCA variance estimation,
and hyperparameter recommendation generation.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from pulsar.characterization import (
    CharacterizationResult,
    DatasetProfile,
    GeometryRecommendations,
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
    df = pd.DataFrame(rng.standard_normal((200, 5)), columns=[f"f{i}" for i in range(5)])
    # Introduce 20% NaN in f0
    na_indices = rng.choice(len(df), size=40, replace=False)
    df.loc[na_indices, "f0"] = np.nan
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_high_missing(tmp_path):
    """CSV with 35% missing values."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(rng.standard_normal((200, 5)), columns=[f"f{i}" for i in range(5)])
    na_indices = rng.choice(len(df), size=70, replace=False)
    df.loc[na_indices, "f0"] = np.nan
    path = tmp_path / "high_missing.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def categorical_csv(tmp_path):
    """CSV with only categorical columns (no numeric columns)."""
    df = pd.DataFrame(
        {"cat": ["a", "b", "c"] * 67, "dog": ["x", "y", "z"] * 67}
    )
    path = tmp_path / "categorical.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def large_numeric_csv(tmp_path):
    """Large CSV: 2000 samples, 5 features."""
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        rng.standard_normal((2000, 5)),
        columns=[f"f{i}" for i in range(5)],
    )
    path = tmp_path / "large.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_returns_characterization_result(numeric_csv):
    """Assert function returns CharacterizationResult."""
    result = characterize_dataset(numeric_csv)
    assert isinstance(result, CharacterizationResult)
    assert isinstance(result.profile, DatasetProfile)
    assert isinstance(result.recommendations, GeometryRecommendations)


def test_profile_fields(numeric_csv):
    """Check profile basic fields."""
    result = characterize_dataset(numeric_csv)
    profile = result.profile

    assert profile.n_samples == 200
    assert profile.n_features == 5
    assert profile.missingness_pct == pytest.approx(0.0, abs=0.1)


def test_knn_distances_monotone(numeric_csv):
    """Assert k-NN distances are monotone: k5 ≤ k10 ≤ k20."""
    result = characterize_dataset(numeric_csv)
    profile = result.profile

    assert profile.knn_k5_mean <= profile.knn_k10_mean
    assert profile.knn_k10_mean <= profile.knn_k20_mean


def test_epsilon_range_valid(numeric_csv):
    """Assert epsilon_min > 0 and epsilon_min ≤ epsilon_max."""
    result = characterize_dataset(numeric_csv)
    recs = result.recommendations

    assert recs.epsilon_min > 0.0
    assert recs.epsilon_max > 0.0
    assert recs.epsilon_min <= recs.epsilon_max


def test_pca_dims_in_valid_range(numeric_csv):
    """Assert all recommended PCA dims are in valid range [2, n_features]."""
    result = characterize_dataset(numeric_csv)
    recs = result.recommendations

    assert len(recs.pca_dims) > 0
    for d in recs.pca_dims:
        assert 2 <= d <= result.profile.n_features


def test_pca_cumulative_variance_monotone(numeric_csv):
    """Assert cumulative PCA variance is non-decreasing."""
    result = characterize_dataset(numeric_csv)
    profile = result.profile

    prev_var = 0.0
    for dim, cumvar in profile.pca_cumulative_variance:
        assert cumvar >= prev_var, f"Non-monotone at dim={dim}: {cumvar} < {prev_var}"
        prev_var = cumvar


def test_yaml_template_parseable(numeric_csv):
    """Assert suggested_params_yaml is valid YAML."""
    result = characterize_dataset(numeric_csv)
    recs = result.recommendations

    parsed = yaml.safe_load(recs.suggested_params_yaml)
    assert isinstance(parsed, dict)
    assert "run" in parsed
    assert "sweep" in parsed
    assert "cosmic_graph" in parsed


def test_no_warning_without_issues(numeric_csv):
    """Assert no warnings generated on clean data."""
    result = characterize_dataset(numeric_csv)
    recs = result.recommendations
    # Clean data with good properties should have no warnings
    assert len(recs.warnings) == 0


def test_no_numeric_raises(categorical_csv):
    """Assert ValueError when CSV has no numeric columns."""
    with pytest.raises(ValueError, match="numeric"):
        characterize_dataset(categorical_csv)


def test_subsample_warning(large_numeric_csv):
    """Assert subsampling warning when data > subsample."""
    result = characterize_dataset(large_numeric_csv, subsample=100)
    recs = result.recommendations

    assert any("subsample" in w.lower() for w in recs.warnings)


def test_threshold_strategy_valid(numeric_csv):
    """Assert threshold_strategy is either 'auto' or '0.0'."""
    result = characterize_dataset(numeric_csv)
    recs = result.recommendations

    assert recs.threshold_strategy in ("auto", "0.0")


def test_suggested_dims_at_80pct_in_probed(numeric_csv):
    """Assert suggested_dims_at_80pct matches one of the probed dimensions."""
    result = characterize_dataset(numeric_csv)
    profile = result.profile
    recs = result.recommendations

    probed_dims = [d for d, _ in profile.pca_cumulative_variance]
    assert recs.suggested_dims_at_80pct in probed_dims


def test_epsilon_steps_fixed(numeric_csv):
    """Assert epsilon_steps is always 15."""
    result = characterize_dataset(numeric_csv)
    assert result.recommendations.epsilon_steps == 15


def test_rationale_nonempty(numeric_csv):
    """Assert rationale string is non-empty."""
    result = characterize_dataset(numeric_csv)
    assert len(result.recommendations.rationale) > 0


def test_low_dimensionality_warning(numeric_csv):
    """Verify low dimensionality case generates appropriate warning."""
    # Create a very low-dim CSV: 2D data
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((100, 2)),
        columns=["x", "y"],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/low_dim.csv"
        df.to_csv(path, index=False)
        result = characterize_dataset(path)
        recs = result.recommendations

        # May have low-dim warning
        if result.profile.n_features <= 2:
            # Low-dim warning would only appear if suggested_dims_at_80pct <= 2
            if recs.suggested_dims_at_80pct <= 2:
                assert any("dimensionality" in w.lower() for w in recs.warnings)


def test_high_dimensionality_warning():
    """Verify high dimensionality case generates appropriate warning."""
    # Create high-dim data
    rng = np.random.default_rng(99)
    df = pd.DataFrame(rng.standard_normal((100, 20)), columns=[f"f{i}" for i in range(20)])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/high_dim.csv"
        df.to_csv(path, index=False)
        result = characterize_dataset(path)
        recs = result.recommendations

        # High-dim warning if suggested_dims >= 15
        if recs.suggested_dims_at_80pct >= 15:
            assert any("dimensionality" in w.lower() for w in recs.warnings)
