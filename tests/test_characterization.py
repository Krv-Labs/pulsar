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
    ColumnProfile,
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


@pytest.fixture
def mixed_csv(tmp_path):
    """CSV with numeric, string, and missing columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(200)],
        "age": rng.integers(18, 90, size=200).astype(float),
        "gender": rng.choice(["M", "F", "NB"], size=200),
        "weight": rng.normal(70, 15, size=200),
        "score1": rng.standard_normal(200),
        "score2": rng.standard_normal(200),
    })
    # Introduce 10% NaN in age
    na_idx = rng.choice(200, size=20, replace=False)
    df.loc[na_idx, "age"] = np.nan
    path = tmp_path / "mixed.csv"
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
    assert "preprocessing" in parsed


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


# ---------------------------------------------------------------------------
# Column profile tests
# ---------------------------------------------------------------------------


def test_column_profiles_present(mixed_csv):
    """Assert column_profiles covers ALL original columns."""
    result = characterize_dataset(mixed_csv)
    profile = result.profile
    assert profile.n_columns_total == 6
    assert len(profile.column_profiles) == 6
    assert all(isinstance(cp, ColumnProfile) for cp in profile.column_profiles)


def test_column_profiles_non_numeric_detected(mixed_csv):
    """Assert non-numeric columns are correctly flagged."""
    result = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in result.profile.column_profiles}
    assert not by_name["patient_id"].is_numeric
    assert by_name["patient_id"].dtype in ("object", "str")
    assert not by_name["gender"].is_numeric
    assert by_name["age"].is_numeric
    assert by_name["weight"].is_numeric


def test_column_profiles_cardinality(mixed_csv):
    """Assert cardinality distinguishes IDs from categoricals."""
    result = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in result.profile.column_profiles}
    # patient_id: 200 unique values (high cardinality = likely ID)
    assert by_name["patient_id"].n_unique == 200
    # gender: 3 unique values (low cardinality = categorical)
    assert by_name["gender"].n_unique == 3


def test_column_profiles_missing_detected(mixed_csv):
    """Assert per-column missingness is reported."""
    result = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in result.profile.column_profiles}
    assert by_name["age"].n_missing == 20
    assert by_name["age"].missing_pct == pytest.approx(10.0, abs=0.1)
    assert by_name["weight"].n_missing == 0
    assert by_name["weight"].missing_pct == 0.0


def test_column_profiles_top_values_for_strings(mixed_csv):
    """Assert top_values populated for non-numeric columns."""
    result = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in result.profile.column_profiles}
    assert by_name["gender"].top_values is not None
    assert len(by_name["gender"].top_values) == 3  # M, F, NB
    for val, count in by_name["gender"].top_values:
        assert isinstance(val, str)
        assert isinstance(count, int)
    # Numeric columns should have no top_values
    assert by_name["weight"].top_values is None


def test_column_profiles_numeric_stats(mixed_csv):
    """Assert numeric stats populated for numeric columns."""
    result = characterize_dataset(mixed_csv)
    by_name = {cp.name: cp for cp in result.profile.column_profiles}
    assert by_name["weight"].mean is not None
    assert by_name["weight"].std is not None
    assert by_name["weight"].min_val is not None
    assert by_name["weight"].max_val is not None
    # Non-numeric should have None for these
    assert by_name["gender"].mean is None
    assert by_name["gender"].std is None


def test_sample_values_are_strings(mixed_csv):
    """Assert sample_values are always strings (JSON-safe)."""
    result = characterize_dataset(mixed_csv)
    for cp in result.profile.column_profiles:
        for sv in cp.sample_values:
            assert isinstance(sv, str)


def test_yaml_template_drop_columns_empty(mixed_csv):
    """Assert YAML template starts with empty drop_columns — agent decides."""
    result = characterize_dataset(mixed_csv)
    parsed = yaml.safe_load(result.recommendations.suggested_params_yaml)
    assert parsed["preprocessing"]["drop_columns"] == []


def test_yaml_template_has_impute_block(mixed_csv):
    """Assert YAML template includes impute entries for missing numeric cols."""
    result = characterize_dataset(mixed_csv)
    parsed = yaml.safe_load(result.recommendations.suggested_params_yaml)
    impute = parsed["preprocessing"]["impute"]
    assert "age" in impute
    assert impute["age"]["method"] == "fill_mean"
    # weight has no missing values — should not appear
    assert "weight" not in impute


def test_non_numeric_warning_neutral(mixed_csv):
    """Assert warning reports non-numeric columns factually without judgments."""
    result = characterize_dataset(mixed_csv)
    warnings = result.recommendations.warnings
    # Should mention non-numeric columns exist
    non_num_warnings = [w for w in warnings if "non-numeric" in w.lower()]
    assert len(non_num_warnings) == 1
    w = non_num_warnings[0]
    # Should list column names with cardinality
    assert "patient_id" in w
    assert "gender" in w
    # Should NOT contain judgment language
    assert "likely IDs" not in w
    assert "high-cardinality" not in w.lower()
    assert "low-cardinality" not in w.lower()
    # Should direct agent to column_profiles
    assert "column_profiles" in w


def test_missing_impute_warning(mixed_csv):
    """Assert warning about imputed numeric columns."""
    result = characterize_dataset(mixed_csv)
    warnings = result.recommendations.warnings
    assert any("fill_mean" in w for w in warnings)


def test_backward_compat_numeric_only(numeric_csv):
    """Assert pure-numeric CSV produces empty drop_columns and no impute."""
    result = characterize_dataset(numeric_csv)
    profile = result.profile
    assert profile.n_columns_total == 5
    assert all(cp.is_numeric for cp in profile.column_profiles)
    parsed = yaml.safe_load(result.recommendations.suggested_params_yaml)
    assert parsed["preprocessing"]["drop_columns"] == []
    assert "impute" not in parsed["preprocessing"]


# ---------------------------------------------------------------------------
# PCA dim cap tests (Fix 2)
# ---------------------------------------------------------------------------


def test_pca_dims_capped_by_sample_size(tmp_path):
    """Assert PCA dims are capped by sqrt(n_samples) for small datasets."""
    rng = np.random.default_rng(0)
    # 100 rows, 50 features — spread variance across many dims
    df = pd.DataFrame(rng.standard_normal((100, 50)), columns=[f"f{i}" for i in range(50)])
    path = tmp_path / "small_wide.csv"
    df.to_csv(path, index=False)
    result = characterize_dataset(str(path))
    # sqrt(100) = 10 → all dims should be ≤ 10
    for d in result.recommendations.pca_dims:
        assert d <= 10, f"PCA dim {d} exceeds sqrt(100)=10 cap"


def test_pca_dims_uncapped_large_sample(tmp_path):
    """Assert large sample size does not restrict dims below feature count."""
    rng = np.random.default_rng(0)
    # 10000 rows, 8 features — cap would be sqrt(10000)=100, well above 8
    df = pd.DataFrame(rng.standard_normal((10000, 8)), columns=[f"f{i}" for i in range(8)])
    path = tmp_path / "large_narrow.csv"
    df.to_csv(path, index=False)
    result = characterize_dataset(str(path))
    # All dims should be ≤ n_features-1 = 7 (not further restricted by sample cap)
    for d in result.recommendations.pca_dims:
        assert d <= 7
