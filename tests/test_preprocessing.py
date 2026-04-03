"""Tests for pulsar.preprocessing — preprocess_dataframe() and helpers."""

import numpy as np
import pandas as pd
import pytest

from pulsar.config import EncodeSpec, ImputeSpec, PulsarConfig, PCASpec, BallMapperSpec, CosmicGraphSpec
from pulsar.preprocessing import (
    impute_string_column,
    preprocess_dataframe,
)


def _minimal_cfg(**overrides) -> PulsarConfig:
    """Build a PulsarConfig with sensible defaults, overriding specific fields."""
    defaults = dict(
        data="",
        impute={},
        encode={},
        drop_columns=[],
        pca=PCASpec(),
        ball_mapper=BallMapperSpec(),
        cosmic_graph=CosmicGraphSpec(),
    )
    defaults.update(overrides)
    return PulsarConfig(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_all_numeric_passthrough(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        cfg = _minimal_cfg()
        result_df, layout = preprocess_dataframe(df, cfg)
        assert result_df.shape == (3, 2)
        assert not result_df.isna().any().any()
        assert layout.feature_names == ("a", "b")
        assert layout.n_rows == 3

    def test_row_count_preserved(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        cfg = _minimal_cfg()
        result_df, layout = preprocess_dataframe(df, cfg)
        assert len(result_df) == 5
        assert layout.n_rows == 5


# ---------------------------------------------------------------------------
# Drop columns
# ---------------------------------------------------------------------------


class TestDropColumns:
    def test_drop_removes_column(self):
        df = pd.DataFrame({"id": [1, 2], "val": [3.0, 4.0]})
        cfg = _minimal_cfg(drop_columns=["id"])
        result_df, layout = preprocess_dataframe(df, cfg)
        assert "id" not in result_df.columns
        assert result_df.shape == (2, 1)

    def test_drop_nonexistent_column_is_silent(self):
        df = pd.DataFrame({"val": [1.0, 2.0]})
        cfg = _minimal_cfg(drop_columns=["nonexistent"])
        result_df, _ = preprocess_dataframe(df, cfg)
        assert list(result_df.columns) == ["val"]


# ---------------------------------------------------------------------------
# Imputation flags
# ---------------------------------------------------------------------------


class TestImputationFlags:
    def test_was_missing_flag_created(self):
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0]})
        cfg = _minimal_cfg(impute={"age": ImputeSpec(method="fill_mean")})
        result_df, _ = preprocess_dataframe(df, cfg)
        assert "age_was_missing" in result_df.columns
        assert result_df["age_was_missing"].tolist() == [0.0, 1.0, 0.0]

    def test_imputed_value_filled(self):
        df = pd.DataFrame({"age": [10.0, np.nan, 30.0]})
        cfg = _minimal_cfg(impute={"age": ImputeSpec(method="fill_mean")})
        result_df, _ = preprocess_dataframe(df, cfg)
        assert not result_df["age"].isna().any()


# ---------------------------------------------------------------------------
# NaN validation (replaces dropna)
# ---------------------------------------------------------------------------


class TestNaNValidation:
    def test_nan_after_partial_impute_raises(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        cfg = _minimal_cfg(impute={"a": ImputeSpec(method="fill_mean")})
        with pytest.raises(ValueError, match=r"'b' \(1 rows\)"):
            preprocess_dataframe(df, cfg)

    def test_no_nan_after_full_impute(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
        cfg = _minimal_cfg(
            impute={
                "a": ImputeSpec(method="fill_mean"),
                "b": ImputeSpec(method="fill_mean"),
            }
        )
        result_df, _ = preprocess_dataframe(df, cfg)
        assert not result_df.isna().any().any()


# ---------------------------------------------------------------------------
# Non-numeric validation
# ---------------------------------------------------------------------------


class TestNumericValidation:
    def test_non_numeric_column_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "name": ["alice", "bob"]})
        cfg = _minimal_cfg()
        with pytest.raises(TypeError, match=r"'name' \(dtype=(object|str)\)"):
            preprocess_dataframe(df, cfg)


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------


class TestNumericCoercion:
    def test_coercible_strings_succeed(self):
        df = pd.DataFrame({"age": ["25", "30", "35"]})
        cfg = _minimal_cfg(impute={"age": ImputeSpec(method="fill_mean")})
        result_df, _ = preprocess_dataframe(df, cfg)
        assert result_df["age"].tolist() == [25.0, 30.0, 35.0]

    def test_mostly_non_numeric_fails(self):
        df = pd.DataFrame({"age": ["young", "old", "25", "middle"]})
        cfg = _minimal_cfg(impute={"age": ImputeSpec(method="fill_mean")})
        with pytest.raises(ValueError, match="non-numeric values"):
            preprocess_dataframe(df, cfg)


# ---------------------------------------------------------------------------
# All-missing guard
# ---------------------------------------------------------------------------


class TestAllMissingGuard:
    def test_all_missing_impute_column_raises(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        cfg = _minimal_cfg(impute={"x": ImputeSpec(method="fill_mean")})
        with pytest.raises(ValueError, match="all-missing"):
            preprocess_dataframe(df, cfg)

    def test_all_missing_encode_column_raises(self):
        df = pd.DataFrame({"cat": [np.nan, np.nan], "v": [1.0, 2.0]})
        cfg = _minimal_cfg(encode={"cat": EncodeSpec(method="one_hot")})
        with pytest.raises(ValueError, match="all-missing"):
            preprocess_dataframe(df, cfg)

    def test_impute_string_column_all_missing(self):
        df = pd.DataFrame({"name": [None, None, None]})
        with pytest.raises(ValueError, match="all-missing"):
            impute_string_column(df, "name", ImputeSpec(method="fill_mode"))


# ---------------------------------------------------------------------------
# Encoding & cardinality
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_one_hot_encoding(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"], "v": [1.0, 2.0, 3.0]})
        cfg = _minimal_cfg(encode={"color": EncodeSpec(method="one_hot")})
        result_df, layout = preprocess_dataframe(df, cfg)
        assert "color_red" in result_df.columns
        assert "color_blue" in result_df.columns
        assert "color" not in result_df.columns

    def test_cardinality_warning(self):
        cats = [f"cat_{i}" for i in range(60)]
        df = pd.DataFrame({"code": cats, "val": range(60)})
        cfg = _minimal_cfg(encode={"code": EncodeSpec(method="one_hot")})
        with pytest.warns(UserWarning, match="60 categories"):
            result_df, _ = preprocess_dataframe(df, cfg)
        assert result_df.shape[1] == 61  # 60 dummies + val

    def test_max_categories_hard_error(self):
        cats = [f"cat_{i}" for i in range(60)]
        df = pd.DataFrame({"code": cats, "val": range(60)})
        cfg = _minimal_cfg(
            encode={"code": EncodeSpec(method="one_hot", max_categories=10)}
        )
        with pytest.raises(ValueError, match="max_categories=10"):
            preprocess_dataframe(df, cfg)

    def test_unsupported_encode_method_raises(self):
        df = pd.DataFrame({"color": ["red", "blue"], "v": [1.0, 2.0]})
        cfg = _minimal_cfg(encode={"color": EncodeSpec(method="label")})
        with pytest.raises(ValueError, match="Unsupported encode method"):
            preprocess_dataframe(df, cfg)


# ---------------------------------------------------------------------------
# Shared vocab (fit_multi pattern)
# ---------------------------------------------------------------------------


class TestSharedVocab:
    def test_vocab_produces_consistent_columns(self):
        df1 = pd.DataFrame({"color": ["red", "blue"], "v": [1.0, 2.0]})
        df2 = pd.DataFrame({"color": ["blue", "green"], "v": [3.0, 4.0]})
        vocab = {"color": ["blue", "green", "red"]}
        cfg = _minimal_cfg(encode={"color": EncodeSpec(method="one_hot")})

        r1, layout1 = preprocess_dataframe(df1, cfg, vocab=vocab)
        r2, layout2 = preprocess_dataframe(df2, cfg, vocab=vocab)

        assert layout1.feature_names == layout2.feature_names
        assert r1.shape[1] == r2.shape[1] == 4  # v + 3 color dummies


# ---------------------------------------------------------------------------
# Layout enforcement
# ---------------------------------------------------------------------------


class TestLayoutEnforcement:
    def test_matching_layout_succeeds(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [5.0, 6.0], "b": [7.0, 8.0]})
        cfg = _minimal_cfg()

        _, ref_layout = preprocess_dataframe(df1, cfg)
        result_df, _ = preprocess_dataframe(df2, cfg, expected_layout=ref_layout)
        assert result_df.shape == (2, 2)

    def test_mismatched_columns_raises(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [5.0, 6.0], "c": [7.0, 8.0]})
        cfg = _minimal_cfg()

        _, ref_layout = preprocess_dataframe(df1, cfg)
        with pytest.raises(ValueError, match="Missing columns"):
            preprocess_dataframe(df2, cfg, expected_layout=ref_layout)

    def test_mismatched_row_count_raises(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        cfg = _minimal_cfg()

        _, ref_layout = preprocess_dataframe(df1, cfg)
        with pytest.raises(ValueError, match="Row count"):
            preprocess_dataframe(df2, cfg, expected_layout=ref_layout)
