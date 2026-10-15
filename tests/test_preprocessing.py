"""Tests for pulsar.preprocessing — preprocess_dataframe() and helpers."""

import numpy as np
import pandas as pd
import pytest

from pulsar.config import (
    EncodeSpec,
    ImputeSpec,
    PulsarConfig,
    PCASpec,
    BallMapperSpec,
    CosmicGraphSpec,
)
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


# ---------------------------------------------------------------------------
# A3 & A4: Config validation (ImputeSpec and missing_indicator_groups)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_impute_spec_fill_value_requires_value(self):
        with pytest.raises(ValueError, match="requires 'value'"):
            ImputeSpec(method="fill_value")

    @pytest.mark.parametrize(
        "bad_val", [True, False, float("nan"), float("inf"), float("-inf")]
    )
    def test_impute_spec_fill_value_rejects_bool_and_non_finite(self, bad_val):
        with pytest.raises(ValueError, match="must be finite"):
            ImputeSpec(method="fill_value", value=bad_val)

    def test_impute_spec_value_rejected_for_other_methods(self):
        with pytest.raises(ValueError, match="only valid with method 'fill_value'"):
            ImputeSpec(method="fill_mean", value=0.0)

    def test_impute_spec_valid_fill_values(self):
        s_num = ImputeSpec(method="fill_value", value=0.0, indicator=False)
        assert s_num.value == 0.0
        assert s_num.indicator is False
        s_int = ImputeSpec(method="fill_value", value=-1)
        assert s_int.value == -1
        assert s_int.indicator is True
        s_str = ImputeSpec(method="fill_value", value="missing")
        assert s_str.value == "missing"

    def test_missing_indicator_groups_duplicate_column_raises(self):
        with pytest.raises(
            ValueError, match="appears in more than one missing_indicator_group"
        ):
            _minimal_cfg(
                missing_indicator_groups={
                    "g1": ["a", "b"],
                    "g2": ["b", "c"],
                }
            )

    def test_missing_indicator_groups_empty_group_or_col_raises(self):
        with pytest.raises(ValueError, match="group names must be non-empty"):
            _minimal_cfg(missing_indicator_groups={"": ["a"]})
        with pytest.raises(ValueError, match="non-empty strings"):
            _minimal_cfg(missing_indicator_groups={"g1": [""]})


# ---------------------------------------------------------------------------
# A5: fill_value, indicator opt-out, and missing_indicator_groups execution
# ---------------------------------------------------------------------------


class TestFillValueAndIndicators:
    def test_numeric_fill_value_known_answer(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan]})
        cfg = _minimal_cfg(
            impute={"x": ImputeSpec(method="fill_value", value=0.0, indicator=True)}
        )
        result_df, _ = preprocess_dataframe(df, cfg)
        assert result_df["x"].tolist() == [1.0, 0.0, 3.0, 0.0]
        assert result_df["x_was_missing"].tolist() == [0.0, 1.0, 0.0, 1.0]

    def test_numeric_fill_value_all_missing_allowed(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})
        cfg = _minimal_cfg(
            impute={"x": ImputeSpec(method="fill_value", value=-1.0, indicator=False)}
        )
        result_df, _ = preprocess_dataframe(df, cfg)
        assert result_df["x"].tolist() == [-1.0, -1.0, -1.0]
        assert "x_was_missing" not in result_df.columns

    def test_numeric_fill_value_rejects_string_value(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        cfg = _minimal_cfg(
            impute={"x": ImputeSpec(method="fill_value", value="missing")}
        )
        with pytest.raises(
            ValueError, match="fill_value for numeric column 'x' must be numeric"
        ):
            preprocess_dataframe(df, cfg)

    def test_string_fill_value_and_one_hot(self):
        df = pd.DataFrame({"cat": ["a", None, "b", np.nan], "v": [1.0, 2.0, 3.0, 4.0]})
        cfg = _minimal_cfg(
            impute={
                "cat": ImputeSpec(method="fill_value", value="missing", indicator=False)
            },
            encode={"cat": EncodeSpec(method="one_hot")},
        )
        result_df, _ = preprocess_dataframe(df, cfg)
        assert "cat_was_missing" not in result_df.columns
        assert "cat_missing" in result_df.columns
        assert result_df["cat_missing"].tolist() == [0.0, 1.0, 0.0, 1.0]
        assert result_df["cat_a"].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert result_df["cat_b"].tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_string_fill_value_all_missing_allowed(self):
        df = pd.DataFrame(
            {"cat": pd.Series([None, None], dtype=object), "v": [1.0, 2.0]}
        )
        cfg = _minimal_cfg(
            impute={
                "cat": ImputeSpec(method="fill_value", value="unknown", indicator=False)
            },
            encode={"cat": EncodeSpec(method="one_hot")},
        )
        result_df, _ = preprocess_dataframe(df, cfg)
        assert "cat_unknown" in result_df.columns
        assert result_df["cat_unknown"].tolist() == [1.0, 1.0]

    def test_indicator_false_suppresses_per_column_indicator(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        cfg = _minimal_cfg(
            impute={"a": ImputeSpec(method="fill_median", indicator=False)}
        )
        result_df, _ = preprocess_dataframe(df, cfg)
        assert "a_was_missing" not in result_df.columns
        assert result_df["a"].tolist() == [1.0, 2.0, 3.0]

    def test_missing_indicator_groups_computed_before_drop_and_suppresses_individual(
        self,
    ):
        df = pd.DataFrame(
            {
                "lab_a": [1.0, np.nan, 3.0, 4.0],
                "lab_b": [10.0, 20.0, np.nan, 40.0],
                "unlisted": [5.0, np.nan, 7.0, 8.0],
            }
        )
        cfg = _minimal_cfg(
            drop_columns=["lab_b"],
            impute={
                "lab_a": ImputeSpec(method="fill_median", indicator=True),
                "unlisted": ImputeSpec(method="fill_median", indicator=True),
            },
            missing_indicator_groups={
                "labs_panel": ["lab_a", "lab_b", "nonexistent_col"],
                "ghost_panel": ["completely_absent_1", "completely_absent_2"],
            },
        )
        result_df, _ = preprocess_dataframe(df, cfg)
        # Group indicator reflects NaNs in both lab_a (row 1) and dropped lab_b (row 2)
        assert "labs_panel_was_missing" in result_df.columns
        assert result_df["labs_panel_was_missing"].tolist() == [0.0, 1.0, 1.0, 0.0]
        # Grouped column does not also get individual indicator
        assert "lab_a_was_missing" not in result_df.columns
        assert "lab_b_was_missing" not in result_df.columns
        # Ungrouped column with indicator=True still gets individual indicator
        assert "unlisted_was_missing" in result_df.columns
        assert result_df["unlisted_was_missing"].tolist() == [0.0, 1.0, 0.0, 0.0]
        # Group with no columns present in input is skipped
        assert "ghost_panel_was_missing" not in result_df.columns

    def test_missing_indicator_groups_emitted_when_constant(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        cfg = _minimal_cfg(missing_indicator_groups={"panel": ["a", "b"]})
        result_df, _ = preprocess_dataframe(df, cfg)
        assert "panel_was_missing" in result_df.columns
        assert result_df["panel_was_missing"].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# A6: Lossless YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_config_yaml_lossless_round_trip(self):
        import yaml
        from pulsar.config import config_to_yaml, load_config

        raw = {
            "run": {"name": "clinical_run", "data": "clinical.parquet"},
            "preprocessing": {
                "drop_columns": ["patient_id", "sparse_lab"],
                "impute": {
                    "Zero_Count": {
                        "method": "fill_value",
                        "value": 0.0,
                        "indicator": False,
                    },
                    "Lab_X": {
                        "method": "fill_median",
                        "seed": 42,
                        "indicator": True,
                    },
                    "Stage": {
                        "method": "fill_value",
                        "value": "missing",
                        "indicator": False,
                    },
                    "Code_Str": {
                        "method": "fill_value",
                        "value": "0",
                        "indicator": False,
                    },
                    "Bool_Str": {
                        "method": "fill_value",
                        "value": "true",
                        "indicator": False,
                    },
                },
                "encode": {
                    "Stage": {"method": "one_hot", "max_categories": 20},
                },
                "missing_indicator_groups": {
                    "bmp_panel": ["Lab_X", "sparse_lab"],
                },
            },
            "sweep": {
                "projection": {
                    "method": "pca",
                    "dimensions": {"values": [2, 4]},
                    "seed": {"values": [42, 99]},
                    "center": False,
                },
                "ball_mapper": {
                    "epsilon": {"values": [0.123456789, 0.5]},
                },
            },
            "cosmic_graph": {
                "construction_threshold": 0.25,
                "neighborhood": "edge",
                "sparsify": True,
                "sparsify_epsilon": 0.8,
                "sparsify_seed": 7,
                "sparsify_sketch_dim": 32,
                "sparsify_sample_count": 128,
                "sparsify_pcg_tol": 1e-5,
                "sparsify_max_iter": 500,
                "construction": "exact",
                "minhash_d": 128,
                "minhash_seed": 11,
            },
            "output": {"n_reps": 6},
        }
        cfg1 = load_config(raw)
        yaml_str = config_to_yaml(cfg1)
        cfg2 = load_config(yaml.safe_load(yaml_str))
        cfg3 = load_config(yaml.safe_load(config_to_yaml(cfg2)))

        for other in (cfg2, cfg3):
            assert cfg1.impute == other.impute
            assert cfg1.missing_indicator_groups == other.missing_indicator_groups
            assert cfg1.encode == other.encode
            assert cfg1.drop_columns == other.drop_columns
            assert cfg1.pca == other.pca
            assert cfg1.projection == other.projection
            assert cfg1.ball_mapper == other.ball_mapper
            assert cfg1.cosmic_graph == other.cosmic_graph
            assert cfg1.n_reps == other.n_reps
            assert cfg1.run_name == other.run_name
            # String values that look like numbers/booleans preserve str type
            assert other.impute["Code_Str"].value == "0"
            assert isinstance(other.impute["Code_Str"].value, str)
            assert other.impute["Bool_Str"].value == "true"
            assert isinstance(other.impute["Bool_Str"].value, str)
