from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pulsar.config import (
    ImputeSpec,
    PulsarConfig,
    config_to_yaml,
    expand_preprocessing_variants,
    load_config,
)
from pulsar.mcp.config_tools import apply_overrides, validate_config_yaml
from pulsar.mcp.preprocessing_plan import (
    PlanQuestion,
    QuestionOption,
    apply_answers_and_edits,
    diff_plan_entries,
    draft_plan,
    plan_to_preprocessing_dict,
)
from pulsar.pipeline import ThemaRS
from pulsar.runtime.fingerprint import projection_fingerprint


@pytest.fixture
def sample_df_with_missing() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(5.0, 2.0, size=n)
    x3 = rng.normal(-2.0, 0.5, size=n)
    cat = rng.choice(["A", "B", "C"], size=n).astype(object)

    # Inject ~20% missingness
    x1[rng.choice(n, size=12, replace=False)] = np.nan
    x2[rng.choice(n, size=10, replace=False)] = np.nan
    cat[rng.choice(n, size=8, replace=False)] = None

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat})


def test_load_config_multi_impute_candidates_and_parameter_sweep():
    raw = {
        "run": {"name": "multi_imp", "data": "data.csv"},
        "preprocessing": {
            "n_imputations": 3,
            "impute": {
                "x1": [
                    {"method": "fill_median", "indicator": True},
                    {"method": "sample_normal", "seed": 42, "indicator": True},
                ],
                "x2": {
                    "method": ["fill_mean", "fill_median"],
                    "indicator": [False, True],
                },
            },
            "encode": {"cat": {"method": "one_hot"}},
        },
        "sweep": {
            "projection": {
                "method": "pca",
                "dimensions": {"values": [2]},
                "seed": {"values": [0]},
            },
            "ball_mapper": {"epsilon": {"values": [0.5, 1.0]}},
        },
    }

    cfg = load_config(raw)
    assert cfg.n_imputations == 3
    assert cfg.impute["x1"] == ImputeSpec(method="fill_median", indicator=True)
    assert len(cfg.impute_candidates["x1"]) == 2
    assert cfg.impute_candidates["x1"][1] == ImputeSpec(
        method="sample_normal", seed=42, indicator=True
    )
    # Cartesian product of 2 methods x 2 indicator flags = 4 candidates for x2
    assert len(cfg.impute_candidates["x2"]) == 4

    variants = expand_preprocessing_variants(cfg)
    # max_candidates is 4, n_imputations is 3 -> total_draws = 4
    assert len(variants) == 4
    assert cfg.preprocessing_variants() == variants


def test_expand_preprocessing_variants_stochastic_seed_increment():
    cfg = PulsarConfig(
        data="dummy.csv",
        encode={},
        drop_columns=[],
        n_imputations=4,
        impute={
            "x1": ImputeSpec(method="sample_normal", seed=10, indicator=True),
            "cat": ImputeSpec(method="sample_categorical", seed=100),
        },
    )
    variants = expand_preprocessing_variants(cfg)
    assert len(variants) == 4
    seeds_x1 = [v.impute["x1"].seed for v in variants]
    seeds_cat = [v.impute["cat"].seed for v in variants]
    assert seeds_x1 == [10, 11, 12, 13]
    assert seeds_cat == [100, 101, 102, 103]


def test_expand_preprocessing_variants_deduplicates_deterministic():
    cfg = PulsarConfig(
        data="dummy.csv",
        encode={},
        drop_columns=[],
        n_imputations=5,
        impute={
            "x1": ImputeSpec(method="fill_median", indicator=True),
            "x2": ImputeSpec(method="fill_mean", indicator=False),
        },
    )
    # Deterministic specs with only 1 candidate each should not produce 5 duplicate variants
    variants = expand_preprocessing_variants(cfg)
    assert len(variants) == 1


def test_config_yaml_round_trip_preserves_multi_imputation():
    import yaml

    raw = {
        "run": {"name": "rt_test", "data": "dummy.csv"},
        "preprocessing": {
            "n_imputations": 3,
            "impute": {
                "x1": [
                    {"method": "fill_median", "indicator": True},
                    {"method": "sample_normal", "seed": 7, "indicator": False},
                ],
                "x2": {"method": "fill_mean", "indicator": False},
            },
        },
        "sweep": {
            "projection": {
                "method": "pca",
                "dimensions": {"values": [2]},
                "seed": {"values": [42]},
            },
            "ball_mapper": {"epsilon": {"values": [0.5]}},
        },
    }
    cfg1 = load_config(raw)
    yaml_str = config_to_yaml(cfg1)
    assert "n_imputations: 3" in yaml_str
    cfg2 = load_config(yaml.safe_load(yaml_str))
    assert cfg2.n_imputations == 3
    assert cfg2.impute_candidates == cfg1.impute_candidates
    assert expand_preprocessing_variants(cfg2) == expand_preprocessing_variants(cfg1)


def test_config_group_collision_checks_all_candidates():
    raw = {
        "run": {"name": "collision_test", "data": "dummy.csv"},
        "preprocessing": {
            "impute": {
                "grp1": [
                    {"method": "fill_median", "indicator": False},
                    {"method": "sample_normal", "seed": 42, "indicator": True},
                ]
            },
            "missing_indicator_groups": {"grp1": ["x1", "x2"]},
        },
    }
    with pytest.raises(ValueError, match="missing indicator name collision"):
        load_config(raw)


@pytest.mark.parametrize("construction_mode", ["exact", "minhash"])
def test_pipeline_fit_multi_imputation_ensemble(
    sample_df_with_missing: pd.DataFrame, construction_mode: str
):
    cfg = load_config(
        {
            "run": {"name": "ensemble_fit", "data": "in_memory.csv"},
            "preprocessing": {
                "n_imputations": 3,
                "impute": {
                    "x1": [
                        {"method": "fill_median", "indicator": True},
                        {"method": "sample_normal", "seed": 42, "indicator": True},
                    ],
                    "x2": [
                        {"method": "fill_mean"},
                        {"method": "sample_normal", "seed": 99},
                    ],
                    "cat": [
                        {"method": "fill_mode"},
                        {"method": "sample_categorical", "seed": 11},
                    ],
                },
                "encode": {"cat": {"method": "one_hot"}},
            },
            "sweep": {
                "projection": {
                    "method": "pca",
                    "dimensions": {"values": [2, 3]},
                    "seed": {"values": [0]},
                },
                "ball_mapper": {"epsilon": {"values": [0.6, 1.0]}},
            },
            "cosmic_graph": {
                "construction_threshold": 0.1,
                "construction": construction_mode,
                "minhash_d": 64,
            },
        }
    )

    variants = expand_preprocessing_variants(cfg)
    assert len(variants) == 3

    model = ThemaRS(cfg)
    model.fit(sample_df_with_missing)

    # 3 imputation variants * 2 dimensions * 1 seed = 6 embeddings
    assert len(model._embeddings) == 3 * 2 * 1
    # 6 embeddings * 2 epsilons = 12 BallMapper graphs
    assert len(model._ball_maps) == 3 * 2 * 1 * 2
    assert model.cosmic_graph.number_of_nodes() == len(sample_df_with_missing)


def test_projection_fingerprint_distinguishes_multi_imputation(
    sample_df_with_missing: pd.DataFrame,
):
    base_cfg = load_config(
        {
            "run": {"name": "fp_test", "data": "data.csv"},
            "preprocessing": {
                "impute": {"x1": {"method": "sample_normal", "seed": 42}},
            },
        }
    )
    multi_n_cfg = load_config(
        {
            "run": {"name": "fp_test", "data": "data.csv"},
            "preprocessing": {
                "n_imputations": 3,
                "impute": {"x1": {"method": "sample_normal", "seed": 42}},
            },
        }
    )
    multi_cand_cfg = load_config(
        {
            "run": {"name": "fp_test", "data": "data.csv"},
            "preprocessing": {
                "impute": {
                    "x1": [
                        {"method": "sample_normal", "seed": 42},
                        {"method": "fill_median"},
                    ]
                },
            },
        }
    )

    n_rows = len(sample_df_with_missing)
    fp_base = projection_fingerprint(base_cfg, n_rows, sample_df_with_missing)
    fp_multi_n = projection_fingerprint(multi_n_cfg, n_rows, sample_df_with_missing)
    fp_multi_cand = projection_fingerprint(
        multi_cand_cfg, n_rows, sample_df_with_missing
    )

    assert fp_base != fp_multi_n
    assert fp_base != fp_multi_cand
    assert fp_multi_n != fp_multi_cand


def test_mcp_validate_and_override_multi_imputation(tmp_path):
    import yaml

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x1,x2\n1.0,2.0\n,3.0\n4.0,\n")

    yaml_str = f"""
run:
  name: mcp_test
  data: "{csv_path}"
preprocessing:
  n_imputations: 2
  impute:
    x1:
      - method: fill_median
        indicator: true
      - method: sample_normal
        seed: 42
        indicator: true
sweep:
  projection:
    method: pca
    dimensions:
      values: [2]
    seed:
      values: [42]
  ball_mapper:
    epsilon:
      values: [0.5]
"""
    report = validate_config_yaml(yaml_str)
    assert report.ok, f"Validation failed: {report.issues}"
    assert "n_imputations: 2" in (report.normalized_yaml or "")

    override_res = apply_overrides(
        yaml_str,
        {
            "n_imputations": 4,
            "preprocessing.impute.x2": [
                {"method": "fill_mean"},
                {"method": "fill_median"},
            ],
        },
    )
    cfg = load_config(yaml.safe_load(override_res.config_yaml))
    assert cfg.n_imputations == 4
    assert len(cfg.impute_candidates["x2"]) == 2


def test_preprocessing_plan_multi_imputation_edits_and_render(
    sample_df_with_missing: pd.DataFrame,
):
    plan = draft_plan(sample_df_with_missing, "ds_1", stakes=False, seed=0)
    plan.questions["q_x1"] = PlanQuestion(
        question_id="q_x1",
        topic="blank_meaning",
        text="How should x1 blanks be filled?",
        applies_to=["x1"],
        rows_affected=12,
        options=[
            QuestionOption(
                option_id="med",
                label="Median",
                effect="Fill with median",
                edits=[
                    {
                        "column": "x1",
                        "impute": {"method": "fill_median"},
                        "indicator": "own",
                    }
                ],
            ),
            QuestionOption(
                option_id="norm",
                label="Sample normal",
                effect="Draw from normal distribution",
                edits=[
                    {
                        "column": "x1",
                        "impute": {"method": "sample_normal", "seed": 42},
                        "indicator": "own",
                    }
                ],
            ),
        ],
        default_option_id="med",
    )

    updated = apply_answers_and_edits(
        plan,
        answers=[{"question_id": "q_x1", "option_ids": ["med", "norm"]}],
        edits=[
            {
                "column": "x2",
                "impute_method": ["fill_mean", "fill_median"],
                "indicator": "own",
            }
        ],
    )
    updated.n_imputations = 3

    e_x1 = updated.entries["x1"]
    e_x2 = updated.entries["x2"]
    assert e_x1.impute_candidates is not None and len(e_x1.impute_candidates) == 2
    assert e_x2.impute_candidates is not None and len(e_x2.impute_candidates) == 2

    diffs = diff_plan_entries(plan, updated)
    assert any(d["column"] == "x1" and d["field"] == "impute_candidates" for d in diffs)
    assert any(d["column"] == "x2" and d["field"] == "impute_candidates" for d in diffs)

    prep_dict = plan_to_preprocessing_dict(updated)
    assert prep_dict["n_imputations"] == 3
    assert isinstance(prep_dict["impute"]["x1"], list)
    assert len(prep_dict["impute"]["x1"]) == 2
    assert isinstance(prep_dict["impute"]["x2"], list)
    assert len(prep_dict["impute"]["x2"]) == 2

    # Ensure load_config accepts the rendered preprocessing dict directly
    cfg = load_config(
        {
            "run": {"name": "from_plan", "data": "dummy.csv"},
            "preprocessing": prep_dict,
        }
    )
    assert len(expand_preprocessing_variants(cfg)) == 3
