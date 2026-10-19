from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from pulsar.config import (
    ImputeSpec,
    PulsarConfig,
    config_to_yaml,
    expand_preprocessing_variants,
    load_config,
)
from pulsar.mcp.config_tools import apply_overrides, validate_config_yaml
from pulsar.mcp.preprocessing_plan import (
    PlanEditError,
    PlanQuestion,
    QuestionOption,
    apply_answers_and_edits,
    diff_plan_entries,
    draft_plan,
    plan_to_preprocessing_dict,
)
from pulsar.mcp.preprocessing_plan.render import build_marker_legend
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


@pytest.mark.parametrize("draws", [1, 3])
def test_explicit_singleton_candidate_is_authoritative_and_cache_sensitive(draws):
    def make_config(seed):
        return PulsarConfig(
            data="in_memory.csv",
            drop_columns=[],
            encode={},
            impute={"x": ImputeSpec("fill_mean")},
            impute_candidates={"x": [ImputeSpec("sample_normal", seed=seed)]},
            n_imputations=draws,
        )

    cfg = make_config(7)
    assert cfg.impute["x"] == ImputeSpec("sample_normal", seed=7)
    variants = expand_preprocessing_variants(cfg)
    assert [v.impute["x"].seed for v in variants] == list(range(7, 7 + draws))
    restored = load_config(yaml.safe_load(config_to_yaml(cfg)))
    assert restored.preprocessing_variants() == variants
    assert projection_fingerprint(cfg, 10) == projection_fingerprint(restored, 10)
    assert projection_fingerprint(cfg, 10) != projection_fingerprint(
        make_config(99), 10
    )


@pytest.mark.parametrize(
    "spec",
    [
        {"method": {"values": ["fill_mean", "fill_median"]}},
        {"method": ["fill_median", "fill_value"], "value": 0},
        {
            "method": "sample_normal",
            "seed": {"range": {"min": 7, "max": 8, "steps": 2}},
            "indicator": {"values": [False, True]},
        },
    ],
)
def test_mcp_and_loader_accept_the_same_imputation_sweeps(tmp_path, spec):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n1\n2\n3\n")
    raw = {"run": {"data": str(csv_path)}, "preprocessing": {"impute": {"x": spec}}}
    expected = load_config(raw)
    report = validate_config_yaml(yaml.safe_dump(raw))
    assert report.ok, report.issues
    restored = load_config(yaml.safe_load(report.normalized_yaml))
    assert restored.preprocessing_variants() == expected.preprocessing_variants()


@pytest.mark.parametrize(
    "spec",
    [
        {"method": {"values": []}},
        {"method": {"values": ["unsupported"]}},
        {"method": {"unexpected": ["fill_mean"]}},
        {"method": "fill_mean", "value": 0},
        {"method": "sample_normal", "seed": {"range": {}}},
        {"method": "fill_mean", "indicator": [False, "bad"]},
    ],
)
def test_mcp_rejects_invalid_imputation_sweeps_with_structured_issues(tmp_path, spec):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n1\n2\n3\n")
    raw = {"run": {"data": str(csv_path)}, "preprocessing": {"impute": {"x": spec}}}
    report = validate_config_yaml(yaml.safe_dump(raw))
    assert not report.ok
    assert report.agent_action
    assert any(i.path.startswith("preprocessing.impute.x") for i in report.issues)


@pytest.fixture
def ensemble_plan(sample_df_with_missing):
    plan = draft_plan(sample_df_with_missing, "ds_ensemble", stakes=False)
    plan.questions["ensemble"] = PlanQuestion(
        question_id="ensemble",
        topic="blank_meaning",
        text="Which treatments should be explored?",
        applies_to=["x1"],
        rows_affected=12,
        options=[
            QuestionOption(
                option_id="zero",
                label="Zero",
                effect="Zero without marker",
                edits=[
                    {
                        "column": "x1",
                        "impute_method": "fill_value",
                        "impute_value": 0,
                        "indicator": "none",
                    }
                ],
            ),
            QuestionOption(
                option_id="median",
                label="Median",
                effect="Median with marker",
                edits=[
                    {
                        "column": "x1",
                        "impute_method": "fill_median",
                        "indicator": "own",
                    }
                ],
            ),
        ],
        default_option_id="median",
    )
    return plan


@pytest.mark.parametrize("option_ids", [["zero", "median"], ["median", "zero"]])
def test_multi_option_answers_preserve_treatments_in_either_order(
    ensemble_plan, option_ids
):
    updated = apply_answers_and_edits(
        ensemble_plan, answers=[{"question_id": "ensemble", "option_ids": option_ids}]
    )
    specs = plan_to_preprocessing_dict(updated)["impute"]["x1"]
    assert {s["method"]: s.get("indicator", True) for s in specs} == {
        "fill_value": False,
        "fill_median": True,
    }
    assert next(s for s in specs if s["method"] == "fill_value")["value"] == 0
    assert any(
        m["marker_column"] == "x1_was_missing" for m in build_marker_legend(updated)
    )
    assert ensemble_plan.entries["x1"].impute_candidates is None


@pytest.mark.parametrize(
    "methods", [["fill_median", "fill_value"], ["fill_value", "fill_median"]]
)
def test_mixed_method_edits_apply_value_only_to_constant_fill(ensemble_plan, methods):
    updated = apply_answers_and_edits(
        ensemble_plan,
        edits=[{"column": "x1", "impute_method": methods, "impute_value": 0}],
    )
    specs = plan_to_preprocessing_dict(updated)["impute"]["x1"]
    assert next(s for s in specs if s["method"] == "fill_value")["value"] == 0
    assert "value" not in next(s for s in specs if s["method"] == "fill_median")


def test_value_edit_without_constant_fill_still_fails(ensemble_plan):
    with pytest.raises(
        PlanEditError, match="only valid with impute_method='fill_value'"
    ):
        apply_answers_and_edits(
            ensemble_plan,
            edits=[
                {
                    "column": "x1",
                    "impute_method": ["fill_mean", "fill_median"],
                    "impute_value": 0,
                }
            ],
        )


def test_explicit_candidate_indicators_and_entry_override(ensemble_plan):
    updated = apply_answers_and_edits(
        ensemble_plan,
        edits=[
            {
                "column": "x1",
                "impute": [
                    {"method": "fill_value", "value": 0, "indicator": False},
                    {"method": "fill_median", "indicator": True},
                ],
            }
        ],
    )
    assert [
        s["indicator"] for s in plan_to_preprocessing_dict(updated)["impute"]["x1"]
    ] == [False, True]
    overridden = apply_answers_and_edits(
        updated, edits=[{"column": "x1", "indicator": "none"}]
    )
    assert [
        s["indicator"] for s in plan_to_preprocessing_dict(overridden)["impute"]["x1"]
    ] == [False, False]


@pytest.mark.parametrize("conflict", ["role", "encode", "indicator"])
def test_multi_option_answers_reject_incompatible_shared_policies(
    ensemble_plan, conflict
):
    first_edit = ensemble_plan.questions["ensemble"].options[0].edits[0]
    first_edit[conflict] = {
        "role": "drop",
        "encode": "one_hot",
        "indicator": "group:x1",
    }[conflict]
    with pytest.raises(
        PlanEditError, match="incompatible roles, encoding, or missingness groups"
    ):
        apply_answers_and_edits(
            ensemble_plan,
            answers=[
                {
                    "question_id": "ensemble",
                    "option_ids": ["zero", "median"],
                }
            ],
        )


def test_mcp_multi_imputation_real_data_penguins_end_to_end():
    import asyncio
    from pathlib import Path

    import yaml
    from pulsar.mcp.server import mcp

    penguins_csv = Path(__file__).resolve().parents[1] / "demos" / "penguins" / "penguins.csv"

    async def _call(tool_name: str, **kwargs):
        res = await mcp.call_tool(tool_name, kwargs)
        sc = res.structured_content
        assert sc is not None and sc.get("status") != "error", f"{tool_name} failed: {sc}"
        return sc

    async def _workflow():
        ing = await _call("ingest_dataset", path=str(penguins_csv))
        ds_id = ing["dataset_id"]

        plan = await _call(
            "draft_preprocessing_plan",
            dataset_id=ds_id,
            target_columns=["species"],
        )
        plan_id = plan["plan_id"]

        cfg_base = await _call("create_config", dataset_id=ds_id, plan_id=plan_id)
        assert cfg_base["sweep_strategy"]["imputation_variants"] == 1
        sweep_base = await _call(
            "run_topological_sweep",
            config_yaml=cfg_base["config_yaml"],
            dataset_id=ds_id,
        )

        edited = await _call(
            "edit_preprocessing_plan",
            plan_id=plan_id,
            answers=[{"question_id": "q1", "option_ids": ["unknown", "none"]}],
            edits=[
                {
                    "column": "bill_length_mm",
                    "impute": [
                        {"method": "fill_median"},
                        {"method": "fill_mean"},
                        {"method": "sample_normal", "seed": 42},
                    ],
                },
                {
                    "column": "sex",
                    "impute": [
                        {"method": "fill_value", "value": "(blank)"},
                        {"method": "sample_categorical", "seed": 17},
                    ],
                },
            ],
        )
        assert edited["validation"]["valid"] is True
        assert any(ch["field"] == "impute_candidates" for ch in edited["changes"])

        cfg_ens = await _call("create_config", dataset_id=ds_id, plan_id=plan_id)
        assert cfg_ens["sweep_strategy"]["imputation_variants"] == 3
        parsed = yaml.safe_load(cfg_ens["config_yaml"])
        assert len(parsed["preprocessing"]["impute"]["bill_length_mm"]) == 3
        assert len(parsed["preprocessing"]["impute"]["flipper_length_mm"]) == 2

        val_cfg = await _call("validate_config", config_yaml=cfg_ens["config_yaml"])
        assert val_cfg["status"] == "ok" and val_cfg["issues"] == []

        val_prep = await _call(
            "validate_preprocessing_config",
            config_yaml=cfg_ens["config_yaml"],
            dataset_id=ds_id,
        )
        assert val_prep["valid"] is True and val_prep["nan_remaining"] == 0

        ref_cfg = await _call(
            "refine_config",
            config_yaml=cfg_ens["config_yaml"],
            overrides={"preprocessing": {"n_imputations": 4}},
        )
        assert (
            yaml.safe_load(ref_cfg["config_yaml"])["preprocessing"]["n_imputations"]
            == 4
        )

        sweep_ens = await _call(
            "run_topological_sweep",
            config_yaml=cfg_ens["config_yaml"],
            dataset_id=ds_id,
        )
        ens_run_id = sweep_ens["run_id"]

        diag = await _call("diagnose_cosmic_graph", run_id=ens_run_id)
        assert diag["status"] == "ok"

        dossier = await _call("generate_cluster_dossier", run_id=ens_run_id)
        assert len(dossier.get("clusters", [])) > 0

        comp = await _call(
            "compare_sweeps",
            run_a=sweep_base["run_id"],
            run_b=ens_run_id,
        )
        assert comp["status"] == "ok"

    asyncio.run(_workflow())

