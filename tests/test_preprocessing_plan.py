from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from pulsar.config import load_config
from pulsar.mcp.preprocessing_plan import (
    PlanEditError,
    apply_answers_and_edits,
    build_marker_legend,
    draft_plan,
    dry_run_processed_space,
    plan_fingerprint,
    plan_to_preprocessing_dict,
)
from pulsar.preprocessing import preprocess_dataframe
from tests.fixtures.preprocessing_traps import make_preprocessing_traps


FORBIDDEN_JARGON = ("impute", "mcar", "one-hot", "z-score", "encode", "feature")


def test_draft_plan_no_fabricated_imputation_and_plain_language() -> None:
    df = make_preprocessing_traps(n=400, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=True, stakes_sample_rows=200, seed=0)

    forbidden_methods = {"sample_normal", "sample_categorical", "fill_mean"}
    for col, entry in plan.entries.items():
        assert entry.impute_method not in forbidden_methods, (
            f"Column {col} drafted forbidden impute_method {entry.impute_method}"
        )

    pre_dict = plan_to_preprocessing_dict(plan)
    for col, spec in pre_dict.get("impute", {}).items():
        assert spec["method"] not in forbidden_methods, (
            f"YAML impute for {col} used forbidden method {spec['method']}"
        )

    # Scan every drafted question text and option label/effect for forbidden jargon
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in FORBIDDEN_JARGON) + r")s?\b",
        re.IGNORECASE,
    )
    assert len(plan.questions) >= 5
    for qid, q in plan.questions.items():
        assert not pattern.search(q.text), (
            f"Question {qid} text contains forbidden jargon: {q.text!r}"
        )
        for opt in q.options:
            assert not pattern.search(opt.label), (
                f"Question {qid}/{opt.option_id} label contains jargon: {opt.label!r}"
            )
            assert not pattern.search(opt.effect), (
                f"Question {qid}/{opt.option_id} effect contains jargon: {opt.effect!r}"
            )


def test_trap_frame_defaults_and_questions() -> None:
    df = make_preprocessing_traps(n=400, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=True, stakes_sample_rows=200, seed=0)

    # Identifiers and constants
    assert plan.entries["patient_id"].role == "id"
    assert plan.entries["patient_id"].status == "resolved"
    assert plan.entries["cohort"].role == "drop"
    assert plan.entries["cohort"].status == "resolved"
    assert plan.entries["sex"].role == "drop"
    assert plan.entries["sex"].status == "resolved"
    assert {c.name for c in plan.evidence.columns if c.is_constant} == {"cohort", "sex"}

    # Categorical with missing -> one_hot + fill_value "(blank)" + no indicator
    race_entry = plan.entries["race"]
    assert race_entry.role == "feature"
    assert race_entry.encode == "one_hot"
    assert race_entry.impute_method == "fill_value"
    assert race_entry.impute_value == "(blank)"
    assert race_entry.indicator == "none"

    # Non-integer zero-free continuous column (bmi) must NOT trigger blank_meaning
    bmi_entry = plan.entries["bmi"]
    assert bmi_entry.role == "feature"
    assert bmi_entry.status == "resolved"
    assert bmi_entry.impute_method == "fill_median"
    assert bmi_entry.indicator == "own"
    for q in plan.questions.values():
        if q.topic == "blank_meaning":
            assert "bmi" not in q.applies_to

    # Inpatient group -> one open blank_meaning question
    blank_qs = [q for q in plan.questions.values() if q.topic == "blank_meaning"]
    assert len(blank_qs) == 1
    inp_q = blank_qs[0]
    assert set(inp_q.applies_to) == {
        "n_inpatient_claims",
        "inpatient_payment",
        "inpatient_days",
    }
    assert inp_q.default_option_id == "unknown"
    assert inp_q.stakes is not None
    assert 0.0 <= inp_q.stakes["neighbors_kept"] <= 1.0

    # Answering 'none' gives fill_value 0 and no marker
    plan2 = apply_answers_and_edits(
        plan,
        answers=[{"question_id": inp_q.question_id, "option_id": "none"}],
        edits=[],
        source="user",
    )
    assert plan2.revision == 2
    assert plan2.questions[inp_q.question_id].status == "answered"
    assert plan2.questions[inp_q.question_id].answer is not None
    assert plan2.questions[inp_q.question_id].answer["option_id"] == "none"
    for col in ("n_inpatient_claims", "inpatient_payment", "inpatient_days"):
        e = plan2.entries[col]
        assert e.impute_method == "fill_value"
        assert e.impute_value == 0
        assert e.indicator == "none"
        assert e.decided_by == "user"
        assert e.status == "resolved"
    pre2 = plan_to_preprocessing_dict(plan2)
    assert "missing_indicator_groups" not in pre2

    # Conditional column question on days_to_death
    cond_qs = [q for q in plan.questions.values() if q.topic == "conditional_column"]
    assert len(cond_qs) == 1
    cond_q = cond_qs[0]
    assert cond_q.applies_to == ["days_to_death"]
    assert cond_q.default_option_id == "leave_out"
    assert plan.entries["days_to_death"].role == "drop"
    assert any("died" in opt.effect for opt in cond_q.options)

    # Duplicate signal question on cost_total
    dup_qs = [q for q in plan.questions.values() if q.topic == "duplicate_signal"]
    assert len(dup_qs) == 1
    dup_q = dup_qs[0]
    assert "cost_total" in dup_q.applies_to
    assert all(plan.entries[c].status == "resolved" for c in dup_q.applies_to)

    # Date question on death_date
    date_qs = [q for q in plan.questions.values() if q.topic == "date_columns"]
    assert len(date_qs) == 1
    date_q = date_qs[0]
    assert date_q.applies_to == ["death_date"]
    assert any("died" in opt.effect for opt in date_q.options)

    # Drafted plan loads through load_config and preprocess_dataframe without NaN
    cfg = load_config(
        {
            "run": {"name": "test", "data": "in_memory"},
            "preprocessing": plan_to_preprocessing_dict(plan),
        }
    )
    processed_df, layout = preprocess_dataframe(df, cfg)
    assert not processed_df.isna().any().any()
    assert "n_inpatient_claims_and_2_more_was_missing" in layout.feature_names


def test_has_date_reuses_existing_group_and_avoids_duplicate_group_columns() -> None:
    df = make_preprocessing_traps(n=120, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=False)

    date_q = next(q for q in plan.questions.values() if q.topic == "date_columns")
    cond_q = next(q for q in plan.questions.values() if q.topic == "conditional_column")

    # 1. Answer has_date alone (death_date shares mg1 with days_to_death)
    plan_date = apply_answers_and_edits(
        plan,
        answers=[{"question_id": date_q.question_id, "option_id": "has_date"}],
        edits=[],
        source="agent",
    )
    assert plan_date.entries["death_date"].indicator == "group:mg1"
    cfg_date = load_config(
        {
            "run": {"name": "test", "data": "in_memory"},
            "preprocessing": plan_to_preprocessing_dict(plan_date),
        }
    )
    out_date, layout_date = preprocess_dataframe(df, cfg_date)
    assert not out_date.isna().any().any()
    assert "death_date_was_missing" in layout_date.feature_names

    # 2. Answer BOTH has_date on death_date AND estimate_and_mark on days_to_death
    plan_both = apply_answers_and_edits(
        plan_date,
        answers=[{"question_id": cond_q.question_id, "option_id": "estimate_and_mark"}],
        edits=[],
        source="agent",
    )
    pre_both = plan_to_preprocessing_dict(plan_both)
    # Neither column appears in two groups; days_to_death has indicator: true (own)
    # and death_date is the sole referencing member of its group.
    seen_cols: list[str] = []
    for cols in pre_both.get("missing_indicator_groups", {}).values():
        seen_cols.extend(cols)
    assert len(seen_cols) == len(set(seen_cols))
    cfg_both = load_config(
        {
            "run": {"name": "test", "data": "in_memory"},
            "preprocessing": pre_both,
        }
    )
    out_both, _ = preprocess_dataframe(df, cfg_both)
    assert not out_both.isna().any().any()


def test_panel_shared_missing_rows_emit_single_marker() -> None:
    rng = np.random.default_rng(7)
    n_rows = 120
    data = {f"col_{i}": rng.normal(0.0, 1.0, size=n_rows) for i in range(10)}
    df = pd.DataFrame(data)
    # 30 rows completely empty across all 10 numeric columns
    df.iloc[:30, :] = np.nan

    plan = draft_plan(df, "ds_panel", stakes=False)
    pre_dict = plan_to_preprocessing_dict(plan)
    groups = pre_dict.get("missing_indicator_groups", {})
    assert len(groups) == 1
    assert "col_0_and_9_more" in groups
    assert len(groups["col_0_and_9_more"]) == 10

    legend = build_marker_legend(plan)
    assert len(legend) == 1
    assert legend[0]["marker_column"] == "col_0_and_9_more_was_missing"
    assert legend[0]["columns_count"] == 10

    space, validation = dry_run_processed_space(df, plan)
    assert validation["valid"] is True
    assert validation["error"] is None
    assert space is not None
    assert space["missing_marker_count"] == 1
    assert space["feature_count"] == 11


def test_entity_time_target_excluded_and_edit_validation() -> None:
    df = make_preprocessing_traps(n=400, seed=0)
    plan = draft_plan(
        df,
        "ds_traps",
        target_columns=["died"],
        entity_column="patient_id",
        time_column="age",
        stakes=False,
    )
    assert plan.entries["patient_id"].role == "entity"
    assert plan.entries["age"].role == "time"
    assert plan.entries["died"].role == "target"
    assert all(
        plan.entries[c].decided_by == "agent" for c in ("patient_id", "age", "died")
    )
    # Outcome question should NOT be opened when target_columns is provided
    assert not any(q.topic == "outcome_columns" for q in plan.questions.values())

    pre = plan_to_preprocessing_dict(plan)
    for col in ("patient_id", "age", "died"):
        assert col in pre["drop_columns"]

    # Stakes budget check
    plan_budget = draft_plan(
        df, "ds_traps", stakes=True, stakes_max_questions=2, stakes_sample_rows=80
    )
    assert plan_budget.cost["stakes_questions_computed"] == 2
    assert plan_budget.cost["stakes_questions_skipped"] >= 1

    # Invalid arguments raise PlanEditError
    with pytest.raises(PlanEditError) as exc_info:
        draft_plan(df, "ds_traps", target_columns=["nonexistent"], stakes=False)
    assert exc_info.value.code == "INVALID_ARGUMENT"

    with pytest.raises(PlanEditError) as exc_info:
        apply_answers_and_edits(
            plan,
            answers=[{"question_id": "q999", "option_id": "none"}],
            edits=[],
            source="agent",
        )
    assert exc_info.value.code == "INVALID_ARGUMENT"

    first_qid = next(iter(plan.questions))
    with pytest.raises(PlanEditError) as exc_info:
        apply_answers_and_edits(
            plan,
            answers=[{"question_id": first_qid, "option_id": "bad_option"}],
            edits=[],
            source="agent",
        )
    assert exc_info.value.code == "INVALID_ARGUMENT"

    with pytest.raises(PlanEditError) as exc_info:
        apply_answers_and_edits(
            plan,
            answers=[],
            edits=[{"column": "no_such_col", "role": "drop"}],
            source="agent",
        )
    assert exc_info.value.code == "INVALID_ARGUMENT"

    with pytest.raises(PlanEditError) as exc_info:
        apply_answers_and_edits(
            plan,
            answers=[],
            edits=[{"column": "bmi", "role": "invalid_role"}],
            source="agent",
        )
    assert exc_info.value.code == "INVALID_ARGUMENT"

    fp1 = plan_fingerprint(plan)
    assert len(fp1) == 12


def test_dry_run_and_stakes_failure_resilience() -> None:
    df = make_preprocessing_traps(n=100, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=False)

    # Force an invalid edit: keep a text column (patient_id) as feature without encoding
    bad_plan = apply_answers_and_edits(
        plan,
        answers=[],
        edits=[{"column": "patient_id", "role": "feature", "encode": None}],
        source="agent",
    )
    space, validation = dry_run_processed_space(df, bad_plan)
    assert space is None
    assert validation["valid"] is False
    assert validation["error"]

    # Stakes computation when baseline fails must not crash; returns None stakes + note
    from pulsar.mcp.preprocessing_plan import compute_stakes

    stakes_res = compute_stakes(df, bad_plan)
    assert all(v is None for v in stakes_res.values())
    assert bad_plan.cost.get("stakes_notes")
