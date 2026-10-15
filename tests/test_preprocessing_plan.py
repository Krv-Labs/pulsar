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
FORBIDDEN_SCHEMA_LEAKS = ("role=", "impute_method", "impute_value", "in edits")


def test_draft_plan_no_fabricated_imputation_and_plain_language() -> None:
    from pulsar.mcp.preprocessing import build_draft_plan_payload

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
        for leak in FORBIDDEN_SCHEMA_LEAKS:
            assert leak not in q.text, (
                f"Question {qid} text leaks internal field syntax {leak!r}: {q.text!r}"
            )
        for opt in q.options:
            assert not pattern.search(opt.label), (
                f"Question {qid}/{opt.option_id} label contains jargon: {opt.label!r}"
            )
            assert not pattern.search(opt.effect), (
                f"Question {qid}/{opt.option_id} effect contains jargon: {opt.effect!r}"
            )
            for leak in FORBIDDEN_SCHEMA_LEAKS:
                assert leak not in opt.label and leak not in opt.effect, (
                    f"Question {qid}/{opt.option_id} leaks {leak!r}: {opt.effect!r}"
                )

    payload = build_draft_plan_payload(plan, processed_space=None)
    assert "role" in payload["how_to_use"] and "target" in payload["how_to_use"]
    assert "target_columns" in payload["how_to_use"]


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


def test_cost_evidence_ms_and_stakes_false_skipped_count() -> None:
    df = make_preprocessing_traps(n=120, seed=0)
    plan_with_stakes = draft_plan(df, "ds_traps", stakes=True, stakes_sample_rows=80)
    assert plan_with_stakes.cost["evidence_ms"] > 0.0
    assert plan_with_stakes.cost["stakes_questions_computed"] == 5
    assert plan_with_stakes.cost["stakes_questions_skipped"] == 0

    plan_no_stakes = draft_plan(df, "ds_traps", stakes=False)
    assert plan_no_stakes.cost["evidence_ms"] > 0.0
    assert plan_no_stakes.cost["stakes_ms"] == 0.0
    assert plan_no_stakes.cost["stakes_rows_sampled"] == 0
    assert plan_no_stakes.cost["stakes_questions_computed"] == 0
    # 5 non-outcome_columns questions are skipped when stakes=False
    assert plan_no_stakes.cost["stakes_questions_skipped"] == 5


def test_outcome_columns_always_last_stakes_none_and_weak_hints_cap() -> None:
    from pulsar.mcp.preprocessing import build_draft_plan_payload

    df = make_preprocessing_traps(n=120, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=True, stakes_max_questions=5, seed=0)
    q_list = list(plan.questions.values())
    assert q_list[-1].topic == "outcome_columns"
    assert q_list[-1].question_id == f"q{len(q_list)}"
    assert q_list[-1].stakes is None
    assert q_list[-1].options[0].effect == "Keeps all current columns as map inputs"
    # None of the 5 real questions were bumped by outcome_columns
    assert all(q.stakes is not None for q in q_list[:-1])

    # Add a second boolean event partner to verify weak_hints capping
    df2 = df.copy()
    df2["surgery_flag"] = (df2["age"] > 70).astype(int)
    df2["days_to_surgery"] = np.where(df2["surgery_flag"] == 1, 15.0, np.nan)
    plan2 = draft_plan(df2, "ds_traps2", stakes=False)
    payload_capped = build_draft_plan_payload(
        plan2, processed_space=None, detail="summary", evidence_limit=1
    )
    assert len(payload_capped["weak_hints"]["event_flag_columns"]) == 1
    assert payload_capped["weak_hints"]["event_flag_columns_omitted"] == 1
    assert (
        payload_capped["weak_hints"]["description"]
        == "complete 0/1 or count columns whose value exactly explains other columns' blanks (often events or exposures; only you can say whether any is an outcome)"
    )
    assert "preprocessing_yaml" not in payload_capped
    assert payload_capped["preprocessing_yaml_chars"] > 0
    payload_full = build_draft_plan_payload(plan2, processed_space=None, detail="full")
    assert len(payload_full["weak_hints"]["event_flag_columns"]) == 2
    assert payload_full["weak_hints"]["event_flag_columns_omitted"] == 0
    assert "preprocessing_yaml" in payload_full
    assert payload_full["preprocessing_yaml_chars"] == len(
        payload_full["preprocessing_yaml"]
    )


def test_object_boolean_columns_profiled_as_boolean_and_preprocessed() -> None:
    from pulsar.analysis.preprocessing_evidence import measure_preprocessing_evidence

    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "obj_bool_complete": pd.Series(
                [True, False, True, False, True, False], dtype=object
            ),
            "obj_bool_missing": pd.Series(
                [True, False, None, True, False, None], dtype=object
            ),
        }
    )
    ev = measure_preprocessing_evidence(df)
    ev_by_col = {c.name: c for c in ev.columns}
    assert ev_by_col["obj_bool_complete"].kind == "boolean"
    assert ev_by_col["obj_bool_missing"].kind == "boolean"

    plan = draft_plan(df, "ds_bool", stakes=False)
    assert plan.entries["obj_bool_complete"].encode is None
    assert plan.entries["obj_bool_complete"].impute is None
    assert plan.entries["obj_bool_missing"].encode is None
    assert plan.entries["obj_bool_missing"].impute_method == "fill_median"
    assert plan.entries["obj_bool_missing"].indicator == "own"

    space, val = dry_run_processed_space(df, plan)
    assert val["valid"] is True
    assert space is not None
    assert space["dummy_column_count"] == 0
    assert space["missing_marker_count"] == 1
    assert space["feature_count"] == 4


def test_duplicate_signal_bundling_and_blank_meaning_group_with_partner() -> None:
    from dataclasses import replace

    from pulsar.analysis.preprocessing_evidence import (
        ColumnRelation,
        measure_preprocessing_evidence,
    )

    rng = np.random.default_rng(42)
    n = 80
    a1 = rng.integers(1, 10, size=n).astype(float)
    a2 = rng.integers(1, 10, size=n).astype(float)
    b1 = rng.integers(1, 10, size=n).astype(float)
    b2 = rng.integers(1, 10, size=n).astype(float)
    # Partner with >2 unique values (not boolean) that is 0 when claim group is blank
    visits = rng.integers(1, 5, size=n)
    visits[:25] = 0
    claim_cnt = rng.integers(1, 8, size=n).astype(float)
    claim_pay = rng.uniform(10.0, 200.0, size=n)
    claim_pay[30:35] = 0.0
    claim_cnt[:25] = np.nan
    claim_pay[:25] = np.nan

    df = pd.DataFrame(
        {
            "a1": a1,
            "a2": a2,
            "a_sum": a1 + a2,
            "b1": b1,
            "b2": b2,
            "b_sum": b1 + b2,
            "dup_orig": a1.copy(),
            "dup_copy1": a1.copy(),
            "dup_copy2": a1.copy(),
            "visits": visits,
            "claim_cnt": claim_cnt,
            "claim_pay": claim_pay,
        }
    )
    ev = measure_preprocessing_evidence(df)
    # Inject an overlapping sum_of relation where a_sum (already a total) is a source
    ev_with_overlap = replace(
        ev,
        relations=(
            *ev.relations,
            ColumnRelation(
                kind="sum_of",
                target="b_sum",
                sources=("a_sum", "b1"),
                value=1.0,
                rows_used=n,
            ),
        ),
    )
    plan = draft_plan(df, "ds_bundle", evidence=ev_with_overlap, stakes=False)
    dup_qs = [q for q in plan.questions.values() if q.topic == "duplicate_signal"]
    # Exactly 1 sum_of question and 1 duplicate question
    assert len(dup_qs) == 2
    sum_q = next(q for q in dup_qs if q.default_option_id == "keep_all")
    assert [o.option_id for o in sum_q.options] == [
        "keep_all",
        "keep_totals",
        "keep_parts",
    ]
    assert "2 totals in this table" in sum_q.text
    keep_totals_opt = next(o for o in sum_q.options if o.option_id == "keep_totals")
    dropped_by_keep_totals = {
        e["column"] for e in keep_totals_opt.edits if e["role"] == "drop"
    }
    kept_by_keep_totals = {
        e["column"] for e in keep_totals_opt.edits if e["role"] == "feature"
    }
    assert dropped_by_keep_totals == {"a1", "a2", "b1", "b2"}
    assert kept_by_keep_totals == {"a_sum", "b_sum"}
    assert "a_sum" not in dropped_by_keep_totals

    dup_q = next(q for q in dup_qs if q.default_option_id == "keep_both")
    assert [o.option_id for o in dup_q.options] == ["keep_both", "keep_one"]
    # Transitive duplicates: answering keep_one keeps a1 and drops dup_orig, dup_copy1, dup_copy2
    plan_dedup = apply_answers_and_edits(
        plan,
        answers=[{"question_id": dup_q.question_id, "option_id": "keep_one"}],
        edits=[],
        source="agent",
    )
    assert plan_dedup.entries["a1"].role == "feature"
    for dcol in ("dup_orig", "dup_copy1", "dup_copy2"):
        assert plan_dedup.entries[dcol].role == "drop"

    # Multi-column group with mask_dependency stays a single blank_meaning question
    blank_qs = [q for q in plan.questions.values() if q.topic == "blank_meaning"]
    assert len(blank_qs) == 1
    assert set(blank_qs[0].applies_to) == {"claim_cnt", "claim_pay"}
    assert (
        "`claim_cnt` never records a 0 (1 other column sometimes does)."
        in blank_qs[0].text
    )
    assert "blank exactly where `visits` is 0" in blank_qs[0].text
    none_opt = next(o for o in blank_qs[0].options if o.option_id == "none")
    assert "Both columns get 0 where blank" in none_opt.effect
    assert not any(q.topic == "conditional_column" for q in plan.questions.values())


def test_date_columns_has_date_skips_complete_dates() -> None:
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "num2": [2.0, 4.0, 1.0, 3.0, 5.0, 7.0, 6.0, 8.0, 10.0, 9.0, 12.0, 11.0],
            "birth_date": [f"1950-01-{i:02d}" for i in range(1, 13)],
            "death_date": [
                f"2020-01-{i:02d}" if i % 3 == 0 else None for i in range(1, 13)
            ],
        }
    )
    plan = draft_plan(df, "ds_dates", stakes=True, seed=0)
    date_q = next(q for q in plan.questions.values() if q.topic == "date_columns")
    has_date_opt = next(o for o in date_q.options if o.option_id == "has_date")
    assert "1 date column has no blanks and stays out" in has_date_opt.effect
    assert [e["column"] for e in has_date_opt.edits] == ["death_date"]

    plan_hd = apply_answers_and_edits(
        plan,
        answers=[{"question_id": date_q.question_id, "option_id": "has_date"}],
        edits=[],
        source="agent",
    )
    cfg = load_config(
        {
            "run": {"name": "test", "data": "in_memory"},
            "preprocessing": plan_to_preprocessing_dict(plan_hd),
        }
    )
    out_df, layout = preprocess_dataframe(df, cfg)
    assert "death_date_was_missing" in layout.feature_names
    assert "birth_date_was_missing" not in layout.feature_names
    assert out_df["death_date_was_missing"].nunique() == 2

    # When ALL date columns have 0 blanks, has_date has empty edits and stakes.neighbors_kept == 1.0
    df_all_complete = df.drop(columns=["death_date"])
    plan_complete = draft_plan(
        df_all_complete, "ds_dates_complete", stakes=True, seed=0
    )
    date_q_comp = next(
        q for q in plan_complete.questions.values() if q.topic == "date_columns"
    )
    has_date_comp = next(o for o in date_q_comp.options if o.option_id == "has_date")
    assert has_date_comp.edits == []
    assert date_q_comp.stakes is not None
    assert date_q_comp.stakes["neighbors_kept"] == 1.0
    assert date_q_comp.stakes["feature_count_delta"] == 0


def test_edit_plan_rejects_empty_and_tracks_confirmed_without_change() -> None:
    from pulsar.mcp.preprocessing import build_edit_plan_payload
    from pulsar.mcp.preprocessing_plan import diff_plan_entries

    df = make_preprocessing_traps(n=80, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=False)

    # Empty answers + empty edits -> INVALID_ARGUMENT
    with pytest.raises(PlanEditError) as exc_info:
        apply_answers_and_edits(plan, answers=[], edits=[], source="agent")
    assert exc_info.value.code == "INVALID_ARGUMENT"

    # Answering with the default option records decided_by and confirmed_without_change,
    # while payload['changes'] and payload['change_summary'] omit status/decided_by rows.
    q1 = plan.questions["q1"]
    plan_confirmed = apply_answers_and_edits(
        plan,
        answers=[{"question_id": "q1", "option_id": q1.default_option_id}],
        edits=[],
        source="agent",
    )
    changes = diff_plan_entries(plan, plan_confirmed)
    assert any(ch["field"] == "decided_by" and ch["now"] == "agent" for ch in changes)
    space, val = dry_run_processed_space(df, plan_confirmed)
    payload = build_edit_plan_payload(
        plan_confirmed, changes=changes, processed_space=space, validation=val
    )
    assert payload["counts"]["confirmed_without_change"] == len(q1.applies_to)
    assert payload["changes"] == []
    assert payload["changes_omitted"] == 0
    assert payload["change_summary"] == []
    assert payload["change_summary_omitted"] == 0
    assert "preprocessing_yaml" not in payload
    assert payload["preprocessing_yaml_chars"] > 0

    # Answering a non-default option on a multi-column question populates change_summary
    blank_q = next(q for q in plan.questions.values() if q.topic == "blank_meaning")
    plan_edited = apply_answers_and_edits(
        plan,
        answers=[{"question_id": blank_q.question_id, "option_id": "none"}],
        edits=[],
        source="user",
    )
    changes_eff = diff_plan_entries(plan, plan_edited)
    space_eff, val_eff = dry_run_processed_space(df, plan_edited)
    payload_eff = build_edit_plan_payload(
        plan_edited, changes=changes_eff, processed_space=space_eff, validation=val_eff
    )
    assert all(
        ch["field"] in ("role", "impute", "indicator", "encode")
        for ch in payload_eff["changes"]
    )
    assert any(
        s["field"] == "impute"
        and s["now"] == {"method": "fill_value", "value": 0}
        and s["count"] == len(blank_q.applies_to)
        for s in payload_eff["change_summary"]
    )


def test_edit_validation_rejects_incompatible_impute_encode_indicator() -> None:
    df = make_preprocessing_traps(n=80, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=False)

    bad_edits = [
        # Numeric impute_method on text column
        {"column": "race", "impute_method": "fill_median"},
        # Categorical impute_method on numeric column
        {"column": "bmi", "impute_method": "fill_mode"},
        # String impute_value on numeric column
        {"column": "bmi", "impute_method": "fill_value", "impute_value": "abc"},
        # NaN impute_value on numeric column
        {"column": "bmi", "impute_method": "fill_value", "impute_value": float("nan")},
        # Adding imputation on a column with 0 blanks
        {"column": "age", "impute_method": "fill_median"},
        # Indicator on a column with 0 blanks
        {"column": "age", "indicator": "own"},
        {"column": "age", "indicator": "group:age"},
        {"column": "age", "indicator": "group:mg2"},
        # Unknown indicator group
        {"column": "bmi", "indicator": "group:zzz"},
    ]
    for edit in bad_edits:
        with pytest.raises(PlanEditError) as exc_info:
            apply_answers_and_edits(plan, answers=[], edits=[edit], source="agent")
        assert exc_info.value.code == "INVALID_ARGUMENT", f"Failed on {edit}"
        assert edit["column"] in exc_info.value.message

    # Numeric fill_value on text column is allowed and converted to str
    plan_txt_val = apply_answers_and_edits(
        plan,
        answers=[],
        edits=[{"column": "race", "impute_method": "fill_value", "impute_value": 0}],
        source="agent",
    )
    assert plan_txt_val.entries["race"].impute_value == "0"

    # one_hot on numeric column is allowed
    plan_num_oh = apply_answers_and_edits(
        plan,
        answers=[],
        edits=[{"column": "age", "encode": "one_hot"}],
        source="agent",
    )
    assert plan_num_oh.entries["age"].encode == "one_hot"


def test_d1_mixed_values_esrd_not_coerced_to_zero() -> None:
    n_rows = 1200
    n_y = 200
    rng = np.random.default_rng(11)
    esrd_2008 = ["Y"] * 150 + ["0"] * (n_rows - 150)
    esrd_2009 = ["Y"] * n_y + ["0"] * (n_rows - n_y)
    # Also include a numeric_text column whose only non-numeric entries are whitespace/blanks
    pure_num_text = [str(i % 10) for i in range(n_rows - 20)] + ["  "] * 20
    df = pd.DataFrame(
        {
            "age": rng.integers(50, 85, size=n_rows).astype(float),
            "bene_esrd_ind_2008": pd.Series(esrd_2008, dtype=object),
            "bene_esrd_ind_2009": pd.Series(esrd_2009, dtype=object),
            "pure_num_text": pd.Series(pure_num_text, dtype=object),
        }
    )

    plan = draft_plan(df, "ds_esrd", stakes=True, stakes_sample_rows=200, seed=0)

    # 1. pure_num_text has n_non_numeric == 0 -> treated as numeric as before, resolved
    pure_ev = next(c for c in plan.evidence.columns if c.name == "pure_num_text")
    assert pure_ev.kind == "numeric_text"
    assert pure_ev.n_non_numeric == 0
    assert pure_ev.non_numeric_values == ()
    assert plan.entries["pure_num_text"].status == "resolved"
    assert plan.entries["pure_num_text"].impute == {"method": "fill_median"}
    assert plan.entries["pure_num_text"].encode is None

    # 2. bene_esrd_ind_2009 has n_non_numeric == 200 and n_unique == 2 <= sqrt(1200) -> as_categories default
    esrd_ev = next(c for c in plan.evidence.columns if c.name == "bene_esrd_ind_2009")
    assert esrd_ev.kind == "numeric_text"
    assert esrd_ev.n_non_numeric == n_y
    assert esrd_ev.non_numeric_values == (("Y", n_y),)

    esrd_entry = plan.entries["bene_esrd_ind_2009"]
    assert esrd_entry.status == "open"
    assert esrd_entry.encode == "one_hot"
    assert esrd_entry.impute is None
    assert esrd_entry.indicator == "none"

    mixed_qs = [q for q in plan.questions.values() if q.topic == "mixed_values"]
    assert len(mixed_qs) == 1
    mq = mixed_qs[0]
    assert mq.applies_to == ["bene_esrd_ind_2008", "bene_esrd_ind_2009"]
    assert mq.default_option_id == "as_categories"
    total_y = n_y + 150
    assert (
        f"2 columns hold the number 0 and the value 'Y' ({total_y:,} entries in total; e.g. `bene_esrd_ind_2009`: {n_y:,})."
        in mq.text
    )
    assert [o.option_id for o in mq.options] == [
        "as_categories",
        "as_blank",
        "leave_out",
    ]
    as_blank_opt = next(o for o in mq.options if o.option_id == "as_blank")
    assert (
        f"{total_y:,} 'Y' entries become blanks and are estimated"
        in as_blank_opt.effect
    )

    # 3. Default plan keeps 'Y' distinguishable after preprocess_dataframe
    cfg = load_config(
        {
            "run": {"name": "test", "data": "in_memory"},
            "preprocessing": plan_to_preprocessing_dict(plan),
        }
    )
    out_df, layout = preprocess_dataframe(df, cfg)
    assert "bene_esrd_ind_2009_Y" in layout.feature_names
    assert int(out_df["bene_esrd_ind_2009_Y"].sum()) == n_y
    assert "bene_esrd_ind_2009_0" in layout.feature_names
    assert int(out_df["bene_esrd_ind_2009_0"].sum()) == n_rows - n_y

    # 4. High-cardinality numeric_text column (n_unique > sqrt(n_rows)) defaults to as_blank
    high_card_vals = [str(i) for i in range(n_rows - 15)] + ["N/A"] * 15
    df_high = pd.DataFrame(
        {
            "age": rng.integers(50, 85, size=n_rows).astype(float),
            "lab_reading": pd.Series(high_card_vals, dtype=object),
        }
    )
    plan_high = draft_plan(df_high, "ds_lab", stakes=False)
    lab_entry = plan_high.entries["lab_reading"]
    assert lab_entry.status == "open"
    assert lab_entry.encode is None
    assert lab_entry.impute == {"method": "fill_median"}
    assert lab_entry.indicator == "own"
    high_qs = [q for q in plan_high.questions.values() if q.topic == "mixed_values"]
    assert len(high_qs) == 1
    assert high_qs[0].default_option_id == "as_blank"
    assert (
        "`lab_reading` mostly holds numbers but also the value 'N/A' (15 rows). "
        "Should those values count as their own category, or as unreadable entries "
        "treated like blanks?"
    ) == high_qs[0].text


def test_r1_unique_plan_id_per_draft() -> None:
    df = make_preprocessing_traps(n=60, seed=0)
    plan1 = draft_plan(df, "ds_traps", stakes=False)
    plan2 = draft_plan(df, "ds_traps", stakes=False)
    assert plan1.plan_id.startswith("pp_")
    assert plan2.plan_id.startswith("pp_")
    assert plan1.plan_id != plan2.plan_id
    # Content fingerprint is still identical for identical decisions
    assert plan_fingerprint(plan1) == plan_fingerprint(plan2)


def test_r2_protected_roles_not_overridden_by_question_answers() -> None:
    df = make_preprocessing_traps(n=120, seed=0)
    df["index_date"] = [f"2008-01-{(i % 28) + 1:02d}" for i in range(len(df))]
    plan = draft_plan(df, "ds_traps", stakes=False)

    cond_q = next(q for q in plan.questions.values() if q.topic == "conditional_column")
    date_q = next(q for q in plan.questions.values() if q.topic == "date_columns")
    assert cond_q.applies_to == ["days_to_death"]
    assert set(date_q.applies_to) == {"death_date", "index_date"}

    # 1. Mark days_to_death and death_date as target
    plan_target = apply_answers_and_edits(
        plan,
        answers=[],
        edits=[
            {"column": "days_to_death", "role": "target"},
            {"column": "death_date", "role": "target"},
        ],
        source="agent",
    )
    assert plan_target.entries["days_to_death"].role == "target"
    assert plan_target.entries["death_date"].role == "target"
    assert plan_target.questions[cond_q.question_id].status == "answered"

    # 2. Answering conditional_column after marking days_to_death target is rejected; role stays target
    with pytest.raises(PlanEditError) as exc_info:
        apply_answers_and_edits(
            plan_target,
            answers=[{"question_id": cond_q.question_id, "option_id": "leave_out"}],
            edits=[],
            source="agent",
        )
    assert exc_info.value.code == "INVALID_ARGUMENT"
    assert (
        f"question {cond_q.question_id} no longer applies (columns now: days_to_death=target)"
        in exc_info.value.message
    )
    assert plan_target.entries["days_to_death"].role == "target"

    # 3. Answering multi-column date_columns (death_date + index_date) skips protected death_date
    plan_after_date = apply_answers_and_edits(
        plan_target,
        answers=[{"question_id": date_q.question_id, "option_id": "ignore"}],
        edits=[],
        source="agent",
    )
    assert plan_after_date.entries["death_date"].role == "target"
    assert plan_after_date.entries["index_date"].role == "drop"


def test_r3_outcome_proxies_and_target_proxy_warnings() -> None:
    from pulsar.mcp.preprocessing import (
        build_draft_plan_payload,
        build_edit_plan_payload,
    )
    from pulsar.mcp.preprocessing_plan import diff_plan_entries

    df = make_preprocessing_traps(n=200, seed=0)
    # 1. Even with no target declared, boolean event_flag_columns (died) are checked.
    # Add a non-constant column that is blank on a subset of died==True rows (like 2009/2010 enrollment in SynPUF).
    died_idx = np.flatnonzero(df["died"].astype(bool).to_numpy())
    cov = np.arange(len(df), dtype=float)
    cov[died_idx[:10]] = np.nan
    df["post_year_coverage"] = cov

    plan_no_target = draft_plan(df, "ds_traps", stakes=False)
    space0, _ = dry_run_processed_space(df, plan_no_target)
    payload0 = build_draft_plan_payload(plan_no_target, processed_space=space0)
    proxies0 = payload0["evidence"]["target_proxies"]
    assert any(
        p["column"] == "post_year_coverage_was_missing"
        and p["target"] == "died"
        and p["share_with_target"] == 1.0
        for p in proxies0
    )
    assert any(
        w["code"] == "TARGET_PROXY_COLUMNS"
        and w["target"] == "died"
        and "post_year_coverage_was_missing" in w["columns"]
        and w["message"]
        == "these columns are 1 only for rows where died is set; they would let the map encode the outcome"
        for w in payload0["warnings"]
    )

    # 2. On traps with target_columns=["died"], answering date_columns with has_date on death_date
    # flags death_date_was_missing with share_with_target == 1.0.
    df_traps = make_preprocessing_traps(n=200, seed=0)
    plan_t = draft_plan(df_traps, "ds_traps", target_columns=["died"], stakes=False)
    date_q = next(q for q in plan_t.questions.values() if q.topic == "date_columns")
    plan_has_date = apply_answers_and_edits(
        plan_t,
        answers=[{"question_id": date_q.question_id, "option_id": "has_date"}],
        edits=[],
        source="agent",
    )
    space_edit, val_edit = dry_run_processed_space(df_traps, plan_has_date)
    edit_payload = build_edit_plan_payload(
        plan_has_date,
        changes=diff_plan_entries(plan_t, plan_has_date),
        processed_space=space_edit,
        validation=val_edit,
    )
    edit_proxies = edit_payload["evidence"]["target_proxies"]
    assert any(
        p["column"] == "death_date_was_missing"
        and p["target"] == "died"
        and p["share_with_target"] == 1.0
        for p in edit_proxies
    )
    assert any(
        w["code"] == "TARGET_PROXY_COLUMNS"
        and w["target"] == "died"
        and "death_date_was_missing" in w["columns"]
        for w in edit_payload["warnings"]
    )


def test_r4_mixed_values_single_and_multi_number_phrasing() -> None:
    rng = np.random.default_rng(7)
    n = 100
    # Single column holding only '0' and 'Y'
    df_single = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "esrd": ["Y" if i < 20 else "0" for i in range(n)],
        }
    )
    p_single = draft_plan(df_single, "ds_s", stakes=False)
    mq_s = next(q for q in p_single.questions.values() if q.topic == "mixed_values")
    assert "`esrd` holds the number 0 and the value 'Y' (20 rows)." in mq_s.text

    # Single column with multiple distinct numbers, mostly numbers vs mostly text
    df_multi_num = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "mostly_num": ["N/A" if i < 15 else str(i % 5) for i in range(n)],
        }
    )
    p_mn = draft_plan(df_multi_num, "ds_mn", stakes=False)
    mq_mn = next(q for q in p_mn.questions.values() if q.topic == "mixed_values")
    assert "`mostly_num` mostly holds numbers" in mq_mn.text


def test_r5_code_columns_question() -> None:
    rng = np.random.default_rng(11)
    n = 100
    df = pd.DataFrame(
        {
            "continuous": rng.normal(size=n),
            "sp_state_code": [int(i % 8) + 1 for i in range(n)],
            "bene_race_cd": [int(i % 4) + 1 for i in range(n)],
            "binary_flag": [int(i % 2) for i in range(n)],
        }
    )
    plan = draft_plan(df, "ds_codes", stakes=True, seed=0)
    q_topics = [q.topic for q in plan.questions.values()]
    assert "code_columns" in q_topics
    # Framing questions (code_columns, outcome_columns) appear at the end
    assert q_topics[-2:] == ["code_columns", "outcome_columns"]

    code_q = next(q for q in plan.questions.values() if q.topic == "code_columns")
    assert code_q.stakes is None
    assert code_q.default_option_id == "all_amounts"
    assert code_q.applies_to == ["sp_state_code", "bene_race_cd"]
    assert (
        "Some whole-number columns might be labels rather than amounts "
        "(e.g. `sp_state_code`: 8 distinct values, `bene_race_cd`: 4 distinct values; 2 columns)."
        in code_q.text
    )

    # Answering all_amounts + overriding sp_state_code to one_hot
    edited = apply_answers_and_edits(
        plan,
        answers=[{"question_id": code_q.question_id, "option_id": "all_amounts"}],
        edits=[{"column": "sp_state_code", "encode": "one_hot"}],
        source="agent",
    )
    assert edited.questions[code_q.question_id].status == "answered"
    assert edited.entries["sp_state_code"].encode == "one_hot"
    assert edited.entries["bene_race_cd"].encode is None


def test_r6_ux_improvements() -> None:
    from pulsar.mcp.preprocessing import (
        build_draft_plan_payload,
        build_edit_plan_payload,
        edit_plan_payload_to_markdown,
    )
    from pulsar.mcp.preprocessing_plan import diff_plan_entries, summarize_plan_counts

    df = make_preprocessing_traps(n=80, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=False)

    # 1. Invalid role error message lists valid roles and hints at 'target'
    with pytest.raises(PlanEditError) as exc_info:
        apply_answers_and_edits(
            plan,
            answers=[],
            edits=[{"column": "died", "role": "outcome"}],
            source="agent",
        )
    assert exc_info.value.code == "INVALID_ARGUMENT"
    assert "Invalid role 'outcome' for column 'died'" in exc_info.value.message
    assert "valid roles are" in exc_info.value.message
    assert "for outcomes use role 'target'" in exc_info.value.message

    # 2. counts uses feature_columns instead of features
    counts = summarize_plan_counts(plan)
    assert "feature_columns" in counts
    assert "features" not in counts
    draft_payload = build_draft_plan_payload(plan, processed_space=None)
    assert "feature_columns" in draft_payload["counts"]
    assert "features" not in draft_payload["counts"]

    # 3. Edit summary markdown lists open question IDs and topics
    first_qid = next(iter(plan.questions))
    first_opt = plan.questions[first_qid].default_option_id
    edited = apply_answers_and_edits(
        plan,
        answers=[{"question_id": first_qid, "option_id": first_opt}],
        edits=[],
        source="agent",
    )
    space, val = dry_run_processed_space(df, edited)
    edit_payload = build_edit_plan_payload(
        edited,
        changes=diff_plan_entries(plan, edited),
        processed_space=space,
        validation=val,
    )
    md = edit_plan_payload_to_markdown(edit_payload)
    assert "- Open questions: " in md
    for q in edited.questions.values():
        if q.status == "open":
            assert f"`{q.question_id}` [{q.topic}]" in md


def test_mostly_blank_and_most_missing_columns_evidence() -> None:
    from pulsar.mcp.preprocessing import (
        build_draft_plan_payload,
        draft_plan_payload_to_markdown,
    )

    rng = np.random.default_rng(123)
    n = 100
    # Complete baseline features
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)

    # 97%-blank numeric column with random mask (has negative values so not count-like)
    lab_x = np.full(n, np.nan)
    filled_idx = rng.choice(n, size=3, replace=False)
    lab_x[filled_idx] = [-1.5, 0.0, 2.5]

    # 30%-blank numeric column with random mask (has negative values so not count-like)
    bmi_30 = rng.normal(loc=0.0, scale=1.0, size=n)
    blank_30_idx = rng.choice(n, size=30, replace=False)
    bmi_30[blank_30_idx] = np.nan

    df_single = pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "lab_x": lab_x,
            "bmi_30": bmi_30,
        }
    )

    plan = draft_plan(df_single, "ds_mostly_blank", stakes=True, seed=0)

    # 1. 97%-blank column -> default role "drop", status "open", never fill_median by default
    assert plan.entries["lab_x"].role == "drop"
    assert plan.entries["lab_x"].impute is None
    assert plan.entries["lab_x"].indicator == "none"
    assert plan.entries["lab_x"].status == "open"
    assert plan.entries["lab_x"].decided_by == "default"

    # 2. 30%-blank column -> unchanged default (fill_median + marker, resolved)
    assert plan.entries["bmi_30"].role == "feature"
    assert plan.entries["bmi_30"].impute == {"method": "fill_median"}
    assert plan.entries["bmi_30"].indicator == "own"
    assert plan.entries["bmi_30"].status == "resolved"

    # 3. Single grouped mostly_blank question with plain language and stakes
    mb_qs = [q for q in plan.questions.values() if q.topic == "mostly_blank"]
    assert len(mb_qs) == 1
    mb_q = mb_qs[0]
    assert mb_q.applies_to == ["lab_x"]
    assert mb_q.rows_affected == 97
    assert mb_q.default_option_id == "leave_out"
    assert [o.option_id for o in mb_q.options] == [
        "leave_out",
        "has_value",
        "estimate_and_mark",
    ]
    assert (
        "`lab_x` is blank for 97% of rows, and the data doesn't show why." in mb_q.text
    )
    est_opt = next(o for o in mb_q.options if o.option_id == "estimate_and_mark")
    assert "about 97% of these values would be estimates" in est_opt.effect
    assert mb_q.stakes is not None
    assert 0.0 <= mb_q.stakes["neighbors_kept"] <= 1.0
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in FORBIDDEN_JARGON) + r")s?\b",
        re.IGNORECASE,
    )
    assert not pattern.search(mb_q.text)
    for leak in FORBIDDEN_SCHEMA_LEAKS:
        assert leak not in mb_q.text
    for opt in mb_q.options:
        assert not pattern.search(opt.label)
        assert not pattern.search(opt.effect)
        for leak in FORBIDDEN_SCHEMA_LEAKS:
            assert leak not in opt.label and leak not in opt.effect

    # 4. Answering has_value runs cleanly through load_config + preprocess_dataframe
    plan_hv = apply_answers_and_edits(
        plan,
        answers=[{"question_id": mb_q.question_id, "option_id": "has_value"}],
        edits=[],
        source="user",
    )
    assert plan_hv.entries["lab_x"].role == "drop"
    assert plan_hv.entries["lab_x"].impute is None
    assert plan_hv.entries["lab_x"].indicator == "group:lab_x"
    assert plan_hv.entries["lab_x"].status == "resolved"
    cfg_hv = load_config(
        {
            "run": {"name": "test_hv", "data": "in_memory"},
            "preprocessing": plan_to_preprocessing_dict(plan_hv),
        }
    )
    out_hv, layout_hv = preprocess_dataframe(df_single, cfg_hv)
    assert "lab_x" not in layout_hv.feature_names
    assert "lab_x_was_missing" in layout_hv.feature_names
    assert int(out_hv["lab_x_was_missing"].sum()) == 97

    # Answering estimate_and_mark also runs cleanly
    plan_est = apply_answers_and_edits(
        plan,
        answers=[{"question_id": mb_q.question_id, "option_id": "estimate_and_mark"}],
        edits=[],
        source="user",
    )
    assert plan_est.entries["lab_x"].role == "feature"
    assert plan_est.entries["lab_x"].impute == {"method": "fill_median"}
    assert plan_est.entries["lab_x"].indicator == "own"
    cfg_est = load_config(
        {
            "run": {"name": "test_est", "data": "in_memory"},
            "preprocessing": plan_to_preprocessing_dict(plan_est),
        }
    )
    _, layout_est = preprocess_dataframe(df_single, cfg_est)
    assert "lab_x" in layout_est.feature_names
    assert "lab_x_was_missing" in layout_est.feature_names

    # 5. Multi-column mostly_blank (shared blank group + singleton) produces ONE grouped question
    shared_mask = np.ones(n, dtype=bool)
    shared_mask[rng.choice(n, size=20, replace=False)] = False  # 80% blank
    grp_a = np.where(shared_mask, np.nan, rng.normal(size=n))
    grp_b = np.where(shared_mask, np.nan, rng.normal(size=n))
    df_multi = df_single.assign(grp_a=grp_a, grp_b=grp_b)
    plan_multi = draft_plan(df_multi, "ds_multi_mb", stakes=True, seed=0)
    mb_multi_qs = [
        q for q in plan_multi.questions.values() if q.topic == "mostly_blank"
    ]
    assert len(mb_multi_qs) == 1
    assert set(mb_multi_qs[0].applies_to) == {"lab_x", "grp_a", "grp_b"}
    assert (
        "`lab_x` is blank for 97% of rows (and 2 other columns are blank for more than half of rows: `grp_a` and `grp_b`)"
        in mb_multi_qs[0].text
    )
    plan_multi_hv = apply_answers_and_edits(
        plan_multi,
        answers=[{"question_id": mb_multi_qs[0].question_id, "option_id": "has_value"}],
        edits=[],
        source="agent",
    )
    cfg_multi_hv = load_config(
        {
            "run": {"name": "test_multi_hv", "data": "in_memory"},
            "preprocessing": plan_to_preprocessing_dict(plan_multi_hv),
        }
    )
    _, layout_multi_hv = preprocess_dataframe(df_multi, cfg_multi_hv)
    assert "lab_x_was_missing" in layout_multi_hv.feature_names
    assert "grp_a_and_1_more_was_missing" in layout_multi_hv.feature_names

    # 6. Evidence slice: most_missing_columns ranked by missing_fraction desc then name, capped at evidence_limit
    space, _ = dry_run_processed_space(df_multi, plan_multi)
    payload_capped = build_draft_plan_payload(
        plan_multi, processed_space=space, detail="summary", evidence_limit=2
    )
    mm = payload_capped["evidence"]["most_missing_columns"]
    assert len(mm) == 2
    assert payload_capped["evidence"]["most_missing_columns_omitted"] == 2
    assert [item["column"] for item in mm] == ["lab_x", "grp_a"]
    assert mm[0] == {
        "column": "lab_x",
        "missing_fraction": 0.97,
        "n_missing": 97,
        "missing_group": None,
        "plan_role": "drop",
        "plan_impute_method": None,
    }
    assert mm[1]["column"] == "grp_a"
    assert mm[1]["missing_fraction"] == 0.8
    assert mm[1]["missing_group"] is not None
    assert mm[1]["plan_role"] == "drop"
    assert mm[1]["plan_impute_method"] is None

    payload_full = build_draft_plan_payload(
        plan_multi, processed_space=space, detail="full", evidence_limit=2
    )
    mm_full = payload_full["evidence"]["most_missing_columns"]
    assert [item["column"] for item in mm_full] == ["lab_x", "grp_a", "grp_b", "bmi_30"]
    assert mm_full[-1]["plan_role"] == "feature"
    assert mm_full[-1]["plan_impute_method"] == "fill_median"
    assert payload_full["evidence"]["most_missing_columns_omitted"] == 0

    md_str = draft_plan_payload_to_markdown(payload_capped)
    assert "- Most-missing columns: 2 (+2 omitted)" in md_str
