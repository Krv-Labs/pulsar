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
    assert plan_with_stakes.cost["stakes_questions_computed"] == 4
    assert plan_with_stakes.cost["stakes_questions_skipped"] == 0

    plan_no_stakes = draft_plan(df, "ds_traps", stakes=False)
    assert plan_no_stakes.cost["evidence_ms"] > 0.0
    assert plan_no_stakes.cost["stakes_ms"] == 0.0
    assert plan_no_stakes.cost["stakes_rows_sampled"] == 0
    assert plan_no_stakes.cost["stakes_questions_computed"] == 0
    # 4 non-outcome_columns questions are skipped when stakes=False
    assert plan_no_stakes.cost["stakes_questions_skipped"] == 4


def test_outcome_columns_always_last_stakes_none_and_weak_hints_cap() -> None:
    from pulsar.mcp.preprocessing import build_draft_plan_payload

    df = make_preprocessing_traps(n=120, seed=0)
    plan = draft_plan(df, "ds_traps", stakes=True, stakes_max_questions=4, seed=0)
    q_list = list(plan.questions.values())
    assert q_list[-1].topic == "outcome_columns"
    assert q_list[-1].question_id == f"q{len(q_list)}"
    assert q_list[-1].stakes is None
    assert q_list[-1].options[0].effect == "Keeps all current columns as map inputs"
    # None of the 4 real questions were bumped by outcome_columns
    assert all(q.stakes is not None for q in q_list[:-1])

    # Add a second boolean event partner to verify weak_hints capping
    df2 = df.copy()
    df2["surgery_flag"] = (df2["age"] > 70).astype(int)
    df2["days_to_surgery"] = np.where(df2["surgery_flag"] == 1, 15.0, np.nan)
    plan2 = draft_plan(df2, "ds_traps2", stakes=False)
    payload_capped = build_draft_plan_payload(
        plan2, processed_space=None, detail="summary", evidence_limit=1
    )
    assert len(payload_capped["weak_hints"]["event_partner_columns"]) == 1
    assert payload_capped["weak_hints"]["event_partner_columns_omitted"] == 1
    payload_full = build_draft_plan_payload(plan2, processed_space=None, detail="full")
    assert len(payload_full["weak_hints"]["event_partner_columns"]) == 2
    assert payload_full["weak_hints"]["event_partner_columns_omitted"] == 0


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
        "`claim_cnt` never records a 0 (the other 1 column sometimes does)."
        in blank_qs[0].text
    )
    assert "blank exactly where `visits` is 0" in blank_qs[0].text
    none_opt = next(o for o in blank_qs[0].options if o.option_id == "none")
    assert "All 2 columns get 0 where blank" in none_opt.effect
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

    # Answering with the default option records decided_by and confirmed_without_change
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
