"""Counterfactual k-NN neighborhood overlap stakes and question ordering."""

from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd

from pulsar.analysis.preprocessing_evidence import neighborhood_overlap
from pulsar.config import load_config
from pulsar.mcp.preprocessing_plan.edits import apply_answers_and_edits
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    PreprocessingPlan,
)
from pulsar.mcp.preprocessing_plan.render import plan_to_preprocessing_dict
from pulsar.preprocessing import preprocess_dataframe


def _framing_rank(topic: str) -> int:
    if topic == "code_columns":
        return 1
    if topic == "outcome_columns":
        return 2
    return 0


def _plan_to_matrix(sample_df: pd.DataFrame, plan: PreprocessingPlan) -> np.ndarray:
    pre_dict = plan_to_preprocessing_dict(plan)
    cfg = load_config(
        {
            "run": {"name": "stakes_eval", "data": "in_memory"},
            "preprocessing": pre_dict,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df_proc, _ = preprocess_dataframe(sample_df, cfg)
    return df_proc.to_numpy(dtype=np.float64)


def _padded_overlap(X_base: np.ndarray, X_opt: np.ndarray, *, k: int) -> float:
    if X_base.ndim != 2 or X_opt.ndim != 2:
        return 0.0
    if X_base.shape[1] == 0 and X_opt.shape[1] == 0:
        return 1.0
    if X_base.shape[1] == 0 or X_opt.shape[1] == 0:
        return 0.0
    max_d = max(X_base.shape[1], X_opt.shape[1])
    if X_base.shape[1] < max_d:
        X_a = np.pad(X_base, ((0, 0), (0, max_d - X_base.shape[1])), mode="constant")
    else:
        X_a = X_base
    if X_opt.shape[1] < max_d:
        X_b = np.pad(X_opt, ((0, 0), (0, max_d - X_opt.shape[1])), mode="constant")
    else:
        X_b = X_opt
    return neighborhood_overlap(X_a, X_b, k=k)


def _evaluate_question_stakes(
    qid: str,
    q: PlanQuestion,
    sample_df: pd.DataFrame,
    plan: PreprocessingPlan,
    X_base: np.ndarray,
    *,
    k_eff: int,
    n_sampled: int,
) -> dict[str, Any] | None:
    non_default_opts = [
        opt for opt in q.options if opt.option_id != q.default_option_id
    ]
    counterfactual_edit_sets: list[list[dict[str, Any]]] = [
        opt.edits for opt in non_default_opts
    ]
    if not counterfactual_edit_sets:
        return None

    min_overlap: float | None = None
    best_delta: int = 0
    for edit_list in counterfactual_edit_sets:
        try:
            if not edit_list:
                overlap = 1.0
                delta = 0
            else:
                alt_plan = apply_answers_and_edits(
                    plan, answers=(), edits=edit_list, source="agent"
                )
                X_opt = _plan_to_matrix(sample_df, alt_plan)
                overlap = _padded_overlap(X_base, X_opt, k=k_eff)
                delta = int(X_opt.shape[1] - X_base.shape[1])
            if min_overlap is None or (
                overlap < min_overlap
                or (overlap == min_overlap and abs(delta) > abs(best_delta))
            ):
                min_overlap = overlap
                best_delta = delta
        except Exception as exc:
            plan.cost.setdefault("stakes_notes", []).append(
                f"Stakes option failed for {qid}: {exc}"
            )

    if min_overlap is None:
        return None
    return {
        "neighbors_kept": round(float(min_overlap), 4),
        "feature_count_delta": int(best_delta),
        "rows_sampled": int(n_sampled),
        "k": int(k_eff),
    }


def compute_stakes(
    df: pd.DataFrame,
    plan: PreprocessingPlan,
    question_ids: Sequence[str] | None = None,
    *,
    sample_rows: int = 2000,
    k: int = 10,
    seed: int = 0,
) -> dict[str, dict[str, Any] | None]:
    """Compute k-NN neighborhood overlap stakes for each question in *question_ids*."""
    target_qids = (
        list(question_ids)
        if question_ids is not None
        else [qid for qid, q in plan.questions.items() if q.status == "open"]
    )
    results: dict[str, dict[str, Any] | None] = {qid: None for qid in target_qids}
    if not target_qids:
        return results

    n_total = len(df)
    if n_total < 2:
        plan.cost.setdefault("stakes_notes", []).append(
            "Dataset has fewer than 2 rows; stakes not computed."
        )
        return results

    if n_total > sample_rows > 0:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=sample_rows, replace=False))
        sample_df = df.iloc[idx].reset_index(drop=True)
    else:
        sample_df = df

    n_sampled = len(sample_df)
    k_eff = min(int(k), n_sampled - 1)
    if k_eff < 1:
        plan.cost.setdefault("stakes_notes", []).append(
            "Effective k < 1 on row sample; stakes not computed."
        )
        return results

    try:
        X_base = _plan_to_matrix(sample_df, plan)
    except Exception as exc:
        plan.cost.setdefault("stakes_notes", []).append(
            f"Baseline dry-run failed during stakes computation: {exc}"
        )
        return results

    for qid in target_qids:
        q = plan.questions.get(qid)
        if q is None or q.topic == "outcome_columns":
            continue
        results[qid] = _evaluate_question_stakes(
            qid,
            q,
            sample_df,
            plan,
            X_base,
            k_eff=k_eff,
            n_sampled=n_sampled,
        )

    return results


def assign_provisional_questions(
    pending_questions: list[PendingQuestion],
) -> tuple[dict[str, PlanQuestion], list[PlanQuestion]]:
    """Pre-sort pending questions, assign provisional IDs, and return eligible stakes questions."""
    pending_questions.sort(
        key=lambda item: (
            _framing_rank(item[0].topic),
            -item[0].rows_affected,
        )
    )
    provisional_questions: dict[str, PlanQuestion] = {}
    for idx, (q, _linked_cols) in enumerate(pending_questions, start=1):
        prov_id = f"_prov_{idx}"
        q.question_id = prov_id
        provisional_questions[prov_id] = q

    eligible_stakes_questions = [
        q
        for q, _ in pending_questions
        if q.topic not in ("outcome_columns", "code_columns")
    ]
    return provisional_questions, eligible_stakes_questions


def evaluate_and_finalize_questions(
    df: pd.DataFrame,
    plan: PreprocessingPlan,
    pending_questions: list[PendingQuestion],
    eligible_stakes_questions: list[PlanQuestion],
    *,
    stakes: bool,
    stakes_sample_rows: int,
    stakes_k: int,
    stakes_max_questions: int,
    seed: int,
) -> None:
    """Compute question stakes (when enabled), sort questions into final order, and assign q1..qN."""
    if stakes and eligible_stakes_questions and stakes_max_questions > 0:
        budget_ids = [
            q.question_id
            for q in eligible_stakes_questions[: int(stakes_max_questions)]
        ]
        skipped_count = len(eligible_stakes_questions) - len(budget_ids)
        t0 = time.perf_counter()
        stakes_by_id = compute_stakes(
            df,
            plan,
            budget_ids,
            sample_rows=stakes_sample_rows,
            k=stakes_k,
            seed=seed,
        )
        stakes_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        computed_count = sum(
            1 for qid in budget_ids if stakes_by_id.get(qid) is not None
        )
        failed_count = len(budget_ids) - computed_count
        sampled_rows = 0
        for qid, st in stakes_by_id.items():
            plan.questions[qid].stakes = st
            if st is not None and sampled_rows == 0:
                sampled_rows = int(st.get("rows_sampled", 0))
        plan.cost["stakes_ms"] = stakes_ms
        plan.cost["stakes_rows_sampled"] = sampled_rows
        plan.cost["stakes_questions_computed"] = computed_count
        plan.cost["stakes_questions_skipped"] = skipped_count
        plan.cost["stakes_questions_failed"] = failed_count

    sorted_prov = sorted(
        pending_questions,
        key=lambda item: (
            _framing_rank(item[0].topic),
            0 if item[0].stakes is not None else 1,
            float(item[0].stakes["neighbors_kept"])
            if item[0].stakes is not None
            else 1.0,
            -item[0].rows_affected,
        ),
    )

    final_questions: dict[str, PlanQuestion] = {}
    for idx, (q, linked_cols) in enumerate(sorted_prov, start=1):
        qid = f"q{idx}"
        q.question_id = qid
        final_questions[qid] = q
        for col in linked_cols:
            if col in plan.entries:
                plan.entries[col].question_id = qid

    plan.questions = final_questions
