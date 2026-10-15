"""Top-level draft_plan orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import uuid

import pandas as pd

from pulsar.analysis.preprocessing_evidence import (
    PreprocessingEvidence,
    measure_preprocessing_evidence,
)
from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.defaults import (
    populate_default_entries,
    validate_role_designations,
)
from pulsar.mcp.preprocessing_plan.model import PreprocessingPlan
from pulsar.mcp.preprocessing_plan.proxies import dry_run_processed_space
from pulsar.mcp.preprocessing_plan.questions import build_pending_questions
from pulsar.mcp.preprocessing_plan.stakes import (
    assign_provisional_questions,
    evaluate_and_finalize_questions,
)


def draft_plan(
    df: pd.DataFrame,
    dataset_id: str,
    *,
    target_columns: Sequence[str] = (),
    entity_column: str = "",
    time_column: str = "",
    stakes: bool = True,
    stakes_sample_rows: int = 2000,
    stakes_k: int = 10,
    stakes_max_questions: int = 16,
    seed: int = 0,
    evidence: PreprocessingEvidence | None = None,
) -> PreprocessingPlan:
    """Measure *df* and construct a revision-1 :class:`PreprocessingPlan`."""
    target_list, entity_col, time_col = validate_role_designations(
        df, target_columns, entity_column, time_column
    )
    if evidence is None:
        evidence = measure_preprocessing_evidence(df)

    ctx = DraftContext.from_inputs(
        df,
        evidence,
        target_list=target_list,
        entity_col=entity_col,
        time_col=time_col,
    )
    populate_default_entries(ctx)
    pending_questions = build_pending_questions(ctx)

    provisional_questions, eligible_stakes_questions = assign_provisional_questions(
        pending_questions
    )
    plan_id = f"pp_{uuid.uuid4().hex[:10]}"

    plan = PreprocessingPlan(
        plan_id=plan_id,
        dataset_id=dataset_id,
        revision=1,
        entries=ctx.entries,
        questions=provisional_questions,
        missing_groups=ctx.missing_groups,
        evidence=evidence,
        created_at=datetime.now(timezone.utc).isoformat(),
        cost={
            "rows": int(evidence.n_rows),
            "columns": int(evidence.n_columns),
            "evidence_ms": round(float(evidence.cost.get("elapsed_ms", 0.0)), 2),
            "stakes_ms": 0.0,
            "stakes_rows_sampled": 0,
            "stakes_questions_computed": 0,
            "stakes_questions_skipped": len(eligible_stakes_questions),
            "stakes_questions_failed": 0,
        },
    )

    evaluate_and_finalize_questions(
        df,
        plan,
        pending_questions,
        eligible_stakes_questions,
        stakes=stakes,
        stakes_sample_rows=stakes_sample_rows,
        stakes_k=stakes_k,
        stakes_max_questions=stakes_max_questions,
        seed=seed,
    )
    dry_run_processed_space(df, plan)
    return plan
