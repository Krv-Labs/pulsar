"""Question builder for topic='duplicate_signal' (exact sums and duplicate columns)."""

from __future__ import annotations

from typing import Any

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)
from pulsar.mcp.preprocessing_plan.text import _format_col_list


def _collect_sum_and_duplicate_relations(
    ctx: DraftContext,
) -> tuple[list[str], list[str], list[Any], list[str], list[str], list[Any]]:
    """Partition eligible ``sum_of`` and ``duplicate`` relations on active feature columns."""
    sum_targets: list[str] = []
    sum_targets_set: set[str] = set()
    sum_sources: list[str] = []
    sum_sources_set: set[str] = set()
    accepted_sums: list[Any] = []

    dup_kept: list[str] = []
    dup_kept_set: set[str] = set()
    dup_dropped: list[str] = []
    dup_dropped_set: set[str] = set()
    accepted_dups: list[Any] = []

    for rel in ctx.evidence.relations:
        if rel.kind not in ("duplicate", "sum_of"):
            continue
        rel_cols = [rel.target, *rel.sources]
        if any(
            c not in ctx.entries or ctx.entries[c].role != "feature" for c in rel_cols
        ):
            continue
        if rel.kind == "sum_of":
            if (
                rel.target in sum_targets_set
                or rel.target in sum_sources_set
                or any(s in sum_targets_set for s in rel.sources)
            ):
                continue
            sum_targets_set.add(rel.target)
            sum_targets.append(rel.target)
            for s in rel.sources:
                if s not in sum_sources_set:
                    sum_sources_set.add(s)
                    sum_sources.append(s)
            accepted_sums.append(rel)
        elif rel.kind == "duplicate":
            col_a = rel.sources[0]
            col_b = rel.target
            if (
                col_a in dup_dropped_set
                or col_b in dup_kept_set
                or col_b in dup_dropped_set
            ):
                continue
            if col_a not in dup_kept_set:
                dup_kept_set.add(col_a)
                dup_kept.append(col_a)
            dup_dropped_set.add(col_b)
            dup_dropped.append(col_b)
            accepted_dups.append(rel)

    return (
        sum_targets,
        sum_sources,
        accepted_sums,
        dup_kept,
        dup_dropped,
        accepted_dups,
    )


def _build_sum_question(
    sum_targets: list[str],
    sum_sources: list[str],
    accepted_sums: list[Any],
) -> PlanQuestion:
    """Build the ``duplicate_signal`` question for exact ``sum_of`` relations."""
    r0 = accepted_sums[0]
    if len(r0.sources) <= 2:
        rhs = " + ".join(f"`{s}`" for s in r0.sources)
    else:
        rhs = " + ".join(f"`{s}`" for s in r0.sources[:2]) + " + …"
    ex_formula = f"`{r0.target}` = {rhs}"
    if len(accepted_sums) == 1:
        q_sum_text = (
            f"1 total in this table is an exact sum of other columns ({ex_formula}). "
            "Counting both gives those signals extra weight. Keep everything, "
            "keep only the totals, or keep only the parts?"
        )
    else:
        q_sum_text = (
            f"{len(accepted_sums)} totals in this table are exact sums of other "
            f"columns (e.g. {ex_formula}). Counting both gives those signals extra "
            "weight. Keep everything, keep only the totals, or keep only the parts?"
        )
    sum_applies = [*sum_targets, *sum_sources]
    sum_rows_aff = max(int(r.rows_used) for r in accepted_sums)
    targets_str = _format_col_list(sum_targets, limit=3)
    sources_str = _format_col_list(sum_sources, limit=3)
    return PlanQuestion(
        question_id="",
        topic="duplicate_signal",
        text=q_sum_text,
        applies_to=sum_applies,
        rows_affected=sum_rows_aff,
        options=[
            QuestionOption(
                option_id="keep_all",
                label="Keep everything",
                effect="Keeps both the total columns and the part columns in the map",
                edits=[{"column": c, "role": "feature"} for c in sum_applies],
            ),
            QuestionOption(
                option_id="keep_totals",
                label="Keep only the totals",
                effect=f"Keeps {targets_str} and leaves {sources_str} out of the map",
                edits=[
                    *[{"column": t, "role": "feature"} for t in sum_targets],
                    *[{"column": s, "role": "drop"} for s in sum_sources],
                ],
            ),
            QuestionOption(
                option_id="keep_parts",
                label="Keep only the parts",
                effect=f"Keeps {sources_str} and leaves {targets_str} out of the map",
                edits=[
                    *[{"column": t, "role": "drop"} for t in sum_targets],
                    *[{"column": s, "role": "feature"} for s in sum_sources],
                ],
            ),
        ],
        default_option_id="keep_all",
    )


def _build_duplicate_question(
    dup_kept: list[str],
    dup_dropped: list[str],
    accepted_dups: list[Any],
) -> PlanQuestion:
    """Build the ``duplicate_signal`` question for exact ``duplicate`` relations."""
    dup_applies = [*dup_kept, *dup_dropped]
    dup_rows_aff = max(int(r.rows_used) for r in accepted_dups)
    if len(accepted_dups) == 1:
        col_a = accepted_dups[0].sources[0]
        col_b = accepted_dups[0].target
        q_dup_text = (
            f"`{col_b}` has identical values to `{col_a}`, so the map counts "
            f"that signal twice. Keep both columns, or keep only `{col_a}`?"
        )
        keep_both_effect = f"Keeps both `{col_a}` and `{col_b}` in the map"
        keep_one_effect = (
            f"Keeps `{col_a}` and leaves duplicate `{col_b}` out of the map"
        )
    else:
        r0 = accepted_dups[0]
        ex_pair = f"`{r0.target}` = `{r0.sources[0]}`"
        q_dup_text = (
            f"{len(accepted_dups)} columns have identical values to another column "
            f"(e.g. {ex_pair}), so the map counts those signals twice. Keep all of "
            "them, or leave out the duplicates?"
        )
        kept_str = _format_col_list(dup_kept, limit=3)
        dropped_str = _format_col_list(dup_dropped, limit=3)
        keep_both_effect = "Keeps all duplicate columns in the map"
        keep_one_effect = (
            f"Keeps {kept_str} and leaves duplicate {dropped_str} out of the map"
        )
    return PlanQuestion(
        question_id="",
        topic="duplicate_signal",
        text=q_dup_text,
        applies_to=dup_applies,
        rows_affected=dup_rows_aff,
        options=[
            QuestionOption(
                option_id="keep_both",
                label="Keep both",
                effect=keep_both_effect,
                edits=[{"column": c, "role": "feature"} for c in dup_applies],
            ),
            QuestionOption(
                option_id="keep_one",
                label="Keep only one",
                effect=keep_one_effect,
                edits=[
                    *[{"column": c, "role": "feature"} for c in dup_kept],
                    *[{"column": c, "role": "drop"} for c in dup_dropped],
                ],
            ),
        ],
        default_option_id="keep_both",
    )


def build(ctx: DraftContext) -> list[PendingQuestion]:
    """Build up to two ``duplicate_signal`` questions (one for sums, one for duplicates)."""
    (
        sum_targets,
        sum_sources,
        accepted_sums,
        dup_kept,
        dup_dropped,
        accepted_dups,
    ) = _collect_sum_and_duplicate_relations(ctx)
    pending: list[PendingQuestion] = []
    if accepted_sums:
        pending.append(
            (_build_sum_question(sum_targets, sum_sources, accepted_sums), [])
        )
    if accepted_dups:
        pending.append(
            (_build_duplicate_question(dup_kept, dup_dropped, accepted_dups), [])
        )
    return pending
