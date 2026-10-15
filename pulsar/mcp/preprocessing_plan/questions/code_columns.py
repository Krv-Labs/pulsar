"""Question builder for topic='code_columns'."""

from __future__ import annotations

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)
from pulsar.mcp.preprocessing_plan.text import _is_count_like_zero_free


def build(ctx: DraftContext) -> list[PendingQuestion]:
    """Build the ``code_columns`` framing question for low-cardinality nonnegative integer columns."""
    ev_by_col = ctx.ev_by_col
    additive_cols = {
        col
        for rel in ctx.evidence.relations
        if rel.kind in ("sum_of", "duplicate")
        for col in (rel.target, *rel.sources)
    }
    code_cols = sorted(
        [
            col
            for col, entry in ctx.entries.items()
            if entry.role == "feature"
            and ev_by_col[col].kind == "numeric"
            and ev_by_col[col].integer_valued is True
            and ev_by_col[col].nonnegative is True
            and 3 <= ev_by_col[col].n_unique <= ctx.cat_threshold
            and not _is_count_like_zero_free(ev_by_col[col])
            and col not in ctx.blank_group_cols
            and col not in additive_cols
        ],
        key=lambda c: (-ev_by_col[c].n_unique, c),
    )
    if not code_cols:
        return []

    top_code_preview = ", ".join(
        f"`{c}`: {ev_by_col[c].n_unique} distinct values" for c in code_cols[:3]
    )
    code_paren = (
        f"e.g. {top_code_preview}"
        if len(code_cols) == 1
        else f"e.g. {top_code_preview}; {len(code_cols)} columns"
    )
    q_code = PlanQuestion(
        question_id="",
        topic="code_columns",
        text=(
            f"Some whole-number columns might be labels rather than amounts "
            f"({code_paren}). For labels, bigger numbers don't mean 'more'. "
            "Are any of these labels or codes?"
        ),
        applies_to=list(code_cols),
        rows_affected=int(ctx.n_rows),
        options=[
            QuestionOption(
                option_id="all_amounts",
                label="All are amounts",
                effect="Keeps all as numeric amounts in the map",
                edits=[],
            )
        ],
        default_option_id="all_amounts",
    )
    return [(q_code, [])]
