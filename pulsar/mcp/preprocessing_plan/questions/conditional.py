"""Question builder for topic='conditional_column'."""

from __future__ import annotations

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)
from pulsar.mcp.preprocessing_plan.text import _format_col_list


def _build_conditional_options(
    ctx: DraftContext,
    partner: str,
    cols: list[str],
    cols_preview: str,
) -> list[QuestionOption]:
    """Construct the three QuestionOptions for a ``conditional_column`` question."""
    ev_by_col = ctx.ev_by_col
    return [
        QuestionOption(
            option_id="leave_out",
            label="Leave out",
            effect=(
                f"Leaves {cols_preview} out of the map because the blank "
                f"pattern largely repeats `{partner}`"
            ),
            edits=[
                {
                    "column": c,
                    "role": "drop",
                    "impute_method": None,
                    "indicator": "none",
                }
                for c in cols
            ],
        ),
        QuestionOption(
            option_id="blank_as_zero",
            label="Treat blanks as 0",
            effect=(
                "Keeps in the map with blanks set to 0 and no blank marker "
                f"(largely repeats `{partner}`)"
            ),
            edits=[
                {
                    "column": c,
                    "role": "feature",
                    "impute_method": "fill_value",
                    "impute_value": 0,
                    "indicator": "none",
                }
                for c in cols
            ],
        ),
        QuestionOption(
            option_id="estimate_and_mark",
            label="Fill with median and mark blanks",
            effect=(
                "Keeps in the map with blanks filled by the median and adds "
                f"a 0/1 blank marker that largely repeats `{partner}`"
            ),
            edits=[
                {
                    "column": c,
                    "role": "feature",
                    "impute_method": "fill_median",
                    "indicator": (
                        f"group:{ev_by_col[c].missing_group}"
                        if ev_by_col[c].missing_group
                        else "own"
                    ),
                }
                for c in cols
            ],
        ),
    ]


def build(ctx: DraftContext) -> list[PendingQuestion]:
    """Build one ``conditional_column`` question per boolean event partner."""
    ev_by_col = ctx.ev_by_col
    pending: list[PendingQuestion] = []

    for partner, cols in ctx.conditional_by_partner.items():
        first_md = ev_by_col[cols[0]].mask_dependency
        assert first_md is not None
        when_phrase = (
            f"when `{partner}` is not 0"
            if first_md.relation == "missing_when_partner_zero"
            else f"when `{partner}` is 0"
        )
        cols_preview = _format_col_list(cols, limit=3)
        if len(cols) == 1:
            q_text = (
                f"`{cols[0]}` only has a value {when_phrase} — it is blank for "
                "everyone else. Should the map use it?"
            )
        else:
            q_text = (
                f"{len(cols)} columns ({cols_preview}) only have values {when_phrase} "
                "— they are blank for everyone else. Should the map use them?"
            )
        rows_aff = max(ev_by_col[c].n_missing for c in cols)
        q_cond = PlanQuestion(
            question_id="",
            topic="conditional_column",
            text=q_text,
            applies_to=list(cols),
            rows_affected=int(rows_aff),
            options=_build_conditional_options(ctx, partner, cols, cols_preview),
            default_option_id="leave_out",
        )
        pending.append((q_cond, list(cols)))

    return pending
