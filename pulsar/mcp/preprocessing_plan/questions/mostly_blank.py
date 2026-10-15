"""Question builder for topic='mostly_blank'."""

from __future__ import annotations

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)
from pulsar.mcp.preprocessing_plan.text import _format_col_list


def _format_mostly_blank_copy(
    mostly_blank_cols: list[str],
    pct0: int,
) -> tuple[str, str]:
    """Return ``(q_mb_text, has_value_effect)`` for a ``mostly_blank`` question."""
    c0 = mostly_blank_cols[0]
    if len(mostly_blank_cols) == 1:
        q_mb_text = (
            f"`{c0}` is blank for {pct0}% of rows, and the data doesn't show why. "
            "Leave it out, keep only whether it has a value, or estimate the missing values?"
        )
        has_value_effect = (
            f"Leaves `{c0}` out of the map and adds a 0/1 marker for whether "
            "it has a value"
        )
        return q_mb_text, has_value_effect

    others = mostly_blank_cols[1:]
    if len(others) == 1:
        other_clause = (
            f"and 1 other column, `{others[0]}`, is blank for more than half of rows"
        )
    elif len(others) == 2:
        other_clause = (
            "and 2 other columns are blank for more than half of rows: "
            f"`{others[0]}` and `{others[1]}`"
        )
    else:
        other_clause = (
            f"and {len(others)} other columns are blank for more than "
            f"half of rows, including `{others[0]}` and `{others[1]}`"
        )
    q_mb_text = (
        f"`{c0}` is blank for {pct0}% of rows ({other_clause}), and the data "
        "doesn't show why. Leave them out, keep only whether each has a value, "
        "or estimate the missing values?"
    )
    mb_preview = _format_col_list(mostly_blank_cols, limit=3)
    has_value_effect = (
        f"Leaves {mb_preview} out of the map and adds 0/1 markers for "
        "whether values are filled in"
    )
    return q_mb_text, has_value_effect


def _build_mostly_blank_options(
    ctx: DraftContext,
    mostly_blank_cols: list[str],
    *,
    mb_preview: str,
    has_value_effect: str,
    est_pct: int,
) -> list[QuestionOption]:
    """Construct the three QuestionOptions for ``mostly_blank``."""
    ev_by_col = ctx.ev_by_col
    mb_leave_out_edits = [
        {
            "column": c,
            "role": "drop",
            "impute_method": None,
            "indicator": "none",
        }
        for c in mostly_blank_cols
    ]
    mb_has_value_edits = [
        {
            "column": c,
            "role": "drop",
            "impute_method": None,
            "indicator": (
                f"group:{ev_by_col[c].missing_group}"
                if ev_by_col[c].missing_group is not None
                else f"group:{c}"
            ),
        }
        for c in mostly_blank_cols
    ]
    mb_estimate_edits = [
        {
            "column": c,
            "role": "feature",
            "impute_method": "fill_median",
            "indicator": (
                f"group:{ev_by_col[c].missing_group}"
                if ev_by_col[c].missing_group is not None
                else "own"
            ),
        }
        for c in mostly_blank_cols
    ]
    return [
        QuestionOption(
            option_id="leave_out",
            label="Leave out",
            effect=f"Leaves {mb_preview} out of the map",
            edits=mb_leave_out_edits,
        ),
        QuestionOption(
            option_id="has_value",
            label="Keep only whether value is filled in",
            effect=has_value_effect,
            edits=mb_has_value_edits,
        ),
        QuestionOption(
            option_id="estimate_and_mark",
            label="Fill with median and mark blanks",
            effect=(
                "Keeps in the map with blanks filled by the median of "
                "observed rows and a 0/1 blank marker "
                f"(about {est_pct}% of these values would be estimates)"
            ),
            edits=mb_estimate_edits,
        ),
    ]


def build(ctx: DraftContext) -> list[PendingQuestion]:
    """Build the ``mostly_blank`` question for remaining >50% blank numeric/boolean columns."""
    ev_by_col = ctx.ev_by_col
    mostly_blank_raw: list[str] = []
    for _gid, cols in ctx.grouped_7b.items():
        if cols[0] in ctx.blank_group_cols:
            continue
        if ev_by_col[cols[0]].n_missing > ctx.n_rows - ev_by_col[cols[0]].n_missing:
            mostly_blank_raw.extend(cols)

    mostly_blank_cols = sorted(
        mostly_blank_raw,
        key=lambda c: (-ev_by_col[c].missing_fraction, c),
    )
    if not mostly_blank_cols:
        return []

    for c in mostly_blank_cols:
        ctx.entries[c].role = "drop"
        ctx.entries[c].impute = None
        ctx.entries[c].indicator = "none"
        ctx.entries[c].encode = None
        ctx.entries[c].decided_by = "default"
        ctx.entries[c].status = "open"

    c0 = mostly_blank_cols[0]
    pct0 = max(51, round(100.0 * ev_by_col[c0].missing_fraction))
    q_mb_text, has_value_effect = _format_mostly_blank_copy(mostly_blank_cols, pct0)
    mb_preview = _format_col_list(mostly_blank_cols, limit=3)
    mb_rows_aff = max(ev_by_col[c].n_missing for c in mostly_blank_cols)
    est_pct = max(
        51,
        round(
            100.0
            * sum(ev_by_col[c].missing_fraction for c in mostly_blank_cols)
            / len(mostly_blank_cols)
        ),
    )
    q_mb = PlanQuestion(
        question_id="",
        topic="mostly_blank",
        text=q_mb_text,
        applies_to=list(mostly_blank_cols),
        rows_affected=int(mb_rows_aff),
        options=_build_mostly_blank_options(
            ctx,
            mostly_blank_cols,
            mb_preview=mb_preview,
            has_value_effect=has_value_effect,
            est_pct=est_pct,
        ),
        default_option_id="leave_out",
    )
    return [(q_mb, list(mostly_blank_cols))]
