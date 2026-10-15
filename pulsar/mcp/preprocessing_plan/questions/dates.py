"""Question builder for topic='date_columns'."""

from __future__ import annotations

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)
from pulsar.mcp.preprocessing_plan.text import _format_col_list


def _format_has_date_effect(
    dates_with_blanks: list[str],
    n_complete_dates: int,
    partners: list[str],
) -> str:
    """Build the option effect description for 'has_date'."""
    if dates_with_blanks:
        if partners:
            partner_str = _format_col_list(partners, limit=3)
            has_date_effect = (
                f"Leaves raw dates out of the map and adds a 0/1 blank marker "
                f"(note: this marker would repeat {partner_str})"
            )
        else:
            has_date_effect = (
                "Leaves raw dates out of the map and adds a 0/1 marker for "
                "whether each date is filled in"
            )
        if n_complete_dates > 0:
            complete_clause = (
                "1 date column has no blanks and stays out"
                if n_complete_dates == 1
                else f"{n_complete_dates} date columns have no blanks and stay out"
            )
            has_date_effect = f"{has_date_effect}; {complete_clause}"
        return has_date_effect

    complete_clause = (
        "1 date column has no blanks and stays out"
        if n_complete_dates == 1
        else f"{n_complete_dates} date columns have no blanks and stay out"
    )
    return f"Leaves raw dates out of the map ({complete_clause})"


def build(ctx: DraftContext) -> list[PendingQuestion]:
    """Build the shared ``date_columns`` question when datetime-like columns exist."""
    date_cols = ctx.date_cols
    if not date_cols:
        return []

    ev_by_col = ctx.ev_by_col
    dates_with_blanks = [
        c for c in date_cols if 0 < ev_by_col[c].n_missing < ev_by_col[c].n_rows
    ]
    n_complete_dates = len(date_cols) - len(dates_with_blanks)
    rows_aff = max(
        ev_by_col[c].n_missing
        if ev_by_col[c].n_missing > 0
        else (ev_by_col[c].n_rows - ev_by_col[c].n_missing)
        for c in date_cols
    )
    partners = sorted(
        {
            ev_by_col[c].mask_dependency.partner
            for c in dates_with_blanks
            if ev_by_col[c].mask_dependency is not None
        }
    )
    has_date_effect = _format_has_date_effect(
        dates_with_blanks, n_complete_dates, partners
    )

    ignore_edits = [
        {"column": c, "role": "drop", "indicator": "none"} for c in date_cols
    ]
    has_date_edits = [
        {
            "column": c,
            "role": "drop",
            "indicator": (
                f"group:{ev_by_col[c].missing_group}"
                if ev_by_col[c].missing_group is not None
                else f"group:{c}"
            ),
        }
        for c in dates_with_blanks
    ]
    date_preview = _format_col_list(date_cols, limit=3)
    q_date = PlanQuestion(
        question_id="",
        topic="date_columns",
        text=(
            f"Some columns hold dates ({date_preview}). Should the map ignore "
            "them, or use only whether each date is filled in?"
        ),
        applies_to=list(date_cols),
        rows_affected=int(rows_aff),
        options=[
            QuestionOption(
                option_id="ignore",
                label="Ignore dates",
                effect="Leaves date columns out of the map",
                edits=ignore_edits,
            ),
            QuestionOption(
                option_id="has_date",
                label="Use whether date is filled in",
                effect=has_date_effect,
                edits=has_date_edits,
            ),
        ],
        default_option_id="ignore",
    )
    return [(q_date, list(date_cols))]
