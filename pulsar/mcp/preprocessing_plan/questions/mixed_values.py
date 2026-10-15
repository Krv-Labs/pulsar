"""Question builder for topic='mixed_values'."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)
from pulsar.mcp.preprocessing_plan.text import _format_col_list


def _detect_single_numeric_value(
    df: pd.DataFrame,
    mixed_value_cols: list[str],
) -> float | None:
    """Return the shared numeric value if all numeric tokens across *mixed_value_cols* are identical."""
    parsed_ranges: list[tuple[float, float] | None] = []
    for c in mixed_value_cols:
        s_clean = (
            df[c]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"(?<=\d),(?=\d{3}(?!\d))", "", regex=True)
        )
        s_num = pd.to_numeric(s_clean[s_clean != ""], errors="coerce").dropna()
        s_num_arr = s_num.to_numpy(dtype=np.float64)
        s_num_arr = s_num_arr[np.isfinite(s_num_arr)]
        if len(s_num_arr) > 0:
            parsed_ranges.append((float(s_num_arr.min()), float(s_num_arr.max())))
        else:
            parsed_ranges.append(None)

    if (
        parsed_ranges
        and parsed_ranges[0] is not None
        and all(
            r is not None and r[0] == r[1] == parsed_ranges[0][0] for r in parsed_ranges
        )
    ):
        return parsed_ranges[0][0]
    return None


def _format_mixed_question_text(
    mixed_value_cols: list[str],
    *,
    c_top: str,
    top_val: str,
    top_n: int,
    total_non_num: int,
    single_num_val: float | None,
    mostly_num_all: bool,
) -> str:
    """Format the question prompt text for ``mixed_values``."""
    row_word = "row" if top_n == 1 else "rows"
    entry_word = "entry" if total_non_num == 1 else "entries"
    if len(mixed_value_cols) == 1:
        if single_num_val is not None:
            num_str = (
                str(int(single_num_val))
                if float(single_num_val).is_integer()
                else str(single_num_val)
            )
            return (
                f"`{c_top}` holds the number {num_str} and the value {top_val!r} "
                f"({top_n:,} {row_word}). Should those values count as their own "
                "category, or as unreadable entries treated like blanks?"
            )
        if mostly_num_all:
            return (
                f"`{c_top}` mostly holds numbers but also the value {top_val!r} "
                f"({top_n:,} {row_word}). Should those values count as their own "
                "category, or as unreadable entries treated like blanks?"
            )
        return (
            f"`{c_top}` holds numbers and the value {top_val!r} "
            f"({top_n:,} {row_word}). Should those values count as their own "
            "category, or as unreadable entries treated like blanks?"
        )

    if single_num_val is not None:
        num_str = (
            str(int(single_num_val))
            if float(single_num_val).is_integer()
            else str(single_num_val)
        )
        return (
            f"{len(mixed_value_cols)} columns hold the number {num_str} and "
            f"the value {top_val!r} ({total_non_num:,} {entry_word} in total; "
            f"e.g. `{c_top}`: {top_n:,}). Should those values count as their "
            "own category, or as unreadable entries treated like blanks?"
        )
    if mostly_num_all:
        return (
            f"{len(mixed_value_cols)} columns mostly hold numbers but also "
            f"the value {top_val!r} ({total_non_num:,} {entry_word} in total; "
            f"e.g. `{c_top}`: {top_n:,}). Should those values count as their "
            "own category, or as unreadable entries treated like blanks?"
        )
    return (
        f"{len(mixed_value_cols)} columns hold numbers and the value "
        f"{top_val!r} ({total_non_num:,} {entry_word} in total; e.g. "
        f"`{c_top}`: {top_n:,}). Should those values count as their own "
        "category, or as unreadable entries treated like blanks?"
    )


def _build_mixed_options(
    ctx: DraftContext,
    mixed_value_cols: list[str],
    *,
    total_non_num: int,
    top_val: str,
) -> list[QuestionOption]:
    """Construct the three QuestionOptions for ``mixed_values``."""
    ev_by_col = ctx.ev_by_col
    entry_word = "entry" if total_non_num == 1 else "entries"
    verb_phrase = (
        "becomes a blank and is estimated"
        if total_non_num == 1
        else "become blanks and are estimated"
    )
    cols_preview = _format_col_list(mixed_value_cols, limit=3)
    as_cat_edits = [
        {
            "column": c,
            "role": "feature",
            "impute_method": ("fill_value" if ev_by_col[c].n_missing > 0 else None),
            **({"impute_value": "(blank)"} if ev_by_col[c].n_missing > 0 else {}),
            "indicator": "none",
            "encode": "one_hot",
        }
        for c in mixed_value_cols
    ]
    as_blank_edits = [
        {
            "column": c,
            "role": "feature",
            "impute_method": "fill_median",
            "indicator": (
                f"group:{ev_by_col[c].missing_group}"
                if ev_by_col[c].missing_group is not None
                else "own"
            ),
            "encode": None,
        }
        for c in mixed_value_cols
    ]
    leave_out_edits = [
        {
            "column": c,
            "role": "drop",
            "impute_method": None,
            "indicator": "none",
            "encode": None,
        }
        for c in mixed_value_cols
    ]
    return [
        QuestionOption(
            option_id="as_categories",
            label="Treat as categories",
            effect=(
                "Each distinct value (including non-numeric text) gets its "
                "own 0/1 category column; blanks become '(blank)'"
            ),
            edits=as_cat_edits,
        ),
        QuestionOption(
            option_id="as_blank",
            label="Treat non-numeric text as blank",
            effect=(
                f"{total_non_num:,} {top_val!r} {entry_word} {verb_phrase} with "
                "the median of observed numbers, plus a 0/1 blank marker"
            ),
            edits=as_blank_edits,
        ),
        QuestionOption(
            option_id="leave_out",
            label="Leave out",
            effect=f"Leaves {cols_preview} out of the map",
            edits=leave_out_edits,
        ),
    ]


def build(ctx: DraftContext) -> list[PendingQuestion]:
    """Build the shared ``mixed_values`` question for numeric_text columns with non-numeric tokens."""
    mixed_value_cols = ctx.mixed_value_cols
    if not mixed_value_cols:
        return []

    ev_by_col = ctx.ev_by_col
    total_non_num = sum(int(ev_by_col[c].n_non_numeric or 0) for c in mixed_value_cols)
    c_top = max(
        mixed_value_cols,
        key=lambda c: int(ev_by_col[c].n_non_numeric or 0),
    )
    ev_top = ev_by_col[c_top]
    top_val, _ = (
        ev_top.non_numeric_values[0]
        if ev_top.non_numeric_values
        else ("", int(ev_top.n_non_numeric or 0))
    )
    top_n = int(ev_top.n_non_numeric or 0)
    single_num_val = _detect_single_numeric_value(ctx.df, mixed_value_cols)
    mostly_num_all = all(
        (
            ev_by_col[c].n_rows
            - ev_by_col[c].n_missing
            - int(ev_by_col[c].n_non_numeric or 0)
        )
        > int(ev_by_col[c].n_non_numeric or 0)
        for c in mixed_value_cols
    )
    q_mixed_text = _format_mixed_question_text(
        mixed_value_cols,
        c_top=c_top,
        top_val=top_val,
        top_n=top_n,
        total_non_num=total_non_num,
        single_num_val=single_num_val,
        mostly_num_all=mostly_num_all,
    )
    rows_aff = max(
        int(ev_by_col[c].n_non_numeric or 0) + ev_by_col[c].n_missing
        for c in mixed_value_cols
    )
    default_opt = (
        "as_categories"
        if all(ev_by_col[c].n_unique <= ctx.cat_threshold for c in mixed_value_cols)
        else "as_blank"
    )
    q_mixed = PlanQuestion(
        question_id="",
        topic="mixed_values",
        text=q_mixed_text,
        applies_to=list(mixed_value_cols),
        rows_affected=int(rows_aff),
        options=_build_mixed_options(
            ctx,
            mixed_value_cols,
            total_non_num=total_non_num,
            top_val=top_val,
        ),
        default_option_id=default_opt,
    )
    return [(q_mixed, list(mixed_value_cols))]
