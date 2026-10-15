"""Question builder for topic='blank_meaning'."""

from __future__ import annotations

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)
from pulsar.mcp.preprocessing_plan.text import (
    _format_col_list,
    _is_count_like_zero_free,
)


def _format_blank_meaning_copy(
    cols: list[str],
    zero_free_cols: list[str],
    *,
    rows_aff: int,
    n_with_zero: int,
    partner_sentence: str,
) -> tuple[str, str, str]:
    """Return ``(q_text, none_effect, unknown_effect)`` for a ``blank_meaning`` question."""
    row_word = "row" if rows_aff == 1 else "rows"
    if len(cols) == 1:
        q_text = (
            f"`{cols[0]}` is blank for {rows_aff:,} {row_word} and never records a 0."
            f"{partner_sentence} Does a blank mean none (zero), or that the value is unknown?"
        )
        none_effect = "Blanks become 0; no extra marker column"
        unknown_effect = (
            "Blanks are filled with the median of observed rows, plus a 0/1 "
            "marker column for which rows were blank"
        )
        return q_text, none_effect, unknown_effect

    zf_str = _format_col_list(zero_free_cols, limit=3)
    verb = "never records a 0" if len(zero_free_cols) == 1 else "never record a 0"
    same_rows_phrase = (
        f"Both are blank for the same {rows_aff:,} {row_word}."
        if len(cols) == 2
        else f"All {len(cols)} are blank for the same {rows_aff:,} {row_word}."
    )
    if n_with_zero > 0:
        other_phrase = (
            "1 other column sometimes does"
            if n_with_zero == 1
            else f"the other {n_with_zero} columns sometimes do"
        )
        q_text = (
            f"{zf_str} {verb} ({other_phrase}). {same_rows_phrase}"
            f"{partner_sentence} Does a blank mean none (zero), or that the value is unknown?"
        )
    else:
        q_text = (
            f"{zf_str} {verb}. {same_rows_phrase}"
            f"{partner_sentence} Does a blank mean none (zero), or that the value is unknown?"
        )
    none_effect = (
        "Both columns get 0 where blank; no extra marker column"
        if len(cols) == 2
        else f"All {len(cols)} columns get 0 where blank; no extra marker column"
    )
    unknown_effect = (
        "Blanks are filled with the median of observed rows, plus one "
        "shared 0/1 marker column for the blank pattern"
    )
    return q_text, none_effect, unknown_effect


def _build_blank_meaning_options(
    ctx: DraftContext,
    cols: list[str],
    none_effect: str,
    unknown_effect: str,
) -> list[QuestionOption]:
    """Construct the two QuestionOptions for ``blank_meaning``."""
    ev_by_col = ctx.ev_by_col
    return [
        QuestionOption(
            option_id="none",
            label="Blank means zero (none)",
            effect=none_effect,
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
            option_id="unknown",
            label="Blank means unknown",
            effect=unknown_effect,
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
    """Build ``blank_meaning`` questions for missingness groups with count-like zero-free columns."""
    ev_by_col = ctx.ev_by_col
    pending: list[PendingQuestion] = []

    for _gid, cols in ctx.grouped_7b.items():
        count_like_cols = [c for c in cols if _is_count_like_zero_free(ev_by_col[c])]
        if not count_like_cols:
            continue
        ctx.blank_group_cols.update(cols)
        for c in cols:
            ctx.entries[c].status = "open"
        rows_aff = ev_by_col[cols[0]].n_missing

        zero_free_cols = list(count_like_cols) + [
            c for c in cols if ev_by_col[c].n_zero == 0 and c not in count_like_cols
        ]
        n_with_zero = len(cols) - len(zero_free_cols)

        dep_md = next(
            (
                ev_by_col[c].mask_dependency
                for c in cols
                if ev_by_col[c].mask_dependency is not None
            ),
            None,
        )
        if dep_md is not None:
            cond_val = (
                "0" if dep_md.relation == "missing_when_partner_zero" else "not 0"
            )
            subj = "It is" if len(cols) == 1 else "They are"
            partner_sentence = (
                f" {subj} blank exactly where `{dep_md.partner}` is {cond_val}."
            )
        else:
            partner_sentence = ""

        q_text, none_effect, unknown_effect = _format_blank_meaning_copy(
            cols,
            zero_free_cols,
            rows_aff=rows_aff,
            n_with_zero=n_with_zero,
            partner_sentence=partner_sentence,
        )
        q_blank = PlanQuestion(
            question_id="",
            topic="blank_meaning",
            text=q_text,
            applies_to=list(cols),
            rows_affected=int(rows_aff),
            options=_build_blank_meaning_options(
                ctx, cols, none_effect, unknown_effect
            ),
            default_option_id="unknown",
        )
        pending.append((q_blank, list(cols)))

    return pending
