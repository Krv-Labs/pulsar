"""Per-column default PlanEntry rules (Rules 1-8) for draft_plan."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from pulsar.analysis.preprocessing_evidence import ColumnEvidence
from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import PlanEditError, PlanEntry
from pulsar.mcp.preprocessing_plan.text import _format_missing_reason


def validate_role_designations(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    entity_column: str,
    time_column: str,
) -> tuple[list[str], str, str]:
    """Validate explicit role arguments against *df.columns* and return normalized names."""
    col_set = set(map(str, df.columns))
    target_list = [str(c) for c in target_columns if str(c)]
    entity_col = str(entity_column).strip() if entity_column else ""
    time_col = str(time_column).strip() if time_column else ""

    for label, col_name in (("entity_column", entity_col), ("time_column", time_col)):
        if col_name and col_name not in col_set:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Unknown {label}: {col_name!r}.",
                {
                    label: col_name,
                    "available_columns": [str(c) for c in df.columns[:20]],
                },
            )
    unknown_targets = [c for c in target_list if c not in col_set]
    if unknown_targets:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Unknown target_columns: {unknown_targets!r}.",
            {
                "unknown_target_columns": unknown_targets,
                "available_columns": [str(c) for c in df.columns[:20]],
            },
        )
    return target_list, entity_col, time_col


def _build_mixed_numeric_text_entry(
    col: str,
    ev: ColumnEvidence,
    ctx: DraftContext,
    n_non_num: int,
) -> PlanEntry:
    """Rule 5a: numeric_text with non-numeric tokens -> never silently coerce."""
    top_val = (
        ev.non_numeric_values[0][0] if ev.non_numeric_values else "non-numeric text"
    )
    row_word = "row" if n_non_num == 1 else "rows"
    blank_word = "blank" if ev.n_missing == 1 else "blanks"
    ctx.mixed_value_cols.append(col)
    if ev.n_unique <= ctx.cat_threshold:
        impute_spec = (
            {"method": "fill_value", "value": "(blank)"} if ev.n_missing > 0 else None
        )
        return PlanEntry(
            column=col,
            role="feature",
            impute=impute_spec,
            indicator="none",
            encode="one_hot",
            decided_by="default",
            status="open",
            reason=(
                f"mostly numbers with {top_val!r} ({n_non_num:,} {row_word}; "
                f"{ev.n_missing:,} {blank_word}); kept as {ev.n_unique:,} categories"
            ),
        )
    ind = f"group:{ev.missing_group}" if ev.missing_group else "own"
    return PlanEntry(
        column=col,
        role="feature",
        impute={"method": "fill_median"},
        indicator=ind,
        encode=None,
        decided_by="default",
        status="open",
        reason=(
            f"mostly numbers with {top_val!r} ({n_non_num:,} {row_word}; "
            f"{ev.n_missing:,} {blank_word}); non-numeric text treated as blank "
            "with median fill and blank marker"
        ),
    )


def _build_numeric_text_entry(
    col: str,
    ev: ColumnEvidence,
    ctx: DraftContext,
) -> PlanEntry:
    """Rule 5: numeric_text column."""
    n_non_num = int(ev.n_non_numeric or 0)
    if n_non_num > 0:
        return _build_mixed_numeric_text_entry(col, ev, ctx, n_non_num)

    md = ev.mask_dependency
    partner_ev = ctx.ev_by_col.get(md.partner) if md is not None else None
    if md is not None and partner_ev is not None and partner_ev.kind == "boolean":
        cond_str = (
            f"{md.partner} is 0"
            if md.relation == "missing_when_partner_zero"
            else f"{md.partner} is not 0"
        )
        blank_word = "blank" if ev.n_missing == 1 else "blanks"
        row_word = "row" if ev.n_missing == 1 else "rows"
        ctx.conditional_by_partner.setdefault(md.partner, []).append(col)
        return PlanEntry(
            column=col,
            role="drop",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="open",
            reason=f"{ev.n_missing:,} {blank_word}; blank on the exact {row_word} where {cond_str}",
        )

    ind = (
        (f"group:{ev.missing_group}" if ev.missing_group else "own")
        if ev.n_missing > 0
        else "none"
    )
    blank_word = "blank" if ev.n_missing == 1 else "blanks"
    if ev.n_missing > 0:
        ctx.rule7b_cols.append(col)
    return PlanEntry(
        column=col,
        role="feature",
        impute={"method": "fill_median"},
        indicator=ind,
        encode=None,
        decided_by="default",
        status="resolved",
        reason=(
            f"numeric text ({ev.n_missing:,} {blank_word}); "
            "coerced to numbers with median fill"
        ),
    )


def _build_text_entry(
    col: str,
    ev: ColumnEvidence,
    ctx: DraftContext,
) -> PlanEntry:
    """Rule 6: text categorical column."""
    if ev.n_unique > ctx.cat_threshold:
        return PlanEntry(
            column=col,
            role="drop",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="resolved",
            reason=(
                f"{ev.n_unique:,} distinct text categories (exceeds sqrt({ctx.n_rows:,}) = "
                f"{int(ctx.cat_threshold)}); would add {ev.n_unique:,} columns"
            ),
        )
    impute_spec = (
        {"method": "fill_value", "value": "(blank)"} if ev.n_missing > 0 else None
    )
    blank_word = "blank" if ev.n_missing == 1 else "blanks"
    reason = (
        f"text column with {ev.n_unique:,} categories and {ev.n_missing:,} {blank_word}; "
        "blanks become '(blank)' category"
        if ev.n_missing > 0
        else f"complete text column with {ev.n_unique:,} categories"
    )
    return PlanEntry(
        column=col,
        role="feature",
        impute=impute_spec,
        indicator="none",
        encode="one_hot",
        decided_by="default",
        status="resolved",
        reason=reason,
    )


def _build_numeric_or_boolean_entry(
    col: str,
    ev: ColumnEvidence,
    ctx: DraftContext,
) -> PlanEntry:
    """Rule 7 & 8: numeric / boolean column."""
    if ev.n_missing == 0:
        return PlanEntry(
            column=col,
            role="feature",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="resolved",
            reason=f"complete {ev.kind} column ({ev.n_unique:,} distinct values)",
        )

    md = ev.mask_dependency
    partner_ev = ctx.ev_by_col.get(md.partner) if md is not None else None
    if md is not None and partner_ev is not None and partner_ev.kind == "boolean":
        cond_str = (
            f"{md.partner} is 0"
            if md.relation == "missing_when_partner_zero"
            else f"{md.partner} is not 0"
        )
        blank_word = "blank" if ev.n_missing == 1 else "blanks"
        row_word = "row" if ev.n_missing == 1 else "rows"
        ctx.conditional_by_partner.setdefault(md.partner, []).append(col)
        return PlanEntry(
            column=col,
            role="drop",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="open",
            reason=f"{ev.n_missing:,} {blank_word}; blank on the exact {row_word} where {cond_str}",
        )

    ind = f"group:{ev.missing_group}" if ev.missing_group else "own"
    ctx.rule7b_cols.append(col)
    return PlanEntry(
        column=col,
        role="feature",
        impute={"method": "fill_median"},
        indicator=ind,
        encode=None,
        decided_by="default",
        status="resolved",
        reason=_format_missing_reason(ev, ctx.missing_groups, ctx.ev_by_col),
    )


def _build_explicit_or_constant_entry(
    col: str,
    ev: ColumnEvidence,
    ctx: DraftContext,
) -> PlanEntry | None:
    """Rules 1-2: explicit role designations, all-missing, or constant columns."""
    if col == ctx.entity_col:
        return PlanEntry(
            column=col,
            role="entity",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="agent",
            status="resolved",
            reason="designated as entity column",
        )
    if col == ctx.time_col:
        return PlanEntry(
            column=col,
            role="time",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="agent",
            status="resolved",
            reason="designated as time column",
        )
    if col in ctx.target_set:
        return PlanEntry(
            column=col,
            role="target",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="agent",
            status="resolved",
            reason="designated as target column",
        )
    if ev.n_missing >= ev.n_rows and ev.n_rows > 0:
        return PlanEntry(
            column=col,
            role="drop",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="resolved",
            reason=f"all {ev.n_rows:,} rows are blank",
        )
    if ev.is_constant:
        return PlanEntry(
            column=col,
            role="drop",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="resolved",
            reason="one distinct value; adds no distance information",
        )
    return None


def _build_default_entry(
    col: str,
    ev: ColumnEvidence,
    ctx: DraftContext,
) -> PlanEntry:
    """Dispatch per-column default rules (Rules 1-8) for *col*."""
    early = _build_explicit_or_constant_entry(col, ev, ctx)
    if early is not None:
        return early

    n_observed = ev.n_rows - ev.n_missing
    if (
        ev.kind == "text"
        and ev.unique_ratio == 1.0
        and ev.n_unique == n_observed
        and n_observed > 1
    ):
        return PlanEntry(
            column=col,
            role="id",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="resolved",
            reason=f"unique text value per observed row ({ev.n_unique:,} distinct values)",
        )
    if ev.kind == "datetime_like":
        blank_word = "blank" if ev.n_missing == 1 else "blanks"
        ctx.date_cols.append(col)
        return PlanEntry(
            column=col,
            role="drop",
            impute=None,
            indicator="none",
            encode=None,
            decided_by="default",
            status="open",
            reason=f"date/time text ({ev.n_unique:,} distinct values; {ev.n_missing:,} {blank_word})",
        )
    if ev.kind == "numeric_text":
        return _build_numeric_text_entry(col, ev, ctx)
    if ev.kind == "text":
        return _build_text_entry(col, ev, ctx)
    return _build_numeric_or_boolean_entry(col, ev, ctx)


def populate_default_entries(ctx: DraftContext) -> None:
    """Populate ``ctx.entries`` and rule-specific column lists for every column in ``ctx.df``."""
    for raw_col in ctx.df.columns:
        col = str(raw_col)
        ev = ctx.ev_by_col[col]
        ctx.entries[col] = _build_default_entry(col, ev, ctx)

    for col in ctx.rule7b_cols:
        gid = ctx.ev_by_col[col].missing_group or f"singleton:{col}"
        ctx.grouped_7b.setdefault(gid, []).append(col)
