"""Wording, column list formatting, and missingness reason helpers."""

from __future__ import annotations

from collections.abc import Sequence

from pulsar.analysis.preprocessing_evidence import ColumnEvidence


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """Return *singular* when ``n == 1``, else *plural* (or ``singular + 's'``)."""
    if n == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


def _is_count_like_zero_free(ev: ColumnEvidence) -> bool:
    """Return True iff *ev* is a numeric count-like column with blanks and no 0s."""
    return (
        ev.kind == "numeric"
        and ev.n_missing > 0
        and ev.integer_valued is True
        and ev.nonnegative is True
        and ev.n_zero == 0
    )


def _format_col_list(cols: Sequence[str], limit: int = 3) -> str:
    """Format a sequence of column names in backticks, capped at *limit* names."""
    items = [f"`{c}`" for c in cols]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) <= limit:
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"
    return f"{', '.join(items[:limit])}, and {len(items) - limit} more"


def _format_missing_reason(
    ev: ColumnEvidence,
    missing_groups: dict[str, list[str]],
    ev_by_col: dict[str, ColumnEvidence] | None = None,
) -> str:
    blank_word = _plural(ev.n_missing, "blank")
    parts = [f"{ev.n_missing:,} {blank_word}"]
    if ev.n_zero == 0 and ev.nonnegative is True:
        parts.append("no recorded 0")
    if ev.missing_group is not None and ev.missing_group in missing_groups:
        n_others = max(0, len(missing_groups[ev.missing_group]) - 1)
        if n_others > 0:
            noun = _plural(n_others, "column")
            parts.append(
                f"blank on the same rows as {n_others} other {noun} ({ev.missing_group})"
            )
    if ev.mask_dependency is not None:
        md = ev.mask_dependency
        partner_ev = ev_by_col.get(md.partner) if ev_by_col is not None else None
        if partner_ev is None or partner_ev.kind != "boolean":
            cond_str = (
                f"{md.partner} is 0"
                if md.relation == "missing_when_partner_zero"
                else f"{md.partner} is not 0"
            )
            row_word = _plural(ev.n_missing, "row")
            parts.append(f"blank on the exact {row_word} where {cond_str}")
    return "; ".join(parts)
