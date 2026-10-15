"""Evidence-first preprocessing plan ledger for Pulsar MCP.

This module is pure Python/pandas/numpy (no FastMCP imports) and turns
measured dataset properties from :mod:`pulsar.analysis.preprocessing_evidence`
into a structured, revisioned :class:`PreprocessingPlan` containing:

- Per-column :class:`PlanEntry` records with explicit provenance
  (``decided_by="default" | "agent" | "user"``) and plain measurement reasons.
- Plain-language :class:`PlanQuestion` items with counterfactual geometry
  stakes (k-NN neighborhood overlap) so agents and users can focus on
  decisions that materially move the map.
- Deterministic conversion to ``PulsarConfig``'s ``preprocessing:`` dictionary
  and YAML block, including readable shared missingness indicator groups.
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Any, Literal
import uuid
import warnings

import numpy as np
import pandas as pd
import yaml

from pulsar.analysis.preprocessing_evidence import (
    ColumnEvidence,
    PreprocessingEvidence,
    measure_preprocessing_evidence,
    neighborhood_overlap,
)
from pulsar.config import load_config
from pulsar.preprocessing import preprocess_dataframe

Role = Literal["feature", "drop", "id", "entity", "time", "target"]
DecidedBy = Literal["default", "agent", "user"]
QuestionTopic = Literal[
    "blank_meaning",
    "code_columns",
    "conditional_column",
    "date_columns",
    "duplicate_signal",
    "mixed_values",
    "outcome_columns",
]

VALID_ROLES: frozenset[str] = frozenset(
    {"feature", "drop", "id", "entity", "time", "target"}
)
VALID_DECIDED_BY: frozenset[str] = frozenset({"default", "agent", "user"})
VALID_IMPUTE_METHODS: frozenset[str] = frozenset(
    {
        "fill_median",
        "fill_mean",
        "fill_mode",
        "fill_value",
        "sample_normal",
        "sample_categorical",
    }
)
VALID_ENCODE_METHODS: frozenset[str] = frozenset({"one_hot"})
_EDIT_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "column",
        "role",
        "impute",
        "impute_method",
        "impute_value",
        "indicator",
        "encode",
    }
)


class PlanEditError(ValueError):
    """Structured error raised when plan parameters, answers, or edits are invalid."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details or {}


@dataclass
class PlanEntry:
    column: str
    role: Role
    impute: dict[str, Any] | None
    indicator: str  # "none" | "own" | "group:<gid>"
    encode: str | None  # "one_hot" | None
    decided_by: DecidedBy
    status: Literal["resolved", "open"]
    reason: str
    question_id: str | None = None

    @property
    def impute_method(self) -> str | None:
        return self.impute.get("method") if self.impute else None

    @property
    def impute_value(self) -> Any:
        return self.impute.get("value") if self.impute else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "role": self.role,
            "impute": dict(self.impute) if self.impute is not None else None,
            "indicator": self.indicator,
            "encode": self.encode,
            "decided_by": self.decided_by,
            "status": self.status,
            "reason": self.reason,
            "question_id": self.question_id,
        }


@dataclass
class QuestionOption:
    option_id: str
    label: str
    effect: str
    edits: list[dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "label": self.label,
            "effect": self.effect,
        }

    def to_full_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "label": self.label,
            "effect": self.effect,
            "edits": [dict(e) for e in self.edits],
        }


@dataclass
class PlanQuestion:
    question_id: str
    topic: QuestionTopic
    text: str
    applies_to: list[str]
    rows_affected: int
    options: list[QuestionOption]
    default_option_id: str | None
    stakes: dict[str, Any] | None = None
    status: Literal["open", "answered"] = "open"
    answer: dict[str, Any] | None = None

    @property
    def columns(self) -> list[str]:
        return self.applies_to


@dataclass
class PreprocessingPlan:
    plan_id: str
    dataset_id: str
    revision: int
    entries: dict[str, PlanEntry]
    questions: dict[str, PlanQuestion]
    missing_groups: dict[str, list[str]]
    evidence: PreprocessingEvidence
    created_at: str
    cost: dict[str, Any] = field(default_factory=dict)
    target_proxies: list[dict[str, Any]] = field(default_factory=list)


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
    blank_word = "blank" if ev.n_missing == 1 else "blanks"
    parts = [f"{ev.n_missing:,} {blank_word}"]
    if ev.n_zero == 0 and ev.nonnegative is True:
        parts.append("no recorded 0")
    if ev.missing_group is not None and ev.missing_group in missing_groups:
        n_others = max(0, len(missing_groups[ev.missing_group]) - 1)
        if n_others > 0:
            noun = "column" if n_others == 1 else "columns"
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
            row_word = "row" if ev.n_missing == 1 else "rows"
            parts.append(f"blank on the exact {row_word} where {cond_str}")
    return "; ".join(parts)


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

    if evidence is None:
        evidence = measure_preprocessing_evidence(df)
    ev_by_col: dict[str, ColumnEvidence] = {c.name: c for c in evidence.columns}
    missing_groups: dict[str, list[str]] = {
        mg.group_id: list(mg.columns) for mg in evidence.missing_groups
    }

    entries: dict[str, PlanEntry] = {}
    date_cols: list[str] = []
    mixed_value_cols: list[str] = []
    conditional_by_partner: dict[str, list[str]] = {}
    rule7b_cols: list[str] = []

    target_set = set(target_list)
    n_rows = evidence.n_rows
    cat_threshold = math.sqrt(max(1, n_rows))

    for raw_col in df.columns:
        col = str(raw_col)
        ev = ev_by_col[col]

        # Rule 1: explicit role designations
        if col == entity_col:
            entries[col] = PlanEntry(
                column=col,
                role="entity",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="agent",
                status="resolved",
                reason="designated as entity column",
            )
            continue
        if col == time_col:
            entries[col] = PlanEntry(
                column=col,
                role="time",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="agent",
                status="resolved",
                reason="designated as time column",
            )
            continue
        if col in target_set:
            entries[col] = PlanEntry(
                column=col,
                role="target",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="agent",
                status="resolved",
                reason="designated as target column",
            )
            continue

        # Rule 2: all-missing or constant
        if ev.n_missing >= ev.n_rows and ev.n_rows > 0:
            entries[col] = PlanEntry(
                column=col,
                role="drop",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="resolved",
                reason=f"all {ev.n_rows:,} rows are blank",
            )
            continue
        if ev.is_constant:
            entries[col] = PlanEntry(
                column=col,
                role="drop",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="resolved",
                reason="one distinct value; adds no distance information",
            )
            continue

        # Rule 3: text unique identifier
        n_observed = ev.n_rows - ev.n_missing
        if (
            ev.kind == "text"
            and ev.unique_ratio == 1.0
            and ev.n_unique == n_observed
            and n_observed > 1
        ):
            entries[col] = PlanEntry(
                column=col,
                role="id",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="resolved",
                reason=f"unique text value per observed row ({ev.n_unique:,} distinct values)",
            )
            continue

        # Rule 4: datetime_like -> drop, open, shared date_columns question
        if ev.kind == "datetime_like":
            blank_word = "blank" if ev.n_missing == 1 else "blanks"
            entries[col] = PlanEntry(
                column=col,
                role="drop",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="open",
                reason=f"date/time text ({ev.n_unique:,} distinct values; {ev.n_missing:,} {blank_word})",
            )
            date_cols.append(col)
            continue

        # Rule 5: numeric_text -> never silently coerce when n_non_numeric > 0
        if ev.kind == "numeric_text":
            n_non_num = int(ev.n_non_numeric or 0)
            if n_non_num > 0:
                top_val = (
                    ev.non_numeric_values[0][0]
                    if ev.non_numeric_values
                    else "non-numeric text"
                )
                row_word = "row" if n_non_num == 1 else "rows"
                blank_word = "blank" if ev.n_missing == 1 else "blanks"
                if ev.n_unique <= cat_threshold:
                    impute_spec = (
                        {"method": "fill_value", "value": "(blank)"}
                        if ev.n_missing > 0
                        else None
                    )
                    entries[col] = PlanEntry(
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
                else:
                    ind = f"group:{ev.missing_group}" if ev.missing_group else "own"
                    entries[col] = PlanEntry(
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
                mixed_value_cols.append(col)
                continue

            md = ev.mask_dependency
            partner_ev = ev_by_col.get(md.partner) if md is not None else None
            if (
                md is not None
                and partner_ev is not None
                and partner_ev.kind == "boolean"
            ):
                cond_str = (
                    f"{md.partner} is 0"
                    if md.relation == "missing_when_partner_zero"
                    else f"{md.partner} is not 0"
                )
                blank_word = "blank" if ev.n_missing == 1 else "blanks"
                row_word = "row" if ev.n_missing == 1 else "rows"
                entries[col] = PlanEntry(
                    column=col,
                    role="drop",
                    impute=None,
                    indicator="none",
                    encode=None,
                    decided_by="default",
                    status="open",
                    reason=f"{ev.n_missing:,} {blank_word}; blank on the exact {row_word} where {cond_str}",
                )
                conditional_by_partner.setdefault(md.partner, []).append(col)
            else:
                ind = (
                    (f"group:{ev.missing_group}" if ev.missing_group else "own")
                    if ev.n_missing > 0
                    else "none"
                )
                blank_word = "blank" if ev.n_missing == 1 else "blanks"
                entries[col] = PlanEntry(
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
            continue

        # Rule 6: text categorical
        if ev.kind == "text":
            if ev.n_unique > cat_threshold:
                entries[col] = PlanEntry(
                    column=col,
                    role="drop",
                    impute=None,
                    indicator="none",
                    encode=None,
                    decided_by="default",
                    status="resolved",
                    reason=(
                        f"{ev.n_unique:,} distinct text categories (exceeds sqrt({n_rows:,}) = "
                        f"{int(cat_threshold)}); would add {ev.n_unique:,} columns"
                    ),
                )
            else:
                impute_spec = (
                    {"method": "fill_value", "value": "(blank)"}
                    if ev.n_missing > 0
                    else None
                )
                blank_word = "blank" if ev.n_missing == 1 else "blanks"
                reason = (
                    f"text column with {ev.n_unique:,} categories and {ev.n_missing:,} {blank_word}; "
                    "blanks become '(blank)' category"
                    if ev.n_missing > 0
                    else f"complete text column with {ev.n_unique:,} categories"
                )
                entries[col] = PlanEntry(
                    column=col,
                    role="feature",
                    impute=impute_spec,
                    indicator="none",
                    encode="one_hot",
                    decided_by="default",
                    status="resolved",
                    reason=reason,
                )
            continue

        # Rule 7 & 8: numeric / boolean
        if ev.n_missing == 0:
            entries[col] = PlanEntry(
                column=col,
                role="feature",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="resolved",
                reason=f"complete {ev.kind} column ({ev.n_unique:,} distinct values)",
            )
            continue

        # Rule 7a: mask_dependency on a boolean event partner -> conditional_column
        md = ev.mask_dependency
        partner_ev = ev_by_col.get(md.partner) if md is not None else None
        if md is not None and partner_ev is not None and partner_ev.kind == "boolean":
            cond_str = (
                f"{md.partner} is 0"
                if md.relation == "missing_when_partner_zero"
                else f"{md.partner} is not 0"
            )
            blank_word = "blank" if ev.n_missing == 1 else "blanks"
            row_word = "row" if ev.n_missing == 1 else "rows"
            entries[col] = PlanEntry(
                column=col,
                role="drop",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="open",
                reason=f"{ev.n_missing:,} {blank_word}; blank on the exact {row_word} where {cond_str}",
            )
            conditional_by_partner.setdefault(md.partner, []).append(col)
            continue

        # Rule 7b: standard missing numeric/boolean (including non-boolean mask_dependency)
        ind = f"group:{ev.missing_group}" if ev.missing_group else "own"
        entries[col] = PlanEntry(
            column=col,
            role="feature",
            impute={"method": "fill_median"},
            indicator=ind,
            encode=None,
            decided_by="default",
            status="resolved",
            reason=_format_missing_reason(ev, missing_groups, ev_by_col),
        )
        rule7b_cols.append(col)

    # Draft questions (before final sorting and q1..qN numbering)
    pending_questions: list[tuple[PlanQuestion, list[str]]] = []

    # Question type 4: date_columns (one shared question for all datetime_like columns)
    if date_cols:
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
        else:
            complete_clause = (
                "1 date column has no blanks and stays out"
                if n_complete_dates == 1
                else f"{n_complete_dates} date columns have no blanks and stay out"
            )
            has_date_effect = f"Leaves raw dates out of the map ({complete_clause})"

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
        pending_questions.append((q_date, list(date_cols)))

    # Question type 5: mixed_values (one shared question for all numeric_text columns with n_non_numeric > 0)
    if mixed_value_cols:
        total_non_num = sum(
            int(ev_by_col[c].n_non_numeric or 0) for c in mixed_value_cols
        )
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
        row_word = "row" if top_n == 1 else "rows"
        entry_word = "entry" if total_non_num == 1 else "entries"
        verb_phrase = (
            "becomes a blank and is estimated"
            if total_non_num == 1
            else "become blanks and are estimated"
        )
        cols_preview = _format_col_list(mixed_value_cols, limit=3)
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
        single_num_val: float | None = None
        if (
            parsed_ranges
            and parsed_ranges[0] is not None
            and all(
                r is not None and r[0] == r[1] == parsed_ranges[0][0]
                for r in parsed_ranges
            )
        ):
            single_num_val = parsed_ranges[0][0]
        mostly_num_all = all(
            (
                ev_by_col[c].n_rows
                - ev_by_col[c].n_missing
                - int(ev_by_col[c].n_non_numeric or 0)
            )
            > int(ev_by_col[c].n_non_numeric or 0)
            for c in mixed_value_cols
        )
        if len(mixed_value_cols) == 1:
            if single_num_val is not None:
                num_str = (
                    str(int(single_num_val))
                    if float(single_num_val).is_integer()
                    else str(single_num_val)
                )
                q_mixed_text = (
                    f"`{c_top}` holds the number {num_str} and the value {top_val!r} "
                    f"({top_n:,} {row_word}). Should those values count as their own "
                    "category, or as unreadable entries treated like blanks?"
                )
            elif mostly_num_all:
                q_mixed_text = (
                    f"`{c_top}` mostly holds numbers but also the value {top_val!r} "
                    f"({top_n:,} {row_word}). Should those values count as their own "
                    "category, or as unreadable entries treated like blanks?"
                )
            else:
                q_mixed_text = (
                    f"`{c_top}` holds numbers and the value {top_val!r} "
                    f"({top_n:,} {row_word}). Should those values count as their own "
                    "category, or as unreadable entries treated like blanks?"
                )
        else:
            if single_num_val is not None:
                num_str = (
                    str(int(single_num_val))
                    if float(single_num_val).is_integer()
                    else str(single_num_val)
                )
                q_mixed_text = (
                    f"{len(mixed_value_cols)} columns hold the number {num_str} and "
                    f"the value {top_val!r} ({total_non_num:,} {entry_word} in total; "
                    f"e.g. `{c_top}`: {top_n:,}). Should those values count as their "
                    "own category, or as unreadable entries treated like blanks?"
                )
            elif mostly_num_all:
                q_mixed_text = (
                    f"{len(mixed_value_cols)} columns mostly hold numbers but also "
                    f"the value {top_val!r} ({total_non_num:,} {entry_word} in total; "
                    f"e.g. `{c_top}`: {top_n:,}). Should those values count as their "
                    "own category, or as unreadable entries treated like blanks?"
                )
            else:
                q_mixed_text = (
                    f"{len(mixed_value_cols)} columns hold numbers and the value "
                    f"{top_val!r} ({total_non_num:,} {entry_word} in total; e.g. "
                    f"`{c_top}`: {top_n:,}). Should those values count as their own "
                    "category, or as unreadable entries treated like blanks?"
                )
        rows_aff = max(
            int(ev_by_col[c].n_non_numeric or 0) + ev_by_col[c].n_missing
            for c in mixed_value_cols
        )
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
        default_opt = (
            "as_categories"
            if all(ev_by_col[c].n_unique <= cat_threshold for c in mixed_value_cols)
            else "as_blank"
        )
        q_mixed = PlanQuestion(
            question_id="",
            topic="mixed_values",
            text=q_mixed_text,
            applies_to=list(mixed_value_cols),
            rows_affected=int(rows_aff),
            options=[
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
            ],
            default_option_id=default_opt,
        )
        pending_questions.append((q_mixed, list(mixed_value_cols)))

    # Question type 7a: conditional_column (one per boolean event partner)
    for partner, cols in conditional_by_partner.items():
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
            options=[
                QuestionOption(
                    option_id="leave_out",
                    label="Leave out",
                    effect=f"Leaves {cols_preview} out of the map because the blank pattern largely repeats `{partner}`",
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
                    effect=f"Keeps in the map with blanks set to 0 and no blank marker (largely repeats `{partner}`)",
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
                    effect=f"Keeps in the map with blanks filled by the median and adds a 0/1 blank marker that largely repeats `{partner}`",
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
            ],
            default_option_id="leave_out",
        )
        pending_questions.append((q_cond, list(cols)))

    # Question type 7b: blank_meaning (grouped by missing_group or singleton)
    # C-AMENDMENT 4: trigger only if >=1 member in the group is count-like zero-free
    grouped_7b: dict[str, list[str]] = {}
    for col in rule7b_cols:
        gid = ev_by_col[col].missing_group or f"singleton:{col}"
        grouped_7b.setdefault(gid, []).append(col)

    blank_group_cols: set[str] = set()
    for gid, cols in grouped_7b.items():
        count_like_cols = [c for c in cols if _is_count_like_zero_free(ev_by_col[c])]
        if not count_like_cols:
            continue
        blank_group_cols.update(cols)
        for c in cols:
            entries[c].status = "open"
        rows_aff = ev_by_col[cols[0]].n_missing
        row_word = "row" if rows_aff == 1 else "rows"

        # Order zero-free columns with count-like columns first
        zero_free_cols = list(count_like_cols) + [
            c for c in cols if ev_by_col[c].n_zero == 0 and c not in count_like_cols
        ]
        n_with_zero = len(cols) - len(zero_free_cols)

        # Check for non-boolean mask_dependency on any column in the group
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
        else:
            zf_str = _format_col_list(zero_free_cols, limit=3)
            verb = (
                "never records a 0" if len(zero_free_cols) == 1 else "never record a 0"
            )
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
        q_blank = PlanQuestion(
            question_id="",
            topic="blank_meaning",
            text=q_text,
            applies_to=list(cols),
            rows_affected=int(rows_aff),
            options=[
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
            ],
            default_option_id="unknown",
        )
        pending_questions.append((q_blank, list(cols)))

    # Question type 9: duplicate_signal (ONE question for sum_of, ONE question for duplicate)
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

    for rel in evidence.relations:
        if rel.kind not in ("duplicate", "sum_of"):
            continue
        rel_cols = [rel.target, *rel.sources]
        if any(c not in entries or entries[c].role != "feature" for c in rel_cols):
            continue
        if rel.kind == "sum_of":
            # Skip if target overlaps existing totals/sources or any source is already a total
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

    if accepted_sums:
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
        q_sum = PlanQuestion(
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
        pending_questions.append((q_sum, []))

    if accepted_dups:
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
        q_dup = PlanQuestion(
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
        pending_questions.append((q_dup, []))

    # Question type 9b: code_columns (low-cardinality nonnegative integer columns).
    # Columns in an exact sum/duplicate relation are additive amounts, not labels.
    additive_cols = {
        col
        for rel in evidence.relations
        if rel.kind in ("sum_of", "duplicate")
        for col in (rel.target, *rel.sources)
    }
    code_cols = sorted(
        [
            col
            for col, entry in entries.items()
            if entry.role == "feature"
            and ev_by_col[col].kind == "numeric"
            and ev_by_col[col].integer_valued is True
            and ev_by_col[col].nonnegative is True
            and 3 <= ev_by_col[col].n_unique <= cat_threshold
            and not _is_count_like_zero_free(ev_by_col[col])
            and col not in blank_group_cols
            and col not in additive_cols
        ],
        key=lambda c: (-ev_by_col[c].n_unique, c),
    )
    if code_cols:
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
            rows_affected=int(n_rows),
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
        pending_questions.append((q_code, []))

    # Question type 10: outcome_columns (when target_columns is empty)
    if not target_list:
        q_out = PlanQuestion(
            question_id="",
            topic="outcome_columns",
            text=(
                "Is any column a result you want to explain after the map is drawn "
                "(for example a diagnosis, outcome, or survival status)? Columns "
                "used to draw the map cannot fairly be used to explain it afterwards."
            ),
            applies_to=[],
            rows_affected=int(n_rows),
            options=[
                QuestionOption(
                    option_id="none_of_them",
                    label="None of them",
                    effect="Keeps all current columns as map inputs",
                    edits=[],
                )
            ],
            default_option_id="none_of_them",
        )
        pending_questions.append((q_out, []))

    def _framing_rank(topic: str) -> int:
        if topic == "code_columns":
            return 1
        if topic == "outcome_columns":
            return 2
        return 0

    # C-AMENDMENT 5: Pre-sort open questions by rows_affected desc, with framing questions at the end
    pending_questions.sort(
        key=lambda item: (
            _framing_rank(item[0].topic),
            -item[0].rows_affected,
        )
    )

    # Temporarily assign provisional IDs so compute_stakes can address them
    provisional_questions: dict[str, PlanQuestion] = {}
    provisional_links: dict[str, list[str]] = {}
    for idx, (q, linked_cols) in enumerate(pending_questions, start=1):
        prov_id = f"_prov_{idx}"
        q.question_id = prov_id
        provisional_questions[prov_id] = q
        provisional_links[prov_id] = linked_cols

    plan_id = f"pp_{uuid.uuid4().hex[:10]}"

    eligible_stakes_questions = [
        q
        for q, _ in pending_questions
        if q.topic not in ("outcome_columns", "code_columns")
    ]

    plan = PreprocessingPlan(
        plan_id=plan_id,
        dataset_id=dataset_id,
        revision=1,
        entries=entries,
        questions=provisional_questions,
        missing_groups=missing_groups,
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

    if stakes and eligible_stakes_questions and stakes_max_questions > 0:
        budget_ids = [
            q.question_id
            for q in eligible_stakes_questions[: int(stakes_max_questions)]
        ]
        skipped_count = len(eligible_stakes_questions) - len(budget_ids)
        t0 = time.perf_counter()
        stakes_by_id = compute_stakes(
            df,
            plan,
            budget_ids,
            sample_rows=stakes_sample_rows,
            k=stakes_k,
            seed=seed,
        )
        stakes_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        computed_count = sum(
            1 for qid in budget_ids if stakes_by_id.get(qid) is not None
        )
        failed_count = len(budget_ids) - computed_count
        sampled_rows = 0
        for qid, st in stakes_by_id.items():
            provisional_questions[qid].stakes = st
            if st is not None and sampled_rows == 0:
                sampled_rows = int(st.get("rows_sampled", 0))
        plan.cost["stakes_ms"] = stakes_ms
        plan.cost["stakes_rows_sampled"] = sampled_rows
        plan.cost["stakes_questions_computed"] = computed_count
        plan.cost["stakes_questions_skipped"] = skipped_count
        plan.cost["stakes_questions_failed"] = failed_count

    # Final sort: framing questions (code_columns, outcome_columns) at the end; otherwise
    # computed stakes first by neighbors_kept asc (ties by rows_affected desc), then null-stakes by rows_affected desc.
    sorted_prov = sorted(
        pending_questions,
        key=lambda item: (
            _framing_rank(item[0].topic),
            0 if item[0].stakes is not None else 1,
            float(item[0].stakes["neighbors_kept"])
            if item[0].stakes is not None
            else 1.0,
            -item[0].rows_affected,
        ),
    )

    final_questions: dict[str, PlanQuestion] = {}
    for idx, (q, linked_cols) in enumerate(sorted_prov, start=1):
        qid = f"q{idx}"
        q.question_id = qid
        final_questions[qid] = q
        for col in linked_cols:
            if col in entries:
                entries[col].question_id = qid

    plan.questions = final_questions
    dry_run_processed_space(df, plan)
    return plan


def _plan_to_matrix(sample_df: pd.DataFrame, plan: PreprocessingPlan) -> np.ndarray:
    pre_dict = plan_to_preprocessing_dict(plan)
    cfg = load_config(
        {
            "run": {"name": "stakes_eval", "data": "in_memory"},
            "preprocessing": pre_dict,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df_proc, _ = preprocess_dataframe(sample_df, cfg)
    return df_proc.to_numpy(dtype=np.float64)


def _padded_overlap(X_base: np.ndarray, X_opt: np.ndarray, *, k: int) -> float:
    if X_base.ndim != 2 or X_opt.ndim != 2:
        return 0.0
    if X_base.shape[1] == 0 and X_opt.shape[1] == 0:
        return 1.0
    if X_base.shape[1] == 0 or X_opt.shape[1] == 0:
        return 0.0
    max_d = max(X_base.shape[1], X_opt.shape[1])
    if X_base.shape[1] < max_d:
        X_a = np.pad(X_base, ((0, 0), (0, max_d - X_base.shape[1])), mode="constant")
    else:
        X_a = X_base
    if X_opt.shape[1] < max_d:
        X_b = np.pad(X_opt, ((0, 0), (0, max_d - X_opt.shape[1])), mode="constant")
    else:
        X_b = X_opt
    return neighborhood_overlap(X_a, X_b, k=k)


def compute_stakes(
    df: pd.DataFrame,
    plan: PreprocessingPlan,
    question_ids: Sequence[str] | None = None,
    *,
    sample_rows: int = 2000,
    k: int = 10,
    seed: int = 0,
) -> dict[str, dict[str, Any] | None]:
    """Compute k-NN neighborhood overlap stakes for each question in *question_ids*."""
    target_qids = (
        list(question_ids)
        if question_ids is not None
        else [qid for qid, q in plan.questions.items() if q.status == "open"]
    )
    results: dict[str, dict[str, Any] | None] = {qid: None for qid in target_qids}
    if not target_qids:
        return results

    n_total = len(df)
    if n_total < 2:
        plan.cost.setdefault("stakes_notes", []).append(
            "Dataset has fewer than 2 rows; stakes not computed."
        )
        return results

    if n_total > sample_rows > 0:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=sample_rows, replace=False))
        sample_df = df.iloc[idx].reset_index(drop=True)
    else:
        sample_df = df

    n_sampled = len(sample_df)
    k_eff = min(int(k), n_sampled - 1)
    if k_eff < 1:
        plan.cost.setdefault("stakes_notes", []).append(
            "Effective k < 1 on row sample; stakes not computed."
        )
        return results

    try:
        X_base = _plan_to_matrix(sample_df, plan)
    except Exception as exc:
        plan.cost.setdefault("stakes_notes", []).append(
            f"Baseline dry-run failed during stakes computation: {exc}"
        )
        return results

    for qid in target_qids:
        q = plan.questions.get(qid)
        if q is None or q.topic == "outcome_columns":
            continue
        non_default_opts = [
            opt for opt in q.options if opt.option_id != q.default_option_id
        ]
        counterfactual_edit_sets: list[list[dict[str, Any]]] = [
            opt.edits for opt in non_default_opts
        ]
        if not counterfactual_edit_sets:
            continue

        min_overlap: float | None = None
        best_delta: int = 0
        for edit_list in counterfactual_edit_sets:
            try:
                if not edit_list:
                    overlap = 1.0
                    delta = 0
                else:
                    alt_plan = apply_answers_and_edits(
                        plan, answers=(), edits=edit_list, source="agent"
                    )
                    X_opt = _plan_to_matrix(sample_df, alt_plan)
                    overlap = _padded_overlap(X_base, X_opt, k=k_eff)
                    delta = int(X_opt.shape[1] - X_base.shape[1])
                if min_overlap is None or (
                    overlap < min_overlap
                    or (overlap == min_overlap and abs(delta) > abs(best_delta))
                ):
                    min_overlap = overlap
                    best_delta = delta
            except Exception as exc:
                plan.cost.setdefault("stakes_notes", []).append(
                    f"Stakes option failed for {qid}: {exc}"
                )

        if min_overlap is not None:
            results[qid] = {
                "neighbors_kept": round(float(min_overlap), 4),
                "feature_count_delta": int(best_delta),
                "rows_sampled": int(n_sampled),
                "k": int(k_eff),
            }

    return results


def _validate_indicator(
    indicator: str,
    col: str,
    ev: ColumnEvidence | None,
    missing_groups: dict[str, list[str]],
) -> str:
    ind = str(indicator).strip()
    if ind == "none":
        return "none"
    if ind == "own":
        normalized = "own"
    elif ind.startswith("group:"):
        gid = ind.split(":", 1)[1].strip()
        if not gid or (gid not in missing_groups and gid != col):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Invalid indicator {indicator!r} for column {col!r}: group must be an evidence missingness group or {col!r}.",
                {
                    "column": col,
                    "indicator": indicator,
                    "known_groups": sorted(missing_groups.keys()),
                },
            )
        normalized = f"group:{gid}"
    else:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Invalid indicator {indicator!r} for column {col!r}. Expected 'none', 'own', or 'group:<gid>'.",
            {
                "column": col,
                "indicator": indicator,
                "known_groups": sorted(missing_groups.keys()),
            },
        )
    if (
        ev is not None
        and ev.n_missing == 0
        and not (ev.kind == "numeric_text" and (ev.n_non_numeric or 0) > 0)
    ):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Column {col!r} has no blanks; indicator must be 'none'.",
            {"column": col, "indicator": indicator, "n_missing": 0},
        )
    return normalized


def _apply_single_edit(
    entries: dict[str, PlanEntry],
    edit: dict[str, Any],
    *,
    source: DecidedBy,
    missing_groups: dict[str, list[str]],
    ev_by_col: dict[str, ColumnEvidence],
) -> None:
    if not isinstance(edit, dict):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            "Each edit must be a mapping.",
            {"edit": edit},
        )
    unknown_keys = set(edit.keys()) - _EDIT_ALLOWED_KEYS
    if unknown_keys:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Unsupported edit field(s) for column {edit.get('column')!r}: {sorted(unknown_keys)}.",
            {
                "column": edit.get("column"),
                "unsupported_fields": sorted(unknown_keys),
                "allowed_fields": sorted(_EDIT_ALLOWED_KEYS),
            },
        )
    col = edit.get("column")
    if not isinstance(col, str) or col not in entries:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Unknown column: {col!r}.",
            {"column": col},
        )

    entry = entries[col]
    ev = ev_by_col.get(col)

    if "role" in edit and edit["role"] is not None:
        role_val = str(edit["role"])
        if role_val not in VALID_ROLES:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                (
                    f"Invalid role {role_val!r} for column {col!r}; "
                    f"valid roles are {sorted(VALID_ROLES)} (for outcomes use role 'target')."
                ),
                {"column": col, "role": role_val, "valid_roles": sorted(VALID_ROLES)},
            )
        entry.role = role_val  # type: ignore[assignment]

    if "encode" in edit:
        enc = edit["encode"]
        if enc is None or enc == "none" or enc == "":
            entry.encode = None
        else:
            enc_str = str(enc)
            if enc_str not in VALID_ENCODE_METHODS:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"Invalid encode method {enc_str!r} for column {col!r}.",
                    {
                        "column": col,
                        "encode": enc_str,
                        "valid_methods": sorted(VALID_ENCODE_METHODS),
                    },
                )
            entry.encode = enc_str

    # Handle nested 'impute' dict or flat 'impute_method' / 'impute_value'
    has_impute_edit = False
    if "impute" in edit:
        has_impute_edit = True
        raw_imp = edit["impute"]
        if raw_imp is None:
            entry.impute = None
        elif isinstance(raw_imp, dict):
            method = raw_imp.get("method")
            if method is None or method == "none":
                entry.impute = None
            else:
                if method not in VALID_IMPUTE_METHODS:
                    raise PlanEditError(
                        "INVALID_ARGUMENT",
                        f"Invalid impute method {method!r} for column {col!r}.",
                        {
                            "column": col,
                            "impute_method": method,
                            "valid_methods": sorted(VALID_IMPUTE_METHODS),
                        },
                    )
                imp_dict: dict[str, Any] = {"method": str(method)}
                if "value" in raw_imp and raw_imp["value"] is not None:
                    if isinstance(raw_imp["value"], bool):
                        raise PlanEditError(
                            "INVALID_ARGUMENT",
                            f"impute_value for column {col!r} cannot be a boolean.",
                            {"column": col, "impute_value": raw_imp["value"]},
                        )
                    imp_dict["value"] = raw_imp["value"]
                entry.impute = imp_dict
        else:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute for column {col!r} must be a mapping or null.",
                {"column": col, "impute": raw_imp},
            )

    if "impute_method" in edit:
        has_impute_edit = True
        method = edit["impute_method"]
        if method is None or method == "none" or method == "":
            entry.impute = None
        else:
            method_str = str(method)
            if method_str not in VALID_IMPUTE_METHODS:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"Invalid impute_method {method_str!r} for column {col!r}.",
                    {
                        "column": col,
                        "impute_method": method_str,
                        "valid_methods": sorted(VALID_IMPUTE_METHODS),
                    },
                )
            prev_val = entry.impute.get("value") if entry.impute else None
            imp_dict = {"method": method_str}
            if method_str == "fill_value" and prev_val is not None:
                imp_dict["value"] = prev_val
            entry.impute = imp_dict

    if "impute_value" in edit and edit["impute_value"] is not None:
        has_impute_edit = True
        val = edit["impute_value"]
        if isinstance(val, bool):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_value for column {col!r} cannot be a boolean.",
                {"column": col, "impute_value": val},
            )
        if entry.impute is None:
            entry.impute = {"method": "fill_value", "value": val}
        else:
            entry.impute = dict(entry.impute)
            entry.impute["value"] = val

    if (
        has_impute_edit
        and entry.impute is not None
        and ev is not None
        and ev.n_missing == 0
        and not (
            ev.kind == "numeric_text"
            and (ev.n_non_numeric or 0) > 0
            and entry.encode is None
        )
    ):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Column {col!r} has no blanks; impute is not applicable.",
            {"column": col, "n_missing": 0},
        )

    if entry.impute is not None:
        method_now = entry.impute.get("method")
        is_text_mode = ev is not None and (
            ev.kind in ("text", "datetime_like")
            or (ev.kind == "numeric_text" and entry.encode is not None)
        )
        is_numeric_mode = ev is not None and (
            ev.kind in ("numeric", "boolean")
            or (ev.kind == "numeric_text" and entry.encode is None)
        )
        if method_now == "fill_value":
            if "value" not in entry.impute or entry.impute["value"] is None:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"impute_method 'fill_value' for column {col!r} requires 'impute_value'.",
                    {"column": col},
                )
            val_now = entry.impute["value"]
            if isinstance(val_now, (float, np.floating)) and (
                math.isnan(float(val_now)) or math.isinf(float(val_now))
            ):
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"impute_value for column {col!r} must be finite, got {val_now!r}.",
                    {"column": col, "impute_value": str(val_now)},
                )
            if ev is not None:
                if is_numeric_mode:
                    if isinstance(val_now, str) or not isinstance(
                        val_now, (int, float, np.number)
                    ):
                        raise PlanEditError(
                            "INVALID_ARGUMENT",
                            f"Column {col!r} is {ev.kind}; impute_value for 'fill_value' must be numeric, got {val_now!r}.",
                            {"column": col, "kind": ev.kind, "impute_value": val_now},
                        )
                elif is_text_mode:
                    if isinstance(val_now, (int, float, np.number)) and not isinstance(
                        val_now, bool
                    ):
                        entry.impute["value"] = str(val_now)
        elif "value" in entry.impute:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_value is only valid with impute_method='fill_value' (column {col!r}).",
                {"column": col, "impute_method": method_now},
            )
        if ev is not None:
            if is_text_mode and method_now in (
                "fill_median",
                "fill_mean",
                "sample_normal",
            ):
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"Column {col!r} is {ev.kind}; numeric impute_method {method_now!r} cannot be used on text.",
                    {"column": col, "kind": ev.kind, "impute_method": method_now},
                )
            if is_numeric_mode and method_now in (
                "fill_mode",
                "sample_categorical",
            ):
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"Column {col!r} is {ev.kind}; categorical impute_method {method_now!r} cannot be used on numbers.",
                    {"column": col, "kind": ev.kind, "impute_method": method_now},
                )

    if "indicator" in edit and edit["indicator"] is not None:
        entry.indicator = _validate_indicator(
            str(edit["indicator"]), col, ev, missing_groups
        )

    # When a column is set to a non-feature role without explicit impute/encode/indicator,
    # clear feature-only settings so dropped/id/target columns do not linger in impute/groups.
    if entry.role != "feature":
        if not has_impute_edit:
            entry.impute = None
        if "encode" not in edit:
            entry.encode = None
        if "indicator" not in edit:
            entry.indicator = "none"

    entry.decided_by = source
    entry.status = "resolved"


def apply_answers_and_edits(
    plan: PreprocessingPlan,
    answers: Sequence[dict[str, Any]] = (),
    edits: Sequence[dict[str, Any]] = (),
    source: DecidedBy = "agent",
) -> PreprocessingPlan:
    """Apply question *answers* and column *edits* to *plan*, returning a new revision."""
    if source not in ("agent", "user"):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Invalid source {source!r}; expected 'agent' or 'user'.",
            {"source": source, "valid_sources": ["agent", "user"]},
        )

    ans_list = list(answers or ())
    edit_list = list(edits or ())
    if not ans_list and not edit_list:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            "Nothing to change: provide at least one answer or edit.",
            {"answers": [], "edits": []},
        )

    ev_by_col: dict[str, ColumnEvidence] = {c.name: c for c in plan.evidence.columns}
    new_entries: dict[str, PlanEntry] = {
        col: copy.deepcopy(entry) for col, entry in plan.entries.items()
    }
    new_questions: dict[str, PlanQuestion] = {
        qid: copy.deepcopy(q) for qid, q in plan.questions.items()
    }

    _protected_roles = frozenset({"target", "entity", "time"})
    # Track columns explicitly assigned a protected role in this call's edit_list
    incoming_protected_roles: dict[str, str] = {
        str(e["column"]): str(e["role"])
        for e in edit_list
        if isinstance(e, dict)
        and isinstance(e.get("column"), str)
        and e.get("role") in _protected_roles
    }

    for ans in ans_list:
        if not isinstance(ans, dict):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                "Each answer must be a mapping with 'question_id' and 'option_id'.",
                {"answer": ans},
            )
        qid = ans.get("question_id")
        oid = ans.get("option_id")
        if not isinstance(qid, str) or qid not in new_questions:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Unknown question_id: {qid!r}.",
                {
                    "question_id": qid,
                    "known_question_ids": list(new_questions.keys()),
                },
            )
        q = new_questions[qid]
        matched_opt: QuestionOption | None = None
        for opt in q.options:
            if opt.option_id == oid:
                matched_opt = opt
                break
        if matched_opt is None:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Unknown option_id {oid!r} for question {qid!r}.",
                {
                    "question_id": qid,
                    "topic": q.topic,
                    "option_id": oid,
                    "valid_options": [o.option_id for o in q.options],
                },
            )

        effective_roles = {
            c: incoming_protected_roles.get(c, new_entries[c].role)
            for c in q.applies_to
            if c in new_entries
        }
        all_protected = bool(effective_roles) and all(
            r in _protected_roles for r in effective_roles.values()
        )
        if q.status != "open" or all_protected:
            roles_str = (
                ", ".join(f"{c}={r}" for c, r in effective_roles.items())
                or ", ".join(
                    f"{c}={incoming_protected_roles.get(c, e.role)}"
                    for c, e in new_entries.items()
                    if incoming_protected_roles.get(c, e.role) in _protected_roles
                )
                or "resolved"
            )
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"question {qid} no longer applies (columns now: {roles_str})",
                {
                    "question_id": qid,
                    "topic": q.topic,
                    "status": q.status,
                    "columns": effective_roles,
                },
            )

        for edit_spec in matched_opt.edits:
            col_name = edit_spec.get("column") if isinstance(edit_spec, dict) else None
            if (
                isinstance(col_name, str)
                and col_name in new_entries
                and incoming_protected_roles.get(col_name, new_entries[col_name].role)
                in _protected_roles
            ):
                continue
            _apply_single_edit(
                new_entries,
                edit_spec,
                source=source,
                missing_groups=plan.missing_groups,
                ev_by_col=ev_by_col,
            )
        for col in q.applies_to:
            if (
                col in new_entries
                and incoming_protected_roles.get(col, new_entries[col].role)
                not in _protected_roles
            ):
                new_entries[col].decided_by = source
                new_entries[col].status = "resolved"
        q.status = "answered"
        q.answer = {"option_id": oid, "decided_by": source}

    target_role_added = False
    for edit_spec in edit_list:
        _apply_single_edit(
            new_entries,
            edit_spec,
            source=source,
            missing_groups=plan.missing_groups,
            ev_by_col=ev_by_col,
        )
        if isinstance(edit_spec, dict) and edit_spec.get("role") == "target":
            target_role_added = True

    # Synchronize open question statuses if direct edits resolved all their columns
    for q in new_questions.values():
        if q.status != "open":
            continue
        if q.topic == "outcome_columns" and target_role_added:
            q.status = "answered"
            q.answer = {"option_id": "custom_edit", "decided_by": source}
        elif q.applies_to and all(
            new_entries[c].status == "resolved"
            and new_entries[c].decided_by != "default"
            for c in q.applies_to
            if c in new_entries
        ):
            q.status = "answered"
            q.answer = {"option_id": "custom_edit", "decided_by": source}

    return PreprocessingPlan(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        revision=plan.revision + 1,
        entries=new_entries,
        questions=new_questions,
        missing_groups=copy.deepcopy(plan.missing_groups),
        evidence=plan.evidence,
        created_at=plan.created_at,
        cost=copy.deepcopy(plan.cost),
    )


def diff_plan_entries(
    before: PreprocessingPlan, after: PreprocessingPlan
) -> list[dict[str, Any]]:
    """Return a list of ``{column, field, was, now}`` diffs between two plan revisions."""
    changes: list[dict[str, Any]] = []
    tracked_fields = ("role", "impute", "indicator", "encode", "status", "decided_by")
    for col, old_entry in before.entries.items():
        new_entry = after.entries.get(col)
        if new_entry is None:
            continue
        for fld in tracked_fields:
            old_val = getattr(old_entry, fld)
            new_val = getattr(new_entry, fld)
            if old_val != new_val:
                changes.append(
                    {
                        "column": col,
                        "field": fld,
                        "was": copy.deepcopy(old_val),
                        "now": copy.deepcopy(new_val),
                    }
                )
    return changes


def _readable_group_key(
    ref_cols: list[str], used_keys: set[str], fallback_gid: str
) -> str:
    """C-AMENDMENT 3: ``<col>`` if 1 member, else ``<first>_and_<k-1>_more``."""
    if len(ref_cols) == 1:
        base = ref_cols[0]
    else:
        base = f"{ref_cols[0]}_and_{len(ref_cols) - 1}_more"
    if base not in used_keys:
        return base
    return f"{base}_{fallback_gid}"


def _collect_group_references(
    plan: PreprocessingPlan,
) -> list[tuple[str, str, list[str]]]:
    """Return ``[(internal_gid, readable_yaml_key, referencing_columns), ...]``.

    Follows C-AMENDMENTS 1–3:
    - Emits only groups referenced by >=1 feature or has_date (role=='drop' with
      ``indicator=='group:<gid>'``) entry.
    - Lists ONLY those referencing columns (never non-referencing members).
    - Uses readable YAML keys (`<col>` or `<first>_and_<k-1>_more`).
    """
    refs_by_gid: dict[str, list[str]] = {}
    for col, entry in plan.entries.items():
        if not entry.indicator.startswith("group:"):
            continue
        if entry.role not in ("feature", "drop"):
            continue
        gid = entry.indicator.split(":", 1)[1]
        if gid:
            refs_by_gid.setdefault(gid, []).append(col)

    used_keys: set[str] = set()
    out: list[tuple[str, str, list[str]]] = []
    for gid, ref_cols in refs_by_gid.items():
        if not ref_cols:
            continue
        yaml_key = _readable_group_key(ref_cols, used_keys, gid)
        used_keys.add(yaml_key)
        out.append((gid, yaml_key, ref_cols))
    return out


def build_marker_legend(
    plan: PreprocessingPlan,
    *,
    preview_limit: int = 5,
) -> list[dict[str, Any]]:
    """Build ``marker_legend`` entries for all emitted missingness marker columns."""
    legend: list[dict[str, Any]] = []
    grouped_cols: set[str] = set()
    for _gid, yaml_key, ref_cols in _collect_group_references(plan):
        grouped_cols.update(ref_cols)
        legend.append(
            {
                "marker_column": f"{yaml_key}_was_missing",
                "columns_preview": list(ref_cols[:preview_limit]),
                "columns_count": len(ref_cols),
            }
        )
    for col, entry in plan.entries.items():
        if (
            entry.role == "feature"
            and entry.impute is not None
            and entry.indicator == "own"
            and col not in grouped_cols
        ):
            legend.append(
                {
                    "marker_column": f"{col}_was_missing",
                    "columns_preview": [col],
                    "columns_count": 1,
                }
            )
    return legend


def plan_to_preprocessing_dict(plan: PreprocessingPlan) -> dict[str, Any]:
    """Convert *plan* to the canonical ``preprocessing:`` dict consumed by ``PulsarConfig``."""
    drop_columns: list[str] = [
        col for col, entry in plan.entries.items() if entry.role != "feature"
    ]

    missing_indicator_groups: dict[str, list[str]] = {}
    for _gid, yaml_key, ref_cols in _collect_group_references(plan):
        missing_indicator_groups[yaml_key] = list(ref_cols)

    impute_dict: dict[str, dict[str, Any]] = {}
    encode_dict: dict[str, dict[str, Any]] = {}

    for col, entry in plan.entries.items():
        if entry.role != "feature":
            continue
        if entry.impute is not None:
            spec: dict[str, Any] = {"method": entry.impute["method"]}
            if "value" in entry.impute and entry.impute["value"] is not None:
                spec["value"] = entry.impute["value"]
            if entry.indicator != "own":
                spec["indicator"] = False
            impute_dict[col] = spec
        if entry.encode is not None:
            encode_dict[col] = {"method": entry.encode}

    pre: dict[str, Any] = {
        "drop_columns": drop_columns,
    }
    if missing_indicator_groups:
        pre["missing_indicator_groups"] = missing_indicator_groups
    if impute_dict:
        pre["impute"] = impute_dict
    if encode_dict:
        pre["encode"] = encode_dict
    return pre


def plan_fingerprint(plan: PreprocessingPlan) -> str:
    """Return a 12-hex SHA-256 fingerprint of ``plan_to_preprocessing_dict(plan)``."""
    pre = plan_to_preprocessing_dict(plan)
    canonical = json.dumps(pre, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def plan_to_preprocessing_yaml(
    plan: PreprocessingPlan,
    *,
    include_header: bool = False,
) -> str:
    """Render the ``preprocessing:`` YAML block for *plan*."""
    pre_dict = plan_to_preprocessing_dict(plan)
    rendered = yaml.safe_dump(
        {"preprocessing": pre_dict},
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    if include_header:
        fp = plan_fingerprint(plan)
        header = f"# preprocessing plan {plan.plan_id} revision {plan.revision} ({fp})"
        return f"{header}\n{rendered}"
    return rendered


def summarize_plan_counts(plan: PreprocessingPlan) -> dict[str, Any]:
    """Compute summary counts for *plan*."""
    decided_by_counts = {"default": 0, "agent": 0, "user": 0}
    n_features = 0
    n_dropped = 0
    n_open_entries = 0

    for entry in plan.entries.values():
        decided_by_counts[entry.decided_by] = (
            decided_by_counts.get(entry.decided_by, 0) + 1
        )
        if entry.role == "feature":
            n_features += 1
        else:
            n_dropped += 1
        if entry.status == "open":
            n_open_entries += 1

    n_open_questions = sum(1 for q in plan.questions.values() if q.status == "open")
    return {
        "columns": len(plan.entries),
        "feature_columns": n_features,
        "dropped": n_dropped,
        "open_entries": n_open_entries,
        "open_questions": n_open_questions,
        "decided_by": decided_by_counts,
    }


def _target_non_zero_mask(series: pd.Series) -> np.ndarray:
    """Return boolean mask of rows where *series* is non-zero / True."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num.fillna(0.0).to_numpy(dtype=np.float64) != 0.0
    vals = series.dropna().astype(str).str.strip().str.lower()
    mask = np.zeros(len(series), dtype=bool)
    notna_idx = np.flatnonzero(series.notna().to_numpy())
    mask[notna_idx] = ~vals.isin(
        {"", "0", "0.0", "false", "no", "f", "n", "none", "nan"}
    ).to_numpy(dtype=bool)
    return mask


def _measure_target_proxies(
    df: pd.DataFrame,
    df_proc: pd.DataFrame,
    output_names: Sequence[str],
    encode_prefixes: tuple[str, ...],
    plan: PreprocessingPlan,
) -> list[dict[str, Any]]:
    """Measure overlap between processed marker/dummy columns and target or boolean event-flag columns."""
    n_rows = len(df)
    if n_rows == 0 or df_proc.shape[0] != n_rows:
        return []

    target_cols = [
        col
        for col, entry in plan.entries.items()
        if entry.role == "target" and col in df.columns
    ]
    if not target_cols:
        ev_by_col = {c.name: c for c in plan.evidence.columns}
        for md in plan.evidence.mask_dependencies:
            p = md.partner
            if (
                p not in target_cols
                and p in df.columns
                and p in ev_by_col
                and ev_by_col[p].kind == "boolean"
            ):
                target_cols.append(p)
    if not target_cols:
        return []

    input_cols = set(df.columns)
    candidate_cols: list[str] = []
    for c in output_names:
        c_str = str(c)
        is_marker = c_str.endswith("_was_missing")
        is_dummy = (
            not is_marker
            and c not in input_cols
            and bool(encode_prefixes)
            and c_str.startswith(encode_prefixes)
        )
        if (is_marker or is_dummy) and c in df_proc.columns:
            candidate_cols.append(c_str)
    if not candidate_cols:
        return []

    proxies: list[dict[str, Any]] = []
    for tgt in target_cols:
        t_mask = _target_non_zero_mask(df[tgt])
        n_target = int(np.sum(t_mask))
        base_rate = round(float(n_target / n_rows), 4)
        for col in candidate_cols:
            arr = df_proc[col].to_numpy(dtype=np.float64)
            m1 = arr == 1.0
            m1_rows = int(np.sum(m1))
            n_both_1 = int(np.sum(m1 & t_mask))
            if col.endswith("_was_missing") and n_both_1 == 0:
                m0 = arr == 0.0
                m0_rows = int(np.sum(m0))
                n_both_0 = int(np.sum(m0 & t_mask))
                if m0_rows > 0 and n_both_0 == m0_rows:
                    proxies.append(
                        {
                            "column": col,
                            "target": tgt,
                            "marker_rows": m0_rows,
                            "share_with_target": 1.0,
                            "target_base_rate": base_rate,
                        }
                    )
                    continue
            if m1_rows <= 0:
                continue
            share = (
                1.0
                if n_both_1 == m1_rows
                else min(0.9999, round(float(n_both_1 / m1_rows), 4))
            )
            proxies.append(
                {
                    "column": col,
                    "target": tgt,
                    "marker_rows": m1_rows,
                    "share_with_target": share,
                    "target_base_rate": base_rate,
                }
            )

    proxies.sort(
        key=lambda item: (
            -item["share_with_target"],
            -item["marker_rows"],
            item["column"],
            item["target"],
        )
    )
    return proxies


def dry_run_processed_space(
    df: pd.DataFrame, plan: PreprocessingPlan
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    """Run ``preprocess_dataframe`` on *df* using *plan* without raising on failure."""
    try:
        pre_dict = plan_to_preprocessing_dict(plan)
        cfg = load_config(
            {
                "run": {"name": "dry_run", "data": "in_memory"},
                "preprocessing": pre_dict,
            }
        )
        df_proc, layout = preprocess_dataframe(df, cfg)
        output_names = layout.feature_names
        marker_count = sum(1 for c in output_names if str(c).endswith("_was_missing"))
        encode_prefixes = tuple(f"{col}_" for col in cfg.encode.keys())
        input_cols = set(df.columns)
        dummy_count = (
            sum(
                1
                for c in output_names
                if not str(c).endswith("_was_missing")
                and c not in input_cols
                and str(c).startswith(encode_prefixes)
            )
            if encode_prefixes
            else 0
        )
        plan.target_proxies = _measure_target_proxies(
            df, df_proc, output_names, encode_prefixes, plan
        )
        return (
            {
                "feature_count": int(df_proc.shape[1]),
                "missing_marker_count": int(marker_count),
                "dummy_column_count": int(dummy_count),
            },
            {"valid": True, "error": None},
        )
    except Exception as exc:
        plan.target_proxies = []
        return None, {"valid": False, "error": str(exc)}
