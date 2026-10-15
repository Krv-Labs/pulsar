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
    "conditional_column",
    "date_columns",
    "duplicate_signal",
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


def _is_count_like_zero_free(ev: ColumnEvidence) -> bool:
    """Return True iff *ev* is a numeric count-like column with blanks and no 0s."""
    return (
        ev.kind == "numeric"
        and ev.n_missing > 0
        and ev.integer_valued is True
        and ev.nonnegative is True
        and ev.n_zero == 0
    )


def _format_missing_reason(
    ev: ColumnEvidence, missing_groups: dict[str, list[str]]
) -> str:
    parts = [f"{ev.n_missing:,} blanks"]
    if ev.n_zero == 0 and ev.nonnegative is True:
        parts.append("no recorded 0")
    if ev.missing_group is not None and ev.missing_group in missing_groups:
        n_others = max(0, len(missing_groups[ev.missing_group]) - 1)
        if n_others > 0:
            noun = "column" if n_others == 1 else "columns"
            parts.append(
                f"blank on the same rows as {n_others} other {noun} ({ev.missing_group})"
            )
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

    evidence = measure_preprocessing_evidence(df)
    ev_by_col: dict[str, ColumnEvidence] = {c.name: c for c in evidence.columns}
    missing_groups: dict[str, list[str]] = {
        mg.group_id: list(mg.columns) for mg in evidence.missing_groups
    }

    entries: dict[str, PlanEntry] = {}
    date_cols: list[str] = []
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
            entries[col] = PlanEntry(
                column=col,
                role="drop",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="open",
                reason=f"date/time text ({ev.n_unique:,} distinct values; {ev.n_missing:,} blanks)",
            )
            date_cols.append(col)
            continue

        # Rule 5: numeric_text -> treated as numeric (fill_median triggers coercion), then rule 7
        if ev.kind == "numeric_text":
            if ev.mask_dependency is not None:
                md = ev.mask_dependency
                cond_str = (
                    f"{md.partner} is 0"
                    if md.relation == "missing_when_partner_zero"
                    else f"{md.partner} is not 0"
                )
                entries[col] = PlanEntry(
                    column=col,
                    role="drop",
                    impute=None,
                    indicator="none",
                    encode=None,
                    decided_by="default",
                    status="open",
                    reason=f"{ev.n_missing:,} blanks; blank on the exact rows where {cond_str}",
                )
                conditional_by_partner.setdefault(md.partner, []).append(col)
            else:
                ind = f"group:{ev.missing_group}" if ev.missing_group else "own"
                entries[col] = PlanEntry(
                    column=col,
                    role="feature",
                    impute={"method": "fill_median"},
                    indicator=ind,
                    encode=None,
                    decided_by="default",
                    status="resolved",
                    reason=(
                        f"numeric text with non-numeric strings ({ev.n_missing:,} blanks); "
                        "coerced to numbers with median fill and blank marker"
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
                reason = (
                    f"text column with {ev.n_unique:,} categories and {ev.n_missing:,} blanks; "
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

        # Rule 7a: mask_dependency
        if ev.mask_dependency is not None:
            md = ev.mask_dependency
            cond_str = (
                f"{md.partner} is 0"
                if md.relation == "missing_when_partner_zero"
                else f"{md.partner} is not 0"
            )
            entries[col] = PlanEntry(
                column=col,
                role="drop",
                impute=None,
                indicator="none",
                encode=None,
                decided_by="default",
                status="open",
                reason=f"{ev.n_missing:,} blanks; blank on the exact rows where {cond_str}",
            )
            conditional_by_partner.setdefault(md.partner, []).append(col)
            continue

        # Rule 7b: standard missing numeric/boolean
        ind = f"group:{ev.missing_group}" if ev.missing_group else "own"
        entries[col] = PlanEntry(
            column=col,
            role="feature",
            impute={"method": "fill_median"},
            indicator=ind,
            encode=None,
            decided_by="default",
            status="resolved",
            reason=_format_missing_reason(ev, missing_groups),
        )
        rule7b_cols.append(col)

    # Draft questions (before final sorting and q1..qN numbering)
    pending_questions: list[tuple[PlanQuestion, list[str]]] = []

    # Question type 4: date_columns (one shared question for all datetime_like columns)
    if date_cols:
        ex_col = date_cols[0]
        rows_aff = max(
            ev_by_col[c].n_missing
            if ev_by_col[c].n_missing > 0
            else (ev_by_col[c].n_rows - ev_by_col[c].n_missing)
            for c in date_cols
        )
        partners = sorted(
            {
                ev_by_col[c].mask_dependency.partner
                for c in date_cols
                if ev_by_col[c].mask_dependency is not None
            }
        )
        if partners:
            partner_str = ", ".join(f"`{p}`" for p in partners)
            has_date_effect = (
                f"Leaves raw dates out of the map and adds a 0/1 blank marker "
                f"(note: this marker would repeat {partner_str})"
            )
        else:
            has_date_effect = (
                "Leaves raw dates out of the map and adds a 0/1 marker for "
                "whether each date is filled in"
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
            for c in date_cols
        ]
        q_date = PlanQuestion(
            question_id="",
            topic="date_columns",
            text=(
                f"Some columns hold dates (e.g. `{ex_col}`). Should the map ignore "
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

    # Question type 7a: conditional_column (one per partner)
    for partner, cols in conditional_by_partner.items():
        first_md = ev_by_col[cols[0]].mask_dependency
        assert first_md is not None
        when_phrase = (
            f"when `{partner}` is not 0"
            if first_md.relation == "missing_when_partner_zero"
            else f"when `{partner}` is 0"
        )
        if len(cols) == 1:
            q_text = (
                f"`{cols[0]}` only has a value {when_phrase} — it is blank for "
                "everyone else. Should the map use it?"
            )
        else:
            preview = ", ".join(f"`{c}`" for c in cols[:3])
            q_text = (
                f"{len(cols)} columns (e.g. {preview}) only have values {when_phrase} "
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
                    effect=f"Leaves {', '.join(f'`{c}`' for c in cols)} out of the map because the blank pattern largely repeats `{partner}`",
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

    for gid, cols in grouped_7b.items():
        count_like_cols = [c for c in cols if _is_count_like_zero_free(ev_by_col[c])]
        if not count_like_cols:
            continue
        for c in cols:
            entries[c].status = "open"
        ex_col = count_like_cols[0]
        rows_aff = ev_by_col[cols[0]].n_missing
        if len(cols) == 1:
            q_text = (
                f"`{ex_col}` is blank for {rows_aff:,} rows and never records a 0. "
                "Does a blank mean none (zero), or that the value is unknown?"
            )
            unknown_effect = (
                "Blanks are filled with the median of observed rows, plus a 0/1 "
                "marker column for which rows were blank"
            )
        else:
            q_text = (
                f"{len(cols)} columns (e.g. `{ex_col}`) are blank for the same "
                f"{rows_aff:,} rows, and none of them ever records a 0. Does a blank "
                "mean none (zero), or that the value is unknown?"
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
                    effect="Blanks become 0; no extra marker column",
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

    # Question type 9: duplicate_signal (relations: duplicate, sum_of)
    for rel in evidence.relations:
        if rel.kind not in ("duplicate", "sum_of"):
            continue
        rel_cols = [rel.target, *rel.sources]
        if any(c not in entries or entries[c].role != "feature" for c in rel_cols):
            continue
        if rel.kind == "sum_of":
            sources_str = ", ".join(f"`{s}`" for s in rel.sources)
            q_rel = PlanQuestion(
                question_id="",
                topic="duplicate_signal",
                text=(
                    f"`{rel.target}` is exactly the sum of {sources_str}, so the map "
                    "counts that signal twice. Keep everything, keep only the total, "
                    "or keep only the parts?"
                ),
                applies_to=[rel.target, *rel.sources],
                rows_affected=int(rel.rows_used),
                options=[
                    QuestionOption(
                        option_id="keep_all",
                        label="Keep everything",
                        effect="Keeps both the total column and the part columns in the map",
                        edits=[
                            {"column": c, "role": "feature"}
                            for c in [rel.target, *rel.sources]
                        ],
                    ),
                    QuestionOption(
                        option_id="keep_total",
                        label="Keep only the total",
                        effect=f"Keeps `{rel.target}` and leaves {sources_str} out of the map",
                        edits=[
                            {"column": rel.target, "role": "feature"},
                            *[{"column": s, "role": "drop"} for s in rel.sources],
                        ],
                    ),
                    QuestionOption(
                        option_id="keep_parts",
                        label="Keep only the parts",
                        effect=f"Keeps {sources_str} and leaves `{rel.target}` out of the map",
                        edits=[
                            {"column": rel.target, "role": "drop"},
                            *[{"column": s, "role": "feature"} for s in rel.sources],
                        ],
                    ),
                ],
                default_option_id="keep_all",
            )
            pending_questions.append((q_rel, []))
        elif rel.kind == "duplicate":
            col_a = rel.sources[0]
            col_b = rel.target
            q_rel = PlanQuestion(
                question_id="",
                topic="duplicate_signal",
                text=(
                    f"`{col_b}` has identical values to `{col_a}`, so the map counts "
                    f"that signal twice. Keep both columns, or keep only `{col_a}`?"
                ),
                applies_to=[col_a, col_b],
                rows_affected=int(rel.rows_used),
                options=[
                    QuestionOption(
                        option_id="keep_both",
                        label="Keep both",
                        effect=f"Keeps both `{col_a}` and `{col_b}` in the map",
                        edits=[
                            {"column": col_a, "role": "feature"},
                            {"column": col_b, "role": "feature"},
                        ],
                    ),
                    QuestionOption(
                        option_id="keep_one",
                        label="Keep only one",
                        effect=f"Keeps `{col_a}` and leaves duplicate `{col_b}` out of the map",
                        edits=[
                            {"column": col_a, "role": "feature"},
                            {"column": col_b, "role": "drop"},
                        ],
                    ),
                ],
                default_option_id="keep_both",
            )
            pending_questions.append((q_rel, []))

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
                    effect="Keeps all current columns as map inputs; set role='target' in edits to hold out outcome columns",
                    edits=[],
                )
            ],
            default_option_id="none_of_them",
        )
        pending_questions.append((q_out, []))

    # C-AMENDMENT 5: Pre-sort open questions by rows_affected desc
    # For ties in rows_affected, put questions with concrete applies_to columns before outcome_columns
    pending_questions.sort(
        key=lambda item: (
            -item[0].rows_affected,
            1 if item[0].topic == "outcome_columns" else 0,
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

    plan_id_seed = json.dumps(
        {
            "shape": [int(evidence.n_rows), int(evidence.n_columns)],
            "columns": [str(c) for c in df.columns],
            "target_columns": target_list,
            "entity_column": entity_col,
            "time_column": time_col,
            "entries": {
                col: {
                    "role": e.role,
                    "impute": e.impute,
                    "indicator": e.indicator,
                    "encode": e.encode,
                    "reason": e.reason,
                }
                for col, e in entries.items()
            },
        },
        sort_keys=True,
    )
    plan_id = f"pp_{hashlib.sha256(plan_id_seed.encode('utf-8')).hexdigest()[:10]}"

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
            "evidence_ms": round(
                float(evidence.cost.get("elapsed_seconds", 0.0)) * 1000.0, 2
            ),
            "stakes_ms": 0.0,
            "stakes_rows_sampled": 0,
            "stakes_questions_computed": 0,
            "stakes_questions_skipped": len(pending_questions),
        },
    )

    if stakes and pending_questions and stakes_max_questions > 0:
        budget_ids = [
            q.question_id for q, _ in pending_questions[: int(stakes_max_questions)]
        ]
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
        computed_count = sum(1 for v in stakes_by_id.values() if v is not None)
        skipped_count = len(pending_questions) - computed_count
        sampled_rows = 0
        for qid, st in stakes_by_id.items():
            provisional_questions[qid].stakes = st
            if st is not None and sampled_rows == 0:
                sampled_rows = int(st.get("rows_sampled", 0))
        plan.cost["stakes_ms"] = stakes_ms
        plan.cost["stakes_rows_sampled"] = sampled_rows
        plan.cost["stakes_questions_computed"] = computed_count
        plan.cost["stakes_questions_skipped"] = skipped_count

    # Final sort: computed stakes first by neighbors_kept asc (ties by rows_affected desc),
    # then uncomputed by rows_affected desc.
    sorted_prov = sorted(
        pending_questions,
        key=lambda item: (
            0 if item[0].stakes is not None else 1,
            float(item[0].stakes["neighbors_kept"])
            if item[0].stakes is not None
            else 1.0,
            -item[0].rows_affected,
            1 if item[0].topic == "outcome_columns" else 0,
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
    return plan


def _plan_to_matrix(sample_df: pd.DataFrame, plan: PreprocessingPlan) -> np.ndarray:
    pre_dict = plan_to_preprocessing_dict(plan)
    cfg = load_config(
        {
            "run": {"name": "stakes_eval", "data": "in_memory"},
            "preprocessing": pre_dict,
        }
    )
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

    weak_hint_cols = sorted(
        {
            md.partner
            for md in plan.evidence.mask_dependencies
            if md.partner in plan.entries and plan.entries[md.partner].role == "feature"
        }
    )

    for qid in target_qids:
        q = plan.questions.get(qid)
        if q is None:
            continue
        non_default_opts = [
            opt
            for opt in q.options
            if opt.option_id != q.default_option_id and opt.edits
        ]
        # For outcome_columns (no non-default option edits), measure the impact of
        # holding out event-partner weak_hints if any exist; otherwise 0 delta.
        counterfactual_edit_sets: list[list[dict[str, Any]]] = [
            opt.edits for opt in non_default_opts
        ]
        if not counterfactual_edit_sets and q.topic == "outcome_columns":
            if weak_hint_cols:
                counterfactual_edit_sets = [
                    [{"column": c, "role": "target"} for c in weak_hint_cols]
                ]
            else:
                results[qid] = {
                    "neighbors_kept": 1.0,
                    "feature_count_delta": 0,
                    "rows_sampled": int(n_sampled),
                    "k": int(k_eff),
                }
                continue

        if not counterfactual_edit_sets:
            results[qid] = {
                "neighbors_kept": 1.0,
                "feature_count_delta": 0,
                "rows_sampled": int(n_sampled),
                "k": int(k_eff),
            }
            continue

        min_overlap: float | None = None
        best_delta: int = 0
        for edit_list in counterfactual_edit_sets:
            try:
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
    entries: dict[str, PlanEntry],
    missing_groups: dict[str, list[str]],
) -> str:
    ind = str(indicator).strip()
    if ind in ("none", "own"):
        return ind
    if ind.startswith("group:"):
        gid = ind.split(":", 1)[1].strip()
        if gid and (gid in missing_groups or gid in entries):
            return f"group:{gid}"
    raise PlanEditError(
        "INVALID_ARGUMENT",
        f"Invalid indicator {indicator!r} for column {col!r}. Expected 'none', 'own', or 'group:<gid>'.",
        {
            "column": col,
            "indicator": indicator,
            "known_groups": sorted(missing_groups.keys()),
        },
    )


def _apply_single_edit(
    entries: dict[str, PlanEntry],
    edit: dict[str, Any],
    *,
    source: DecidedBy,
    missing_groups: dict[str, list[str]],
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
            f"Unsupported edit field(s): {sorted(unknown_keys)}.",
            {
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

    if "role" in edit and edit["role"] is not None:
        role_val = str(edit["role"])
        if role_val not in VALID_ROLES:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Invalid role {role_val!r} for column {col!r}.",
                {"column": col, "role": role_val, "valid_roles": sorted(VALID_ROLES)},
            )
        entry.role = role_val  # type: ignore[assignment]

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

    if entry.impute is not None:
        if entry.impute.get("method") == "fill_value":
            if "value" not in entry.impute or entry.impute["value"] is None:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"impute_method 'fill_value' for column {col!r} requires 'impute_value'.",
                    {"column": col},
                )
        elif "value" in entry.impute:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_value is only valid with impute_method='fill_value' (column {col!r}).",
                {"column": col, "impute_method": entry.impute.get("method")},
            )

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

    if "indicator" in edit and edit["indicator"] is not None:
        entry.indicator = _validate_indicator(
            str(edit["indicator"]), col, entries, missing_groups
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

    new_entries: dict[str, PlanEntry] = {
        col: copy.deepcopy(entry) for col, entry in plan.entries.items()
    }
    new_questions: dict[str, PlanQuestion] = {
        qid: copy.deepcopy(q) for qid, q in plan.questions.items()
    }

    for ans in answers or ():
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
                    "option_id": oid,
                    "valid_options": [o.option_id for o in q.options],
                },
            )

        for edit_spec in matched_opt.edits:
            _apply_single_edit(
                new_entries,
                edit_spec,
                source=source,
                missing_groups=plan.missing_groups,
            )
        for col in q.applies_to:
            if col in new_entries:
                new_entries[col].decided_by = source
                new_entries[col].status = "resolved"
        q.status = "answered"
        q.answer = {"option_id": oid, "decided_by": source}

    target_role_added = False
    for edit_spec in edits or ():
        _apply_single_edit(
            new_entries,
            edit_spec,
            source=source,
            missing_groups=plan.missing_groups,
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
        "features": n_features,
        "dropped": n_dropped,
        "open_entries": n_open_entries,
        "open_questions": n_open_questions,
        "decided_by": decided_by_counts,
    }


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
        return (
            {
                "feature_count": int(df_proc.shape[1]),
                "missing_marker_count": int(marker_count),
                "dummy_column_count": int(dummy_count),
            },
            {"valid": True, "error": None},
        )
    except Exception as exc:
        return None, {"valid": False, "error": str(exc)}
