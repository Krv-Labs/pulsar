"""Data models and constants for preprocessing plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pulsar.analysis.preprocessing_evidence import PreprocessingEvidence

Role = Literal["feature", "drop", "id", "entity", "time", "target"]
DecidedBy = Literal["default", "agent", "user"]
QuestionTopic = Literal[
    "blank_meaning",
    "code_columns",
    "conditional_column",
    "date_columns",
    "duplicate_signal",
    "mixed_values",
    "mostly_blank",
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


PendingQuestion = tuple[PlanQuestion, list[str]]


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
