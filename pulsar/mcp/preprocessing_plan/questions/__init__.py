"""Ordered registry of question builders for draft_plan."""

from __future__ import annotations

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import PendingQuestion
from pulsar.mcp.preprocessing_plan.questions import (
    blank_meaning,
    code_columns,
    conditional,
    dates,
    duplicates,
    mixed_values,
    mostly_blank,
    outcome,
)

QUESTION_BUILDERS = (
    dates.build,
    mixed_values.build,
    conditional.build,
    blank_meaning.build,
    mostly_blank.build,
    duplicates.build,
    code_columns.build,
    outcome.build,
)


def build_pending_questions(ctx: DraftContext) -> list[PendingQuestion]:
    """Run all question builders in canonical order and return ``pending_questions``."""
    pending_questions: list[PendingQuestion] = []
    for builder in QUESTION_BUILDERS:
        pending_questions.extend(builder(ctx))
    return pending_questions
