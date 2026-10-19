"""Question builder for topic='outcome_columns'."""

from __future__ import annotations

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.model import (
    PendingQuestion,
    PlanQuestion,
    QuestionOption,
)


def build(ctx: DraftContext) -> list[PendingQuestion]:
    """Build the ``outcome_columns`` framing question when no target columns were specified."""
    if ctx.target_list:
        return []

    q_out = PlanQuestion(
        question_id="",
        topic="outcome_columns",
        text=(
            "Is any column a result you want to explain after the map is drawn "
            "(for example a diagnosis, outcome, or survival status)? Columns "
            "used to draw the map cannot fairly be used to explain it afterwards."
        ),
        applies_to=[],
        rows_affected=int(ctx.n_rows),
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
    return [(q_out, [])]
