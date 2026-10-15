"""Evidence-first preprocessing plan ledger for Pulsar MCP.

This package is pure Python/pandas/numpy (no FastMCP imports) and turns
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

from pulsar.mcp.preprocessing_plan.context import DraftContext
from pulsar.mcp.preprocessing_plan.draft import draft_plan
from pulsar.mcp.preprocessing_plan.edits import (
    apply_answers_and_edits,
    diff_plan_entries,
)
from pulsar.mcp.preprocessing_plan.model import (
    VALID_DECIDED_BY,
    VALID_ENCODE_METHODS,
    VALID_IMPUTE_METHODS,
    VALID_ROLES,
    DecidedBy,
    PendingQuestion,
    PlanEditError,
    PlanEntry,
    PlanQuestion,
    PreprocessingPlan,
    QuestionOption,
    QuestionTopic,
    Role,
)
from pulsar.mcp.preprocessing_plan.proxies import dry_run_processed_space
from pulsar.mcp.preprocessing_plan.render import (
    build_marker_legend,
    plan_fingerprint,
    plan_to_preprocessing_dict,
    plan_to_preprocessing_yaml,
    summarize_plan_counts,
)
from pulsar.mcp.preprocessing_plan.stakes import compute_stakes

__all__ = [
    "DecidedBy",
    "DraftContext",
    "PendingQuestion",
    "PlanEditError",
    "PlanEntry",
    "PlanQuestion",
    "PreprocessingPlan",
    "QuestionOption",
    "QuestionTopic",
    "Role",
    "VALID_DECIDED_BY",
    "VALID_ENCODE_METHODS",
    "VALID_IMPUTE_METHODS",
    "VALID_ROLES",
    "apply_answers_and_edits",
    "build_marker_legend",
    "compute_stakes",
    "diff_plan_entries",
    "draft_plan",
    "dry_run_processed_space",
    "plan_fingerprint",
    "plan_to_preprocessing_dict",
    "plan_to_preprocessing_yaml",
    "summarize_plan_counts",
]
