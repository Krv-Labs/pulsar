from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any, Literal

import yaml
from fastmcp import Context
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.config import load_config
from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.preprocessing import (
    build_draft_plan_payload,
    build_edit_plan_payload,
    draft_plan_payload_to_markdown,
    edit_plan_payload_to_markdown,
)
from pulsar.mcp.preprocessing_plan import (
    PlanEditError,
    PreprocessingPlan,
    apply_answers_and_edits,
    diff_plan_entries,
    draft_plan,
    dry_run_processed_space,
)
from pulsar.mcp.results import mcp_result
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _normalize_data_path,
    _read_dataset_file,
    _resolve_dataset_path,
)
from pulsar.mcp.toolmeta import (
    ANY,
    BOOL,
    INT,
    NUM,
    STR,
    arr,
    default_tool_timeout,
    enum,
    nullable,
    obj,
    tool_meta,
)
from pulsar.preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)


def _compact_json(payload: dict[str, Any]) -> str:
    # Plan payloads are large; keep the JSON text view token-lean.
    return json.dumps(payload, separators=(",", ":"))


_OPEN = obj({}, additional=True)

_COUNTS_SCHEMA = obj(
    {
        "columns": INT,
        "feature_columns": INT,
        "dropped": INT,
        "open_entries": INT,
        "open_questions": INT,
        "decided_by": obj(
            {"default": INT, "agent": INT, "user": INT},
            ["default", "agent", "user"],
        ),
    },
    [
        "columns",
        "feature_columns",
        "dropped",
        "open_entries",
        "open_questions",
        "decided_by",
    ],
)

_EDIT_COUNTS_SCHEMA = obj(
    {
        "columns": INT,
        "feature_columns": INT,
        "dropped": INT,
        "open_entries": INT,
        "open_questions": INT,
        "decided_by": obj(
            {"default": INT, "agent": INT, "user": INT},
            ["default", "agent", "user"],
        ),
        "confirmed_without_change": INT,
    },
    [
        "columns",
        "feature_columns",
        "dropped",
        "open_entries",
        "open_questions",
        "decided_by",
        "confirmed_without_change",
    ],
)

_PROCESSED_SPACE_SCHEMA = nullable(
    obj(
        {
            "feature_count": INT,
            "missing_marker_count": INT,
            "dummy_column_count": INT,
        },
        ["feature_count", "missing_marker_count", "dummy_column_count"],
    )
)

_MARKER_LEGEND_ITEM_SCHEMA = obj(
    {
        "marker_column": STR,
        "columns_preview": arr(STR),
        "columns_count": INT,
    },
    ["marker_column", "columns_preview", "columns_count"],
)

_QUESTION_ITEM_SCHEMA = obj(
    {
        "question_id": STR,
        "topic": enum(
            [
                "blank_meaning",
                "code_columns",
                "conditional_column",
                "date_columns",
                "duplicate_signal",
                "mixed_values",
                "mostly_blank",
                "outcome_columns",
            ]
        ),
        "text": STR,
        "applies_to_preview": arr(STR),
        "applies_to_count": INT,
        "rows_affected": INT,
        "options": arr(
            obj(
                {"option_id": STR, "label": STR, "effect": STR},
                ["option_id", "label", "effect"],
            )
        ),
        "default_option_id": STR,
        "stakes": nullable(
            obj(
                {
                    "neighbors_kept": NUM,
                    "feature_count_delta": INT,
                    "rows_sampled": INT,
                    "k": INT,
                },
                ["neighbors_kept", "feature_count_delta", "rows_sampled", "k"],
            )
        ),
    },
    [
        "question_id",
        "topic",
        "text",
        "applies_to_preview",
        "applies_to_count",
        "rows_affected",
        "options",
        "default_option_id",
        "stakes",
    ],
)

_EVIDENCE_SCHEMA = obj(
    {
        "missing_groups": arr(
            obj(
                {
                    "group_id": STR,
                    "columns_preview": arr(STR),
                    "columns_count": INT,
                    "n_missing": INT,
                    "zero_free_columns_preview": arr(STR),
                },
                [
                    "group_id",
                    "columns_preview",
                    "columns_count",
                    "n_missing",
                    "zero_free_columns_preview",
                ],
            )
        ),
        "missing_groups_omitted": INT,
        "mask_dependencies": arr(
            obj(
                {
                    "column": STR,
                    "partner": STR,
                    "relation": STR,
                    "n_missing": INT,
                    "n_other_partners": INT,
                },
                ["column", "partner", "relation", "n_missing", "n_other_partners"],
            )
        ),
        "mask_dependencies_omitted": INT,
        "relations": arr(
            obj(
                {
                    "kind": STR,
                    "target": STR,
                    "sources": arr(STR),
                    "value": NUM,
                    "rows_used": INT,
                },
                ["kind", "target", "sources", "value", "rows_used"],
            )
        ),
        "relations_omitted": INT,
        "rare_binary_leverage": arr(
            obj(
                {"column": STR, "minority_fraction": NUM, "max_abs_z": NUM},
                ["column", "minority_fraction", "max_abs_z"],
            )
        ),
        "rare_binary_leverage_omitted": INT,
        "constant_columns_preview": arr(STR),
        "constant_columns_count": INT,
        "constant_columns_omitted": INT,
        "datetime_columns_preview": arr(STR),
        "datetime_columns_count": INT,
        "datetime_columns_omitted": INT,
        "integer_coded_preview": arr(
            obj({"column": STR, "n_unique": INT}, ["column", "n_unique"])
        ),
        "integer_coded_omitted": INT,
        "target_proxies": arr(
            obj(
                {
                    "column": STR,
                    "target": STR,
                    "marker_rows": INT,
                    "share_with_target": NUM,
                    "target_base_rate": NUM,
                },
                [
                    "column",
                    "target",
                    "marker_rows",
                    "share_with_target",
                    "target_base_rate",
                ],
            )
        ),
        "target_proxies_omitted": INT,
        "most_missing_columns": arr(
            obj(
                {
                    "column": STR,
                    "missing_fraction": NUM,
                    "n_missing": INT,
                    "missing_group": nullable(STR),
                    "plan_role": STR,
                    "plan_impute_method": nullable(STR),
                },
                [
                    "column",
                    "missing_fraction",
                    "n_missing",
                    "missing_group",
                    "plan_role",
                    "plan_impute_method",
                ],
            )
        ),
        "most_missing_columns_omitted": INT,
    },
    [
        "missing_groups",
        "missing_groups_omitted",
        "mask_dependencies",
        "mask_dependencies_omitted",
        "relations",
        "relations_omitted",
        "rare_binary_leverage",
        "rare_binary_leverage_omitted",
        "constant_columns_preview",
        "constant_columns_count",
        "constant_columns_omitted",
        "datetime_columns_preview",
        "datetime_columns_count",
        "datetime_columns_omitted",
        "integer_coded_preview",
        "integer_coded_omitted",
        "target_proxies",
        "target_proxies_omitted",
        "most_missing_columns",
        "most_missing_columns_omitted",
    ],
)

_TARGET_PROXY_WARNING_SCHEMA = obj(
    {
        "code": STR,
        "target": STR,
        "columns": arr(STR),
        "message": STR,
    },
    ["code", "target", "columns", "message"],
)

_EDIT_EVIDENCE_SCHEMA = obj(
    {
        "target_proxies": arr(
            obj(
                {
                    "column": STR,
                    "target": STR,
                    "marker_rows": INT,
                    "share_with_target": NUM,
                    "target_base_rate": NUM,
                },
                [
                    "column",
                    "target",
                    "marker_rows",
                    "share_with_target",
                    "target_base_rate",
                ],
            )
        ),
        "target_proxies_omitted": INT,
    },
    ["target_proxies", "target_proxies_omitted"],
)

_DRAFT_PLAN_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "plan_id": STR,
        "dataset_id": STR,
        "revision": INT,
        "plan_fingerprint": STR,
        "detail": enum(["summary", "full"]),
        "counts": _COUNTS_SCHEMA,
        "processed_space": _PROCESSED_SPACE_SCHEMA,
        "marker_legend": arr(_MARKER_LEGEND_ITEM_SCHEMA),
        "marker_legend_omitted": INT,
        "questions": arr(_QUESTION_ITEM_SCHEMA),
        "questions_omitted": INT,
        "evidence": _EVIDENCE_SCHEMA,
        "warnings": arr(_TARGET_PROXY_WARNING_SCHEMA),
        "weak_hints": obj(
            {
                "event_flag_columns": {
                    **arr(STR),
                    "description": (
                        "complete 0/1 or count columns whose value exactly explains "
                        "other columns' blanks (often events or exposures; only you "
                        "can say whether any is an outcome)"
                    ),
                },
                "event_flag_columns_omitted": INT,
                "description": STR,
            },
            ["event_flag_columns", "event_flag_columns_omitted", "description"],
        ),
        "preprocessing_yaml_chars": INT,
        "cost": _OPEN,
        "how_to_use": STR,
        "next_tools": arr(STR),
        # detail="full" only
        "preprocessing_yaml": STR,
        "entries": _OPEN,
    },
    [
        "status",
        "plan_id",
        "dataset_id",
        "revision",
        "plan_fingerprint",
        "detail",
        "counts",
        "processed_space",
        "marker_legend",
        "marker_legend_omitted",
        "questions",
        "questions_omitted",
        "evidence",
        "warnings",
        "weak_hints",
        "preprocessing_yaml_chars",
        "cost",
        "how_to_use",
        "next_tools",
    ],
)

_EDIT_PLAN_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "plan_id": STR,
        "dataset_id": STR,
        "revision": INT,
        "plan_fingerprint": STR,
        "changes": arr(
            obj(
                {"column": STR, "field": STR, "was": ANY, "now": ANY},
                ["column", "field", "was", "now"],
            )
        ),
        "changes_omitted": INT,
        "change_summary": arr(
            obj(
                {"field": STR, "was": ANY, "now": ANY, "count": INT},
                ["field", "was", "now", "count"],
            )
        ),
        "change_summary_omitted": INT,
        "counts": _EDIT_COUNTS_SCHEMA,
        "processed_space": _PROCESSED_SPACE_SCHEMA,
        "marker_legend": arr(_MARKER_LEGEND_ITEM_SCHEMA),
        "marker_legend_omitted": INT,
        "evidence": _EDIT_EVIDENCE_SCHEMA,
        "warnings": arr(_TARGET_PROXY_WARNING_SCHEMA),
        "validation": obj(
            {"valid": BOOL, "error": nullable(STR)},
            ["valid", "error"],
        ),
        "open_questions_preview": arr(_QUESTION_ITEM_SCHEMA),
        "open_questions_omitted": INT,
        "preprocessing_yaml_chars": INT,
        "next_tools": arr(STR),
    },
    [
        "status",
        "plan_id",
        "dataset_id",
        "revision",
        "plan_fingerprint",
        "changes",
        "changes_omitted",
        "change_summary",
        "change_summary_omitted",
        "counts",
        "processed_space",
        "marker_legend",
        "marker_legend_omitted",
        "evidence",
        "warnings",
        "validation",
        "open_questions_preview",
        "open_questions_omitted",
        "preprocessing_yaml_chars",
        "next_tools",
    ],
)

_VALIDATE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "valid": BOOL,
        "input_rows": INT,
        "output_rows": INT,
        "output_feature_count": INT,
        "dummy_expansion_count": INT,
        "missingness_flag_count": INT,
        "high_cardinality_encoded_columns": arr(STR),
        "feature_names_preview": arr(STR),
        "nan_remaining": INT,
    },
    [
        "status",
        "valid",
        "input_rows",
        "output_rows",
        "output_feature_count",
        "dummy_expansion_count",
        "missingness_flag_count",
        "high_cardinality_encoded_columns",
        "feature_names_preview",
        "nan_remaining",
    ],
)


def _draft_plan_worker(
    df: Any,
    dataset_id: str,
    target_columns: list[str],
    entity_column: str,
    time_column: str,
    stakes: bool,
    detail: Literal["summary", "full"],
    question_limit: int,
    evidence_limit: int,
) -> tuple[PreprocessingPlan, dict[str, Any]]:
    plan = draft_plan(
        df,
        dataset_id,
        target_columns=target_columns,
        entity_column=entity_column,
        time_column=time_column,
        stakes=stakes,
        stakes_max_questions=2 * int(question_limit),
    )
    processed_space, validation = dry_run_processed_space(df, plan)
    if processed_space is None and validation.get("error"):
        plan.cost.setdefault("stakes_notes", []).append(
            f"Processed-space dry-run failed: {validation['error']}"
        )
    payload = build_draft_plan_payload(
        plan,
        processed_space=processed_space,
        detail=detail,
        question_limit=question_limit,
        evidence_limit=evidence_limit,
    )
    return plan, payload


@tool_meta(
    read_only=False,
    idempotent=False,
    timeout=default_tool_timeout(),
    output_schema=_DRAFT_PLAN_SCHEMA,
    title="Draft Preprocessing Plan",
)
async def draft_preprocessing_plan(
    dataset_id: Annotated[
        str,
        Field(
            description=(
                "Required. Dataset handle returned by ingest_dataset; an empty "
                "value returns MISSING_INPUT."
            )
        ),
    ],
    target_columns: Annotated[
        list[str] | None,
        Field(
            description=(
                "Columns to hold out as evaluation outcomes (role='target', "
                "decided_by='agent', excluded from map features)."
            )
        ),
    ] = None,
    entity_column: Annotated[
        str,
        Field(
            description=(
                "Optional entity identifier column (role='entity', excluded from "
                "map features)."
            )
        ),
    ] = "",
    time_column: Annotated[
        str,
        Field(
            description=(
                "Optional longitudinal time column (role='time', excluded from "
                "map features)."
            )
        ),
    ] = "",
    stakes: Annotated[
        bool,
        Field(
            description=(
                "Compute k-NN neighborhood overlap stakes for open questions on "
                "a deterministic row sample (up to 2 * question_limit questions)."
            )
        ),
    ] = True,
    detail: Annotated[
        Literal["summary", "full"],
        Field(
            description=(
                "summary: capped question and evidence previews. full: uncapped "
                "lists plus every column entry in `entries`."
            )
        ),
    ] = "summary",
    question_limit: Annotated[
        int,
        Field(description="Max open questions returned in summary mode (>= 1)."),
    ] = 8,
    evidence_limit: Annotated[
        int,
        Field(description="Max items per evidence list in summary mode (>= 1)."),
    ] = 12,
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text view: 'markdown' or 'json'. structured_content is identical."
            )
        ),
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Measure dataset evidence and draft a versioned preprocessing plan (`plan_id`, `revision=1`).

    Returns labeled defaults (`decided_by="default"`), open plain-language questions
    ranked by measured k-NN `stakes`, evidence, `marker_legend`, and `preprocessing_yaml`.
    Constant, all-missing, and ID drops are resolved defaults reversible via
    `edit_preprocessing_plan`. Infer what you can (`source="agent"`); ask the user in
    domain words only when needed.
    """
    if not dataset_id:
        return mcp_error(
            "draft_preprocessing_plan",
            "dataset_id is required.",
            error_code="MISSING_INPUT",
            agent_action="Pass dataset_id from ingest_dataset.",
        )
    if question_limit < 1:
        return mcp_error(
            "draft_preprocessing_plan",
            f"question_limit must be >= 1, got '{question_limit}'.",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass question_limit >= 1 (default 8).",
        )
    if evidence_limit < 1:
        return mcp_error(
            "draft_preprocessing_plan",
            f"evidence_limit must be >= 1, got '{evidence_limit}'.",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass evidence_limit >= 1 (default 12).",
        )

    try:
        session = _get_session(ctx)
        df, _ = await _load_session_dataframe(session, dataset_id=dataset_id)
        plan, payload = await asyncio.to_thread(
            _draft_plan_worker,
            df,
            dataset_id,
            list(target_columns or ()),
            entity_column,
            time_column,
            stakes,
            detail,
            question_limit,
            evidence_limit,
        )
        session.preprocessing_plans[plan.plan_id] = plan
        if response_format == "markdown":
            return mcp_result(payload, text=draft_plan_payload_to_markdown(payload))
        return mcp_result(payload, text=_compact_json(payload))
    except PlanEditError as exc:
        return mcp_error(
            "draft_preprocessing_plan",
            exc.message,
            error_code=exc.code,
            agent_action=(
                "Pass only column names present in the dataset for target_columns, "
                "entity_column, and time_column."
            ),
            details=exc.details,
        )
    except LookupError:
        return unknown_handle_error(
            "draft_preprocessing_plan", "dataset_id", dataset_id
        )
    except Exception as exc:
        logger.error("Error in draft_preprocessing_plan: %s", exc)
        return mcp_error(
            "draft_preprocessing_plan",
            str(exc),
            error_code="DRAFT_PLAN_FAILED",
            agent_action=(
                "Verify the dataset loads via characterize_dataset and retry."
            ),
        )


def _edit_plan_worker(
    df: Any,
    plan: PreprocessingPlan,
    answers: list[dict[str, Any]],
    edits: list[dict[str, Any]],
    source: Literal["agent", "user"],
) -> tuple[PreprocessingPlan, dict[str, Any]]:
    updated = apply_answers_and_edits(
        plan,
        answers=answers,
        edits=edits,
        source=source,
    )
    changes = diff_plan_entries(plan, updated)
    processed_space, validation = dry_run_processed_space(df, updated)
    payload = build_edit_plan_payload(
        updated,
        changes=changes,
        processed_space=processed_space,
        validation=validation,
    )
    return updated, payload


@tool_meta(
    read_only=False,
    idempotent=False,
    output_schema=_EDIT_PLAN_SCHEMA,
    title="Edit Preprocessing Plan",
)
async def edit_preprocessing_plan(
    plan_id: Annotated[
        str,
        Field(
            description=(
                "Plan handle returned by draft_preprocessing_plan or create_config."
            )
        ),
    ],
    base_revision: Annotated[
        int | None,
        Field(
            description=(
                "Revision number being edited; mismatches return "
                "SESSION_REVISION_CONFLICT. If omitted, edits the current revision."
            )
        ),
    ] = None,
    answers: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description=(
                "Question answers as [{'question_id': '<question_id>', 'option_id': '<option_id>'}]; "
                "use the question_id and option_id values the payload lists."
            )
        ),
    ] = None,
    edits: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description=(
                "Direct column edits as [{'column': col, 'role'?: ..., "
                "'impute_method'?: ..., 'impute_value'?: ..., 'indicator'?: ..., "
                "'encode'?: ...}]."
            )
        ),
    ] = None,
    source: Annotated[
        Literal["agent", "user"],
        Field(
            description=(
                "Decision provenance: 'agent' when inferred from context/schema; "
                "'user' when relaying a user's answer."
            )
        ),
    ] = "agent",
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text view: 'markdown' or 'json'. structured_content is identical."
            )
        ),
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Apply question `answers` and/or per-column `edits` to a stored preprocessing plan.

    Checks `base_revision` (`SESSION_REVISION_CONFLICT` on mismatch), records
    decision provenance (`source="agent"` or `"user"`), dry-runs preprocessing
    (`validation.valid=false` with error text on schema failure, without raising),
    and stores the new revision in the session.
    """
    if not plan_id:
        return mcp_error(
            "edit_preprocessing_plan",
            "plan_id is required.",
            error_code="MISSING_INPUT",
            agent_action="Pass plan_id from draft_preprocessing_plan or create_config.",
        )

    session = _get_session(ctx)
    plan = session.preprocessing_plans.get(plan_id)
    if plan is None:
        return unknown_handle_error("edit_preprocessing_plan", "plan_id", plan_id)

    if base_revision is not None and int(base_revision) != int(plan.revision):
        return mcp_error(
            "edit_preprocessing_plan",
            (
                f"Plan '{plan_id}' is at revision {plan.revision}, "
                f"but base_revision={base_revision} was supplied."
            ),
            error_code="SESSION_REVISION_CONFLICT",
            agent_action=(
                f"Inspect the current plan or retry edit_preprocessing_plan with "
                f"base_revision={plan.revision}."
            ),
            details={
                "plan_id": plan_id,
                "current_revision": int(plan.revision),
                "base_revision": int(base_revision),
            },
        )

    try:
        df, _ = await _load_session_dataframe(session, dataset_id=plan.dataset_id)
        updated, payload = await asyncio.to_thread(
            _edit_plan_worker,
            df,
            plan,
            list(answers or ()),
            list(edits or ()),
            source,
        )
        session.preprocessing_plans.pop(plan.plan_id, None)
        session.preprocessing_plans[plan.plan_id] = updated
        if response_format == "markdown":
            return mcp_result(payload, text=edit_plan_payload_to_markdown(payload))
        return mcp_result(payload, text=_compact_json(payload))
    except PlanEditError as exc:
        if exc.details.get("topic") == "outcome_columns":
            agent_action = (
                "To mark outcome columns, call edit_preprocessing_plan with "
                "edits=[{column, role: 'target'}]."
            )
        elif exc.details.get("topic") == "code_columns":
            agent_action = (
                "To treat code columns as categories or drop them, call "
                "edit_preprocessing_plan with edits=[{column, encode: 'one_hot'}] "
                "or edits=[{column, role: 'drop'}]."
            )
        elif exc.details.get("n_missing") == 0 or "has no blanks" in exc.message:
            agent_action = (
                "Remove the impute/indicator edit for this column; it has no blanks."
            )
        else:
            agent_action = (
                "Check question_id, option_id, column names, and edit fields in "
                "details, then retry."
            )
        return mcp_error(
            "edit_preprocessing_plan",
            exc.message,
            error_code=exc.code,
            agent_action=agent_action,
            details=exc.details,
        )
    except LookupError:
        return unknown_handle_error(
            "edit_preprocessing_plan", "dataset_id", plan.dataset_id
        )
    except Exception as exc:
        logger.error("Error in edit_preprocessing_plan: %s", exc)
        return mcp_error(
            "edit_preprocessing_plan",
            str(exc),
            error_code="EDIT_PLAN_FAILED",
            agent_action="Check the answers and edits arguments, then retry.",
        )


async def _resolve_validation_dataframe(
    session: Any, dataset_id: str
) -> tuple[Any | None, ToolResult | None]:
    if not dataset_id:
        if session.data is not None:
            return session.data, None
        return None, mcp_error(
            "validate_preprocessing_config",
            "No dataset_id given and no data is loaded in this session.",
            error_code="SESSION_DATA_MISSING",
            agent_action=(
                "Pass dataset_id (from ingest_dataset) to validate against that "
                "dataset; required on stateless transports."
            ),
        )
    try:
        resolved = _resolve_dataset_path(dataset_id)
    except LookupError:
        return None, unknown_handle_error(
            "validate_preprocessing_config", "dataset_id", dataset_id
        )
    if (
        session.data is not None
        and session.data_dataset_id == dataset_id
        and session.data_path == _normalize_data_path(resolved)
    ):
        return session.data, None
    try:
        data = await asyncio.to_thread(_read_dataset_file, resolved)
        return data, None
    except FileNotFoundError:
        return None, path_access_error("validate_preprocessing_config", resolved)


def _build_validation_payload(data: Any, layout: Any) -> dict[str, Any]:
    input_cols = set(data.columns)
    output_names = layout.feature_names
    dummy_count = sum(
        1 for name in output_names if "_" in name and name not in input_cols
    )
    missingness_flag_count = sum(
        1 for name in output_names if name.endswith("_was_missing")
    )
    high_cardinality_encoded = [
        col for col, cats in layout.vocab.items() if len(cats) > 20
    ]
    col_preview = list(output_names[:8])
    if len(output_names) > 8:
        col_preview.append(f"... +{len(output_names) - 8} more")
    return {
        "status": "ok",
        "valid": True,
        "input_rows": len(data),
        "output_rows": layout.n_rows,
        "output_feature_count": len(output_names),
        "dummy_expansion_count": dummy_count,
        "missingness_flag_count": missingness_flag_count,
        "high_cardinality_encoded_columns": high_cardinality_encoded,
        "feature_names_preview": col_preview,
        "nan_remaining": 0,
    }


@tool_meta(
    read_only=True,
    idempotent=True,
    output_schema=_VALIDATE_SCHEMA,
    title="Validate Preprocessing Config",
)
async def validate_preprocessing_config(
    config_yaml: Annotated[
        str,
        Field(
            description=(
                "Full config as YAML text (a mapping with a `preprocessing` "
                "section); only preprocessing is executed."
            )
        ),
    ],
    dataset_id: Annotated[
        str,
        Field(
            description=(
                "Dataset handle from ingest_dataset to validate against; read "
                "without changing the session's bound data. Omit to use the data "
                "the session already loaded (stdio only; stateless transports must "
                "pass it)."
            )
        ),
    ] = "",
    ctx: Context = None,
) -> ToolResult:
    """Dry-run only the preprocessing of a config against a dataset.

    Cheap (no sweep), read-only. Returns row/feature counts and expansion
    diagnostics, or a CONFIG_INVALID error with guidance to fix via
    `edit_preprocessing_plan` or YAML edits. Pass dataset_id, or omit it to use
    the session's loaded data; validate_config does the full schema check.
    """
    session = _get_session(ctx)
    data, err = await _resolve_validation_dataframe(session, dataset_id)
    if err is not None:
        return err

    try:
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            return mcp_error(
                "validate_preprocessing_config",
                "config_yaml must be a valid YAML mapping.",
                error_code="CONFIG_YAML_INVALID",
                agent_action="Pass config_yaml as a YAML mapping with a preprocessing section.",
            )

        cfg = load_config(config_dict)
        _, layout = await asyncio.to_thread(preprocess_dataframe, data, cfg)
        return mcp_result(_build_validation_payload(data, layout))

    except yaml.YAMLError as e:
        return mcp_error(
            "validate_preprocessing_config",
            f"config_yaml is not valid YAML: {e}",
            error_code="CONFIG_YAML_INVALID",
            agent_action=(
                "Pass raw YAML (no Markdown fences) and fix the syntax error, "
                "then retry."
            ),
            details={"valid": False},
        )
    except (ValueError, TypeError) as e:
        return mcp_error(
            "validate_preprocessing_config",
            str(e),
            error_code="CONFIG_INVALID",
            agent_action=(
                "Update the preprocessing plan via edit_preprocessing_plan (then "
                "create_config) or edit the preprocessing block in config_yaml, "
                "then re-run validate_preprocessing_config."
            ),
            details={"valid": False},
        )
    except Exception as e:
        logger.error(f"Error in validate_preprocessing_config: {e}")
        return mcp_error(
            "validate_preprocessing_config",
            str(e),
            error_code="VALIDATE_PREPROCESSING_FAILED",
            agent_action="Check config_yaml and the loaded data, then retry.",
        )
