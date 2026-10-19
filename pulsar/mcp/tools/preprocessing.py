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
from pulsar.mcp.toolmeta import default_tool_timeout, tool_meta
from pulsar.mcp.tools.preprocessing_schemas import (
    DRAFT_PLAN_SCHEMA,
    EDIT_PLAN_SCHEMA,
    VALIDATE_SCHEMA,
)
from pulsar.preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)


def _compact_json(payload: dict[str, Any]) -> str:
    # Plan payloads are large; keep the JSON text view token-lean.
    return json.dumps(payload, separators=(",", ":"))


def _plan_result(
    payload: dict[str, Any],
    response_format: Literal["markdown", "json"],
    to_markdown: Any,
) -> ToolResult:
    if response_format == "markdown":
        return mcp_result(payload, text=to_markdown(payload))
    return mcp_result(payload, text=_compact_json(payload))


def _edit_error_action(exc: PlanEditError) -> str:
    if exc.details.get("topic") == "outcome_columns":
        return (
            "To mark outcome columns, call edit_preprocessing_plan with "
            "edits=[{column, role: 'target'}]."
        )
    elif exc.details.get("topic") == "code_columns":
        return (
            "To treat code columns as categories or drop them, call "
            "edit_preprocessing_plan with edits=[{column, encode: 'one_hot'}] "
            "or edits=[{column, role: 'drop'}]."
        )
    elif exc.details.get("n_missing") == 0 or "has no blanks" in exc.message:
        return "Remove the impute/indicator edit for this column; it has no blanks."
    return (
        "Check question_id, option_id, column names, and edit fields in "
        "details, then retry."
    )


def _revision_conflict_error(
    plan_id: str, current_revision: int, base_revision: int
) -> ToolResult:
    return mcp_error(
        "edit_preprocessing_plan",
        (
            f"Plan '{plan_id}' is at revision {current_revision}, "
            f"but base_revision={base_revision} was supplied."
        ),
        error_code="SESSION_REVISION_CONFLICT",
        agent_action=(
            f"Inspect the current plan or retry edit_preprocessing_plan with "
            f"base_revision={current_revision}."
        ),
        details={
            "plan_id": plan_id,
            "current_revision": int(current_revision),
            "base_revision": int(base_revision),
        },
    )


def _draft_plan_worker(
    df: Any,
    dataset_id: str,
    target_columns: list[str] | None,
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
        target_columns=list(target_columns or ()),
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
    output_schema=DRAFT_PLAN_SCHEMA,
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
        try:
            df, _ = await _load_session_dataframe(session, dataset_id=dataset_id)
        except LookupError:
            return unknown_handle_error(
                "draft_preprocessing_plan", "dataset_id", dataset_id
            )
        plan, payload = await asyncio.to_thread(
            _draft_plan_worker,
            df,
            dataset_id,
            target_columns,
            entity_column,
            time_column,
            stakes,
            detail,
            question_limit,
            evidence_limit,
        )
        session.preprocessing_plans[plan.plan_id] = plan
        return _plan_result(payload, response_format, draft_plan_payload_to_markdown)
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
    answers: list[dict[str, Any]] | None,
    edits: list[dict[str, Any]] | None,
    source: Literal["agent", "user"],
) -> tuple[PreprocessingPlan, dict[str, Any]]:
    updated = apply_answers_and_edits(
        plan,
        answers=list(answers or ()),
        edits=list(edits or ()),
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


def _store_updated_plan(
    session: Any,
    plan_id: str,
    plan: Any,
    updated: Any,
    expected_revision: int,
) -> ToolResult | None:
    current = session.preprocessing_plans.get(plan_id)
    if current is not plan:
        return _revision_conflict_error(
            plan_id,
            int(current.revision if current is not None else plan.revision),
            expected_revision,
        )
    # Pop and re-insert so dict insertion order marks this plan as most recently active for create_config.
    session.preprocessing_plans.pop(plan.plan_id, None)
    session.preprocessing_plans[plan.plan_id] = updated
    return None


@tool_meta(
    read_only=False,
    idempotent=False,
    output_schema=EDIT_PLAN_SCHEMA,
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
                "Answers as [{question_id, option_id}] using listed IDs; use "
                "option_ids: [id, ...] to ensemble compatible imputation treatments, "
                "preserving each option's indicator."
            )
        ),
    ] = None,
    edits: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description=(
                "Direct column edits as [{'column': col, 'role'?: ..., "
                "'impute_method'?: method or list, 'impute_value'?: ..., 'indicator'?: ..., "
                "'encode'?: ...}]. 'impute' accepts a spec or candidate list with "
                "method, value?, seed?, indicator?: bool."
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
    and stores the new revision. `option_ids` ensembles compatible treatments;
    role, encoding, and missingness groups must agree across selected options.
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

    expected_revision = (
        int(plan.revision) if base_revision is None else int(base_revision)
    )
    if expected_revision != int(plan.revision):
        return _revision_conflict_error(plan_id, int(plan.revision), expected_revision)

    try:
        try:
            df, _ = await _load_session_dataframe(session, dataset_id=plan.dataset_id)
        except LookupError:
            return unknown_handle_error(
                "edit_preprocessing_plan", "dataset_id", plan.dataset_id
            )
        updated, payload = await asyncio.to_thread(
            _edit_plan_worker,
            df,
            plan,
            answers,
            edits,
            source,
        )
        conflict = _store_updated_plan(
            session, plan_id, plan, updated, expected_revision
        )
        if conflict is not None:
            return conflict
        return _plan_result(payload, response_format, edit_plan_payload_to_markdown)
    except PlanEditError as exc:
        return mcp_error(
            "edit_preprocessing_plan",
            exc.message,
            error_code=exc.code,
            agent_action=_edit_error_action(exc),
            details=exc.details,
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


def _count_dummy_columns(
    output_names: list[str], input_cols: set[Any], cfg: Any
) -> int:
    encode_prefixes = tuple(f"{col}_" for col in cfg.encode.keys())
    if not encode_prefixes:
        return 0
    return sum(
        1
        for name in output_names
        if not str(name).endswith("_was_missing")
        and name not in input_cols
        and str(name).startswith(encode_prefixes)
    )


def _build_validation_payload(
    data: Any, df_proc: Any, layout: Any, cfg: Any, *, nan_remaining: int | None = None
) -> dict[str, Any]:
    input_cols = set(data.columns)
    output_names = layout.feature_names
    dummy_count = _count_dummy_columns(output_names, input_cols, cfg)
    missingness_flag_count = sum(
        1 for name in output_names if name.endswith("_was_missing")
    )
    high_cardinality_encoded = [
        col for col, cats in layout.vocab.items() if len(cats) > 20
    ]
    col_preview = list(output_names[:8])
    if len(output_names) > 8:
        col_preview.append(f"... +{len(output_names) - 8} more")
    resolved_nans = (
        int(df_proc.isna().sum().sum()) if nan_remaining is None else int(nan_remaining)
    )
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
        "nan_remaining": resolved_nans,
    }


def _dry_run_all_variants(data: Any, cfg: Any) -> tuple[Any, Any, int]:
    variants = cfg.preprocessing_variants()
    df_proc, layout = preprocess_dataframe(data, variants[0])
    max_nans = int(df_proc.isna().sum().sum())
    for variant in variants[1:]:
        df_v, _ = preprocess_dataframe(data, variant)
        max_nans = max(max_nans, int(df_v.isna().sum().sum()))
    return df_proc, layout, max_nans


@tool_meta(
    read_only=True,
    idempotent=True,
    output_schema=VALIDATE_SCHEMA,
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
        df_proc, layout, nan_remaining = await asyncio.to_thread(
            _dry_run_all_variants, data, cfg
        )
        return mcp_result(
            _build_validation_payload(
                data, df_proc, layout, cfg, nan_remaining=nan_remaining
            )
        )

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
