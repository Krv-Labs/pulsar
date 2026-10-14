from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from typing import Annotated, Any, Literal

import yaml
from fastmcp import Context
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.config import load_config
from pulsar.preprocessing import preprocess_dataframe
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.results import mcp_result
from pulsar.mcp.toolmeta import (
    BOOL,
    INT,
    STR,
    arr,
    enum,
    obj,
    tool_meta,
)
from pulsar.mcp.preprocessing import (
    _preprocessing_block_to_yaml,
    _recommend_preprocessing_block,
    build_preprocessing_recommendation_payload,
    enrich_dirty_numeric_samples,
    preprocessing_recommendation_to_markdown,
    repair_config,
)
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _resolve_dataset_path,
)

logger = logging.getLogger(__name__)

_OPEN = obj({}, additional=True)

_RECOMMEND_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "detail": enum(["summary", "full"]),
        "recommended_action": STR,
        "next_tool_call": obj({"tool": STR, "args": _OPEN}, ["tool"], additional=True),
        "preprocessing_yaml": STR,
        "decision_counts": obj(
            {k: INT for k in ("drop", "impute", "encode", "review", "no_action")},
            ["drop", "impute", "encode", "review", "no_action"],
        ),
        "expansion_estimate": INT,
        "warnings": arr(_OPEN),
        "dirty_numeric_detection": _OPEN,
        "actioned_columns_preview": obj(
            {"drop": arr(STR), "impute": arr(STR), "encode": arr(STR)},
            ["drop", "impute", "encode"],
        ),
        "high_missingness_columns": arr(_OPEN),
        "high_missingness_columns_omitted": INT,
        "numeric_categorical_candidates": arr(_OPEN),
        "numeric_categorical_candidates_omitted": INT,
        "rationale_preview": arr(_OPEN),
        "rationale_omitted": INT,
        "recommended_next_tools": arr(STR),
        # detail="full" only
        "rationale": arr(_OPEN),
        "column_decisions": _OPEN,
        "high_missingness_columns_full": arr(_OPEN),
        "numeric_categorical_candidates_full": arr(_OPEN),
        "profile_source": _OPEN,
    },
    [
        "status",
        "detail",
        "recommended_action",
        "next_tool_call",
        "preprocessing_yaml",
        "decision_counts",
        "expansion_estimate",
        "warnings",
        "dirty_numeric_detection",
        "actioned_columns_preview",
        "high_missingness_columns",
        "high_missingness_columns_omitted",
        "numeric_categorical_candidates",
        "numeric_categorical_candidates_omitted",
        "rationale_preview",
        "rationale_omitted",
        "recommended_next_tools",
    ],
)

_REPAIR_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "classification": STR,
        "change_count": INT,
        "changes": arr(
            obj(
                {"column": STR, "was": STR, "now": STR, "rationale": STR},
                ["column", "was", "now", "rationale"],
            )
        ),
        "config_yaml": STR,
        "next_step": STR,
    },
    ["status", "classification", "change_count", "changes", "config_yaml", "next_step"],
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

_REPAIR_ROW = re.compile(r"^\|\s*`(?P<col>[^`]*)`\s*\|(?P<rest>.*)\|\s*$")


def _parse_repair_markdown(markdown: str) -> dict[str, Any]:
    """Split repair_config's markdown report into structured fields."""
    cls = re.search(r"## Error Classification\n(.*?) — (\d+) change", markdown)
    changes = []
    for line in markdown.splitlines():
        m = _REPAIR_ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group("rest").split("|")]
        if len(cells) < 3:
            continue
        changes.append(
            {
                "column": m.group("col"),
                "was": cells[0],
                "now": cells[1],
                "rationale": " | ".join(cells[2:]),
            }
        )
    yml = re.search(r"```yaml\n(.*?)```", markdown, re.S)
    return {
        "status": "ok",
        "classification": cls.group(1).strip("` ") if cls else "",
        "change_count": int(cls.group(2)) if cls else len(changes),
        "changes": changes,
        "config_yaml": yml.group(1) if yml else "",
        "next_step": (
            "Call validate_preprocessing_config with config_yaml before re-running "
            "run_topological_sweep."
        ),
    }


def _recommend_preprocessing_worker(
    normalized_path: str,
    df: Any,
    dataset_id: str,
    detail: Literal["summary", "full"],
    rationale_limit: int,
) -> dict[str, Any]:
    from pulsar.analysis.characterization import characterize_dataset as _char

    result = _char(normalized_path, dataframe=df)
    geo = dataclasses.asdict(result)

    n_samples = geo.get("n_samples", 0)
    column_profiles = geo.get("column_profiles", [])

    column_profiles, dirty_numeric_detection = enrich_dirty_numeric_samples(
        column_profiles, df
    )
    drop, impute, encode, rationale = _recommend_preprocessing_block(
        column_profiles, n_samples
    )

    preprocessing_yaml = _preprocessing_block_to_yaml(drop, impute, encode)

    # Estimate expansion: each encoded column adds ~n_unique dummy columns
    expansion_estimate = 0
    for raw_cp in column_profiles:
        cp = (
            raw_cp
            if isinstance(raw_cp, dict)
            else {"name": raw_cp.name, "n_unique": raw_cp.n_unique}
        )
        name = cp["name"]
        if name in encode:
            expansion_estimate += cp.get("n_unique", 2)

    return build_preprocessing_recommendation_payload(
        drop=drop,
        impute=impute,
        encode=encode,
        rationale=rationale,
        column_profiles=column_profiles,
        preprocessing_yaml=preprocessing_yaml,
        expansion_estimate=expansion_estimate,
        dirty_numeric_detection=dirty_numeric_detection,
        dataset_id=dataset_id,
        n_samples=n_samples,
        detail=detail,
        rationale_limit=rationale_limit,
    )


@tool_meta(read_only=True, idempotent=True, output_schema=_RECOMMEND_SCHEMA)
async def recommend_preprocessing(
    dataset_id: Annotated[
        str,
        Field(
            description=(
                "Dataset handle returned by ingest_dataset. Required despite the "
                "empty default; an empty value returns MISSING_INPUT."
            )
        ),
    ] = "",
    detail: Annotated[
        Literal["summary", "full"],
        Field(
            description=(
                "summary: bounded previews of decisions and flagged columns. "
                "full: also every per-column decision and flagged-column list."
            )
        ),
    ] = "summary",
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text view: 'markdown' or 'json'. structured_content is identical."
            )
        ),
    ] = "markdown",
    rationale_limit: Annotated[
        int,
        Field(description="Max rows in each preview list (>= 1); extras are counted."),
    ] = 20,
    ctx: Context = None,
) -> ToolResult:
    """Recommend a preprocessing policy (drop / impute / encode) from column profiles.

    Returns `preprocessing_yaml` to paste into a config, rationale previews,
    warnings, and a one-hot expansion estimate. A starting point, not a verdict:
    follow up with probe_columns on flagged columns, then validate_config.
    """
    if rationale_limit < 1:
        return mcp_error(
            "recommend_preprocessing",
            f"rationale_limit must be >= 1, got '{rationale_limit}'",
            error_code="INVALID_RATIONALE_LIMIT",
            agent_action="Pass rationale_limit >= 1 (default 20).",
        )
    if not dataset_id:
        return mcp_error(
            "recommend_preprocessing",
            "dataset_id is required.",
            error_code="MISSING_INPUT",
            agent_action="Pass dataset_id from ingest_dataset.",
        )
    try:
        session = _get_session(ctx)
        df, normalized_path = await _load_session_dataframe(
            session,
            dataset_id=dataset_id,
        )
        payload = await asyncio.to_thread(
            _recommend_preprocessing_worker,
            normalized_path,
            df,
            dataset_id,
            detail,
            rationale_limit,
        )
        if response_format == "markdown":
            return mcp_result(
                payload, text=preprocessing_recommendation_to_markdown(payload)
            )
        return mcp_result(payload)

    except LookupError:
        return unknown_handle_error("recommend_preprocessing", "dataset_id", dataset_id)
    except Exception as e:
        logger.error(f"Error in recommend_preprocessing: {e}")
        return mcp_error(
            "recommend_preprocessing",
            str(e),
            error_code="RECOMMEND_PREPROCESSING_FAILED",
            agent_action=(
                "Check the dataset loads (characterize_dataset) and retry; the "
                "reason field has the underlying failure."
            ),
        )


@tool_meta(read_only=True, idempotent=True, output_schema=_REPAIR_SCHEMA)
async def repair_preprocessing_config(
    error_message: Annotated[
        str,
        Field(
            description=(
                "Full text of the preprocessing error from a failed sweep or "
                "validate_preprocessing_config call."
            )
        ),
    ],
    config_yaml: Annotated[
        str,
        Field(description="The config YAML (as text) that produced the error."),
    ],
    dataset_id: Annotated[
        str,
        Field(
            description="Dataset handle from ingest_dataset, used to profile columns."
        ),
    ],
    ctx: Context = None,
) -> ToolResult:
    """Repair a config after a preprocessing error; returns patched `config_yaml` and a change log.

    Handles remaining NaNs, non-numeric columns, numeric-imputation coercion
    failures, all-missing columns and cardinality limits. Does not run anything:
    re-check with validate_preprocessing_config.
    """
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        session = _get_session(ctx)
        try:
            dataset_path = _resolve_dataset_path(dataset_id)
        except LookupError:
            return unknown_handle_error(
                "repair_preprocessing_config", "dataset_id", dataset_id
            )
        df, _ = await _load_session_dataframe(session, dataset_id=dataset_id)
        result = await asyncio.to_thread(_char, dataset_path, dataframe=df)
        geo = dataclasses.asdict(result)

        profiles_by_name: dict[str, Any] = {}
        for cp in geo.get("column_profiles", []):
            pname = cp["name"] if isinstance(cp, dict) else cp.name
            profiles_by_name[pname] = cp
        repaired = repair_config(config_yaml, error_message, profiles_by_name)
        if isinstance(repaired, ToolResult):
            reason = (repaired.structured_content or {}).get("reason", "")
            if "valid YAML mapping" in reason:
                code, action = (
                    "CONFIG_YAML_INVALID",
                    "Pass config_yaml as a YAML mapping with the usual top-level sections.",
                )
            elif "Unrecognized" in reason:
                code, action = (
                    "ERROR_PATTERN_UNRECOGNIZED",
                    "Edit the preprocessing block by hand using the error text, "
                    "then call validate_preprocessing_config.",
                )
            else:
                code, action = (
                    "NO_REPAIR_NEEDED",
                    "Review the error message manually; the config needed no automatic change.",
                )
            return mcp_error(
                "repair_preprocessing_config",
                reason,
                error_code=code,
                agent_action=action,
            )
        return mcp_result(_parse_repair_markdown(repaired), text=repaired)
    except Exception as e:
        logger.error(f"Error in repair_preprocessing_config: {e}")
        return mcp_error(
            "repair_preprocessing_config",
            str(e),
            error_code="REPAIR_FAILED",
            agent_action=(
                "Check that config_yaml parses and dataset_id is valid, then retry."
            ),
        )


@tool_meta(read_only=True, idempotent=True, output_schema=_VALIDATE_SCHEMA)
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
    ctx: Context = None,
) -> ToolResult:
    """Dry-run only the preprocessing of a config against the session's loaded data.

    Cheap (no sweep). Returns row/feature counts and expansion diagnostics, or a
    CONFIG_INVALID error with a repair hint. Needs data loaded by
    characterize_dataset or a sweep; validate_config does the full schema check.
    """
    session = _get_session(ctx)

    if session.data is None:
        return mcp_error(
            "validate_preprocessing_config",
            "No data in session. Run run_topological_sweep (even with a minimal config) or characterize_dataset first to load data into the session.",
            error_code="SESSION_DATA_MISSING",
            agent_action=(
                "Call characterize_dataset(dataset_id=...) to load data into the "
                "session, then retry."
            ),
        )

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
        df_out, layout = await asyncio.to_thread(
            preprocess_dataframe, session.data, cfg
        )

        # Compute expansion diagnostics
        input_cols = set(session.data.columns)
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

        return mcp_result(
            {
                "status": "ok",
                "valid": True,
                "input_rows": len(session.data),
                "output_rows": layout.n_rows,
                "output_feature_count": len(output_names),
                "dummy_expansion_count": dummy_count,
                "missingness_flag_count": missingness_flag_count,
                "high_cardinality_encoded_columns": high_cardinality_encoded,
                "feature_names_preview": list(col_preview),
                "nan_remaining": 0,
            }
        )

    except (ValueError, TypeError) as e:
        return mcp_error(
            "validate_preprocessing_config",
            str(e),
            error_code="CONFIG_INVALID",
            agent_action=(
                "Call repair_preprocessing_config(error_message=..., "
                "config_yaml=..., dataset_id=...) to fix this automatically."
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
