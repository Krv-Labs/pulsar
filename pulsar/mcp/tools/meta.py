from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from typing import Annotated, Literal

from fastmcp import Context
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.mcp.prompts import WORKFLOW_PROMPT
from pulsar.mcp.registry import registry
from pulsar.mcp import session as session_mod
from pulsar.mcp.results import mcp_result
from pulsar.mcp.toolmeta import BOOL, INT, STR, arr, nullable, obj, tool_meta
from pulsar.mcp.session import (
    _get_session,
    _session_key,
    _resolve_dataset_path,
    _load_session_dataframe,
)
from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.characterization import (
    compact_characterization_payload,
    characterization_payload_to_markdown,
)

logger = logging.getLogger(__name__)

_ANY: dict = {}  # unconstrained JSON value


def _tool_timeout_s() -> float:
    try:
        return float(os.environ.get("PULSAR_MCP_TOOL_TIMEOUT_S", "1800"))
    except ValueError:
        return 1800.0


_GUIDE_SCHEMA = obj({"guide": STR}, ["guide"])


@tool_meta(read_only=True, idempotent=True, output_schema=_GUIDE_SCHEMA)
async def get_workflow_guide(ctx: Context = None) -> ToolResult:
    """Return the recommended end-to-end Pulsar workflow as markdown.

    Call once at the start of a session for the procedure (ingest, calibrate,
    run, diagnose, interpret). Structured content is {"guide": <markdown>}; the
    text view is the same markdown."""
    return mcp_result({"guide": WORKFLOW_PROMPT}, text=WORKFLOW_PROMPT)


_MODEL_CONTINUITY = (
    "In-memory model state persists between calls only on stdio and legacy-session "
    "HTTP. Stateless HTTP (no session id) isolates clients but does not preserve "
    "the fitted model between calls; dataset_id / run_id handles are the durable "
    "state."
)


_RUNTIME_SCHEMA = obj(
    {
        "cwd": STR,
        "cache_dir": STR,
        "temp_dir": STR,
        "transport": STR,
        "transport_assumption": STR,
        "model_continuity": STR,
        "session_id": STR,
        "dataset_handle_persistence": STR,
        "run_handle_persistence": STR,
        "model_state_persistence": STR,
        "session_model_loaded": BOOL,
        "session_data_loaded": BOOL,
        "latest_dataset_id": nullable(STR),
        "latest_run_id": nullable(STR),
        "active_config_yaml": nullable(STR),
        "active_config_dataset_id": nullable(STR),
        "workflow_guide_available_via": STR,
        "path_guidance": arr(STR),
        "model_cache": obj(
            {"size": INT, "capacity": INT, "run_ids": arr(STR)},
            ["size", "capacity", "run_ids"],
        ),
    },
    [
        "cwd",
        "cache_dir",
        "transport",
        "session_id",
        "latest_dataset_id",
        "latest_run_id",
        "model_cache",
    ],
)


@tool_meta(read_only=True, idempotent=True, output_schema=_RUNTIME_SCHEMA)
async def get_runtime_context(ctx: Context = None) -> ToolResult:
    """Report the server's runtime context: paths, session handles, transport, model cache.

    Returns cwd, cache_dir, session_id, latest dataset_id/run_id, active config,
    transport, which runs are cached in memory (model_cache), and path guidance.
    Call before file-based operations or when a handle seems stale."""
    session = _get_session(ctx)
    transport = getattr(ctx, "transport", None) or "stdio-single-client"
    payload = {
        "cwd": os.getcwd(),
        "cache_dir": str(registry.cache_dir),
        "temp_dir": os.getenv("TMPDIR", "/tmp"),
        "transport": transport,
        "transport_assumption": transport,
        "model_continuity": _MODEL_CONTINUITY,
        "session_id": _session_key(ctx),
        "dataset_handle_persistence": "on-disk registry under cache_dir",
        "run_handle_persistence": "on-disk registry under cache_dir/runs",
        "model_state_persistence": "in-memory only",
        "session_model_loaded": session.model is not None,
        "session_data_loaded": session.data is not None,
        "latest_dataset_id": session.dataset_id,
        "latest_run_id": session.latest_run_id,
        "active_config_yaml": session.active_config_yaml,
        "active_config_dataset_id": session.active_config_dataset_id,
        "workflow_guide_available_via": "get_workflow_guide",
        "path_guidance": [
            "Use ingest_dataset(path) for host-visible absolute paths.",
            "SANDBOX ISOLATION: If your file is in a sandbox (e.g. /home/claude), use the 'Cache-Bridge' pattern: Copy the file to the `cache_dir` shown above, then call `ingest_dataset(path)` on the destination.",
            "config_path must be server-visible; config_yaml should be raw YAML (no Markdown).",
        ],
        "model_cache": session_mod.model_cache.stats(),
    }
    return mcp_result(payload)


_CHARACTERIZE_SCHEMA = obj(
    {
        "status": STR,
        "n_samples": _ANY,
        "n_features": _ANY,
        "n_columns_total": _ANY,
        "missingness_pct": _ANY,
        "numeric_missingness_pct": _ANY,
        "overall_missingness_pct": _ANY,
        "raw_numeric_geometry": obj({}, additional=True),
        "schema_summary": obj({}, additional=True),
        "column_name_preview": obj({}, additional=True),
        "column_profile_preview": arr(obj({}, additional=True)),
        "omitted_column_profiles": INT,
        "recommended_next_tool": STR,
        "agent_guidance": STR,
        "minhash_advisory": obj({}, additional=True),
    },
    [
        "status",
        "n_samples",
        "raw_numeric_geometry",
        "schema_summary",
        "column_profile_preview",
        "recommended_next_tool",
    ],
)


@tool_meta(
    read_only=True,
    idempotent=True,
    output_schema=_CHARACTERIZE_SCHEMA,
    timeout=_tool_timeout_s(),
)
async def characterize_dataset(
    csv_path: Annotated[
        str,
        Field(
            description="Host-visible absolute path to a CSV/Parquet file. Ignored when dataset_id is given; use either this or dataset_id."
        ),
    ] = "",
    dataset_id: Annotated[
        str,
        Field(
            description="Dataset handle returned by ingest_dataset. Preferred over csv_path; takes precedence when both are given."
        ),
    ] = "",
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description="Text rendering only: 'markdown' (readable report) or 'json'. structured_content is identical in both."
        ),
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Summarize dataset geometry and schema: row/feature counts, variance curve, k-NN scale, column preview.

    Returns a bounded measurement packet (no domain interpretation). On massive
    datasets it adds a minhash_advisory suggesting cosmic_graph.minhash_d. Use
    before create_config to see the data's scale and structure."""
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        if dataset_id:
            csv_path = _resolve_dataset_path(dataset_id)
        if not csv_path:
            return mcp_error(
                "characterize_dataset",
                "Provide either csv_path or dataset_id.",
                error_code="DATASET_NOT_SPECIFIED",
                agent_action="Call ingest_dataset(path) to get a dataset_id, then pass dataset_id.",
            )

        session = _get_session(ctx)
        if dataset_id:
            df, normalized_path = await _load_session_dataframe(
                session,
                dataset_id=dataset_id,
            )
        else:
            df, normalized_path = await _load_session_dataframe(
                session,
                data_path=csv_path,
            )

        result = await asyncio.to_thread(_char, normalized_path, dataframe=df)
        payload = compact_characterization_payload(dataclasses.asdict(result))
        if response_format == "markdown":
            return mcp_result(
                payload, text=characterization_payload_to_markdown(payload)
            )
        return mcp_result(payload)
    except FileNotFoundError:
        return path_access_error(
            "characterize_dataset",
            csv_path,
            missing_action=(
                "Ask the user for a host-visible absolute dataset path, or use "
                "ingest_dataset first and pass dataset_id."
            ),
            sandbox_action=(
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads. "
                "Run a bash script to copy the file to the `cache_dir` (call `get_runtime_context` "
                "to find it), use `ingest_dataset(path)` to get a dataset_id, and then "
                "retry this tool."
            ),
        )
    except LookupError:
        return unknown_handle_error("characterize_dataset", "dataset_id", dataset_id)
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return mcp_error(
            "characterize_dataset",
            str(e),
            error_code="CHARACTERIZATION_FAILED",
            agent_action="Check the dataset is readable tabular data, then retry or re-ingest it.",
        )
