from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os

from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.mcp.prompts import WORKFLOW_PROMPT
from pulsar.mcp.registry import registry
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


async def get_workflow_guide(ctx: Context = None) -> str:
    """
    Return the recommended end-to-end Pulsar analysis workflow as markdown.

    Agents that want an opinionated phase-by-phase procedure (ingest →
    calibrate → run → diagnose → interpret) should call this once at the
    start of a session. Agents that drive Pulsar with their own workflow
    can ignore it.
    """
    return WORKFLOW_PROMPT


async def get_runtime_context(ctx: Context = None) -> str:
    """
    Return the MCP server runtime context so agents can reason about path visibility
    and handle lifecycle before attempting file-based operations.
    """
    session = _get_session(ctx)
    payload = {
        "cwd": os.getcwd(),
        "cache_dir": str(registry.cache_dir),
        "temp_dir": os.getenv("TMPDIR", "/tmp"),
        "transport_assumption": "stdio-single-client",
        "session_id": _session_key(ctx),
        "dataset_handle_persistence": "on-disk registry under cache_dir",
        "run_handle_persistence": "on-disk registry under cache_dir/runs",
        "latest_dataset_id": session.dataset_id,
        "latest_run_id": session.latest_run_id,
        "path_guidance": [
            "Use ingest_dataset(path) for host-visible absolute paths.",
            "SANDBOX ISOLATION: If your file is in a sandbox (e.g. /home/claude), use the 'Cache-Bridge' pattern: Copy the file to the `cache_dir` shown above, then call `ingest_dataset(path)` on the destination.",
            "Chunked/Base64 uploads are a last-resort legacy fallback for remote-only servers. DO NOT use them for local files.",
            "config_path must be server-visible; config_yaml should be raw YAML (no Markdown).",
        ],
    }
    return json.dumps(payload, indent=2)


async def characterize_dataset(
    csv_path: str = "",
    dataset_id: str = "",
    response_format: str = "markdown",
    ctx: Context = None,
) -> str:
    """
    Probes dataset geometry to return raw facts (N, features, variance curve, k-NN mean).
    Prefer dataset_id after ingest. Use csv_path only for host-visible CSV or Parquet files.
    """
    if response_format not in {"json", "markdown"}:
        return mcp_error(
            "characterize_dataset",
            f"response_format must be 'json' or 'markdown', got '{response_format}'",
        )
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        if dataset_id:
            csv_path = _resolve_dataset_path(dataset_id)
        if not csv_path:
            raise ToolError("Provide either csv_path or dataset_id.")

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
            return characterization_payload_to_markdown(payload)
        return json.dumps(payload, indent=2)
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
        return mcp_error("characterize_dataset", str(e))
