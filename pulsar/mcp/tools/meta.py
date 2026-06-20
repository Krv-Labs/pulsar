from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from typing import Literal

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
    """Recommended end-to-end Pulsar workflow as markdown. Call once at start
    of session for the opinionated procedure (ingest → calibrate → run →
    diagnose → interpret)."""
    return WORKFLOW_PROMPT


async def get_runtime_context(ctx: Context = None) -> str:
    """MCP server runtime context (cwd, cache_dir, session_id, latest handles,
    path-visibility guidance). Call before file-based operations."""
    session = _get_session(ctx)
    payload = {
        "cwd": os.getcwd(),
        "cache_dir": str(registry.cache_dir),
        "temp_dir": os.getenv("TMPDIR", "/tmp"),
        "transport_assumption": "stdio-single-client",
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
    }
    return json.dumps(payload, indent=2)


async def characterize_dataset(
    csv_path: str = "",
    dataset_id: str = "",
    response_format: Literal["markdown", "json"] = "markdown",
    ctx: Context = None,
) -> str:
    """Probe dataset geometry: N, features, variance curve, k-NN mean."""
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
