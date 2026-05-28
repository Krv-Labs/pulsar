from __future__ import annotations

import base64
import dataclasses
import json
import logging

from fastmcp import Context

from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.registry import registry
from pulsar.mcp.session import _get_session, _session_key

logger = logging.getLogger(__name__)


async def ingest_dataset(path: str, ctx: Context = None) -> str:
    """
    Register a host-visible absolute dataset path and return a stable dataset_id handle.
    Use this only when the MCP server can read the path directly.
    """
    try:
        record = registry.register_dataset(path)
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except FileNotFoundError:
        return path_access_error(
            "ingest_dataset",
            path,
            missing_action=(
                "Ask the user for a host-visible absolute dataset path, then call "
                "ingest_dataset again."
            ),
            sandbox_action=(
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads. "
                "Run a bash script to copy the file to the `cache_dir` (call `get_runtime_context` "
                "to find it), then retry `ingest_dataset(path)` with the new path."
            ),
        )
    except PermissionError:
        return mcp_error(
            "ingest_dataset",
            "Dataset path exists but is not readable by the MCP server.",
            error_code="FILE_PERMISSION_DENIED",
            agent_action="Provide a readable host-visible dataset path.",
            details={"path_context": {"attempted_path": path}},
        )
    except Exception as e:
        return mcp_error("ingest_dataset", str(e))


async def begin_dataset_upload(
    filename: str,
    media_type: str = "text/csv",
    ctx: Context = None,
) -> str:
    """
    Begin a staged server-side upload for a dataset that is not reachable by path.
    Use this for larger sandboxed uploads, then append chunks and finalize to get dataset_id.
    """
    try:
        record = registry.begin_upload(filename, media_type=media_type)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("begin_dataset_upload", str(e))


async def append_dataset_chunk(
    upload_id: str,
    chunk: str,
    encoding: str = "base64",
    ctx: Context = None,
) -> str:
    """
    Append one chunk to a staged dataset upload.
    Use base64 encoding by default to avoid newline and control-character corruption.
    """
    try:
        if encoding == "base64":
            try:
                chunk_bytes = base64.b64decode(chunk, validate=True)
            except Exception:
                return mcp_error(
                    "append_dataset_chunk",
                    "Chunk payload could not be decoded from base64.",
                    error_code="UPLOAD_DECODE_FAILED",
                    agent_action="Retry with valid base64 chunk data.",
                )
        elif encoding == "utf-8":
            chunk_bytes = chunk.encode("utf-8")
        else:
            return mcp_error(
                "append_dataset_chunk",
                f"Unsupported chunk encoding '{encoding}'.",
                error_code="UPLOAD_ENCODING_UNSUPPORTED",
                agent_action="Use encoding='base64' for binary-safe chunk transport.",
            )

        record = registry.append_upload_chunk(upload_id, chunk_bytes)
        if record is None:
            return unknown_handle_error("append_dataset_chunk", "upload_id", upload_id)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("append_dataset_chunk", str(e))


async def finalize_dataset_upload(upload_id: str, ctx: Context = None) -> str:
    """
    Finalize a staged upload and register it as a dataset_id for downstream tools.
    """
    try:
        record = registry.finalize_upload(upload_id)
        if record is None:
            return unknown_handle_error(
                "finalize_dataset_upload", "upload_id", upload_id
            )
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("finalize_dataset_upload", str(e))
