from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Annotated, Any, Literal
import uuid

from fastmcp import Context
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.registry import registry
from pulsar.mcp.results import mcp_result
from pulsar.mcp.session import _get_session, _read_dataset_file, _resolve_dataset_path
from pulsar.mcp.toolmeta import (
    INT,
    NUM,
    STR,
    nullable,
    obj,
    tool_meta,
)

logger = logging.getLogger(__name__)

_DATASET_RECORD_SCHEMA = obj(
    {
        "dataset_id": STR,
        "path": STR,
        "name": STR,
        "size_bytes": INT,
        "mtime_ns": INT,
        "ingested_at": NUM,
        "source": STR,
    },
    ["dataset_id", "path", "name", "size_bytes", "mtime_ns", "ingested_at", "source"],
)
_UPLOAD_RECORD_SCHEMA = obj(
    {
        "upload_id": STR,
        "filename": STR,
        "media_type": STR,
        "created_at": NUM,
        "bytes_received": INT,
    },
    ["upload_id", "filename", "media_type", "created_at", "bytes_received"],
)
_SAMPLE_SCHEMA = obj(
    {
        "dataset_id": STR,
        "source_dataset_id": STR,
        "entity_column": STR,
        "max_entities": INT,
        "min_observations": INT,
        "time_column": nullable(STR),
        "min_time_points": nullable(INT),
        "eligible_entities": INT,
        "sampled_entities": INT,
        "sampled_rows": INT,
    },
    [
        "dataset_id",
        "source_dataset_id",
        "entity_column",
        "max_entities",
        "min_observations",
        "eligible_entities",
        "sampled_entities",
        "sampled_rows",
    ],
)


def _select_longitudinal_entities(
    frame,
    entity_column: str,
    max_entities: int,
    min_observations: int,
    time_column: str | None,
    min_time_points: int | None,
) -> tuple[list, int]:
    import pandas as pd

    counts = frame.groupby(entity_column, dropna=False).size()
    eligible = [
        entity
        for entity, count in counts.items()
        if count >= min_observations and pd.notna(entity)
    ]
    if min_time_points is not None:
        time_counts = frame.groupby(entity_column, dropna=False)[time_column].nunique(
            dropna=True
        )
        eligible = [
            entity
            for entity in eligible
            if time_counts.get(entity, 0) >= min_time_points
        ]

    def entity_order(entity):
        identity = f"{type(entity).__module__}.{type(entity).__qualname__}:{entity!r}"
        return hashlib.sha256(identity.encode("utf-8")).digest()

    return sorted(eligible, key=entity_order)[:max_entities], len(eligible)


def _sample_longitudinal_entities_worker(
    *,
    dataset_id: str,
    source_path: str,
    entity_column: str,
    max_entities: int,
    min_observations: int,
    time_column: str | None,
    min_time_points: int | None,
) -> dict[str, Any] | ToolResult:
    frame = _read_dataset_file(source_path)
    if entity_column not in frame.columns:
        return mcp_error(
            "sample_longitudinal_entities",
            f"Entity column '{entity_column}' is not present in the dataset.",
            error_code="ENTITY_COLUMN_UNKNOWN",
            agent_action="Call characterize_dataset or probe_columns and pass an existing entity column.",
            details={"dataset_id": dataset_id, "entity_column": entity_column},
        )
    if time_column is not None and time_column not in frame.columns:
        return mcp_error(
            "sample_longitudinal_entities",
            f"Time column '{time_column}' is not present in the dataset.",
            error_code="TIME_COLUMN_UNKNOWN",
            agent_action="Pass an existing discrete time column or omit time_column.",
            details={"dataset_id": dataset_id, "time_column": time_column},
        )

    selected, eligible_count = _select_longitudinal_entities(
        frame,
        entity_column,
        max_entities,
        min_observations,
        time_column,
        min_time_points,
    )
    sampled = frame[frame[entity_column].isin(selected)]
    if sampled.empty:
        return mcp_error(
            "sample_longitudinal_entities",
            "No entities meet the requested minimum observation and timepoint counts.",
            error_code="NO_ELIGIBLE_ENTITIES",
            agent_action="Lower the minimum counts or use different entity and time columns.",
            details={
                "dataset_id": dataset_id,
                "entity_column": entity_column,
                "min_observations": min_observations,
                "time_column": time_column,
                "min_time_points": min_time_points,
            },
        )

    # CSV sources stay CSV: pyarrow rejects the mixed-type object columns
    # read_csv can produce, and a CSV round-trip re-reads like the source.
    suffix = ".parquet" if source_path.lower().endswith(".parquet") else ".csv"
    writer = sampled.to_parquet if suffix == ".parquet" else sampled.to_csv
    staging_path = registry.cache_dir / f"sample_{uuid.uuid4().hex}{suffix}"
    try:
        writer(staging_path, index=False)
        record = registry.register_dataset_from_file(
            f"{Path(source_path).stem}_sample{suffix}",
            staging_path,
            source="entity_sample",
        )
    finally:
        staging_path.unlink(missing_ok=True)
    return {
        "dataset_id": record.dataset_id,
        "source_dataset_id": dataset_id,
        "entity_column": entity_column,
        "max_entities": max_entities,
        "min_observations": min_observations,
        "time_column": time_column,
        "min_time_points": min_time_points,
        "eligible_entities": eligible_count,
        "sampled_entities": len(selected),
        "sampled_rows": int(len(sampled)),
    }


@tool_meta(
    title="Ingest Dataset", read_only=False, output_schema=_DATASET_RECORD_SCHEMA
)
async def ingest_dataset(
    path: Annotated[
        str,
        Field(
            description=(
                "Absolute path to a CSV or Parquet file on the machine running this server (host-visible; sandbox-only paths fail)."
            )
        ),
    ],
    ctx: Context = None,
) -> ToolResult:
    """Register a host-visible CSV/Parquet file and return its `dataset_id`.

    Returns the dataset record (`dataset_id`, resolved path, size). Pass
    `dataset_id` to characterize_dataset, create_config, probe_columns and
    run_topological_sweep. Start here; no other tool reads raw paths.
    """
    try:
        record = await asyncio.to_thread(registry.register_dataset, path)
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return mcp_result(dataclasses.asdict(record))
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
        return mcp_error(
            "ingest_dataset",
            str(e),
            error_code="INGEST_FAILED",
            agent_action=(
                "Check that the file is a readable CSV or Parquet file, then retry "
                "ingest_dataset with a host-visible absolute path."
            ),
        )


@tool_meta(
    title="Sample Longitudinal Entities", read_only=False, output_schema=_SAMPLE_SCHEMA
)
async def sample_longitudinal_entities(
    dataset_id: Annotated[
        str,
        Field(description="Source dataset handle returned by ingest_dataset."),
    ],
    entity_column: Annotated[
        str,
        Field(
            description=(
                "Column that identifies the repeated entity (e.g. a patient or "
                "device id); all rows of a sampled entity are kept."
            )
        ),
    ],
    max_entities: Annotated[
        int,
        Field(description="Maximum number of entities to keep (>= 1)."),
    ] = 2000,
    min_observations: Annotated[
        int,
        Field(description="Minimum number of rows an entity needs to be eligible."),
    ] = 3,
    ctx: Context = None,
    time_column: Annotated[
        str | None,
        Field(
            description=(
                "Optional discrete time column; required when min_time_points is set."
            )
        ),
    ] = None,
    min_time_points: Annotated[
        int | None,
        Field(
            description=(
                "Optional minimum number of distinct time_column values an entity "
                "needs to be eligible."
            )
        ),
    ] = None,
) -> ToolResult:
    """Create a deterministic, row-preserving sample of longitudinal entities.

    Returns a NEW `dataset_id` (registered) plus eligible/sampled counts; no rows
    or identifiers are returned. Use before build_longitudinal_graph on large
    panels. The source must already be ingested.
    """
    if (
        max_entities < 1
        or min_observations < 1
        or (min_time_points is not None and min_time_points < 1)
        or (min_time_points is not None and time_column is None)
    ):
        return mcp_error(
            "sample_longitudinal_entities",
            "max_entities and min_observations must be positive, and min_time_points requires a positive value and time_column.",
            error_code="INVALID_SAMPLE_SIZE",
            agent_action="Use positive values and provide time_column when setting min_time_points.",
        )

    try:
        source_path = _resolve_dataset_path(dataset_id)
    except LookupError:
        return unknown_handle_error(
            "sample_longitudinal_entities", "dataset_id", dataset_id
        )

    try:
        outcome = await asyncio.to_thread(
            _sample_longitudinal_entities_worker,
            dataset_id=dataset_id,
            source_path=source_path,
            entity_column=entity_column,
            max_entities=max_entities,
            min_observations=min_observations,
            time_column=time_column,
            min_time_points=min_time_points,
        )
        if isinstance(outcome, ToolResult):
            return outcome
        return mcp_result(outcome)
    except Exception as e:
        logger.exception("Could not sample longitudinal entities")
        return mcp_error(
            "sample_longitudinal_entities",
            str(e),
            error_code="SAMPLE_FAILED",
            agent_action="Check the dataset file and column names, then retry.",
        )


@tool_meta(
    title="Begin Dataset Upload", read_only=False, output_schema=_UPLOAD_RECORD_SCHEMA
)
async def begin_dataset_upload(
    filename: Annotated[
        str, Field(description="File name to store the upload under, e.g. data.csv.")
    ],
    media_type: Annotated[
        str, Field(description="MIME type of the file, e.g. text/csv.")
    ] = "text/csv",
    ctx: Context = None,
) -> ToolResult:
    """Begin a staged chunked upload (only for clients that cannot share a host path).

    Returns an `upload_id`; then call append_dataset_chunk repeatedly and
    finalize_dataset_upload to get a `dataset_id`. Prefer ingest_dataset when the
    file is visible to the server.
    """
    try:
        record = await asyncio.to_thread(
            registry.begin_upload, filename, media_type=media_type
        )
        return mcp_result(dataclasses.asdict(record))
    except Exception as e:
        return mcp_error(
            "begin_dataset_upload",
            str(e),
            error_code="UPLOAD_BEGIN_FAILED",
            agent_action="Retry with a plain file name and a valid media_type.",
        )


@tool_meta(
    title="Append Dataset Chunk", read_only=False, output_schema=_UPLOAD_RECORD_SCHEMA
)
async def append_dataset_chunk(
    upload_id: Annotated[
        str, Field(description="Upload handle returned by begin_dataset_upload.")
    ],
    chunk: Annotated[
        str, Field(description="The next slice of the file, encoded per `encoding`.")
    ],
    encoding: Annotated[
        Literal["base64", "utf-8"],
        Field(
            description=(
                "How `chunk` is encoded: base64 (binary-safe, default) or utf-8 "
                "(plain text)."
            )
        ),
    ] = "base64",
    ctx: Context = None,
) -> ToolResult:
    """Append one chunk to a staged upload and return the updated byte count.

    Call repeatedly in file order after begin_dataset_upload, then
    finalize_dataset_upload.
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

        record = await asyncio.to_thread(
            registry.append_upload_chunk, upload_id, chunk_bytes
        )
        if record is None:
            return unknown_handle_error("append_dataset_chunk", "upload_id", upload_id)
        return mcp_result(dataclasses.asdict(record))
    except Exception as e:
        return mcp_error(
            "append_dataset_chunk",
            str(e),
            error_code="UPLOAD_APPEND_FAILED",
            agent_action="Retry the chunk, or restart with begin_dataset_upload.",
        )


@tool_meta(
    title="Finalize Dataset Upload",
    read_only=False,
    output_schema=_DATASET_RECORD_SCHEMA,
)
async def finalize_dataset_upload(
    upload_id: Annotated[
        str, Field(description="Upload handle returned by begin_dataset_upload.")
    ],
    ctx: Context = None,
) -> ToolResult:
    """Finalize a staged upload and register it; returns the dataset record.

    The returned `dataset_id` works like one from ingest_dataset.
    """
    try:
        record = await asyncio.to_thread(registry.finalize_upload, upload_id)
        if record is None:
            return unknown_handle_error(
                "finalize_dataset_upload", "upload_id", upload_id
            )
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return mcp_result(dataclasses.asdict(record))
    except Exception as e:
        return mcp_error(
            "finalize_dataset_upload",
            str(e),
            error_code="UPLOAD_FINALIZE_FAILED",
            agent_action="Check that all chunks were appended, then retry.",
        )
