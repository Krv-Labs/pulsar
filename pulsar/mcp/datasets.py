"""Dataset ingest + resolution for the curated artifact-based MCP surface (build-spec §3.1).

A ``dataset_id`` is a CONTENT fingerprint (``ds_`` + sha256 of canonicalized bytes), so
identical data ⇒ identical id (enables dedup). Snapshots are immutable; sandbox edits
produce a NEW dataset_id, re-swept explicitly via ``sync_to_pulsar`` (D13).

Datasets are stored per-user under ``{user_id}/datasets/{dataset_id}/`` in the object
store: ``data.parquet`` + ``meta.json``. Ingest enforces the Spike-3 row cap (clean,
structured error — never a crash).
"""
from __future__ import annotations

import hashlib
import io
import json

import pandas as pd

from pulsar.config import MAX_ROWS


class DatasetAdmissionError(Exception):
    """Dataset exceeds the compute envelope (e.g. row cap)."""


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Content fingerprint: stable column order, normalized to CSV bytes."""
    canon = df.reindex(sorted(df.columns, key=str), axis=1).to_csv(index=False).encode()
    return "ds_" + hashlib.sha256(canon).hexdigest()[:16]


def data_key(user_id: str, dataset_id: str) -> str:
    return f"{user_id}/datasets/{dataset_id}/data.parquet"


def meta_key(user_id: str, dataset_id: str) -> str:
    return f"{user_id}/datasets/{dataset_id}/meta.json"


def ingest_dataframe(
    df: pd.DataFrame,
    store,
    *,
    user_id: str,
    name: str = "",
    source: str = "upload",
    parent_dataset_id: str | None = None,
    max_rows: int | None = None,
) -> dict:
    """Register a dataframe as an immutable snapshot. Raises DatasetAdmissionError over the row cap."""
    cap = int(max_rows or MAX_ROWS)
    if len(df) > cap:
        raise DatasetAdmissionError(
            f"row cap exceeded: {len(df)} rows > MAX_ROWS={cap} "
            f"(Spike-3 memory envelope; n>10k prohibitive). Downsample before ingest."
        )
    ds_id = dataset_fingerprint(df)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    store.put(data_key(user_id, ds_id), buf.getvalue())
    meta = {
        "datasetId": ds_id,
        "userId": user_id,
        "nRows": int(len(df)),
        "nCols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "name": name or ds_id,
        "source": source,
        "parentDatasetId": parent_dataset_id,
    }
    store.put(meta_key(user_id, ds_id), json.dumps(meta).encode())
    return meta


def load_dataset(dataset_id: str, store, *, user_id: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.get(data_key(user_id, dataset_id))))


def load_dataset_meta(dataset_id: str, store, *, user_id: str) -> dict:
    return json.loads(store.get(meta_key(user_id, dataset_id)).decode())


def dataset_exists(dataset_id: str, store, *, user_id: str) -> bool:
    return store.exists(data_key(user_id, dataset_id))
