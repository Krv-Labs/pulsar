"""Utilities for hashing configuration-sensitive pipeline artifacts."""

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os


def pca_fingerprint(cfg, n_rows: int, dataframe=None) -> str:
    """
    Compute a fingerprint for PCA configuration and data shape.

    Used to detect when cached PCA embeddings can be reused (same data + PCA params).
    Includes data path metadata, preprocessing config, PCA config, and raw input schema
    so cached embeddings are only reused when the PCA input matrix is identical.
    """
    # Use 0 if data path doesn't exist (unlikely at this stage)
    mtime = os.path.getmtime(cfg.data) if os.path.exists(cfg.data) else 0
    payload = json.dumps(
        {
            "data_path": cfg.data,
            "mtime": mtime,
            "drop_columns": sorted(getattr(cfg, "drop_columns", []) or []),
            "impute": _normalize_mapping(getattr(cfg, "impute", {})),
            "encode": _normalize_mapping(getattr(cfg, "encode", {})),
            "dimensions": list(cfg.pca.dimensions),
            "seeds": list(cfg.pca.seeds),
            "n_rows": n_rows,
            "input_schema": _schema_signature(dataframe),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _normalize_mapping(mapping) -> dict:
    normalized = {}
    for key, value in sorted((mapping or {}).items()):
        if is_dataclass(value):
            normalized[key] = asdict(value)
        else:
            normalized[key] = value
    return normalized


def _schema_signature(dataframe) -> list[dict[str, str]] | None:
    if dataframe is None:
        return None
    return [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(dataframe.columns, dataframe.dtypes)
    ]
