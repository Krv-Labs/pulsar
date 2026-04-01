"""Utilities for hashing configuration-sensitive pipeline artifacts."""

import hashlib
import json
import os


def pca_fingerprint(cfg, n_rows: int) -> str:
    """
    Compute a fingerprint for PCA configuration and data shape.

    Used to detect when cached PCA embeddings can be reused (same data + PCA params).
    Includes data_path, n_rows, and mtime to invalidate cache on data changes.
    """
    # Use 0 if data path doesn't exist (unlikely at this stage)
    mtime = os.path.getmtime(cfg.data) if os.path.exists(cfg.data) else 0
    payload = json.dumps(
        {
            "data_path": cfg.data,
            "mtime": mtime,
            "dimensions": list(cfg.pca.dimensions),
            "seeds": list(cfg.pca.seeds),
            "n_rows": n_rows,
        }
    )
    return hashlib.sha256(payload.encode()).hexdigest()
