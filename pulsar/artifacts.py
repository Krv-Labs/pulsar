"""Derived-artifact serialization for the Pulsar MCP service (Spike 3 / build-spec §3.1).

Spike 3 (ran against this repo) confirmed the live PyO3 objects on a fitted ``ThemaRS``
— ``_cosmic_rust`` (CosmicGraph), ``_stability_result`` (StabilityResult), ``_ball_maps``
(list[BallMapper]) — **cannot** be pickled/dilled. So we persist only the DERIVED
artifacts, which round-trip cleanly:

  weighted_adjacency (CSR sparse), cosmic_graph (edge list), embeddings (.npy),
  preprocessed_data + raw data (parquet), resolved construction threshold,
  the threshold-stability curve, default cluster labels, and graph metrics.

``ArtifactView`` duck-types the post-fit ``ThemaRS`` read surface (``weighted_adjacency``,
``cosmic_graph``, ``resolved_construction_threshold``, ``data``, ``preprocessed_data``,
``_embeddings``, ``ball_maps`` len, lazily-recomputed ``stability_result``) so the
existing ``pulsar.mcp.interpreter`` / ``diagnostics`` functions run unchanged off a
loaded artifact — with NO live model and NO PyO3 object held across processes.

The on-disk JSON aligns with ``PulsarArtifact`` in ``packages/contracts`` (D17 versioned
contract); ``dataRef``/``dataColumns``/``nBallMaps`` are pulsar-side extensions needed for
faithful interpretation (mirrored as optional fields in the TS contract).
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1


def pulsar_version() -> str:
    """Pinned Pulsar version for the artifact (D17). Falls back to env/local marker."""
    from importlib import metadata

    for name in ("thema-pulsar", "thema_pulsar", "pulsar"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return os.environ.get("PULSAR_VERSION", "0.0.0+local")


# --------------------------------------------------------------------------- #
# blob helpers
# --------------------------------------------------------------------------- #
def _ndarray_bytes(arr) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr))
    return buf.getvalue()


def _load_ndarray(data: bytes):
    import numpy as np

    return np.load(io.BytesIO(data))


def _df_parquet_bytes(df) -> bytes:
    buf = io.BytesIO()
    # Stringify columns for parquet; positional row order is preserved (the
    # row-alignment with graph nodes 0..n-1 is positional).
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(buf, index=False)
    return buf.getvalue()


def _load_df_parquet(data: bytes):
    import pandas as pd

    return pd.read_parquet(io.BytesIO(data))


def _graph_metrics(G, n: int) -> dict[str, float]:
    """density / componentCount / giant + singleton fraction (mirrors diagnose_model)."""
    import networkx as nx

    n_edges = G.number_of_edges()
    max_edges = n * (n - 1) // 2
    density = (n_edges / max_edges) if max_edges else 0.0
    comps = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in comps), reverse=True)
    giant_fraction = (sizes[0] / n) if (sizes and n) else 0.0
    singleton_fraction = (sum(1 for s in sizes if s == 1) / n) if n else 0.0
    return {
        "n": float(n),
        "nEdges": float(n_edges),
        "density": float(density),
        "componentCount": float(len(comps)),
        "giantFraction": float(giant_fraction),
        "singletonFraction": float(singleton_fraction),
    }


def _safe_cluster_labels(model, n: int) -> list[int]:
    """Default clustering (recomputable from adjacency). All-zeros if no stable cut."""
    try:
        from pulsar.mcp.interpreter import resolve_clusters

        cr = resolve_clusters(model, method="auto")
        return [int(x) for x in cr.labels.tolist()]
    except Exception:
        return [0] * n


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def dump_artifact(model, *, dataset_id: str, config_hash: str, prefix: str, store) -> dict[str, Any]:
    """Serialize a fitted ``ThemaRS`` to a derived-artifact dict + blobs in ``store``.

    ``prefix`` is the object-store key prefix (``{user_id}/{dataset_id}/{config_hash}``).
    Returns the JSON-able artifact dict (the caller persists it as ``artifact.json``).
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    W = np.asarray(model._weighted_adjacency, dtype=np.float64)
    n = int(W.shape[0])
    G = model._cosmic_graph

    csr = csr_matrix(W)
    weighted_adjacency = {
        "format": "csr",
        "indptr": [int(x) for x in csr.indptr.tolist()],
        "indices": [int(x) for x in csr.indices.tolist()],
        "data": [float(x) for x in csr.data.tolist()],
        "shape": [n, n],
    }
    edges = [
        {"u": int(u), "v": int(v), "w": float(d.get("weight", 1.0))}
        for u, v, d in G.edges(data=True)
    ]

    embeddings_meta = []
    for i, emb in enumerate(getattr(model, "_embeddings", []) or []):
        arr = np.asarray(emb, dtype=np.float32)
        ref = f"{prefix}/embeddings/emb_{i}.npy"
        store.put(ref, _ndarray_bytes(arr))
        embeddings_meta.append({"seed": i, "dims": int(arr.shape[1]), "ref": ref})

    pre_ref = f"{prefix}/preprocessed.parquet"
    store.put(pre_ref, _df_parquet_bytes(model._preprocessed_data))
    data_ref = f"{prefix}/data.parquet"
    store.put(data_ref, _df_parquet_bytes(model._data))

    sr = model._stability_result
    stability_curve = (
        {
            "optimalThreshold": float(sr.optimal_threshold),
            "thresholds": [float(t) for t in sr.thresholds],
            "componentCounts": [int(c) for c in sr.component_counts],
        }
        if sr is not None
        else None
    )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "datasetId": dataset_id,
        "configHash": config_hash,
        "pulsarVersion": pulsar_version(),
        "n": n,
        "weightedAdjacency": weighted_adjacency,
        "cosmicGraph": {"edges": edges},
        "embeddings": embeddings_meta,
        "preprocessedColumns": [str(c) for c in model._preprocessed_data.columns],
        "preprocessedDataRef": pre_ref,
        "dataColumns": [str(c) for c in model._data.columns],
        "dataRef": data_ref,
        "clusterLabels": _safe_cluster_labels(model, n),
        "stabilityCurve": stability_curve,
        "resolvedConstructionThreshold": float(model._resolved_construction_threshold),
        "nBallMaps": int(len(model._ball_maps)),
        "metrics": _graph_metrics(G, n),
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
class _BallMapsProxy:
    """Length-only stand-in: BallMapper PyO3 objects are not persisted (Spike 3)."""

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        return self._n > 0

    def __iter__(self):
        raise RuntimeError(
            "BallMapper objects are not persisted (Spike 3); only the count is available."
        )


class ArtifactView:
    """Read-only, duck-typed stand-in for a fitted ``ThemaRS``, backed by a persisted artifact.

    Exposes the same attributes/properties the curated interpret/diagnostic functions
    read, so they run unchanged with no live model. ``stability_result`` is recomputed
    lazily from the persisted adjacency via the Rust ``find_stable_thresholds`` (cheap,
    deterministic) — it is NOT the original PyO3 object held in memory.
    """

    def __init__(
        self,
        *,
        weighted_adjacency,
        cosmic_graph,
        embeddings,
        preprocessed_data,
        data,
        resolved_construction_threshold: float,
        stability_curve: dict | None,
        cluster_labels: list[int],
        metrics: dict,
        dataset_id: str | None,
        config_hash: str | None,
        pulsar_version: str | None,
        n: int,
        n_ball_maps: int,
    ) -> None:
        self._weighted_adjacency = weighted_adjacency
        self._cosmic_graph = cosmic_graph
        self._embeddings = embeddings
        self._preprocessed_data = preprocessed_data
        self._data = data
        self._resolved_construction_threshold = float(resolved_construction_threshold)
        self._stability_curve = stability_curve
        self._cluster_labels = cluster_labels
        self.metrics = metrics
        self.dataset_id = dataset_id
        self.config_hash = config_hash
        self.pulsar_version = pulsar_version
        self.n = int(n)
        self._ball_maps = _BallMapsProxy(n_ball_maps)
        self._stability_result_cache = None

    # ThemaRS-compatible read surface ------------------------------------- #
    @property
    def weighted_adjacency(self):
        return self._weighted_adjacency

    @property
    def cosmic_graph(self):
        return self._cosmic_graph

    @property
    def resolved_construction_threshold(self) -> float:
        return self._resolved_construction_threshold

    @property
    def preprocessed_data(self):
        return self._preprocessed_data

    @property
    def data(self):
        return self._data

    @property
    def ball_maps(self):
        return self._ball_maps

    @property
    def stability_result(self):
        """Lazily recomputed from the persisted adjacency (deterministic; not the live object)."""
        if self._stability_result_cache is None:
            from pulsar._pulsar import find_stable_thresholds

            self._stability_result_cache = find_stable_thresholds(self._weighted_adjacency)
        return self._stability_result_cache

    @property
    def _stability_result(self):  # alias for helpers that read the private attr name
        return self.stability_result


def load_artifact(d: dict, store) -> ArtifactView:
    """Reconstruct an ``ArtifactView`` from an artifact dict + blobs in ``store``."""
    import networkx as nx
    import numpy as np
    from scipy.sparse import csr_matrix

    wa = d["weightedAdjacency"]
    shape = tuple(int(x) for x in wa["shape"])
    n = shape[0]
    W = csr_matrix(
        (
            np.asarray(wa["data"], dtype=np.float64),
            np.asarray(wa["indices"], dtype=np.int64),
            np.asarray(wa["indptr"], dtype=np.int64),
        ),
        shape=shape,
    ).toarray().astype(np.float64)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in d["cosmicGraph"]["edges"]:
        G.add_edge(int(e["u"]), int(e["v"]), weight=float(e["w"]))

    embeddings = [_load_ndarray(store.get(m["ref"])) for m in d.get("embeddings", [])]
    preprocessed = _load_df_parquet(store.get(d["preprocessedDataRef"]))
    data = _load_df_parquet(store.get(d["dataRef"])) if d.get("dataRef") else preprocessed

    return ArtifactView(
        weighted_adjacency=W,
        cosmic_graph=G,
        embeddings=embeddings,
        preprocessed_data=preprocessed,
        data=data,
        resolved_construction_threshold=float(d["resolvedConstructionThreshold"]),
        stability_curve=d.get("stabilityCurve"),
        cluster_labels=d.get("clusterLabels", []),
        metrics=d.get("metrics", {}),
        dataset_id=d.get("datasetId"),
        config_hash=d.get("configHash"),
        pulsar_version=d.get("pulsarVersion"),
        n=n,
        n_ball_maps=int(d.get("nBallMaps", 0)),
    )
