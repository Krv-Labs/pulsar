"""Curated, artifact-based MCP tool surface (Stage 1, build-spec §3.2 + §3.3).

The production HTTP surface (``PULSAR_TOOLSET=curated``). Every interpret/diagnostic tool
loads an ``ArtifactView`` from the object store (NO live model, NO session, NO re-sweep)
and calls the PRESERVED statistics in ``pulsar.mcp.interpreter`` / ``diagnostics`` /
``payloads`` unchanged. Sweeps are async: ``run_topological_sweep`` enqueues and returns a
``job_id``; a worker persists the artifact; ``get_sweep_status`` polls.

Each tool returns a JSON string ``{markdown, structured, vizPayload, confidence}`` (the
``ToolResult`` contract; ``confidence`` is D14 / post-MVP → null). Viz payload labels are
locked to the H0 vocabulary (D15): cosmic_graph, threshold_stability, manifold3d,
feature_signal — never "persistence diagram"/"barcode".
"""
from __future__ import annotations

import dataclasses
import io
import json

import pandas as pd
from fastmcp.exceptions import ToolError

from pulsar.artifacts import load_artifact
from pulsar.mcp.datasets import (
    DatasetAdmissionError,
    data_key,
    dataset_exists,
    ingest_dataframe,
    load_dataset,
)
from pulsar.mcp.diagnostics import diagnose_model
from pulsar.mcp.interpreter import (
    build_feature_evidence_index,
    cluster_profile_payload,
    compare_clusters as _compare_clusters_fn,
    comparison_to_markdown,
    feature_signal_payload,
    resolve_clusters,
)
from pulsar.mcp.jobs import config_hash, get_job_queue
from pulsar.mcp.payloads import (
    build_summary_evidence_payload,
    cluster_profile_payload_to_markdown,
    cluster_result_payload,
    feature_signal_payload_to_markdown,
    summary_evidence_payload_to_markdown,
)
from pulsar.mcp.store import get_object_store

# A 3D PCA dim is included so manifold3d has a real 3-D projection.
DEFAULT_SWEEP_CONFIG = {
    "sweep": {"pca": {"dimensions": [2, 3], "seed": [42]}, "ball_mapper": {"epsilon": [0.5]}},
    "cosmic_graph": {"construction_threshold": "auto"},
    "output": {"n_reps": 4},
}


# --------------------------------------------------------------------------- #
# JSON / result helpers
# --------------------------------------------------------------------------- #
def _json_default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)!r}")


def _result(markdown: str, structured, viz=None, confidence=None) -> str:
    return json.dumps(
        {"markdown": markdown, "structured": structured, "vizPayload": viz, "confidence": confidence},
        default=_json_default,
    )


def _load_view(user_id: str, dataset_id: str, config_hash_: str, store):
    key = f"{user_id}/{dataset_id}/{config_hash_}/artifact.json"
    if not store.exists(key):
        raise ToolError(
            f"No artifact for ref (user={user_id}, dataset={dataset_id}, config={config_hash_}). "
            f"Run run_topological_sweep and poll get_sweep_status until 'done'."
        )
    return load_artifact(json.loads(store.get(key)), store)


def _safe_clusters(view, method: str = "auto", max_k: int = 15):
    try:
        return resolve_clusters(view, method=method, max_k=max_k)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# viz builders (H0 vocabulary — D15)
# --------------------------------------------------------------------------- #
def _project_3d(embeddings):
    import numpy as np

    if not embeddings:
        return None
    emb = np.asarray(max(embeddings, key=lambda a: a.shape[1]), dtype=float)
    d = emb.shape[1]
    if d >= 3:
        return emb[:, :3]
    if d == 2:
        return np.column_stack([emb, np.zeros(len(emb))])
    return np.column_stack([emb, np.zeros((len(emb), 2))])


def _viz_cosmic_graph(view, labels, max_edges: int = 2500):
    lab = [int(x) for x in labels] if labels is not None else []
    nodes = [{"id": i, "cluster": (lab[i] if i < len(lab) else -1)} for i in range(view.n)]
    edges = sorted(
        ((int(u), int(v), float(d.get("weight", 1.0))) for u, v, d in view.cosmic_graph.edges(data=True)),
        key=lambda e: -e[2],
    )[:max_edges]
    return {"kind": "cosmic_graph", "nodes": nodes, "edges": [{"u": u, "v": v, "w": w} for u, v, w in edges]}


def _viz_threshold_stability(view):
    sc = view._stability_curve or {}
    return {
        "kind": "threshold_stability",
        "thresholds": sc.get("thresholds", []),
        "componentCounts": sc.get("componentCounts", []),
        "optimal": sc.get("optimalThreshold", view.resolved_construction_threshold),
    }


def _viz_manifold3d(view, labels):
    pts = _project_3d(view._embeddings)
    if pts is None:
        return None
    lab = [int(x) for x in labels] if labels is not None else [0] * view.n
    return {
        "kind": "manifold3d",
        "points": [[float(x) for x in row] for row in pts],
        "cluster": lab,
        "ids": list(range(view.n)),
    }


def _viz_feature_signal(numeric_rows):
    feats = [
        {
            "name": r.get("column"),
            "z": float(r.get("z_score", 0.0) or 0.0),
            "p": float(r.get("one_vs_rest_p", 1.0) or 1.0),
            "q": float(r.get("one_vs_rest_q", 1.0) or 1.0),
            "tier": r.get("signal_tier", "noise"),
        }
        for r in numeric_rows
        if r.get("column") is not None
    ]
    return {"kind": "feature_signal", "features": feats}


def _diagnose_markdown(gm) -> str:
    return (
        f"**Cosmic graph** — {gm.n_nodes} nodes, {gm.n_edges} edges, density {gm.density:.3f}.\n"
        f"Components: {gm.component_count} (giant fraction {gm.giant_fraction:.2f}, "
        f"singleton fraction {gm.singleton_fraction:.2f}). "
        f"Construction threshold {gm.resolved_construction_threshold:.4f}.\n"
        + ("Advisories: " + "; ".join(a.get("code", "") for a in (gm.advisories or [])) if gm.advisories else "")
    )


# --------------------------------------------------------------------------- #
# ingest / characterize / sweep / status
# --------------------------------------------------------------------------- #
async def ingest_dataset(file_ref: str, name: str = "", user_id: str = "local") -> str:
    """Register an uploaded file (object-store key, csv/parquet) as an immutable snapshot.

    Row-cap admission: over-envelope ⇒ structured error (never a crash).
    """
    store = get_object_store()
    raw = store.get(file_ref)
    df = pd.read_parquet(io.BytesIO(raw)) if file_ref.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))
    try:
        meta = ingest_dataframe(df, store, user_id=user_id, name=name, source="upload")
    except DatasetAdmissionError as e:
        raise ToolError(str(e))
    md = f"Ingested `{meta['datasetId']}` — {meta['nRows']} rows × {meta['nCols']} cols."
    return _result(md, meta)


async def characterize_dataset(dataset_id: str, user_id: str = "local") -> str:
    """Raw-geometry characterization (no sweep). Used to gate whether a sweep is worthwhile."""
    store = get_object_store()
    if not dataset_exists(dataset_id, store, user_id=user_id):
        raise ToolError(f"dataset {dataset_id} not found; ingest it first.")
    df = load_dataset(dataset_id, store, user_id=user_id)

    from pulsar.analysis.characterization import characterize_dataset as _char
    from pulsar.mcp.characterization import (
        characterization_payload_to_markdown,
        compact_characterization_payload,
    )

    profile = _char("", dataframe=df)
    payload = compact_characterization_payload(dataclasses.asdict(profile))
    md = characterization_payload_to_markdown(payload)
    return _result(md, payload)


async def run_topological_sweep(dataset_id: str, config: dict | None = None, user_id: str = "local") -> str:
    """Enqueue an async sweep (NEVER blocks). Returns a job_id + the artifact_ref to poll for."""
    store = get_object_store()
    queue = get_job_queue()
    if not dataset_exists(dataset_id, store, user_id=user_id):
        raise ToolError(f"dataset {dataset_id} not found; ingest it first.")
    cfg = config or DEFAULT_SWEEP_CONFIG
    ch = config_hash(cfg)
    job_id = queue.enqueue(
        {
            "user_id": user_id,
            "dataset_id": dataset_id,
            "config_hash": ch,
            "data_ref": data_key(user_id, dataset_id),
            "config": cfg,
        }
    )
    structured = {
        "jobId": job_id,
        "status": "queued",
        "artifactRef": {"userId": user_id, "datasetId": dataset_id, "configHash": ch},
    }
    md = (
        f"Sweep enqueued (job `{job_id}`). Poll `get_sweep_status` until `done`, "
        f"then interpret with artifact_ref (dataset `{dataset_id}`, config `{ch}`)."
    )
    return _result(md, structured)


async def get_sweep_status(job_id: str) -> str:
    """Poll an async sweep. status ∈ queued|running|done|error; carries artifact_ref when done."""
    rec = get_job_queue().status(job_id)
    if rec is None:
        raise ToolError(f"unknown job {job_id}")
    structured = {
        "status": rec["status"],
        "artifactRef": rec.get("artifact_ref"),
        "structureStatus": rec.get("structure_status"),
        "error": rec.get("error"),
        "peakRssMb": rec.get("peak_rss_mb"),
        "vcpuMs": rec.get("vcpu_ms"),
    }
    md = f"Job `{job_id}`: **{rec['status']}**"
    if rec.get("structure_status"):
        md += f" — {rec['structure_status']}"
    if rec.get("error"):
        md += f" — error: {rec['error']}"
    return _result(md, structured)


# --------------------------------------------------------------------------- #
# interpret tools (artifact_ref = userId/datasetId/configHash)
# --------------------------------------------------------------------------- #
async def diagnose_cosmic_graph(dataset_id: str, config_hash: str, user_id: str = "local") -> str:
    """H0 cosmic-graph diagnostics off the persisted artifact. viz: cosmic_graph."""
    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    gm = diagnose_model(view)
    structured = dataclasses.asdict(gm)
    cr = _safe_clusters(view)
    labels = list(cr.labels) if cr else None
    return _result(_diagnose_markdown(gm), structured, _viz_cosmic_graph(view, labels))


async def generate_cluster_dossier(
    dataset_id: str,
    config_hash: str,
    method: str = "auto",
    max_k: int = 15,
    user_id: str = "local",
) -> str:
    """Cluster dossier off the persisted artifact. viz: manifold3d (3-D embedding by cluster)."""
    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    cr = _safe_clusters(view, method=method, max_k=max_k)
    if cr is None:
        return _result(
            "No reliable connected-component structure detected (no stable H0 threshold plateau). "
            "Interpretation is not advised on this snapshot.",
            {"structureStatus": "no_reliable_structure"},
            _viz_threshold_stability(view),
        )
    fei = build_feature_evidence_index(view, view.data, cr.labels)
    gm = diagnose_model(view)
    cmeta = cluster_result_payload(cr)
    structured = build_summary_evidence_payload(fei, cmeta, dataclasses.asdict(gm))
    md = summary_evidence_payload_to_markdown(structured)
    return _result(md, structured, _viz_manifold3d(view, cr.labels))


async def get_feature_signal(
    dataset_id: str,
    config_hash: str,
    feature_names: list[str],
    cluster_ids: list[int] | None = None,
    user_id: str = "local",
) -> str:
    """Per-feature H0 enrichment signal off the persisted artifact. viz: feature_signal."""
    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    cr = _safe_clusters(view)
    if cr is None:
        raise ToolError("No reliable structure detected; run generate_cluster_dossier first.")
    fei = build_feature_evidence_index(view, view.data, cr.labels)
    signals = feature_signal_payload(fei, feature_names, cluster_ids=cluster_ids)
    rows = []
    for c in signals.get("clusters", []):
        rows.extend(c.get("numeric_features", []))
    best: dict = {}
    for r in rows:
        col = r.get("column")
        if col is None:
            continue
        if col not in best or abs(r.get("z_score", 0.0) or 0.0) > abs(best[col].get("z_score", 0.0) or 0.0):
            best[col] = r
    md = feature_signal_payload_to_markdown(signals)
    return _result(md, {"signals": signals}, _viz_feature_signal(list(best.values())))


async def get_cluster_profile(
    dataset_id: str,
    config_hash: str,
    cluster_id: int,
    detail: str = "standard",
    max_features: int = 16,
    user_id: str = "local",
) -> str:
    """Full profile of one cluster off the persisted artifact."""
    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    cr = _safe_clusters(view)
    if cr is None:
        raise ToolError("No reliable structure detected; run generate_cluster_dossier first.")
    fei = build_feature_evidence_index(view, view.data, cr.labels)
    cluster = cluster_profile_payload(fei, cluster_id, detail=detail, max_features=max_features)
    payload = {
        "status": "ok",
        "cluster_result": cluster_result_payload(cr),
        "detail": detail,
        "max_features": max_features,
        "cluster": cluster,
    }
    md = cluster_profile_payload_to_markdown(payload)
    return _result(md, payload, _viz_feature_signal(cluster.get("numeric_features", [])))


async def compare_clusters(
    dataset_id: str, config_hash: str, cluster_a: int, cluster_b: int, user_id: str = "local"
) -> str:
    """Pairwise cluster comparison off the persisted artifact."""
    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    cr = _safe_clusters(view)
    if cr is None:
        raise ToolError("No reliable structure detected; run generate_cluster_dossier first.")
    results = _compare_clusters_fn(view.data, cr.labels, cluster_a, cluster_b)
    md = comparison_to_markdown(cluster_a, cluster_b, results)
    return _result(md, {"comparison": results, "clusterA": cluster_a, "clusterB": cluster_b})


async def sync_to_pulsar(
    parent_dataset_id: str,
    sandbox_file_ref: str,
    config: dict | None = None,
    user_id: str = "local",
) -> str:
    """D13 explicit sync: register a sandbox-modified file as a NEW snapshot + enqueue ONE sweep."""
    store = get_object_store()
    queue = get_job_queue()
    raw = store.get(sandbox_file_ref)
    df = (
        pd.read_parquet(io.BytesIO(raw))
        if sandbox_file_ref.endswith(".parquet")
        else pd.read_csv(io.BytesIO(raw))
    )
    try:
        meta = ingest_dataframe(
            df, store, user_id=user_id, source="sync", parent_dataset_id=parent_dataset_id
        )
    except DatasetAdmissionError as e:
        raise ToolError(str(e))
    new_id = meta["datasetId"]
    cfg = config or DEFAULT_SWEEP_CONFIG
    ch = config_hash(cfg)
    job_id = queue.enqueue(
        {
            "user_id": user_id,
            "dataset_id": new_id,
            "config_hash": ch,
            "data_ref": data_key(user_id, new_id),
            "config": cfg,
        }
    )
    structured = {
        "newDatasetId": new_id,
        "jobId": job_id,
        "artifactRef": {"userId": user_id, "datasetId": new_id, "configHash": ch},
    }
    md = (
        f"Synced sandbox file → new snapshot `{new_id}` (parent `{parent_dataset_id}`); "
        f"sweep enqueued (`{job_id}`)."
    )
    return _result(md, structured)


CURATED_TOOLS_LIST = [
    ingest_dataset,
    characterize_dataset,
    run_topological_sweep,
    get_sweep_status,
    diagnose_cosmic_graph,
    generate_cluster_dossier,
    get_feature_signal,
    get_cluster_profile,
    compare_clusters,
    sync_to_pulsar,
]
