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

import numpy as np
import pandas as pd
import yaml
from fastmcp.exceptions import ToolError

from pulsar.artifacts import load_artifact
from pulsar.mcp.config_tools import _build_initial_config_yaml, validate_config_yaml
from pulsar.mcp.config_refs import load_config_ref, save_config_ref
from pulsar.mcp.datasets import (
    DatasetAdmissionError,
    data_key,
    dataset_exists,
    ingest_dataframe,
    load_dataset,
)
from pulsar.mcp.diagnostics import (
    _build_graph_summary,
    _skeleton_graph_payload,
    diagnose_model,
)
from pulsar.mcp.interpreter import (
    build_feature_evidence_index,
    cluster_profile_payload,
    compare_clusters as _compare_clusters_fn,
    comparison_to_markdown,
    feature_signal_payload,
    resolve_clusters,
    signal_matrix_payload,
)
from pulsar.mcp.jobs import config_hash, get_job_queue
from pulsar.mcp.payloads import (
    build_summary_evidence_payload,
    cluster_profile_payload_to_markdown,
    cluster_result_payload,
    feature_signal_payload_to_markdown,
    singleton_count_at_threshold,
    summary_evidence_payload_to_markdown,
)
from pulsar.mcp.preprocessing import _calibrate_processed_space
from pulsar.mcp.store import get_object_store
from pulsar.mcp.thresholds import (
    THRESHOLD_CANDIDATE_POLICIES,
    agent_threshold_options,
    component_mass_profile,
    mass_profile_hint,
    prepare_threshold_graph,
    structural_breakpoints,
)


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


def _dataset_validation_path(store, user_id: str, dataset_id: str) -> str:
    return str(store.root / data_key(user_id, dataset_id))


def _validated_config_for_dataset(
    config: dict, *, dataset_id: str, user_id: str, store
) -> tuple[dict, str]:
    data_path = _dataset_validation_path(store, user_id, dataset_id)
    report = validate_config_yaml(yaml.safe_dump(config, sort_keys=False), dataset_path=data_path)
    if not report.ok or report.normalized_yaml is None:
        issues = "; ".join(f"{i.path}: {i.message}" for i in report.issues)
        raise ToolError(f"Config validation failed for dataset {dataset_id}: {issues}")
    return yaml.safe_load(report.normalized_yaml), report.normalized_yaml


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


def _sparse_threshold_curve(
    thresholds: list[float],
    component_counts: list[int],
    singleton_counts: list[int],
    *,
    max_points: int,
) -> list[dict]:
    n_points = len(thresholds)
    if n_points == 0:
        return []
    indices = (
        range(n_points)
        if n_points <= max_points
        else sorted({int(i) for i in np.linspace(0, n_points - 1, num=max_points)})
    )
    return [
        {
            "threshold": thresholds[i],
            "component_count": component_counts[i],
            "singleton_count": singleton_counts[i],
        }
        for i in indices
    ]


def _threshold_agent_readout(selected_profile: dict | None, breakpoints: list[dict]) -> str:
    if not selected_profile:
        return "No threshold mass profile was available."

    largest_pct = float(selected_profile["largest_component_fraction"]) * 100
    small_pct = float(selected_profile["small_component_mass_fraction"]) * 100
    large_breakpoints = [row for row in breakpoints if row["event"] == "large_component_transition"]

    if largest_pct >= 95:
        if large_breakpoints:
            return (
                f"Auto threshold is stable but giant-component dominated ({largest_pct:.2f}% of rows in the largest component); "
                "inspect large structural breakpoints before naming cohorts."
            )
        return (
            f"Auto threshold is stable but produces one giant component containing {largest_pct:.2f}% of rows; "
            f"only {small_pct:.2f}% of rows sit in small components."
        )
    if large_breakpoints:
        return (
            "Auto threshold exposes nontrivial component structure and large threshold transitions "
            "are available for parameter discussion."
        )
    return "Auto threshold exposes nontrivial component structure without large split markers."


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


async def prepare_sweep(
    dataset_id: str,
    intent: str = "",
    config: dict | None = None,
    config_ref: str = "",
    user_id: str = "local",
) -> str:
    """Create/reuse a dataset-calibrated config and validate it before any sweep."""
    store = get_object_store()
    if not dataset_exists(dataset_id, store, user_id=user_id):
        raise ToolError(f"dataset {dataset_id} not found; ingest it first.")

    df = load_dataset(dataset_id, store, user_id=user_id)
    data_path = _dataset_validation_path(store, user_id, dataset_id)
    source = "provided_config" if config is not None else "config_ref" if config_ref else "create_config"

    if config is None and config_ref:
        try:
            config_yaml = load_config_ref(
                store, user_id=user_id, dataset_id=dataset_id, config_ref=config_ref
            )
        except FileNotFoundError as e:
            raise ToolError(str(e)) from e
        config = yaml.safe_load(config_yaml)

    if config is None:
        from pulsar.analysis.characterization import characterize_dataset as _char

        profile = _char(data_path, dataframe=df)
        geo = dataclasses.asdict(profile)
        processed = _calibrate_processed_space(df, geo["column_profiles"], geo["n_samples"], data_path)
        config_yaml = _build_initial_config_yaml(
            geo,
            data_path=data_path,
            run_name=intent.strip() or "initial_sweep",
            processed_profile=processed,
        )
        config = yaml.safe_load(config_yaml)

    validated_config, normalized_yaml = _validated_config_for_dataset(
        config, dataset_id=dataset_id, user_id=user_id, store=store
    )
    ch = config_hash(validated_config)
    save_config_ref(store, user_id=user_id, dataset_id=dataset_id, config_yaml=normalized_yaml)
    structured = {
        "status": "ok",
        "datasetId": dataset_id,
        "configHash": ch,
        "config": validated_config,
        "configYaml": normalized_yaml,
        "source": source,
        "phaseState": "validated",
    }
    md = f"Prepared dataset `{dataset_id}` for sweeping with validated config `{ch}`."
    return _result(md, structured)


async def run_topological_sweep(dataset_id: str, config: dict | None = None, user_id: str = "local") -> str:
    """Enqueue an async sweep (NEVER blocks). Returns a job_id + the artifact_ref to poll for."""
    store = get_object_store()
    queue = get_job_queue()
    if not dataset_exists(dataset_id, store, user_id=user_id):
        raise ToolError(f"dataset {dataset_id} not found; ingest it first.")
    if config is None:
        raise ToolError(
            "Validated config required. Call prepare_sweep(dataset_id) first and pass its returned config to run_topological_sweep."
        )
    cfg, _ = _validated_config_for_dataset(config, dataset_id=dataset_id, user_id=user_id, store=store)
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
    """Poll an async sweep. status ∈ queued|running|done|error|cancelled."""
    rec = get_job_queue().status(job_id)
    if rec is None:
        raise ToolError(f"unknown job {job_id}")
    structured = {
        "status": rec["status"],
        "artifactRef": rec.get("artifact_ref"),
        "structureStatus": rec.get("structure_status"),
        "error": rec.get("error"),
        "cancelReason": rec.get("cancel_reason"),
        "cancelledAt": rec.get("cancelled_at"),
        "peakRssMb": rec.get("peak_rss_mb"),
        "vcpuMs": rec.get("vcpu_ms"),
    }
    md = f"Job `{job_id}`: **{rec['status']}**"
    if rec.get("structure_status"):
        md += f" — {rec['structure_status']}"
    if rec.get("error"):
        md += f" — error: {rec['error']}"
    if rec.get("cancel_reason"):
        md += f" — cancelled: {rec['cancel_reason']}"
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


async def get_threshold_stability_curve(
    dataset_id: str,
    config_hash: str,
    detail: str = "summary",
    threshold_candidate_policy: str = "balanced",
    user_id: str = "local",
) -> str:
    """H0 component stability vs edge-weight threshold off a persisted artifact. viz: threshold_stability."""
    if detail not in {"summary", "full"}:
        raise ToolError("detail must be one of ['summary', 'full']")
    if threshold_candidate_policy not in THRESHOLD_CANDIDATE_POLICIES:
        raise ToolError(
            "threshold_candidate_policy must be one of "
            f"{sorted(THRESHOLD_CANDIDATE_POLICIES)}, got '{threshold_candidate_policy}'"
        )

    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    adj = view.weighted_adjacency
    stability = view.stability_result
    threshold_graph = prepare_threshold_graph(adj)
    thresholds = [float(t) for t in stability.thresholds]
    component_counts = [int(c) for c in stability.component_counts]
    resolved_construction_threshold = float(view.resolved_construction_threshold)
    optimal_threshold = float(stability.optimal_threshold)
    matches_current = bool(np.isclose(resolved_construction_threshold, optimal_threshold))

    row_max = adj.max(axis=1)
    ts = np.asarray(thresholds, dtype=adj.dtype)
    singleton_counts = [int(c) for c in (row_max[None, :] <= ts[:, None]).sum(axis=1)]

    plateau_limit = 10 if detail == "full" else 4
    plateaus = []
    for p in stability.top_k_plateaus(plateau_limit):
        singleton_count = singleton_count_at_threshold(adj, float(p.midpoint))
        mass_profile = component_mass_profile(threshold_graph, float(p.midpoint))
        plateaus.append(
            {
                "start": float(p.start_threshold),
                "end": float(p.end_threshold),
                "component_count": int(p.component_count),
                "length": float(p.length),
                "midpoint": float(p.midpoint),
                "singleton_count": singleton_count,
                "singleton_fraction": round(singleton_count / max(int(adj.shape[0]), 1), 4),
                "component_mass_profile": mass_profile,
                "interpretation_hint": mass_profile_hint(mass_profile),
            }
        )

    breakpoints = structural_breakpoints(threshold_graph, thresholds, component_counts)
    threshold_options = agent_threshold_options(
        threshold_graph,
        stability.top_k_plateaus(20 if detail == "full" else 10),
        thresholds,
        component_counts,
        policy=threshold_candidate_policy,
        max_candidates=12 if detail == "full" else 7,
    )
    selected_profile = plateaus[0].get("component_mass_profile") if plateaus else None
    if detail == "summary":
        for p in plateaus:
            p.pop("component_mass_profile", None)
        for list_key in ["stable_plateau_candidates", "transition_adjacent_candidates", "candidates"]:
            for cand in threshold_options.get(list_key, []):
                cand.pop("component_mass_profile", None)
                cand.pop("mass_shape", None)

    curve_sample = _sparse_threshold_curve(
        thresholds,
        component_counts,
        singleton_counts,
        max_points=25 if detail == "full" else 15,
    )
    structured = {
        "status": "ok",
        "detail": detail,
        "datasetId": dataset_id,
        "configHash": config_hash,
        "resolved_construction_threshold": resolved_construction_threshold,
        "optimal_threshold": optimal_threshold,
        "matches_current_threshold": matches_current,
        "threshold_guidance": (
            "Current graph construction matches the stability optimum."
            if matches_current
            else "Stability optimum differs from the current constructed graph; treat it as a rerun candidate, not a change to the fitted graph."
        ),
        "agent_readout": _threshold_agent_readout(selected_profile, breakpoints),
        "agent_threshold_options": threshold_options,
        "plateaus": plateaus,
        "structural_breakpoints": breakpoints,
        "structural_breakpoints_guidance": (
            "Breakpoints are capped structural transition candidates ranked first by large-component transitions, "
            "then by the smaller mass that actually joins or splits."
        ),
        "curve_point_count": len(thresholds),
        "curve_sample": curve_sample,
        "curve_sample_omitted": max(len(thresholds) - len(curve_sample), 0),
        "curve_sample_guidance": (
            "curve_sample is an evenly spaced sketch of the full threshold curve. "
            "Use detail='full' only when exact per-threshold arrays are needed."
        ),
    }
    if detail == "full":
        structured["thresholds"] = thresholds
        structured["component_counts"] = component_counts
        structured["singleton_counts"] = singleton_counts
    md = (
        f"Threshold stability for `{dataset_id}` / `{config_hash}`: "
        f"optimal {optimal_threshold:.4f}, construction {resolved_construction_threshold:.4f}. "
        f"{structured['agent_readout']}"
    )
    return _result(md, structured, _viz_threshold_stability(view))


async def get_topological_skeleton(
    dataset_id: str,
    config_hash: str,
    detail: str = "summary",
    max_edges: int = 100,
    max_nodes: int = 100,
    user_id: str = "local",
) -> str:
    """Structured graph connectivity off a persisted artifact."""
    if detail not in {"summary", "nodes", "edges", "full", "full_nodes"}:
        raise ToolError(
            "detail must be one of ['summary', 'nodes', 'edges', 'full', 'full_nodes']"
        )
    if max_edges < 1:
        raise ToolError(f"max_edges must be >= 1, got '{max_edges}'")
    if max_nodes < 1:
        raise ToolError(f"max_nodes must be >= 1, got '{max_nodes}'")
    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    graph_summary = _build_graph_summary(view)
    structured = {
        "status": "ok",
        "datasetId": dataset_id,
        "configHash": config_hash,
        "config_yaml_omitted": True,
        "config_yaml_unavailable": (
            "Curated HTTP skeleton reads persisted artifacts only; config YAML is not stored in the artifact."
        ),
        "resolved_construction_threshold": graph_summary.get("resolved_construction_threshold"),
        "graph": _skeleton_graph_payload(graph_summary, detail=detail, max_edges=max_edges, max_nodes=max_nodes),
        "recommended_next_tools": [
            "diagnose_cosmic_graph",
            "get_threshold_stability_curve",
            "generate_cluster_dossier",
        ],
    }
    graph = structured["graph"]
    md = (
        f"Topological skeleton for `{dataset_id}` / `{config_hash}`: "
        f"{graph['node_count']} nodes, {graph['edge_count']} edges, {graph.get('component_count')} components."
    )
    return _result(md, structured, _viz_cosmic_graph(view, view._cluster_labels))


async def get_cluster_signal_matrix(
    dataset_id: str,
    config_hash: str,
    cluster_ids: list[int] | None = None,
    include_context_tier: bool = False,
    max_clusters: int = 8,
    return_markdown: bool = True,
    user_id: str = "local",
) -> str:
    """Cross-cluster feature-signal matrix off a persisted artifact."""
    if max_clusters < 1:
        raise ToolError(f"max_clusters must be >= 1, got '{max_clusters}'")
    view = _load_view(user_id, dataset_id, config_hash, get_object_store())
    cr = _safe_clusters(view)
    if cr is None:
        raise ToolError("No reliable structure detected; run generate_cluster_dossier first.")
    fei = build_feature_evidence_index(view, view.data, cr.labels)
    matrix = signal_matrix_payload(
        fei,
        cluster_ids=cluster_ids,
        include_context_tier=include_context_tier,
        max_clusters=max_clusters,
        return_markdown=return_markdown,
    )
    structured = {
        "status": "ok",
        "datasetId": dataset_id,
        "configHash": config_hash,
        "cluster_result": cluster_result_payload(cr),
        "signal_matrix": matrix,
    }
    md = (
        matrix["markdown_report"]
        if return_markdown and isinstance(matrix, dict)
        else "Cluster signal matrix computed."
    )
    return _result(md, structured)


async def sync_to_pulsar(
    parent_dataset_id: str,
    sandbox_file_ref: str,
    user_id: str = "local",
) -> str:
    """D13 explicit sync: register a sandbox-modified file as a NEW snapshot.

    Sweeps are phase-gated: callers must run prepare_sweep(newDatasetId) before
    run_topological_sweep. This prevents derived datasets from falling back to a
    generic default config.
    """
    store = get_object_store()
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
    structured = {
        "status": "synced",
        "newDatasetId": new_id,
        "parentDatasetId": parent_dataset_id,
        "recommendedNextTool": "prepare_sweep",
    }
    md = (
        f"Synced sandbox file → new snapshot `{new_id}` (parent `{parent_dataset_id}`). "
        "Call `prepare_sweep` on the new dataset before sweeping."
    )
    return _result(md, structured)


# Tenant-safe config/preprocessing helpers + workflow guide (object-store loader, user_id-scoped).
# These give the HTTP agent self-correction parity with the stdio surface; see curated_preprocessing.py.
from pulsar.mcp.tools.curated_preprocessing import (  # noqa: E402
    create_config,
    get_config,
    get_workflow_guide,
    list_sweeps,
    probe_columns,
    recommend_preprocessing,
    refine_config,
    repair_preprocessing_config,
    validate_config,
    validate_preprocessing_config,
)

CURATED_TOOLS_LIST = [
    ingest_dataset,
    characterize_dataset,
    list_sweeps,
    prepare_sweep,
    run_topological_sweep,
    get_sweep_status,
    diagnose_cosmic_graph,
    get_threshold_stability_curve,
    get_topological_skeleton,
    generate_cluster_dossier,
    get_feature_signal,
    get_cluster_profile,
    get_cluster_signal_matrix,
    compare_clusters,
    sync_to_pulsar,
    # Config build/edit/validate + self-correction + workflow (tenant-safe ports).
    get_workflow_guide,
    create_config,
    get_config,
    refine_config,
    validate_config,
    recommend_preprocessing,
    validate_preprocessing_config,
    repair_preprocessing_config,
    probe_columns,
]
