"""MCP tools for longitudinal (panel) analysis.

Build a longitudinal panel from an ingested long-format table, then interrogate it
through whichever representation answers the question:

- ``TemporalCosmicGraph`` — entity nodes, intra-time edges, tensor aggregations.
- ``CosmicTrajectory`` — observation ``(entity, t)`` nodes, edges that span time.

Artifacts live in the session only, like the fitted model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import uuid
from typing import Any, Literal

import yaml
from fastmcp import Context

from pulsar.config import load_config
from pulsar.mcp.config_tools import validate_config_yaml
from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.longitudinal import (
    PanelError,
    adapt_config_to_panel,
    archetype_payload,
    build_representations,
    classify_trajectories_payload,
    cross_time_payload,
    estimate_costs,
    guard_representation,
    pivot_panel,
    temporal_diagnosis,
    trajectory_diagnosis,
)
from pulsar.mcp.session import (
    LongitudinalArtifact,
    _get_session,
    _read_dataset_file,
    _resolve_dataset_path,
    _resolve_longitudinal,
)

logger = logging.getLogger(__name__)


def _panel_error(tool: str, exc: PanelError) -> str:
    return mcp_error(
        tool,
        exc.reason,
        error_code=exc.error_code,
        agent_action=exc.agent_action,
        details=exc.details,
    )


def _stale_handle(tool: str, longitudinal_id: str) -> str:
    if not longitudinal_id:
        return mcp_error(
            tool,
            "No longitudinal panel has been built in this session.",
            error_code="LONGITUDINAL_ID_UNKNOWN",
            agent_action="Call build_longitudinal_graph first.",
            details={"longitudinal_id": longitudinal_id},
        )
    return unknown_handle_error(tool, "longitudinal_id", longitudinal_id)


def _require_trajectory(tool: str, artifact: LongitudinalArtifact) -> str | None:
    if artifact.trajectory is None:
        return mcp_error(
            tool,
            "This panel was built without the trajectory representation.",
            error_code="REPRESENTATION_NOT_BUILT",
            agent_action=(
                "Rebuild with build_longitudinal_graph(representation='trajectory') "
                "or representation='both'."
            ),
            details={
                "longitudinal_id": artifact.longitudinal_id,
                "representation": artifact.representation,
            },
        )
    return None


async def build_longitudinal_graph(
    dataset_id: str = "",
    entity_column: str = "",
    time_column: str = "",
    feature_columns: list[str] | None = None,
    representation: Literal["trajectory", "temporal", "both"] = "trajectory",
    config_yaml: str = "",
    on_missing: Literal["drop_entity", "forward_fill", "allow_ragged"] = "drop_entity",
    similarity_threshold: float = 0.0,
    detail: Literal["summary", "full"] = "summary",
    response_format: Literal["json", "markdown"] = "json",
    ctx: Context = None,
) -> str:
    """Pivot a long-format dataset into a panel and build longitudinal representations.

    The dataset must be long: one row per (entity, time) with numeric feature columns.
    ``time_column`` must already be discrete and orderable — continuous timestamps have
    to be binned before ingest.

    ``representation='trajectory'`` (default) is observation-centric and stays sparse,
    so edges may join different time steps. ``'temporal'`` builds the dense (n, n, T)
    tensor whose edges live inside one time slice. ``'both'`` builds each.

    ``on_missing`` controls entities not observed at every step: ``drop_entity``
    (default), ``forward_fill``, or ``allow_ragged`` (trajectory only).

    Returns a ``longitudinal_id`` handle plus panel shape, alignment report, and
    build cost. Does not disturb the static sweep/cluster session state.
    """
    session = _get_session(ctx)

    if detail not in {"summary", "full"}:
        return mcp_error(
            "build_longitudinal_graph", "detail must be 'summary' or 'full'."
        )
    if response_format not in {"json", "markdown"}:
        return mcp_error(
            "build_longitudinal_graph", "response_format must be 'json' or 'markdown'."
        )
    if not entity_column or not time_column:
        return mcp_error(
            "build_longitudinal_graph",
            "entity_column and time_column are required.",
            error_code="PANEL_KEYS_MISSING",
            agent_action=(
                "Call probe_columns or characterize_dataset to identify the entity "
                "and time columns, then retry."
            ),
        )
    if similarity_threshold < 0.0:
        return mcp_error(
            "build_longitudinal_graph", "similarity_threshold must be >= 0."
        )

    current_yaml = config_yaml or session.active_config_yaml or ""
    if not current_yaml:
        return mcp_error(
            "build_longitudinal_graph",
            "No config available for the longitudinal build.",
            error_code="CONFIG_REQUIRED",
            agent_action=(
                "Call create_config(dataset_id) first — it calibrates epsilon and "
                "the projection grid against this dataset — then pass config_yaml."
            ),
            details={"dataset_id": dataset_id},
        )

    try:
        dataset_path = _resolve_dataset_path(dataset_id) if dataset_id else None
    except LookupError:
        return unknown_handle_error(
            "build_longitudinal_graph", "dataset_id", dataset_id
        )
    if dataset_path is None:
        return mcp_error(
            "build_longitudinal_graph",
            "dataset_id is required for a longitudinal build.",
            error_code="DATASET_ID_REQUIRED",
            agent_action="Call ingest_dataset first and pass the returned dataset_id.",
        )

    validation = validate_config_yaml(current_yaml, dataset_path=dataset_path)
    if not validation.ok or validation.normalized_yaml is None:
        return mcp_error(
            "build_longitudinal_graph",
            "Config validation failed.",
            error_code=validation.error_code or "CONFIG_VALIDATION_FAILED",
            agent_action=validation.agent_action,
            details={
                "resolved_dataset_path": validation.resolved_dataset_path,
                "issues": [dataclasses.asdict(issue) for issue in validation.issues],
            },
        )
    current_yaml = validation.normalized_yaml

    async def report(stage: str, fraction: float) -> None:
        if ctx is None:
            return
        await ctx.report_progress(progress=fraction, total=1.0, message=stage)

    try:
        cfg = load_config(yaml.safe_load(current_yaml))

        await report("read dataset", 0.1)
        # Deliberately NOT _bind_session_data: a panel is n*T observation rows and
        # rebinding would invalidate the static cluster and feature-evidence caches.
        df = await asyncio.to_thread(_read_dataset_file, dataset_path)

        await report("pivot panel", 0.25)
        panel = await asyncio.to_thread(
            pivot_panel,
            df,
            entity_column,
            time_column,
            feature_columns,
            on_missing,
        )
        guard_representation(panel, representation)
        cfg, config_notes = adapt_config_to_panel(cfg, panel)
        costs = estimate_costs(panel, cfg)

        await report(f"build {representation}", 0.4)
        trajectory, temporal = await asyncio.to_thread(
            build_representations, panel, cfg, representation, similarity_threshold
        )

        await report("summarize", 0.9)
        longitudinal_id = f"lng_{uuid.uuid4().hex[:12]}"
        artifact = LongitudinalArtifact(
            longitudinal_id=longitudinal_id,
            dataset_id=dataset_id or None,
            representation=representation,
            trajectory=trajectory,
            temporal=temporal,
            panel={
                "entity_column": entity_column,
                "time_column": time_column,
                "feature_columns": panel.feature_columns,
                "times": panel.times,
                "report": panel.report,
            },
            config_yaml=current_yaml,
            created_at=time.time(),
        )
        session.longitudinal[longitudinal_id] = artifact

        surface: dict[str, Any] = {}
        if trajectory is not None:
            cross = trajectory.cross_time()
            total = int(trajectory.similarity.nnz // 2)
            surface["trajectory"] = {
                "n_observations": trajectory.n_observations,
                "n_balls": int(len(trajectory.balls)),
                "n_edges": total,
                "cross_time_edges": int(cross.nnz // 2),
                "cross_time_fraction": (
                    round((cross.nnz // 2) / total, 4) if total else 0.0
                ),
            }
        if temporal is not None:
            surface["temporal"] = {
                "tensor_shape": list(temporal.shape),
                "tensor_mb": round(temporal.tensor.nbytes / (1024 * 1024), 2),
            }

        payload: dict[str, Any] = {
            "status": "ok",
            "detail": detail,
            "longitudinal_id": longitudinal_id,
            "dataset_id": dataset_id or None,
            "representation": representation,
            "panel": {
                "entity_column": entity_column,
                "time_column": time_column,
                "n_features": len(panel.feature_columns),
                "feature_columns_preview": panel.feature_columns[:10],
                **panel.report,
            },
            "build_cost": costs,
            "config_adaptation": config_notes,
            "graph_surface": surface,
            "next_tools": [
                "diagnose_longitudinal_graph",
                "get_trajectory_archetypes",
                "get_cross_time_neighbors",
            ],
        }
        if detail == "full":
            payload["config_yaml"] = current_yaml
            payload["times"] = panel.times

        await report("complete", 1.0)

        if response_format == "json":
            return json.dumps(payload, indent=2, default=str)
        return _build_to_markdown(payload)

    except PanelError as exc:
        return _panel_error("build_longitudinal_graph", exc)
    except FileNotFoundError:
        return path_access_error("build_longitudinal_graph", dataset_path)
    except Exception as exc:  # noqa: BLE001 - surface as a structured envelope
        logger.error("build_longitudinal_graph failed: %s", exc, exc_info=True)
        return mcp_error("build_longitudinal_graph", str(exc))


def _build_to_markdown(payload: dict[str, Any]) -> str:
    panel = payload["panel"]
    cost = payload["build_cost"]
    lines = [
        f"# Longitudinal Panel `{payload['longitudinal_id']}`",
        "",
        f"- Representation: **{payload['representation']}**",
        f"- Panel: {panel['n_entities_kept']} entities x {panel['n_times']} times "
        f"= {panel['observation_count']} observations",
        f"- Features: {panel['n_features']}",
        f"- Alignment: {panel['alignment']} (policy `{panel['policy_applied']}`, "
        f"{panel['dropped_entities']['total']} entities dropped)",
        f"- Missing cells: {panel['cells_missing_fraction']:.2%}",
        f"- Estimated ball maps: {cost['estimated_ball_maps_trajectory']} "
        f"(trajectory) / {cost['estimated_ball_maps_temporal']} (temporal)",
    ]
    surface = payload.get("graph_surface", {})
    if "trajectory" in surface:
        traj = surface["trajectory"]
        lines += [
            "",
            "## Trajectory surface",
            f"- {traj['n_observations']} observations, {traj['n_balls']} balls, "
            f"{traj['n_edges']} edges",
            f"- Cross-time edges: {traj['cross_time_edges']} "
            f"({traj['cross_time_fraction']:.1%})",
        ]
    if "temporal" in surface:
        temp = surface["temporal"]
        lines += [
            "",
            "## Temporal surface",
            f"- Tensor {temp['tensor_shape']} ({temp['tensor_mb']} MB)",
        ]
    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)


async def diagnose_longitudinal_graph(
    longitudinal_id: str = "",
    representation: Literal["auto", "trajectory", "temporal"] = "auto",
    detail: Literal["summary", "full"] = "summary",
    response_format: Literal["json", "markdown"] = "json",
    ctx: Context = None,
) -> str:
    """Measure a built longitudinal panel; does not prescribe an interpretation.

    For the trajectory surface: observation and cover counts, the CO_SIMILAR weight
    distribution, what fraction of edges actually span time, and a threshold ladder
    showing how steeply cluster counts move.

    For the temporal surface: the six tensor aggregations side by side, each cut at
    its own q90 (their value ranges are not comparable), with why/best_for/avoid_for
    so the aggregation can be chosen deliberately rather than by default.

    ``representation='auto'`` reports every surface the panel actually has.
    """
    session = _get_session(ctx)

    if detail not in {"summary", "full"}:
        return mcp_error(
            "diagnose_longitudinal_graph", "detail must be 'summary' or 'full'."
        )
    if response_format not in {"json", "markdown"}:
        return mcp_error(
            "diagnose_longitudinal_graph",
            "response_format must be 'json' or 'markdown'.",
        )

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("diagnose_longitudinal_graph", longitudinal_id)

    try:
        surfaces: dict[str, Any] = {}
        if representation in {"auto", "trajectory"} and artifact.trajectory is not None:
            surfaces["trajectory"] = await asyncio.to_thread(
                trajectory_diagnosis, artifact.trajectory, detail=detail
            )
        if representation in {"auto", "temporal"} and artifact.temporal is not None:
            surfaces["temporal"] = await asyncio.to_thread(
                temporal_diagnosis, artifact.temporal, detail=detail
            )

        if not surfaces:
            return mcp_error(
                "diagnose_longitudinal_graph",
                f"Panel has no '{representation}' surface to diagnose.",
                error_code="REPRESENTATION_NOT_BUILT",
                agent_action=(
                    "Rebuild with build_longitudinal_graph(representation=...) "
                    "including the surface you want to measure."
                ),
                details={
                    "longitudinal_id": artifact.longitudinal_id,
                    "built": artifact.representation,
                    "requested": representation,
                },
            )

        payload = {
            "status": "ok",
            "detail": detail,
            "longitudinal_id": artifact.longitudinal_id,
            "panel": artifact.panel["report"],
            "surfaces": surfaces,
        }
        if response_format == "json":
            return json.dumps(payload, indent=2, default=str)
        return _diagnosis_to_markdown(payload)

    except Exception as exc:  # noqa: BLE001
        logger.error("diagnose_longitudinal_graph failed: %s", exc, exc_info=True)
        return mcp_error("diagnose_longitudinal_graph", str(exc))


def _diagnosis_to_markdown(payload: dict[str, Any]) -> str:
    lines = [f"# Longitudinal Diagnosis `{payload['longitudinal_id']}`"]
    traj = payload["surfaces"].get("trajectory")
    if traj:
        sim = traj["similarity_surface"]
        lines += [
            "",
            "## Trajectory surface (observation nodes)",
            f"- {traj['panel']['n_observations']} observations "
            f"({traj['panel']['n_entities']} entities x {traj['panel']['n_times']} times)",
            f"- {sim['n_edges']} edges; {sim['cross_time_edges']} cross-time "
            f"({sim['cross_time_fraction']:.1%})",
            f"- Weights p25/p50/p95: {sim['weight_distribution']['p25']} / "
            f"{sim['weight_distribution']['p50']} / {sim['weight_distribution']['p95']}",
            f"- Cover: {traj['cover']['n_balls']} balls",
            "",
            "### Threshold ladder",
            "| Threshold | Obs clusters | Distinct trajectories | Largest group |",
            "|---|---|---|---|",
        ]
        for row in traj["threshold_sweep"]["rows"]:
            lines.append(
                f"| {row['threshold']} | {row['observation_clusters']} | "
                f"{row['distinct_trajectories']} | {row['largest_trajectory_group']} |"
            )
        lines += ["", f"_{traj['threshold_sweep']['note']}_"]

    temporal = payload["surfaces"].get("temporal")
    if temporal:
        lines += [
            "",
            "## Temporal surface (entity nodes)",
            f"- Tensor {temporal['tensor']['shape']} ({temporal['tensor']['mb']} MB)",
            f"- _{temporal['cut_basis']}_",
            "",
            "| Aggregation | Range | Edges@cut | Components@cut | Best for | Why |",
            "|---|---|---|---|---|---|",
        ]
        for agg in temporal["aggregations"]:
            lines.append(
                f"| {agg['aggregation']} | {agg['value_range']} | "
                f"{agg['edges_at_cut']} | {agg['components_at_cut']} | "
                f"{', '.join(agg['best_for'])} | {agg['why']} |"
            )
    return "\n".join(lines)


async def get_trajectory_archetypes(
    longitudinal_id: str = "",
    threshold: float = 0.0,
    max_archetypes: int = 10,
    max_entities_per_archetype: int = 5,
    detail: Literal["summary", "full"] = "summary",
    response_format: Literal["json", "markdown"] = "markdown",
    ctx: Context = None,
) -> str:
    """Group entities by the cluster sequence they trace through time.

    Each entity's trajectory is the sequence of cosmic clusters its observations fall
    into, one per time step. Identical sequences form an archetype.

    ``threshold`` is an *interpretation* filter on CO_SIMILAR edges, not the
    construction threshold. Cluster counts move steeply with it, so the response
    always includes a threshold ladder — read the ladder before trusting one row.
    """
    session = _get_session(ctx)

    if response_format not in {"json", "markdown"}:
        return mcp_error(
            "get_trajectory_archetypes", "response_format must be 'json' or 'markdown'."
        )
    if threshold < 0.0:
        return mcp_error("get_trajectory_archetypes", "threshold must be >= 0.")
    if max_archetypes < 1:
        return mcp_error("get_trajectory_archetypes", "max_archetypes must be >= 1.")
    if max_entities_per_archetype < 0:
        return mcp_error(
            "get_trajectory_archetypes",
            "max_entities_per_archetype must be >= 0.",
            error_code="INVALID_ARGUMENT",
        )

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("get_trajectory_archetypes", longitudinal_id)

    unavailable = _require_trajectory("get_trajectory_archetypes", artifact)
    if unavailable:
        return unavailable

    try:
        payload = await asyncio.to_thread(
            archetype_payload,
            artifact.trajectory,
            threshold=threshold,
            max_archetypes=max_archetypes,
            max_entities_per_archetype=max_entities_per_archetype,
        )
        payload["status"] = "ok"
        payload["detail"] = detail
        payload["longitudinal_id"] = artifact.longitudinal_id
        if response_format == "json":
            return json.dumps(payload, indent=2, default=str)
        return _archetypes_to_markdown(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("get_trajectory_archetypes failed: %s", exc, exc_info=True)
        return mcp_error("get_trajectory_archetypes", str(exc))


def _archetypes_to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Trajectory Archetypes `{payload['longitudinal_id']}`",
        "",
        f"- Interpretation threshold: {payload['interpretation_threshold']} "
        f"({payload['threshold_role']})",
        f"- {payload['n_entities']} entities over {payload['n_times']} time steps",
        f"- {payload['distinct_trajectories']} distinct trajectories "
        f"({payload['archetypes_omitted']} not shown)",
        "",
        "| Sequence | Entities | Share | Transitions | Example members |",
        "|---|---|---|---|---|",
    ]
    for arch in payload["archetypes"]:
        members = ", ".join(arch["entities"]["preview"])
        lines.append(
            f"| {arch['sequence']} | {arch['n_entities']} | {arch['fraction']:.1%} | "
            f"{arch['transitions']} | {members} |"
        )
    lines += [
        "",
        "## Threshold ladder",
        "| Threshold | Obs clusters | Distinct trajectories | Largest group |",
        "|---|---|---|---|",
    ]
    for row in payload["threshold_sweep"]["rows"]:
        lines.append(
            f"| {row['threshold']} | {row['observation_clusters']} | "
            f"{row['distinct_trajectories']} | {row['largest_trajectory_group']} |"
        )
    lines += ["", f"_{payload['threshold_sweep']['note']}_"]
    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)


async def get_cross_time_neighbors(
    longitudinal_id: str = "",
    entity_id: str = "",
    t: Any | None = None,
    observation_id: int | None = None,
    threshold: float = 0.0,
    max_neighbors: int = 20,
    direction: Literal["any", "forward", "backward"] = "any",
    response_format: Literal["json", "markdown"] = "markdown",
    ctx: Context = None,
) -> str:
    """Rank observations from *other* time steps that resemble a given observation.

    This is the query the trajectory representation exists for: "who does this entity
    at this time look like, at a different time?" — e.g. a patient at admission
    matched against other patients at discharge.

    Identify the observation either by ``observation_id`` or by ``entity_id`` + ``t``.
    ``direction`` restricts to later (``forward``) or earlier (``backward``) matches.
    """
    session = _get_session(ctx)

    if response_format not in {"json", "markdown"}:
        return mcp_error(
            "get_cross_time_neighbors", "response_format must be 'json' or 'markdown'."
        )
    if direction not in {"any", "forward", "backward"}:
        return mcp_error(
            "get_cross_time_neighbors",
            "direction must be 'any', 'forward', or 'backward'.",
        )
    if max_neighbors < 1:
        return mcp_error("get_cross_time_neighbors", "max_neighbors must be >= 1.")

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("get_cross_time_neighbors", longitudinal_id)

    unavailable = _require_trajectory("get_cross_time_neighbors", artifact)
    if unavailable:
        return unavailable

    trajectory = artifact.trajectory
    obs_id = observation_id
    if obs_id is None:
        if not entity_id or t is None:
            return mcp_error(
                "get_cross_time_neighbors",
                "Provide observation_id, or both entity_id and t.",
                error_code="OBSERVATION_NOT_SPECIFIED",
                agent_action=(
                    "Pass entity_id and t (from the panel's entity/time columns), "
                    "or an observation_id from a previous response."
                ),
            )
        # Entity ids survive the pivot as their original dtype; match on string form
        # so the agent can always pass a plain string.
        panel_times = artifact.panel["times"]
        time_index = next(
            (i for i, value in enumerate(panel_times) if value == t),
            None,
        )
        if time_index is None:
            time_index = next(
                (i for i, value in enumerate(panel_times) if str(value) == str(t)),
                None,
            )
        if time_index is None:
            return mcp_error(
                "get_cross_time_neighbors",
                f"No panel time label matches t={t!r}.",
                error_code="OBSERVATION_NOT_FOUND",
                agent_action="Pass a time label from the panel's time_column.",
                details={
                    "entity_id": entity_id,
                    "t": t,
                    "available_times": panel_times,
                },
            )
        matches = trajectory.obs.index[
            (trajectory.obs["entity_id"].astype(str) == str(entity_id))
            & (trajectory.obs["t"] == time_index)
        ]
        if len(matches) != 1:
            return mcp_error(
                "get_cross_time_neighbors",
                f"No unique observation for entity '{entity_id}' at t={t}.",
                error_code="OBSERVATION_NOT_FOUND",
                agent_action=(
                    "Check the entity_id and t against the panel; call "
                    "diagnose_longitudinal_graph for the panel shape."
                ),
                details={"entity_id": entity_id, "t": t, "matches": int(len(matches))},
            )
        obs_id = int(matches[0])

    if obs_id < 0 or obs_id >= trajectory.n_observations:
        return mcp_error(
            "get_cross_time_neighbors",
            f"observation_id {obs_id} is out of range.",
            error_code="OBSERVATION_NOT_FOUND",
            agent_action=(
                f"Valid observation ids are 0..{trajectory.n_observations - 1}."
            ),
            details={"observation_id": obs_id},
        )

    try:
        payload = await asyncio.to_thread(
            cross_time_payload,
            trajectory,
            obs_id=obs_id,
            threshold=threshold,
            max_neighbors=max_neighbors,
            direction=direction,
        )
        payload["status"] = "ok"
        payload["longitudinal_id"] = artifact.longitudinal_id
        if response_format == "json":
            return json.dumps(payload, indent=2, default=str)
        return _neighbors_to_markdown(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("get_cross_time_neighbors failed: %s", exc, exc_info=True)
        return mcp_error("get_cross_time_neighbors", str(exc))


def _neighbors_to_markdown(payload: dict[str, Any]) -> str:
    source = payload["source"]
    lines = [
        f"# Cross-Time Neighbors of entity `{source['entity_id']}` at t={source['t']}",
        "",
        f"- Surface: {payload['surface']}",
        f"- Interpretation threshold: {payload['interpretation_threshold']}, "
        f"direction `{payload['direction']}`",
        f"- Cross-time degree: {payload['cross_time_degree']} "
        f"({payload['neighbors_omitted']} not shown)",
        "",
        "| Entity | t | delta_t | Weight |",
        "|---|---|---|---|",
    ]
    for neighbor in payload["neighbors"]:
        lines.append(
            f"| {neighbor['entity_id']} | {neighbor['t']} | "
            f"{neighbor['delta_t']:+d} | {neighbor['weight']:.4f} |"
        )
    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)


async def classify_trajectories(
    longitudinal_id: str = "",
    method: Literal[
        "complexity", "transition", "sequence", "levenshtein", "dtw"
    ] = "complexity",
    threshold: float = 0.15,
    response_format: Literal["json", "markdown"] = "markdown",
    ctx: Context = None,
) -> str:
    """Classify patient longitudinal trajectories based on their clinical paths.

    Allows downstream agents or clinical researchers to partition patients into
    actionable cohorts using three distinct paradigms:
      - 'complexity': Groups by Shannon entropy and state-transition counts (Stable, Gradual, Volatile).
      - 'transition': Groups by Markov self-retention probability (Highly Stable, Transitioning, Volatile).
      - 'levenshtein': Groups by Levenshtein edit distance matrix complete-linkage clustering.
      - 'dtw': Groups by Dynamic Time Warping alignment complete-linkage clustering.
      - 'sequence': Groups by structural path type (Singleton, Monostate, Multistate).
    """
    session = _get_session(ctx)

    if response_format not in {"json", "markdown"}:
        return mcp_error(
            "classify_trajectories", "response_format must be 'json' or 'markdown'."
        )
    if method not in {"complexity", "transition", "sequence", "levenshtein", "dtw"}:
        return mcp_error(
            "classify_trajectories",
            "method must be 'complexity', 'transition', 'sequence', 'levenshtein', or 'dtw'.",
        )
    if threshold < 0.0:
        return mcp_error("classify_trajectories", "threshold must be >= 0.")

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("classify_trajectories", longitudinal_id)

    unavailable = _require_trajectory("classify_trajectories", artifact)
    if unavailable:
        return unavailable

    try:
        payload = await asyncio.to_thread(
            classify_trajectories_payload,
            artifact.trajectory,
            method=method,
            threshold=threshold,
        )
        payload["status"] = "ok"
        payload["longitudinal_id"] = artifact.longitudinal_id
        if response_format == "json":
            return json.dumps(payload, indent=2, default=str)
        return _classification_to_markdown(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("classify_trajectories failed: %s", exc, exc_info=True)
        return mcp_error("classify_trajectories", str(exc))


def _classification_to_markdown(payload: dict[str, Any]) -> str:
    method_labels = {
        "complexity": "Shannon Entropy & Transition Counts (Stable vs Volatile)",
        "transition": "Markov Self-Retention Probabilities",
        "sequence": "Structural Path Profile (Singleton vs Multistate)",
        "levenshtein": "Levenshtein Edit Distance Complete-Linkage Clustering",
        "dtw": "Dynamic Time Warping Complete-Linkage Clustering",
    }
    lines = [
        f"# Trajectory Classification `{payload['longitudinal_id']}`",
        "",
        f"- **Method**: `{payload['method']}` ({method_labels[payload['method']]})",
        f"- **Interpretation Threshold**: {payload['interpretation_threshold']}",
        f"- **Total Patients Classified**: {payload['n_entities']}",
        "",
        "## Cohort Summary",
        "",
        "| Cohort Class | Patient Count | Population Share |",
        "|---|---|---|",
    ]
    for p_class, info in sorted(
        payload["classes_summary"].items(), key=lambda x: -x[1]["count"]
    ):
        lines.append(f"| {p_class} | {info['count']} | {info['fraction']:.1%} |")

    lines += [
        "",
        "## Patient Classification Samples (Top 15)",
        "",
        "| Patient ID | Cohort Class | Visits | Sequence (Observed) |",
        "|---|---|---|---|",
    ]
    samples = list(payload["classification"].items())[:15]
    for p_id, info in samples:
        lines.append(
            f"| {p_id} | {info['class']} | {info['length']} | {info['sequence']} |"
        )

    if len(payload["classification"]) > 15:
        lines.append(
            f"| ... | and {len(payload['classification']) - 15} more patients | | |"
        )

    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)
