"""Longitudinal panel support for the MCP surface.

Pulsar ingests flat wide tables, but both longitudinal representations consume
``list[np.ndarray]`` snapshots with per-time row alignment. This module owns the
long -> panel pivot, the alignment policies, the cost guards, and the payload
builders. Tools in :mod:`pulsar.mcp.tools.longitudinal` stay thin over it, the same
split ``thresholds.py`` and ``history.py`` already use.

Two representations, two questions:

- ``TemporalCosmicGraph`` — nodes are *entities*, edges live inside one time slice.
  Answers "which entities are stably similar over time" via tensor aggregations.
- ``CosmicTrajectory`` — nodes are *observations* ``(entity, t)``, edges may span
  time. Answers "which observations resemble each other across time" and "how do
  entities move between cohorts".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pulsar.config import PulsarConfig
from pulsar.mcp.payloads import bounded_list, size_summary
from pulsar.representations import CosmicTrajectory, TemporalCosmicGraph

PIVOT_POLICIES = ("drop_entity", "forward_fill", "allow_ragged")
REPRESENTATIONS = ("trajectory", "temporal", "both")

#: Ceiling on the dense ``(n, n, T)`` float64 tensor TemporalCosmicGraph allocates.
#: n=1000, T=50 is 400 GB, so this guard is load-bearing, not decorative.
_DEFAULT_MAX_TENSOR_BYTES = 2 * 1024**3

# The temporal build keeps the int64 pseudo-Laplacian, a Rust normalization copy,
# and the float64 output alive together. Budget that peak, not only the output.
_TEMPORAL_PEAK_TENSOR_MULTIPLIER = 3

#: Self-scaled cut used to make the six aggregations comparable. Their value ranges
#: differ (persistence/mean/recency are [0,1]; volatility is a variance; trend is a
#: signed slope), so a single global threshold would be meaningless across them.
_AGGREGATION_CUT_QUANTILE = 0.90

_ARCHETYPE_SWEEP_THRESHOLDS = (0.0, 0.1, 0.25, 0.5, 0.75)

AGGREGATION_GUIDANCE: dict[str, dict[str, Any]] = {
    "persistence": {
        "value_range": "[0,1]",
        "measures": "fraction of time steps where the pair exceeds the threshold",
        "why": "Counts how often a relationship holds, ignoring its magnitude.",
        "best_for": ["stable_cohorts", "report_ready"],
        "avoid_for": ["change_detection"],
    },
    "mean": {
        "value_range": "[0,1]",
        "measures": "average affinity across time",
        "why": "General-purpose summary; a strong brief relationship and a weak "
        "persistent one can score alike.",
        "best_for": ["general_structure"],
        "avoid_for": ["change_detection", "outlier_mining"],
    },
    "recency": {
        "value_range": "[0,1]",
        "measures": "exponentially decayed affinity, most recent step weighted 1",
        "why": "Answers what the cohort looks like now rather than on average.",
        "best_for": ["current_state_cohorts"],
        "avoid_for": ["stable_cohorts"],
    },
    "volatility": {
        "value_range": "unbounded>=0",
        "measures": "population variance of the pair's affinity across time",
        "why": "High values mark relationships that come and go.",
        "best_for": ["unstable_pairs", "detail_seeking"],
        "avoid_for": ["report_ready"],
    },
    "trend": {
        "value_range": "signed",
        "measures": "OLS slope of affinity against time",
        "why": "Sign carries the meaning: positive converging, negative diverging.",
        "best_for": ["converging_pairs", "diverging_pairs"],
        "avoid_for": ["stable_cohorts"],
    },
    "change_point": {
        "value_range": "unbounded>=0",
        "measures": "largest single step-to-step jump in affinity",
        "why": "Isolates abrupt transitions that averages smooth away.",
        "best_for": ["abrupt_transitions", "outlier_mining"],
        "avoid_for": ["report_ready"],
    },
}


class PanelError(ValueError):
    """Panel/pivot failure carrying the fields an MCP error envelope needs."""

    def __init__(
        self,
        reason: str,
        *,
        error_code: str,
        agent_action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error_code = error_code
        self.agent_action = agent_action
        self.details = details or {}


@dataclass
class Panel:
    """A pivoted longitudinal panel ready for ``from_snapshots``."""

    snapshots: list[np.ndarray]
    entity_ids: list[Any]
    snapshot_entity_ids: list[list[Any]]
    times: list[Any]
    feature_columns: list[str]
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def n_entities(self) -> int:
        return len(self.entity_ids)

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def is_aligned(self) -> bool:
        """True when every snapshot has the same entities in the same row order."""
        return all(
            ids == self.snapshot_entity_ids[0] for ids in self.snapshot_entity_ids[1:]
        )


def max_tensor_bytes() -> int:
    """Ceiling for the dense temporal tensor, overridable for large-memory hosts."""
    raw = os.environ.get("PULSAR_MCP_MAX_TENSOR_BYTES")
    if not raw:
        return _DEFAULT_MAX_TENSOR_BYTES
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_TENSOR_BYTES
    return parsed if parsed > 0 else _DEFAULT_MAX_TENSOR_BYTES


def _numeric_feature_columns(
    df: pd.DataFrame, entity_column: str, time_column: str
) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if col not in {entity_column, time_column}
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def pivot_panel(
    df: pd.DataFrame,
    entity_column: str,
    time_column: str,
    feature_columns: list[str] | None = None,
    on_missing: Literal["drop_entity", "forward_fill", "allow_ragged"] = "drop_entity",
) -> Panel:
    """Pivot a long-format table into per-time snapshots with aligned rows.

    ``time_column`` must already be discrete and orderable — no binning is applied.
    Rows are keyed by ``(entity, time)``; duplicates are an error rather than a
    silent aggregation, since the choice of aggregator is the caller's to make.
    """
    if on_missing not in PIVOT_POLICIES:
        raise PanelError(
            f"Unknown on_missing policy '{on_missing}'.",
            error_code="INVALID_PIVOT_POLICY",
            agent_action=f"Use one of {list(PIVOT_POLICIES)}.",
            details={"on_missing": on_missing, "supported": list(PIVOT_POLICIES)},
        )

    missing_keys = [
        name for name in (entity_column, time_column) if name not in df.columns
    ]
    if missing_keys:
        raise PanelError(
            f"Column(s) {missing_keys} not present in the dataset.",
            error_code="PANEL_COLUMN_NOT_FOUND",
            agent_action=(
                "Call probe_columns or characterize_dataset to list available "
                "columns, then pass the correct entity_column and time_column."
            ),
            details={
                "missing": missing_keys,
                "available_columns": bounded_list([str(c) for c in df.columns]),
            },
        )

    if feature_columns:
        features = [str(col) for col in feature_columns]
        unknown = [col for col in features if col not in df.columns]
        if unknown:
            raise PanelError(
                f"Feature column(s) {unknown} not present in the dataset.",
                error_code="PANEL_COLUMN_NOT_FOUND",
                agent_action="Pass feature_columns that exist in the dataset.",
                details={
                    "missing": unknown,
                    "available_columns": bounded_list([str(c) for c in df.columns]),
                },
            )
        non_numeric = [
            col for col in features if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise PanelError(
                f"Feature column(s) {non_numeric} are not numeric.",
                error_code="PANEL_FEATURES_NOT_NUMERIC",
                agent_action=(
                    "Longitudinal geometry needs numeric features. Encode or drop "
                    "these columns before building."
                ),
                details={"non_numeric": non_numeric},
            )
    else:
        features = _numeric_feature_columns(df, entity_column, time_column)

    if not features:
        raise PanelError(
            "No numeric feature columns remain after excluding entity and time.",
            error_code="PANEL_NO_FEATURES",
            agent_action=(
                "Pass feature_columns explicitly, or encode categorical columns first."
            ),
            details={"entity_column": entity_column, "time_column": time_column},
        )

    frame = df[[entity_column, time_column, *features]].copy()

    duplicated = frame.duplicated([entity_column, time_column])
    if bool(duplicated.any()):
        raise PanelError(
            f"{int(duplicated.sum())} duplicate (entity, time) rows found.",
            error_code="PANEL_DUPLICATE_OBSERVATIONS",
            agent_action=(
                "Aggregate the duplicates to one row per (entity, time) before "
                "ingesting; Pulsar will not pick an aggregator for you."
            ),
            details={
                "duplicate_row_count": int(duplicated.sum()),
                "entity_column": entity_column,
                "time_column": time_column,
            },
        )

    try:
        times = sorted(frame[time_column].dropna().unique().tolist())
    except TypeError as exc:
        raise PanelError(
            f"Time column '{time_column}' is not orderable.",
            error_code="TIME_COLUMN_NOT_ORDERABLE",
            agent_action=(
                "Provide a discrete orderable time column (integer step index or "
                "sortable label). Continuous timestamps must be binned before ingest."
            ),
            details={
                "time_column": time_column,
                "dtype": str(frame[time_column].dtype),
            },
        ) from exc

    if not times:
        raise PanelError(
            f"Time column '{time_column}' has no non-null values.",
            error_code="PANEL_EMPTY",
            agent_action="Check the time column for an all-null or empty dataset.",
            details={"time_column": time_column},
        )
    if len(times) < 2:
        raise PanelError(
            f"Time column '{time_column}' has only one distinct value.",
            error_code="PANEL_SINGLE_TIME_STEP",
            agent_action=(
                "A longitudinal build needs at least 2 time steps. Use "
                "run_topological_sweep for single-snapshot analysis."
            ),
            details={"time_column": time_column, "n_times": len(times)},
        )

    entities = frame[entity_column].dropna().drop_duplicates().tolist()
    n_entities_input = len(entities)

    indexed = frame.dropna(subset=[entity_column, time_column]).set_index(
        [entity_column, time_column]
    )
    full_index = pd.MultiIndex.from_product(
        [entities, times], names=[entity_column, time_column]
    )
    present = pd.Series(True, index=indexed.index).reindex(full_index, fill_value=False)
    presence = present.to_numpy().reshape(len(entities), len(times))
    cells_missing_fraction = float(1.0 - presence.mean()) if presence.size else 0.0

    aligned = indexed.reindex(full_index)
    dropped: list[Any] = []

    if on_missing == "allow_ragged":
        snapshots = []
        snapshot_entity_ids = []
        for time_index, time_value in enumerate(times):
            rows = aligned.to_numpy(dtype=np.float64).reshape(
                len(entities), len(times), len(features)
            )[:, time_index, :]
            keep = presence[:, time_index]
            snapshots.append(np.ascontiguousarray(rows[keep], dtype=np.float64))
            snapshot_entity_ids.append(
                [entity for entity, is_present in zip(entities, keep) if is_present]
            )
        kept_entities = entities
    else:
        if on_missing == "forward_fill":
            aligned = aligned.groupby(level=0, sort=False).ffill()
            complete = ~aligned.isna().any(axis=1).to_numpy().reshape(
                len(entities), len(times)
            ).any(axis=1)
        else:  # drop_entity
            complete = presence.all(axis=1) & ~aligned.isna().any(
                axis=1
            ).to_numpy().reshape(len(entities), len(times)).any(axis=1)

        kept_entities = [ent for ent, ok in zip(entities, complete) if ok]
        dropped = [ent for ent, ok in zip(entities, complete) if not ok]
        if not kept_entities:
            raise PanelError(
                "No entity is observed at every time step.",
                error_code="PANEL_NO_COMPLETE_ENTITIES",
                agent_action=(
                    "Try on_missing='forward_fill', or on_missing='allow_ragged' "
                    "with representation='trajectory'."
                ),
                details={
                    "n_entities_input": n_entities_input,
                    "n_times": len(times),
                    "cells_missing_fraction": round(cells_missing_fraction, 4),
                },
            )

        cube = aligned.to_numpy(dtype=np.float64).reshape(
            len(entities), len(times), len(features)
        )[complete]
        snapshots = [
            np.ascontiguousarray(cube[:, t, :], dtype=np.float64)
            for t in range(len(times))
        ]
        snapshot_entity_ids = [list(kept_entities) for _ in times]

    report = {
        "policy_applied": on_missing,
        "n_entities_input": n_entities_input,
        "n_entities_kept": len(kept_entities),
        "dropped_entities": bounded_list([str(e) for e in dropped]),
        "n_times": len(times),
        "n_features": len(features),
        "cells_missing_fraction": round(cells_missing_fraction, 4),
        "alignment": "complete" if on_missing != "allow_ragged" else "ragged",
        "observation_count": int(sum(arr.shape[0] for arr in snapshots)),
    }

    return Panel(
        snapshots=snapshots,
        entity_ids=list(kept_entities),
        snapshot_entity_ids=snapshot_entity_ids,
        times=list(times),
        feature_columns=features,
        report=report,
    )


def adapt_config_to_panel(
    config: PulsarConfig, panel: Panel
) -> tuple[PulsarConfig, dict[str, Any]]:
    """Fit the projection grid to the panel's feature count.

    ``create_config`` calibrates against the raw long-format file, whose column
    count includes the entity and time keys and therefore exceeds the panel's
    feature count. A projection dimension above the feature count is a hard error
    inside Rust, so drop those rungs and disclose it rather than failing late.

    Both ``projection`` and ``pca`` are updated: ``CosmicTrajectory`` routes through
    ``projection_grid`` while ``TemporalCosmicGraph.from_snapshots`` still calls
    ``pca_grid`` directly.
    """
    n_features = len(panel.feature_columns)
    if n_features < 2:
        raise PanelError(
            f"Panel has {n_features} feature column(s); need at least 2.",
            error_code="PANEL_TOO_FEW_FEATURES",
            agent_action=(
                "Pass more feature_columns, or check that entity_column and "
                "time_column are not consuming the only numeric columns."
            ),
            details={
                "n_features": n_features,
                "feature_columns": panel.feature_columns,
            },
        )

    adapted = copy.deepcopy(config)
    notes: dict[str, Any] = {"panel_feature_count": n_features}
    for spec_name in ("projection", "pca"):
        spec = getattr(adapted, spec_name, None)
        if spec is None:
            continue
        requested = list(spec.dimensions)
        keep = [dim for dim in requested if dim <= n_features]
        dropped = [dim for dim in requested if dim > n_features]
        if not keep:
            keep = [n_features]
        spec.dimensions = keep
        if dropped:
            notes[f"{spec_name}_dimensions_requested"] = requested
            notes[f"{spec_name}_dimensions_applied"] = keep
            notes[f"{spec_name}_dimensions_dropped"] = dropped

    if any(key.endswith("_dropped") for key in notes):
        notes["projection_reason"] = (
            "Projection dimensions above the panel feature count were dropped; "
            "config was calibrated against the raw file, which also counts the "
            "entity and time columns."
        )

    notes.update(_calibrate_epsilon_to_panel(adapted, panel))
    return adapted, notes


def _calibrate_epsilon_to_panel(config: PulsarConfig, panel: Panel) -> dict[str, Any]:
    """Check the epsilon grid against the *panel's* k-NN domain, not the raw file's.

    ``create_config`` calibrates epsilon on the raw long-format table, whose geometry
    is not the panel's: the entity and time keys inflate it, and the panel is scaled
    before covering. An epsilon grid entirely outside ``[knn_p5, knn_p95]`` produces a
    degenerate cover (one ball holding everything), so recalibrate onto the panel's
    own domain and say so. A partially valid grid is left alone — the agent may have
    chosen it deliberately — with the domain reported either way.
    """
    from pulsar.analysis.characterization import profile_numeric_matrix
    from pulsar._pulsar import StandardScaler

    pooled = np.vstack(panel.snapshots)
    scaled = np.array(StandardScaler().fit_transform(np.ascontiguousarray(pooled)))
    profile = profile_numeric_matrix(scaled)
    low, high = float(profile.knn_p5), float(profile.knn_p95)

    requested = list(config.ball_mapper.epsilons)
    in_domain = [eps for eps in requested if low <= eps <= high]
    notes: dict[str, Any] = {
        "panel_epsilon_domain": {"knn_p5": round(low, 4), "knn_p95": round(high, 4)},
        "epsilons_in_domain": len(in_domain),
    }
    if in_domain or high <= 0:
        return notes

    steps = max(len(requested), 2)
    config.ball_mapper.epsilons = [
        round(float(value), 6) for value in np.linspace(low, high, steps)
    ]
    notes["epsilons_requested"] = bounded_list(requested, preview_limit=5)
    notes["epsilons_applied"] = bounded_list(
        config.ball_mapper.epsilons, preview_limit=5
    )
    notes["epsilon_reason"] = (
        "No requested epsilon fell inside the panel's k-NN domain, which would "
        "produce a degenerate single-ball cover. Recalibrated onto [knn_p5, knn_p95] "
        "of the scaled panel. Pass config_yaml with your own epsilons to override."
    )
    return notes


def estimate_costs(panel: Panel, config: PulsarConfig) -> dict[str, Any]:
    """Visible cost context so the agent chooses a representation deliberately."""
    n_entities = panel.n_entities
    n_times = panel.n_times
    tensor_bytes = n_entities * n_entities * n_times * 8
    peak_tensor_bytes = tensor_bytes * _TEMPORAL_PEAK_TENSOR_MULTIPLIER
    projection = getattr(config, "projection", None)
    dimensions = list(getattr(projection, "dimensions", None) or config.pca.dimensions)
    seeds = list(getattr(projection, "seeds", None) or config.pca.seeds)
    epsilons = list(config.ball_mapper.epsilons)
    return {
        "n_observations": panel.report["observation_count"],
        "n_entities": n_entities,
        "n_times": n_times,
        "temporal_tensor_bytes": int(tensor_bytes),
        "temporal_tensor_mb": round(tensor_bytes / (1024 * 1024), 2),
        "temporal_peak_bytes": int(peak_tensor_bytes),
        "temporal_peak_mb": round(peak_tensor_bytes / (1024 * 1024), 2),
        "temporal_tensor_limit_mb": round(max_tensor_bytes() / (1024 * 1024), 2),
        "estimated_ball_maps_trajectory": len(dimensions) * len(seeds) * len(epsilons),
        "estimated_ball_maps_temporal": (
            len(dimensions) * len(seeds) * len(epsilons) * n_times
        ),
        "projection_dimensions": dimensions,
        "epsilons": epsilons,
    }


def guard_representation(panel: Panel, representation: str) -> None:
    """Raise a structured PanelError when the panel cannot support the request."""
    if representation not in REPRESENTATIONS:
        raise PanelError(
            f"Unknown representation '{representation}'.",
            error_code="INVALID_REPRESENTATION",
            agent_action=f"Use one of {list(REPRESENTATIONS)}.",
            details={"supported": list(REPRESENTATIONS)},
        )

    wants_temporal = representation in {"temporal", "both"}
    if wants_temporal and not panel.is_aligned:
        raise PanelError(
            "TemporalCosmicGraph requires the same entities in the same order at every time step.",
            error_code="RAGGED_PANEL_NOT_SUPPORTED",
            agent_action=(
                "Use representation='trajectory' (tolerates ragged panels), or "
                "rebuild with on_missing='drop_entity' or 'forward_fill'."
            ),
            details={
                "snapshot_row_counts": bounded_list(
                    [int(arr.shape[0]) for arr in panel.snapshots]
                ),
                "policy_applied": panel.report.get("policy_applied"),
            },
        )

    if wants_temporal:
        limit = max_tensor_bytes()
        tensor_bytes = panel.n_entities * panel.n_entities * panel.n_times * 8
        peak_tensor_bytes = tensor_bytes * _TEMPORAL_PEAK_TENSOR_MULTIPLIER
        if peak_tensor_bytes > limit:
            raise PanelError(
                "Temporal tensor exceeds the configured memory ceiling.",
                error_code="TENSOR_TOO_LARGE",
                agent_action=(
                    "Use representation='trajectory' — it stays sparse and never "
                    "allocates an (n, n, T) tensor. Otherwise reduce entities or "
                    "time steps, or raise PULSAR_MCP_MAX_TENSOR_BYTES."
                ),
                details={
                    "tensor_shape": [panel.n_entities, panel.n_entities, panel.n_times],
                    "tensor_bytes": int(tensor_bytes),
                    "tensor_mb": round(tensor_bytes / (1024 * 1024), 2),
                    "peak_required_bytes": int(peak_tensor_bytes),
                    "peak_required_mb": round(peak_tensor_bytes / (1024 * 1024), 2),
                    "limit_bytes": int(limit),
                    "limit_mb": round(limit / (1024 * 1024), 2),
                },
            )


def build_representations(
    panel: Panel,
    config: PulsarConfig,
    representation: str,
    similarity_threshold: float,
) -> tuple[CosmicTrajectory | None, TemporalCosmicGraph | None]:
    """Blocking build of the requested representations. Call under to_thread."""
    trajectory = None
    temporal = None
    if representation in {"trajectory", "both"}:
        trajectory = CosmicTrajectory.from_snapshots(
            panel.snapshots,
            config,
            entity_ids=panel.snapshot_entity_ids,
            timestamps=panel.times,
            similarity_threshold=similarity_threshold,
        )
    if representation in {"temporal", "both"}:
        temporal = TemporalCosmicGraph.from_snapshots(panel.snapshots, config)
    return trajectory, temporal


# --------------------------------------------------------------------- payloads


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    """Off-diagonal upper-triangle values of a square matrix."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float64)
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def _weight_distribution(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "p25": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "p25": round(float(np.percentile(values, 25)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "max": round(float(values.max()), 6),
    }


def _component_summary(matrix: sp.csr_matrix) -> dict[str, Any]:
    n_components, labels = connected_components(matrix, directed=False)
    _, counts = np.unique(labels, return_counts=True)
    total = int(labels.size)
    largest = int(counts.max()) if counts.size else 0
    return {
        "component_count": int(n_components),
        "largest_component_fraction": round(largest / total, 4) if total else 0.0,
        "singleton_count": int((counts == 1).sum()),
        "component_sizes": size_summary(counts.tolist(), preview_limit=10),
    }


def trajectory_diagnosis(
    trajectory: CosmicTrajectory, *, detail: str = "summary"
) -> dict[str, Any]:
    """Measurement payload for the observation-centric surface."""
    similarity = trajectory.similarity
    weights = similarity.tocoo().data
    cross = trajectory.cross_time()
    total_edges = int(similarity.nnz // 2)
    cross_edges = int(cross.nnz // 2)

    balls = trajectory.balls
    cover: dict[str, Any] = {"n_balls": int(len(balls))}
    if len(balls):
        cover["balls_by_eps"] = {
            str(eps): int(count)
            for eps, count in balls["eps"].value_counts().sort_index().items()
        }
        cover["ball_sizes"] = size_summary(balls["size"].tolist(), preview_limit=10)
        memberships = np.asarray(trajectory.incidence.sum(axis=1)).ravel()
        cover["memberships_per_observation"] = _weight_distribution(
            memberships.astype(np.float64)
        )

    payload: dict[str, Any] = {
        "representation": "trajectory",
        "node_identity": "observation (entity, t)",
        "panel": {
            "n_observations": trajectory.n_observations,
            "n_entities": trajectory.n_entities,
            "n_times": trajectory.n_times,
        },
        "similarity_surface": {
            "surface": "CO_SIMILAR on the pooled observation graph",
            "construction": trajectory.meta.get("construction"),
            "similarity_threshold_applied": trajectory.meta.get("similarity_threshold"),
            "n_edges": total_edges,
            "cross_time_edges": cross_edges,
            "cross_time_fraction": (
                round(cross_edges / total_edges, 4) if total_edges else 0.0
            ),
            "weight_distribution": _weight_distribution(weights),
        },
        "cover": cover,
        "threshold_sweep": threshold_sweep(trajectory),
        "next_tools": ["get_trajectory_archetypes", "get_cross_time_neighbors"],
    }
    if detail == "full":
        payload["component_morphology"] = _component_summary(similarity)
        payload["meta"] = dict(trajectory.meta)
    return payload


def threshold_sweep(trajectory: CosmicTrajectory) -> dict[str, Any]:
    """Component and archetype counts across a threshold ladder.

    The component count of a cosmic graph moves steeply with the interpretation
    threshold, so a single reading is not ground truth. Showing the ladder keeps
    the agent from treating one threshold as the answer.
    """
    rows = []
    for threshold in _ARCHETYPE_SWEEP_THRESHOLDS:
        labels = trajectory.cluster_labels(threshold)
        frame = trajectory.obs.assign(cluster=labels)
        sequences = frame.pivot(index="entity_id", columns="t", values="cluster").apply(
            tuple, axis=1
        )
        rows.append(
            {
                "threshold": threshold,
                "observation_clusters": int(np.unique(labels).size),
                "distinct_trajectories": int(sequences.nunique()),
                "largest_trajectory_group": int(sequences.value_counts().iloc[0]),
            }
        )
    return {
        "basis": "connected components on CO_SIMILAR above each threshold",
        "note": (
            "Cluster counts move steeply with threshold; compare rows rather than "
            "trusting any single one."
        ),
        "rows": rows,
    }


def temporal_diagnosis(
    temporal: TemporalCosmicGraph, *, detail: str = "summary"
) -> dict[str, Any]:
    """Per-aggregation comparison so the agent picks the right lens.

    Each aggregation is cut at its own q90 rather than a shared threshold — their
    value ranges are not comparable ([0,1] vs a variance vs a signed slope).
    """
    aggregations = []
    for name, guidance in AGGREGATION_GUIDANCE.items():
        matrix = getattr(temporal, f"{name}_graph")()
        values = _upper_values(matrix)
        magnitude = np.abs(values)
        cut = (
            float(np.quantile(magnitude, _AGGREGATION_CUT_QUANTILE))
            if magnitude.size
            else 0.0
        )
        # Discrete-valued aggregations (persistence at small T takes only a few
        # levels) put q90 exactly on the max, where a strict `>` selects nothing.
        # An inclusive cut keeps the "top decile" reading meaningful for those.
        magnitudes = np.abs(matrix)
        mask = magnitudes >= cut if cut > 0 else magnitudes > 0
        np.fill_diagonal(mask, False)
        sparse_mask = sp.csr_matrix(mask)
        entry = {
            "aggregation": name,
            **guidance,
            "distribution": _weight_distribution(magnitude),
            "self_scaled_cut": round(cut, 6),
            "edges_at_cut": int(sparse_mask.nnz // 2),
            "components_at_cut": int(
                connected_components(sparse_mask, directed=False)[0]
            ),
        }
        if name == "trend":
            entry["converging_pairs"] = int((values >= cut).sum()) if cut > 0 else 0
            entry["diverging_pairs"] = int((values <= -cut).sum()) if cut > 0 else 0
        aggregations.append(entry)

    return {
        "representation": "temporal",
        "node_identity": "entity",
        "panel": {"n_entities": temporal.n, "n_times": temporal.T},
        "tensor": {
            "shape": list(temporal.shape),
            "bytes": int(temporal.tensor.nbytes),
            "mb": round(temporal.tensor.nbytes / (1024 * 1024), 2),
        },
        "cut_basis": (
            f"each aggregation cut at its own q{int(_AGGREGATION_CUT_QUANTILE * 100)} "
            "magnitude; ranges are not comparable across aggregations"
        ),
        "aggregations": aggregations,
        "next_tools": ["diagnose_longitudinal_graph"],
        **(
            {"detail": "full", "threshold": temporal._threshold}
            if detail == "full"
            else {}
        ),
    }


def archetype_payload(
    trajectory: CosmicTrajectory,
    *,
    threshold: float,
    max_archetypes: int,
    max_entities_per_archetype: int,
) -> dict[str, Any]:
    """Distinct cluster sequences and their populations — trajectory classification."""
    labels = trajectory.cluster_labels(threshold)
    frame = trajectory.obs.assign(cluster=labels)
    pivoted = frame.pivot(index="entity_id", columns="t", values="cluster")
    sequences = pivoted.apply(tuple, axis=1)
    counts = sequences.value_counts()

    archetypes = []
    for sequence, population in counts.head(max_archetypes).items():
        members = sequences.index[sequences == sequence].tolist()
        observed_seq = [v for v in sequence if not pd.isna(v)]
        archetypes.append(
            {
                "sequence": [
                    None if pd.isna(value) else int(value) for value in sequence
                ],
                "n_entities": int(population),
                "fraction": round(float(population) / len(sequences), 4),
                "transitions": int(
                    sum(1 for a, b in zip(observed_seq, observed_seq[1:]) if a != b)
                ),
                "entities": bounded_list(
                    [str(member) for member in members],
                    preview_limit=max_entities_per_archetype,
                ),
            }
        )

    return {
        "representation": "trajectory",
        "interpretation_threshold": threshold,
        "threshold_role": "interpretation (edge filter), not construction",
        "n_entities": int(len(sequences)),
        "n_times": int(pivoted.shape[1]),
        "distinct_trajectories": int(counts.size),
        "archetypes_returned": len(archetypes),
        "archetypes_omitted": max(int(counts.size) - len(archetypes), 0),
        "archetypes": archetypes,
        "threshold_sweep": threshold_sweep(trajectory),
        "next_tools": ["get_cross_time_neighbors", "diagnose_longitudinal_graph"],
    }


def cross_time_payload(
    trajectory: CosmicTrajectory,
    *,
    obs_id: int,
    threshold: float,
    max_neighbors: int,
    direction: str,
) -> dict[str, Any]:
    """Ranked cross-time lookalikes for one observation."""
    obs = trajectory.obs
    source = obs.loc[obs_id]
    row = trajectory.cross_time(threshold).getrow(obs_id).tocoo()

    times = obs["t"].to_numpy()
    entities = obs["entity_id"].to_numpy()
    deltas = times[row.col] - int(source["t"])

    keep = np.ones(row.col.shape, dtype=bool)
    if direction == "forward":
        keep = deltas > 0
    elif direction == "backward":
        keep = deltas < 0

    cols = row.col[keep]
    weights = row.data[keep]
    deltas = deltas[keep]
    order = np.argsort(-weights)[:max_neighbors]

    neighbors = [
        {
            "obs_id": int(cols[i]),
            "entity_id": str(entities[cols[i]]),
            "t": int(times[cols[i]]),
            "delta_t": int(deltas[i]),
            "weight": round(float(weights[i]), 6),
        }
        for i in order
    ]

    return {
        "representation": "trajectory",
        "surface": "CO_SIMILAR restricted to cross-time pairs",
        "interpretation_threshold": threshold,
        "direction": direction,
        "source": {
            "obs_id": int(obs_id),
            "entity_id": str(source["entity_id"]),
            "t": int(source["t"]),
        },
        "neighbors_returned": len(neighbors),
        "neighbors_omitted": max(int(cols.size) - len(neighbors), 0),
        "cross_time_degree": int(cols.size),
        "neighbors": neighbors,
        "next_tools": ["get_trajectory_archetypes"],
    }


def levenshtein_distance(seq1: list[int], seq2: list[int]) -> int:
    """Pairwise edit distance (minimum insertions/deletions/substitutions) between sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,
            )  # substitution
    return dp[n][m]


def dtw_distance(seq1: list[int], seq2: list[int]) -> float:
    """Pairwise Dynamic Time Warping alignment distance, allowing time stretching/squeezing."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float(max(n, m))
    dp = np.full((n, m), np.inf)
    dp[0, 0] = 0.0 if seq1[0] == seq2[0] else 1.0

    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + (0.0 if seq1[i] == seq2[0] else 1.0)
    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + (0.0 if seq1[0] == seq2[j] else 1.0)

    for i in range(1, n):
        for j in range(1, m):
            cost = 0.0 if seq1[i] == seq2[j] else 1.0
            dp[i, j] = cost + min(
                dp[i - 1, j],  # insertion
                dp[i, j - 1],  # deletion
                dp[i - 1, j - 1],
            )  # match/mismatch
    return float(dp[n - 1, m - 1])


def classify_trajectories_payload(
    trajectory: CosmicTrajectory,
    *,
    method: str,
    threshold: float,
) -> dict[str, Any]:
    """Classify patients based on their clinical trajectories.

    Supported methods:
      - 'complexity': entropy and transition count based classification (Stable, Gradual, Volatile)
      - 'transition': Markov-chain self-retention probability based classification (Highly Stable, Transitioning, Volatile)
      - 'levenshtein': Levenshtein edit distance matrix hierarchical complete-linkage clustering into 4 cohorts
      - 'dtw': Dynamic Time Warping distance matrix hierarchical complete-linkage clustering into 4 cohorts
      - 'sequence': trajectory structure-based classification (Singleton, Monostate, Multistate)
    """
    labels = trajectory.cluster_labels(threshold)
    frame = trajectory.obs.assign(cluster=labels)
    pivoted = frame.pivot(index="entity_id", columns="t", values="cluster")

    classification = {}
    classes_summary = {}

    if method == "complexity":
        # Calculate Shannon entropy and transition counts for each entity
        for entity_id, row in pivoted.iterrows():
            seq = [v for v in row if not pd.isna(v)]
            if not seq:
                classification[str(entity_id)] = {
                    "class": "Unknown",
                    "entropy": 0.0,
                    "transitions": 0,
                    "length": 0,
                    "sequence": [],
                }
                continue

            # Shannon entropy of cluster visits
            counts = pd.Series(seq).value_counts()
            probs = counts / len(seq)
            entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 1 else 0.0

            # State-to-state transition count
            transitions = sum(1 for a, b in zip(seq, seq[1:]) if a != b)

            # Assign complexity class
            if len(seq) <= 1:
                p_class = "Singleton (1 visit)"
            elif entropy == 0.0:
                p_class = "Stable (0 entropy)"
            elif transitions < 2:
                p_class = "Gradual Transition"
            else:
                p_class = "Volatile / Refractory"

            classification[str(entity_id)] = {
                "class": p_class,
                "entropy": round(entropy, 4),
                "transitions": int(transitions),
                "length": int(len(seq)),
                "sequence": [int(v) for v in seq],
            }

    elif method == "transition":
        # Calculate self-retention rate
        for entity_id, row in pivoted.iterrows():
            seq = [v for v in row if not pd.isna(v)]
            if len(seq) <= 1:
                classification[str(entity_id)] = {
                    "class": "Insufficient Visits (<2)",
                    "self_retention_rate": 1.0,
                    "length": int(len(seq)),
                    "sequence": [int(v) for v in seq],
                }
                continue

            # Transitions that are self-loops (a == b)
            self_loops = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
            self_retention_rate = float(self_loops / (len(seq) - 1))

            if self_retention_rate >= 0.8:
                p_class = "Highly Stable (>=80% retention)"
            elif self_retention_rate >= 0.4:
                p_class = "Transitioning (40%-80% retention)"
            else:
                p_class = "Highly Volatile (<40% retention)"

            classification[str(entity_id)] = {
                "class": p_class,
                "self_retention_rate": round(self_retention_rate, 4),
                "length": int(len(seq)),
                "sequence": [int(v) for v in seq],
            }

    elif method in {"levenshtein", "dtw"}:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Extract sequences
        entities_list = []
        seqs_list = []
        for entity_id, row in pivoted.iterrows():
            seq = [int(v) for v in row if not pd.isna(v)]
            entities_list.append(entity_id)
            seqs_list.append(seq)

        # Pairwise distance matrix
        n_ents = len(entities_list)
        dist_matrix = np.zeros((n_ents, n_ents))
        for i in range(n_ents):
            for j in range(i + 1, n_ents):
                if method == "levenshtein":
                    d = levenshtein_distance(seqs_list[i], seqs_list[j])
                else:
                    d = dtw_distance(seqs_list[i], seqs_list[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Hierarchical complete linkage clustering (robust on edit distances)
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method="complete")
        # Define 4 cohorts based on distance cuts
        cohort_labels = fcluster(Z, t=4, criterion="maxclust")

        # Assign classes
        for idx, entity_id in enumerate(entities_list):
            cohort_id = int(cohort_labels[idx])
            classification[str(entity_id)] = {
                "class": f"Cohort Class {cohort_id}",
                "length": int(len(seqs_list[idx])),
                "sequence": seqs_list[idx],
            }

    elif method == "sequence":
        # Classify by sequence profile type
        for entity_id, row in pivoted.iterrows():
            seq = [v for v in row if not pd.isna(v)]
            unique_states = len(set(seq))

            if len(seq) <= 1:
                p_class = "Singleton Path"
            elif unique_states == 1:
                p_class = "Monostate Path"
            else:
                p_class = f"Multistate Transitioning Path ({unique_states} states)"

            classification[str(entity_id)] = {
                "class": p_class,
                "unique_states": int(unique_states),
                "length": int(len(seq)),
                "sequence": [int(v) for v in seq],
            }
    else:
        raise ValueError(f"Unknown classification method '{method}'")

    # Aggregate classes summary
    classes = [info["class"] for info in classification.values()]
    counts = pd.Series(classes).value_counts()
    for p_class, count in counts.items():
        classes_summary[p_class] = {
            "count": int(count),
            "fraction": round(float(count) / len(classification), 4),
        }

    # Sort classifications by entity_id for deterministic return
    sorted_classification = {
        k: classification[k] for k in sorted(classification.keys())
    }

    return {
        "representation": "trajectory",
        "method": method,
        "interpretation_threshold": threshold,
        "n_entities": len(sorted_classification),
        "classes_summary": classes_summary,
        "classification": sorted_classification,
        "next_tools": ["get_trajectory_archetypes", "diagnose_longitudinal_graph"],
    }
