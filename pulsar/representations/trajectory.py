"""
CosmicTrajectory — observation-centric cosmic representation for longitudinal panels.

``TemporalCosmicGraph`` models a panel as a dense ``(n, n, T)`` tensor whose nodes are
*entities*; every edge lives inside one time slice. It therefore cannot express "patient
A at admission resembles patient B at discharge". ``CosmicTrajectory`` changes the node
identity to the *observation* ``(entity, t)`` and pools every observation into a single
geometry, so BallMapper covers — and the cosmic similarity edges derived from them — span
time by construction.

Storage is sparse matrices plus two metadata frames, not a materialized property graph:

===============  =================================================================
``obs``          ``pd.DataFrame``, N rows indexed by ``obs_id``: entity_id, t, ...
``balls``        ``pd.DataFrame``, B rows indexed by ``ball_id``: scope, eps, dim, size
``similarity``   ``sp.csr_matrix`` (N, N), symmetric CO_SIMILAR weights in (0, 1]
``incidence``    ``sp.csr_matrix`` (N, B) int8, IN_BALL — the hyperedge incidence
===============  =================================================================

The ``Entity`` / ``TimeLayer`` nodes and ``OF_ENTITY`` / ``AT_TIME`` edges of the design
spec carry no information beyond ``obs.entity_id`` and ``obs.t``, so they stay columns.
``TRAJECTORY`` edges are derivable arithmetic (:meth:`CosmicTrajectory.trajectory_edges`)
rather than stored. ``SCHEMA`` records the mapping from this storage to the spec's typed
labels, which is what a Neo4j/parquet export layer needs.

The incidence matrix is the hypergraph view — its columns *are* the cover elements::

    ct.incidence @ ct.incidence.T                    # (N, N) obs-obs co-membership
    ct.incidence.T @ ct.incidence                    # (B, B) ball-ball overlap
    ct.incidence[:, (ct.balls["eps"] < 0.8).values]  # slice the cover by scale

Complements ``TemporalCosmicGraph``; does not replace or modify it.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pulsar._pulsar import StandardScaler, ball_mapper_grid
from pulsar.config import PulsarConfig
from pulsar.pipeline import _CosmicBuilder, projection_grid

__all__ = ["SCHEMA", "CosmicTrajectory"]


#: Mapping from this module's storage onto the typed labels in the CosmicTrajectory
#: design spec (docs/design/cosmic_trajectory.md §4). Kept as data so an export layer
#: does not have to re-derive the contract.
SCHEMA: dict[str, dict[str, Any]] = {
    "Observation": {
        "store": "obs",
        "key": "obs_id",
        "props": ("entity_id", "t", "timestamp"),
    },
    "Ball": {
        "store": "balls",
        "key": "ball_id",
        "props": ("scope", "eps", "dim", "size"),
    },
    "IN_BALL": {
        "store": "incidence",
        "endpoints": ("Observation", "Ball"),
        "directed": False,
    },
    "CO_SIMILAR": {
        "store": "similarity",
        "endpoints": ("Observation", "Observation"),
        "props": ("weight",),
        "directed": False,
    },
    "TRAJECTORY": {
        "store": "derived — trajectory_edges()",
        "endpoints": ("Observation", "Observation"),
        "props": ("delta_t",),
        "directed": True,
    },
}


def _stack_panel(
    snapshots: Sequence[np.ndarray],
    entity_ids: Sequence[Hashable] | Sequence[Sequence[Hashable]] | None,
    timestamps: Sequence[Any] | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack a panel into a pooled ``(N, F)`` matrix plus its observation frame.

    ``obs_id`` is the stacked row position rather than ``t * n + i``, so ragged panels
    (an entity missing at some ``t``) work without a special case.
    """
    if len(snapshots) == 0:
        raise ValueError("snapshots must be non-empty")

    arrays = [np.ascontiguousarray(s, dtype=np.float64) for s in snapshots]
    for t, arr in enumerate(arrays):
        if arr.ndim != 2:
            raise ValueError(f"snapshot {t} must be 2-D, got shape {arr.shape}")

    widths = {arr.shape[1] for arr in arrays}
    if len(widths) != 1:
        raise ValueError(
            f"All snapshots must share a feature count; got {sorted(widths)}"
        )

    counts = [arr.shape[0] for arr in arrays]
    if sum(counts) == 0:
        raise ValueError("snapshots contain no rows")

    if entity_ids is None:
        ids: np.ndarray = np.concatenate([np.arange(c) for c in counts])
    else:
        id_rows = list(entity_ids)
        if id_rows and isinstance(id_rows[0], (list, tuple, np.ndarray, pd.Index)):
            if len(id_rows) != len(arrays):
                raise ValueError(
                    f"entity_ids has {len(id_rows)} time slices but there are "
                    f"{len(arrays)} snapshots"
                )
            per_snapshot = [np.asarray(list(row)) for row in id_rows]
            bad = [
                t
                for t, (row, count) in enumerate(zip(per_snapshot, counts))
                if row.shape[0] != count
            ]
            if bad:
                raise ValueError(
                    "per-snapshot entity_ids must match snapshot row counts; "
                    f"mismatch at snapshots {bad}"
                )
            ids = np.concatenate(per_snapshot)
        else:
            ids_1d = np.asarray(id_rows)
            bad = [t for t, c in enumerate(counts) if c != ids_1d.shape[0]]
            if bad:
                raise ValueError(
                    f"entity_ids has {ids_1d.shape[0]} entries but snapshots {bad} have "
                    "different row counts; pass per-snapshot entity IDs for ragged panels"
                )
            ids = np.tile(ids_1d, len(arrays))

    obs = pd.DataFrame(
        {"entity_id": ids, "t": np.repeat(np.arange(len(arrays)), counts)},
        index=pd.RangeIndex(sum(counts), name="obs_id"),
    )
    if timestamps is not None:
        stamps = list(timestamps)
        if len(stamps) != len(arrays):
            raise ValueError(
                f"timestamps has {len(stamps)} entries but there are {len(arrays)} "
                "snapshots"
            )
        obs["timestamp"] = np.repeat(np.asarray(stamps, dtype=object), counts)

    return np.vstack(arrays), obs


def _incidence_from_ball_maps(
    members: list[list[int]], n_observations: int
) -> sp.csr_matrix:
    """Build the ``(N, B)`` IN_BALL incidence from per-ball membership lists."""
    lengths = np.fromiter((len(m) for m in members), dtype=np.int64, count=len(members))
    nnz = int(lengths.sum())
    indices = np.fromiter(chain.from_iterable(members), dtype=np.int32, count=nnz)
    indptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    csc = sp.csc_matrix(
        (np.ones(nnz, dtype=np.int8), indices, indptr),
        shape=(n_observations, len(members)),
    )
    return csc.tocsr()


def _similarity_from_cosmic(
    cosmic_graph: Any, n_observations: int, threshold: float
) -> sp.csr_matrix:
    """Symmetric CSR from ``CosmicGraph.weighted_edges()`` (upper triangle, w > 0).

    Never touches ``weighted_adj`` / ``adj``, which densify to ``(N, N)``.
    """
    edges = cosmic_graph.weighted_edges()
    if not edges:
        return sp.csr_matrix((n_observations, n_observations), dtype=np.float64)

    # (E, 3) float64: row indices below 2**53 are exact, so the int cast is lossless.
    triples = np.asarray(edges, dtype=np.float64)
    weights = triples[:, 2]
    keep = weights > threshold
    upper = sp.coo_matrix(
        (
            weights[keep],
            (triples[keep, 0].astype(np.int64), triples[keep, 1].astype(np.int64)),
        ),
        shape=(n_observations, n_observations),
    )
    return (upper + upper.T).tocsr()


@dataclass(frozen=True)
class CosmicTrajectory:
    """Typed longitudinal cosmic representation on observation nodes.

    Build with :meth:`from_snapshots`. See the module docstring for the storage layout
    and ``SCHEMA`` for the mapping onto the design spec's node/edge labels.
    """

    obs: pd.DataFrame
    balls: pd.DataFrame
    similarity: sp.csr_matrix
    incidence: sp.csr_matrix
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_snapshots(
        cls,
        snapshots: Sequence[np.ndarray],
        config: PulsarConfig,
        *,
        entity_ids: Sequence[Hashable] | Sequence[Sequence[Hashable]] | None = None,
        timestamps: Sequence[Any] | None = None,
        similarity_threshold: float = 0.0,
        scale: bool = True,
    ) -> CosmicTrajectory:
        """Build a trajectory graph from a list of per-time observation matrices.

        Parameters
        ----------
        snapshots
            Length-``T`` sequence of ``(n_t, F)`` arrays sharing a feature schema. Row
            counts may differ across ``t``. For ragged panels, pass one entity-ID
            sequence per snapshot so rows retain their original identity.
        config
            Reused for the projection grid, BallMapper epsilons, and cosmic
            construction mode (``minhash`` or ``exact``).
        similarity_threshold
            Drop CO_SIMILAR edges at or below this weight during materialization.
        scale
            Fit one ``StandardScaler`` on the *pooled* matrix. Scaling per time step
            would re-center each slice and erase the drift that cross-time edges exist
            to detect. Set ``False`` when snapshots are already a shared embedding.
        """
        if similarity_threshold < 0.0:
            raise ValueError(
                f"similarity_threshold must be >= 0, got {similarity_threshold}"
            )

        pooled, obs = _stack_panel(snapshots, entity_ids, timestamps)
        if scale:
            pooled = np.array(StandardScaler().fit_transform(pooled))
        embeddings = projection_grid(pooled, config)
        return cls._build(embeddings, obs, config, similarity_threshold, scale)

    @classmethod
    def _build(
        cls,
        embeddings: list[np.ndarray],
        obs: pd.DataFrame,
        config: PulsarConfig,
        similarity_threshold: float,
        scaled: bool,
    ) -> CosmicTrajectory:
        n_observations = len(obs)
        builder = _CosmicBuilder(n_observations, config.cosmic_graph)
        members: list[list[int]] = []
        ball_rows: list[tuple[str, float, int, int]] = []

        # One ball_mapper_grid call per embedding — the batch-size-1 form of the loop in
        # ThemaRS.fit. Keeps ball provenance known by construction instead of inferring
        # it from the grid's (undocumented, projection-dependent) output ordering.
        for embedding in embeddings:
            ball_maps = ball_mapper_grid([embedding], config.ball_mapper.epsilons)
            builder.accumulate(ball_maps)
            for ball_map in ball_maps:
                nodes = ball_map.nodes  # clones on access; read once
                members.extend(nodes)
                ball_rows.extend(
                    ("global", ball_map.eps, embedding.shape[1], len(m)) for m in nodes
                )

        cosmic_graph = builder.build()
        balls = pd.DataFrame(
            ball_rows, columns=["scope", "eps", "dim", "size"]
        ).set_index(pd.RangeIndex(len(ball_rows), name="ball_id"))

        return cls(
            obs=obs,
            balls=balls,
            similarity=_similarity_from_cosmic(
                cosmic_graph, n_observations, similarity_threshold
            ),
            incidence=_incidence_from_ball_maps(members, n_observations),
            meta={
                "n_observations": n_observations,
                "n_balls": len(ball_rows),
                "n_embeddings": len(embeddings),
                "construction": config.cosmic_graph.construction,
                "similarity_threshold": similarity_threshold,
                "epsilons": list(config.ball_mapper.epsilons),
                "scaled": scaled,
                "scope": "global",
            },
        )

    # -------------------------------------------------------------------- properties

    @property
    def n_observations(self) -> int:
        return len(self.obs)

    @property
    def n_entities(self) -> int:
        return int(self.obs["entity_id"].nunique())

    @property
    def n_times(self) -> int:
        return int(self.obs["t"].nunique())

    def observation_index(self, entity_id: Hashable, t: int) -> int:
        """``obs_id`` of one observation. Raises ``KeyError`` if absent or ambiguous."""
        match = self.obs.index[
            (self.obs["entity_id"] == entity_id) & (self.obs["t"] == t)
        ]
        if len(match) != 1:
            raise KeyError(
                f"expected exactly one observation for entity {entity_id!r} at t={t}, "
                f"found {len(match)}"
            )
        return int(match[0])

    # ------------------------------------------------------------------------- slices

    def thresholded(self, threshold: float = 0.0) -> sp.csr_matrix:
        """CO_SIMILAR restricted to weights strictly above ``threshold``."""
        if threshold <= 0.0:
            return self.similarity
        coo = self.similarity.tocoo()
        keep = coo.data > threshold
        return sp.coo_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape
        ).tocsr()

    def cross_time(self, threshold: float = 0.0) -> sp.csr_matrix:
        """CO_SIMILAR edges joining observations from *different* time steps."""
        coo = self.thresholded(threshold).tocoo()
        times = self.obs["t"].to_numpy()
        keep = times[coo.row] != times[coo.col]
        return sp.coo_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape
        ).tocsr()

    def within_time(self, t: int, threshold: float = 0.0) -> sp.csr_matrix:
        """Square CO_SIMILAR submatrix over the observations at ``t``.

        Rows/columns follow ``self.obs.index[self.obs["t"] == t]`` order — comparable to
        one page of a ``TemporalCosmicGraph`` tensor, though the weights differ because
        this build pools all times into one cover.
        """
        idx = np.flatnonzero(self.obs["t"].to_numpy() == t)
        if idx.size == 0:
            raise KeyError(f"no observations at t={t}")
        sub = self.thresholded(threshold)
        return sub[idx][:, idx]

    def trajectory_edges(self) -> list[tuple[int, int]]:
        """Directed ``t -> t+1`` same-entity edges as ``(earlier, later)`` obs ids."""
        ordered = self.obs.sort_values(["entity_id", "t"])
        ids = ordered.index.to_numpy()
        entities = ordered["entity_id"].to_numpy()
        times = ordered["t"].to_numpy()
        pairs = np.column_stack([ids[:-1], ids[1:]])[
            (entities[:-1] == entities[1:]) & (times[1:] == times[:-1] + 1)
        ]
        return [(int(a), int(b)) for a, b in pairs]

    # ----------------------------------------------------------------- trajectories

    def cluster_labels(self, threshold: float = 0.0) -> np.ndarray:
        """Connected-component label per observation, aligned to ``obs.index``."""
        _, labels = connected_components(self.thresholded(threshold), directed=False)
        return labels

    def trajectory_frame(self, threshold: float = 0.0) -> pd.DataFrame:
        """Entity x time table of cluster labels — one row is one trajectory."""
        frame = self.obs.assign(cluster=self.cluster_labels(threshold))
        return frame.pivot(index="entity_id", columns="t", values="cluster")

    def trajectory_archetypes(self, threshold: float = 0.0) -> pd.Series:
        """Distinct cluster sequences and how many entities follow each."""
        return self.trajectory_frame(threshold).apply(tuple, axis=1).value_counts()

    # ------------------------------------------------------------------------ views

    def to_networkx(
        self, threshold: float = 0.0, include_trajectory: bool = False
    ) -> nx.Graph:
        """Observation graph with ``weight`` on CO_SIMILAR edges and ``obs`` node attrs.

        ``include_trajectory`` adds same-entity ``t -> t+1`` edges tagged
        ``type="TRAJECTORY"``. The graph stays undirected — tagging every CO_SIMILAR
        edge would cost an O(E) Python loop, and trajectory direction is recoverable
        from the ``t`` node attribute.
        """
        graph = nx.from_scipy_sparse_array(
            self.thresholded(threshold), edge_attribute="weight"
        )
        nx.set_node_attributes(graph, self.obs.to_dict(orient="index"))
        if include_trajectory:
            graph.add_edges_from(self.trajectory_edges(), type="TRAJECTORY")
        return graph

    def __repr__(self) -> str:
        return (
            f"CosmicTrajectory(observations={self.n_observations}, "
            f"entities={self.n_entities}, times={self.n_times}, "
            f"balls={self.incidence.shape[1]}, edges={self.similarity.nnz // 2})"
        )
