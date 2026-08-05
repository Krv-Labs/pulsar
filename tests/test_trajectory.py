"""Tests for CosmicTrajectory — the observation-centric longitudinal representation."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pulsar import CosmicTrajectory, ThemaRS
from pulsar.config import load_config


def _config(
    *,
    dimensions=(2,),
    seeds=(42,),
    epsilons=(0.2, 0.4),
    construction="exact",
    name="test_trajectory",
):
    return load_config(
        {
            "run": {"name": name},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "pca": {
                    "dimensions": {"values": list(dimensions)},
                    "seed": {"values": list(seeds)},
                },
                "ball_mapper": {"epsilon": {"values": list(epsilons)}},
            },
            "cosmic_graph": {
                "construction_threshold": 0.0,
                "construction": construction,
            },
            "output": {"n_reps": 1},
        }
    )


def _migrating_panel(n_stay=10, n_move=10, n_times=4, seed=0):
    """Two blobs; `n_move` entities drift from blob A to blob B over the window.

    Entities ``0..n_stay-1`` sit in blob A the whole time, ``n_stay..2*n_stay-1`` sit in
    blob B, and the last ``n_move`` migrate A -> B. A late observation of a migrating
    entity should therefore resemble an *early* observation of a blob-B entity.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros(3)
    b = np.array([10.0, 10.0, 10.0])
    snapshots = []
    for t in range(n_times):
        frac = t / max(n_times - 1, 1)
        rows = np.vstack(
            [
                a + rng.normal(scale=0.2, size=(n_stay, 3)),
                b + rng.normal(scale=0.2, size=(n_stay, 3)),
                a + frac * (b - a) + rng.normal(scale=0.2, size=(n_move, 3)),
            ]
        )
        snapshots.append(rows)
    return snapshots


class TestPanelInvariants:
    def test_observation_frame_shape_and_lookup(self):
        snapshots = _migrating_panel()
        ct = CosmicTrajectory.from_snapshots(snapshots, _config())

        assert ct.n_observations == sum(s.shape[0] for s in snapshots)
        assert ct.n_entities == 30
        assert ct.n_times == 4
        assert len(ct.obs) == ct.n_observations
        assert ct.obs.index.name == "obs_id"

        # obs_id is stacked row position: entity i at time t sits at t * n + i here.
        assert ct.observation_index(entity_id=5, t=2) == 2 * 30 + 5
        assert ct.obs.loc[ct.observation_index(7, 3), "t"] == 3

    def test_similarity_is_symmetric_with_zero_diagonal(self):
        ct = CosmicTrajectory.from_snapshots(_migrating_panel(), _config())
        difference = ct.similarity - ct.similarity.T
        assert difference.nnz == 0
        assert not ct.similarity.diagonal().any()
        assert ct.similarity.nnz > 0

    def test_ragged_panel_is_supported(self):
        rng = np.random.default_rng(1)
        snapshots = [rng.normal(size=(n, 3)) for n in (6, 4, 5)]
        ct = CosmicTrajectory.from_snapshots(snapshots, _config())

        assert ct.n_observations == 15
        assert ct.obs["t"].value_counts().sort_index().tolist() == [6, 4, 5]

    def test_entity_ids_are_carried_through(self):
        snapshots = _migrating_panel(n_stay=2, n_move=2, n_times=2)
        ids = ["p0", "p1", "p2", "p3", "p4", "p5"]
        ct = CosmicTrajectory.from_snapshots(snapshots, _config(), entity_ids=ids)

        assert ct.obs["entity_id"].tolist() == ids * 2
        assert ct.observation_index("p3", 1) == 6 + 3

    def test_composite_entity_ids_remain_a_flat_shared_sequence(self):
        snapshots = _migrating_panel(n_stay=2, n_move=2, n_times=2)
        ids = [("patient", i) for i in range(6)]

        ct = CosmicTrajectory.from_snapshots(snapshots, _config(), entity_ids=ids)

        assert ct.obs["entity_id"].tolist() == ids * 2

    def test_ragged_snapshot_entity_ids_are_carried_through(self):
        snapshots = _migrating_panel(n_stay=2, n_move=1, n_times=2)
        ct = CosmicTrajectory.from_snapshots(
            [snapshots[0][:2], snapshots[1][:2]],
            _config(),
            snapshot_entity_ids=[["p0", "p1"], ["p1", "p2"]],
        )

        assert ct.obs["entity_id"].tolist() == ["p0", "p1", "p1", "p2"]
        assert ct.observation_index("p1", 1) == 2

    def test_mismatched_entity_ids_raise(self):
        snapshots = _migrating_panel(n_stay=2, n_move=2, n_times=2)
        with pytest.raises(
            ValueError, match="Entity IDs must match snapshot row counts"
        ):
            CosmicTrajectory.from_snapshots(snapshots, _config(), entity_ids=["a", "b"])

    def test_empty_and_malformed_input_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            CosmicTrajectory.from_snapshots([], _config())
        with pytest.raises(ValueError, match="share a feature count"):
            CosmicTrajectory.from_snapshots(
                [np.zeros((3, 2)), np.zeros((3, 4))], _config()
            )
        with pytest.raises(ValueError, match="must be 2-D"):
            CosmicTrajectory.from_snapshots([np.zeros(3)], _config())
        with pytest.raises(ValueError, match="similarity_threshold"):
            CosmicTrajectory.from_snapshots(
                _migrating_panel(), _config(), similarity_threshold=-0.1
            )


class TestIncidence:
    def test_incidence_column_sums_match_ball_sizes(self):
        ct = CosmicTrajectory.from_snapshots(
            _migrating_panel(), _config(epsilons=(0.2, 0.4, 0.6))
        )

        assert ct.incidence.shape == (ct.n_observations, len(ct.balls))
        column_sums = np.asarray(ct.incidence.sum(axis=0)).ravel()
        np.testing.assert_array_equal(column_sums, ct.balls["size"].to_numpy())

    def test_comembership_diagonal_counts_ball_memberships(self):
        ct = CosmicTrajectory.from_snapshots(
            _migrating_panel(), _config(epsilons=(0.2, 0.5))
        )

        comembership = (ct.incidence @ ct.incidence.T).diagonal()
        row_sums = np.asarray(ct.incidence.sum(axis=1)).ravel()
        np.testing.assert_array_equal(comembership, row_sums)
        # BallMapper covers every point, so each observation joins >= 1 ball per map.
        assert row_sums.min() >= 1

    def test_ball_provenance_is_recorded(self):
        ct = CosmicTrajectory.from_snapshots(
            _migrating_panel(), _config(dimensions=(2, 3), epsilons=(0.2, 0.5))
        )

        assert set(ct.balls["scope"]) == {"global"}
        assert set(ct.balls["eps"]) == {0.2, 0.5}
        assert set(ct.balls["dim"]) == {2, 3}


class TestCrossTime:
    def test_cross_time_edges_exist_for_migrating_entities(self):
        snapshots = _migrating_panel(n_times=4)
        ct = CosmicTrajectory.from_snapshots(
            snapshots, _config(epsilons=(0.2, 0.4, 0.6))
        )

        cross = ct.cross_time()
        assert cross.nnz > 0

        # A migrating entity at the final step should reach a blob-B entity at t=0.
        migrated_late = ct.observation_index(entity_id=25, t=3)
        settled_early = ct.observation_index(entity_id=15, t=0)
        assert ct.similarity[migrated_late, settled_early] > 0.0

    def test_cross_time_is_a_strict_subset_of_similarity(self):
        ct = CosmicTrajectory.from_snapshots(_migrating_panel(), _config())

        cross = ct.cross_time().tocoo()
        times = ct.obs["t"].to_numpy()
        assert (times[cross.row] != times[cross.col]).all()
        assert cross.nnz < ct.similarity.nnz

    def test_within_time_slice_has_expected_shape(self):
        snapshots = _migrating_panel(n_times=3)
        ct = CosmicTrajectory.from_snapshots(snapshots, _config())

        slice_1 = ct.within_time(1)
        assert slice_1.shape == (snapshots[1].shape[0], snapshots[1].shape[0])
        with pytest.raises(KeyError, match="no observations at t=9"):
            ct.within_time(9)


class TestThemaRSParity:
    def test_single_snapshot_matches_static_pipeline(self):
        """A T=1 pooled build is the static pipeline — weights must match exactly."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 4))
        config = _config(epsilons=(0.6, 1.0), construction="exact")

        ct = CosmicTrajectory.from_snapshots([X], config)
        model = ThemaRS(config).fit(
            data=pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        )

        expected = nx.to_scipy_sparse_array(
            model.cosmic_graph,
            nodelist=range(X.shape[0]),
            weight="weight",
            format="csr",
        )
        np.testing.assert_allclose(
            ct.similarity.toarray(), expected.toarray(), atol=1e-12
        )


class TestTrajectories:
    def test_trajectory_frame_is_entity_by_time(self):
        snapshots = _migrating_panel(n_times=4)
        ct = CosmicTrajectory.from_snapshots(snapshots, _config())

        frame = ct.trajectory_frame(threshold=0.05)
        assert frame.shape == (30, 4)
        assert frame.index.name == "entity_id"
        assert list(frame.columns) == [0, 1, 2, 3]

        archetypes = ct.trajectory_archetypes(threshold=0.05)
        assert archetypes.sum() == 30
        assert len(archetypes) >= 1

    def test_trajectory_edges_link_consecutive_observations(self):
        snapshots = _migrating_panel(n_stay=3, n_move=3, n_times=4)
        ct = CosmicTrajectory.from_snapshots(snapshots, _config())

        edges = ct.trajectory_edges()
        n_entities, n_times = 9, 4
        assert len(edges) == n_entities * (n_times - 1)
        for src, dst in edges:
            assert ct.obs.loc[src, "entity_id"] == ct.obs.loc[dst, "entity_id"]
            assert ct.obs.loc[dst, "t"] == ct.obs.loc[src, "t"] + 1

    def test_trajectory_edges_do_not_bridge_missing_time_steps(self):
        snapshots = _migrating_panel(n_stay=2, n_move=1, n_times=3)
        ct = CosmicTrajectory.from_snapshots(
            [snapshots[0][:2], snapshots[1][:2], snapshots[2][:2]],
            _config(),
            snapshot_entity_ids=[["p0", "p1"], ["p1", "p2"], ["p0", "p2"]],
        )

        assert all(
            ct.obs.loc[dst, "t"] == ct.obs.loc[src, "t"] + 1
            for src, dst in ct.trajectory_edges()
        )
        p0_times = ct.obs[ct.obs["entity_id"] == "p0"]
        assert not any(
            src in p0_times.index and dst in p0_times.index
            for src, dst in ct.trajectory_edges()
        )

    def test_cluster_labels_align_with_observations(self):
        ct = CosmicTrajectory.from_snapshots(_migrating_panel(), _config())
        labels = ct.cluster_labels(threshold=0.05)
        assert labels.shape == (ct.n_observations,)


class TestToNetworkx:
    def test_graph_carries_observation_attributes(self):
        ct = CosmicTrajectory.from_snapshots(_migrating_panel(), _config())
        graph = ct.to_networkx(threshold=0.05)

        assert graph.number_of_nodes() == ct.n_observations
        obs_id = ct.observation_index(4, 2)
        assert graph.nodes[obs_id]["entity_id"] == 4
        assert graph.nodes[obs_id]["t"] == 2
        for _, _, weight in graph.edges(data="weight"):
            assert weight > 0.05

    def test_trajectory_edges_are_optional(self):
        ct = CosmicTrajectory.from_snapshots(_migrating_panel(), _config())

        plain = ct.to_networkx(threshold=0.9)
        tagged = ct.to_networkx(threshold=0.9, include_trajectory=True)

        assert all(data.get("type") is None for *_, data in plain.edges(data=True))
        assert tagged.number_of_edges() >= plain.number_of_edges()
        for src, dst in ct.trajectory_edges():
            assert tagged[src][dst]["type"] == "TRAJECTORY"


class TestScale:
    def test_large_panel_stays_sparse(self):
        """n=200, T=10 -> N=2000 must complete without an (N, N) dense allocation."""
        rng = np.random.default_rng(3)
        centers = rng.normal(scale=5.0, size=(4, 5))
        snapshots = [
            centers[rng.integers(0, 4, size=200)] + rng.normal(scale=0.5, size=(200, 5))
            for _ in range(10)
        ]

        ct = CosmicTrajectory.from_snapshots(
            snapshots, _config(epsilons=(0.8,), construction="minhash")
        )

        assert ct.n_observations == 2000
        assert sp.issparse(ct.similarity)
        assert sp.issparse(ct.incidence)
        assert ct.similarity.nnz < 2000 * 2000
        assert ct.cross_time().nnz > 0

    def test_similarity_threshold_prunes_edges(self):
        snapshots = _migrating_panel()
        loose = CosmicTrajectory.from_snapshots(snapshots, _config())
        tight = CosmicTrajectory.from_snapshots(
            snapshots, _config(), similarity_threshold=0.5
        )

        assert tight.similarity.nnz < loose.similarity.nnz
        assert (tight.similarity.data > 0.5).all()
