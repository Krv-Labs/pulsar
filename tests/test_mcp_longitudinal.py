"""Tests for the longitudinal MCP surface (panel pivot + both representations)."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from test_trajectory import _migrating_panel  # noqa: E402

from pulsar.config import load_config  # noqa: E402
from pulsar.mcp.longitudinal import PanelError, pivot_panel  # noqa: E402
from pulsar.mcp.session import _get_session, _sessions  # noqa: E402
from pulsar.mcp.tools import ALL_TOOLS_LIST  # noqa: E402
from pulsar.mcp.tools.ingestion import ingest_dataset  # noqa: E402
from pulsar.mcp.tools.longitudinal import (  # noqa: E402
    build_longitudinal_graph,
    diagnose_longitudinal_graph,
    get_cross_time_neighbors,
    get_trajectory_archetypes,
    classify_trajectories,
)


def _long_frame(n_times=4, seed=0) -> pd.DataFrame:
    """The migrating-blobs fixture, melted into long format."""
    snapshots = _migrating_panel(n_times=n_times, seed=seed)
    rows = []
    for t, snapshot in enumerate(snapshots):
        for i, values in enumerate(snapshot):
            rows.append(
                {
                    "patient_id": f"p{i}",
                    "hour": t,
                    "f0": values[0],
                    "f1": values[1],
                    "f2": values[2],
                }
            )
    return pd.DataFrame(rows)


def _write_long(tmp_path, frame: pd.DataFrame | None = None) -> str:
    frame = _long_frame() if frame is None else frame
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _config_yaml(epsilons=(0.05, 0.1), dimensions=(2, 3)) -> str:
    """A small explicit grid so tests build fast and deterministically."""
    return yaml.safe_dump(
        {
            "run": {"name": "test_longitudinal"},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "projection": {
                    "method": "jl",
                    "dimensions": {"values": list(dimensions)},
                    "seed": {"values": [42]},
                },
                "ball_mapper": {"epsilon": {"values": list(epsilons)}},
            },
            "cosmic_graph": {"construction_threshold": 0.0, "construction": "exact"},
            "output": {"n_reps": 1},
        }
    )


def _build(tmp_path, **kwargs) -> dict:
    _sessions.clear()
    _get_session(None)
    path = kwargs.pop("path", None) or _write_long(tmp_path)
    dataset = json.loads(asyncio.run(ingest_dataset(path)))
    params = {
        "dataset_id": dataset["dataset_id"],
        "entity_column": "patient_id",
        "time_column": "hour",
        "config_yaml": _config_yaml(),
        "response_format": "json",
    }
    params.update(kwargs)
    return json.loads(asyncio.run(build_longitudinal_graph(**params)))


class TestPivotPolicies:
    def test_complete_panel_keeps_every_entity(self):
        panel = pivot_panel(_long_frame(), "patient_id", "hour")

        assert panel.n_entities == 30
        assert panel.n_times == 4
        assert panel.feature_columns == ["f0", "f1", "f2"]
        assert panel.report["alignment"] == "complete"
        assert panel.report["dropped_entities"]["total"] == 0
        assert all(snap.shape == (30, 3) for snap in panel.snapshots)

    def test_drop_entity_removes_incomplete_entities(self):
        frame = _long_frame()
        # p0 misses the final hour entirely.
        ragged = frame[~((frame["patient_id"] == "p0") & (frame["hour"] == 3))]

        panel = pivot_panel(ragged, "patient_id", "hour", on_missing="drop_entity")

        assert panel.n_entities == 29
        assert panel.report["dropped_entities"]["total"] == 1
        assert "p0" in panel.report["dropped_entities"]["preview"]
        assert panel.is_aligned

    def test_forward_fill_retains_entities_missing_later_steps(self):
        frame = _long_frame()
        ragged = frame[~((frame["patient_id"] == "p0") & (frame["hour"] == 3))]

        panel = pivot_panel(ragged, "patient_id", "hour", on_missing="forward_fill")

        assert panel.n_entities == 30
        assert panel.is_aligned
        # The carried value equals p0's hour-2 observation.
        source = ragged[(ragged["patient_id"] == "p0") & (ragged["hour"] == 2)]
        carried = panel.snapshots[3][panel.entity_ids.index("p0")]
        np.testing.assert_allclose(carried, source[["f0", "f1", "f2"]].to_numpy()[0])

    def test_allow_ragged_keeps_natural_slice_sizes(self):
        frame = _long_frame()
        ragged = frame[~((frame["patient_id"] == "p0") & (frame["hour"] == 3))]

        panel = pivot_panel(ragged, "patient_id", "hour", on_missing="allow_ragged")

        assert [snap.shape[0] for snap in panel.snapshots] == [30, 30, 30, 29]
        assert not panel.is_aligned
        assert panel.report["alignment"] == "ragged"

    def test_allow_ragged_preserves_each_snapshot_entity_ids(self):
        frame = _long_frame()
        ragged = frame[~((frame["patient_id"] == "p0") & (frame["hour"] == 3))]

        panel = pivot_panel(ragged, "patient_id", "hour", on_missing="allow_ragged")

        assert panel.snapshot_entity_ids[0][:2] == ["p0", "p1"]
        assert panel.snapshot_entity_ids[3][:2] == ["p1", "p2"]

    def test_structured_errors_for_bad_panels(self):
        frame = _long_frame()

        with pytest.raises(PanelError) as missing:
            pivot_panel(frame, "nope", "hour")
        assert missing.value.error_code == "PANEL_COLUMN_NOT_FOUND"

        with pytest.raises(PanelError) as dup:
            pivot_panel(pd.concat([frame, frame]), "patient_id", "hour")
        assert dup.value.error_code == "PANEL_DUPLICATE_OBSERVATIONS"

        single = frame[frame["hour"] == 0]
        with pytest.raises(PanelError) as one_step:
            pivot_panel(single, "patient_id", "hour")
        assert one_step.value.error_code == "PANEL_SINGLE_TIME_STEP"


class TestBuild:
    def test_build_both_representations(self, tmp_path):
        response = _build(tmp_path, representation="both")

        assert response["status"] == "ok"
        assert response["longitudinal_id"].startswith("lng_")
        assert response["panel"]["n_entities_kept"] == 30
        assert response["panel"]["observation_count"] == 120
        assert set(response["graph_surface"]) == {"trajectory", "temporal"}
        assert response["graph_surface"]["temporal"]["tensor_shape"] == [30, 30, 4]
        assert response["graph_surface"]["trajectory"]["n_observations"] == 120

    def test_cross_time_edges_are_produced(self, tmp_path):
        surface = _build(tmp_path)["graph_surface"]["trajectory"]

        # Migrating entities guarantee genuine cross-time structure.
        assert surface["cross_time_edges"] > 0
        assert 0.0 < surface["cross_time_fraction"] <= 1.0

    def test_ragged_panel_rejected_for_temporal(self, tmp_path):
        frame = _long_frame()
        ragged = frame[~((frame["patient_id"] == "p0") & (frame["hour"] == 3))]
        path = _write_long(tmp_path, ragged)

        response = _build(
            tmp_path, path=path, representation="temporal", on_missing="allow_ragged"
        )

        assert response["error_code"] == "RAGGED_PANEL_NOT_SUPPORTED"
        assert "trajectory" in response["agent_action"]

    def test_same_size_ragged_panel_is_rejected_when_entity_ids_differ(self, tmp_path):
        frame = _long_frame()
        ragged = frame[
            ~(
                ((frame["patient_id"] == "p0") & (frame["hour"] == 1))
                | ((frame["patient_id"] == "p1") & (frame["hour"] == 0))
            )
        ]

        response = _build(
            tmp_path,
            path=_write_long(tmp_path, ragged),
            representation="temporal",
            on_missing="allow_ragged",
        )

        assert response["error_code"] == "RAGGED_PANEL_NOT_SUPPORTED"

    def test_tensor_cost_guard(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PULSAR_MCP_MAX_TENSOR_BYTES", "1000")

        blocked = _build(tmp_path, representation="temporal")
        assert blocked["error_code"] == "TENSOR_TOO_LARGE"
        assert blocked["details"]["tensor_bytes"] == 30 * 30 * 4 * 8
        assert blocked["details"]["peak_required_bytes"] == 3 * 30 * 30 * 4 * 8
        assert blocked["details"]["limit_bytes"] == 1000

        # The sparse representation is unaffected by the dense ceiling.
        allowed = _build(tmp_path, representation="trajectory")
        assert allowed["status"] == "ok"

    def test_tensor_guard_budgets_peak_working_memory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PULSAR_MCP_MAX_TENSOR_BYTES", "30000")

        blocked = _build(tmp_path, representation="temporal")

        assert blocked["error_code"] == "TENSOR_TOO_LARGE"
        assert blocked["details"]["tensor_bytes"] < blocked["details"]["limit_bytes"]
        assert (
            blocked["details"]["peak_required_bytes"]
            > blocked["details"]["limit_bytes"]
        )

    def test_missing_keys_and_config_are_structured_errors(self, tmp_path):
        no_keys = _build(tmp_path, entity_column="", time_column="")
        assert no_keys["error_code"] == "PANEL_KEYS_MISSING"

        no_config = _build(tmp_path, config_yaml="")
        assert no_config["error_code"] == "CONFIG_REQUIRED"
        assert "create_config" in no_config["agent_action"]


class TestConfigAdaptation:
    def test_dimensions_above_feature_count_are_dropped(self, tmp_path):
        response = _build(tmp_path, config_yaml=_config_yaml(dimensions=(2, 8, 16)))

        notes = response["config_adaptation"]
        assert notes["panel_feature_count"] == 3
        assert notes["projection_dimensions_dropped"] == [8, 16]
        assert notes["projection_dimensions_applied"] == [2]

    def test_out_of_domain_epsilons_are_recalibrated(self, tmp_path):
        # Epsilons calibrated on the raw file are far outside the scaled panel's
        # k-NN domain and would collapse the cover into a single ball.
        response = _build(tmp_path, config_yaml=_config_yaml(epsilons=(6.0, 9.0)))

        notes = response["config_adaptation"]
        assert notes["epsilons_in_domain"] == 0
        assert "epsilons_applied" in notes
        # The reported domain is rounded for display; allow for that when
        # comparing against the unrounded epsilons actually applied.
        domain = notes["panel_epsilon_domain"]
        tol = 1e-4
        for eps in notes["epsilons_applied"]["preview"]:
            assert domain["knn_p5"] - tol <= eps <= domain["knn_p95"] + tol

    def test_in_domain_epsilons_are_left_alone(self, tmp_path):
        response = _build(tmp_path)

        notes = response["config_adaptation"]
        assert notes["epsilons_in_domain"] > 0
        assert "epsilons_applied" not in notes


class TestDiagnose:
    def test_trajectory_surface_is_measured(self, tmp_path):
        built = _build(tmp_path)
        diagnosis = json.loads(
            asyncio.run(
                diagnose_longitudinal_graph(
                    built["longitudinal_id"], response_format="json"
                )
            )
        )

        traj = diagnosis["surfaces"]["trajectory"]
        assert traj["node_identity"] == "observation (entity, t)"
        assert traj["panel"]["n_observations"] == 120
        assert traj["similarity_surface"]["cross_time_edges"] > 0
        assert traj["similarity_surface"]["weight_distribution"]["count"] > 0
        # Threshold sensitivity must be visible, not implied.
        rows = traj["threshold_sweep"]["rows"]
        assert len(rows) >= 3
        assert rows[0]["observation_clusters"] <= rows[-1]["observation_clusters"]

    def test_temporal_surface_compares_all_aggregations(self, tmp_path):
        built = _build(tmp_path, representation="temporal")
        diagnosis = json.loads(
            asyncio.run(
                diagnose_longitudinal_graph(
                    built["longitudinal_id"], response_format="json"
                )
            )
        )

        temporal = diagnosis["surfaces"]["temporal"]
        names = [agg["aggregation"] for agg in temporal["aggregations"]]
        assert names == [
            "persistence",
            "mean",
            "recency",
            "volatility",
            "trend",
            "change_point",
        ]
        for agg in temporal["aggregations"]:
            assert agg["why"] and agg["best_for"]
            # A self-scaled cut must select something, including for the
            # discrete-valued aggregations where q90 sits on the maximum.
            assert agg["edges_at_cut"] > 0
        trend = next(a for a in temporal["aggregations"] if a["aggregation"] == "trend")
        assert "converging_pairs" in trend and "diverging_pairs" in trend

    def test_requesting_an_unbuilt_surface_is_structured(self, tmp_path):
        built = _build(tmp_path, representation="trajectory")
        response = json.loads(
            asyncio.run(
                diagnose_longitudinal_graph(
                    built["longitudinal_id"], representation="temporal"
                )
            )
        )
        assert response["error_code"] == "REPRESENTATION_NOT_BUILT"


class TestArchetypes:
    def test_archetypes_recover_the_planted_groups(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                get_trajectory_archetypes(
                    built["longitudinal_id"], threshold=0.25, response_format="json"
                )
            )
        )

        # The fixture plants three groups of ten: stay-A, stay-B, and A->B movers.
        assert response["n_entities"] == 30
        assert response["distinct_trajectories"] == 3
        assert [a["n_entities"] for a in response["archetypes"]] == [10, 10, 10]
        # Only the migrating group changes cluster over time.
        transitions = sorted(a["transitions"] for a in response["archetypes"])
        assert transitions[0] == 0 and transitions[-1] > 0

    def test_archetype_payload_is_bounded(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                get_trajectory_archetypes(
                    built["longitudinal_id"],
                    threshold=0.25,
                    max_archetypes=2,
                    max_entities_per_archetype=3,
                    response_format="json",
                )
            )
        )

        assert response["archetypes_returned"] == 2
        assert response["archetypes_omitted"] == 1
        for archetype in response["archetypes"]:
            entities = archetype["entities"]
            assert set(entities) == {"total", "preview", "omitted"}
            assert len(entities["preview"]) <= 3
            assert entities["total"] == 10

    def test_negative_member_limit_is_rejected(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                get_trajectory_archetypes(
                    built["longitudinal_id"],
                    max_entities_per_archetype=-1,
                    response_format="json",
                )
            )
        )

        assert response["error_code"] == "INVALID_ARGUMENT"


class TestCrossTimeNeighbors:
    def test_migrated_entity_matches_its_destination_cohort_earlier(self, tmp_path):
        built = _build(tmp_path)
        # p25 migrates from blob A to blob B; by the last step it should resemble
        # the entities that sat in blob B (p10-p19) from the very beginning.
        response = json.loads(
            asyncio.run(
                get_cross_time_neighbors(
                    built["longitudinal_id"],
                    entity_id="p25",
                    t=3,
                    direction="backward",
                    max_neighbors=5,
                    response_format="json",
                )
            )
        )

        assert response["source"]["entity_id"] == "p25"
        assert response["neighbors"], "expected cross-time matches"
        for neighbor in response["neighbors"]:
            assert neighbor["delta_t"] < 0
            assert neighbor["t"] < 3
        settled = [
            n for n in response["neighbors"] if 10 <= int(n["entity_id"][1:]) < 20
        ]
        assert len(settled) >= 3

    def test_direction_filters_are_applied(self, tmp_path):
        built = _build(tmp_path)
        forward = json.loads(
            asyncio.run(
                get_cross_time_neighbors(
                    built["longitudinal_id"],
                    entity_id="p25",
                    t=0,
                    direction="forward",
                    response_format="json",
                )
            )
        )
        assert all(n["delta_t"] > 0 for n in forward["neighbors"])

    def test_original_time_labels_resolve_to_trajectory_observations(self, tmp_path):
        frame = _long_frame()
        frame["hour"] = frame["hour"].map({0: 1, 1: 2, 2: 3, 3: 4})
        built = _build(tmp_path, path=_write_long(tmp_path, frame))

        response = json.loads(
            asyncio.run(
                get_cross_time_neighbors(
                    built["longitudinal_id"],
                    entity_id="p25",
                    t=4,
                    direction="backward",
                    max_neighbors=5,
                    response_format="json",
                )
            )
        )

        assert response["status"] == "ok"
        assert response["source"]["t"] == 3
        assert response["source"]["time_label"] == 4
        assert response["neighbors"]
        assert all("time_label" in neighbor for neighbor in response["neighbors"])

        neighbor = response["neighbors"][0]
        replayed = json.loads(
            asyncio.run(
                get_cross_time_neighbors(
                    built["longitudinal_id"],
                    entity_id=neighbor["entity_id"],
                    t=neighbor["time_label"],
                    response_format="json",
                )
            )
        )

        assert replayed["source"]["obs_id"] == neighbor["obs_id"]

    def test_unresolvable_observation_is_structured(self, tmp_path):
        built = _build(tmp_path)
        missing = json.loads(
            asyncio.run(
                get_cross_time_neighbors(
                    built["longitudinal_id"], entity_id="nobody", t=0
                )
            )
        )
        assert missing["error_code"] == "OBSERVATION_NOT_FOUND"

        unspecified = json.loads(
            asyncio.run(get_cross_time_neighbors(built["longitudinal_id"]))
        )
        assert unspecified["error_code"] == "OBSERVATION_NOT_SPECIFIED"

    def test_unmatched_time_label_returns_bounded_preview(self, tmp_path):
        built = _build(tmp_path)

        response = json.loads(
            asyncio.run(
                get_cross_time_neighbors(
                    built["longitudinal_id"], entity_id="p0", t=999
                )
            )
        )

        available = response["details"]["available_times"]
        assert set(available) == {"total", "preview", "omitted"}
        assert available["total"] == 4


class TestHandles:
    def test_unknown_handle_returns_structured_error(self):
        _sessions.clear()
        _get_session(None)
        for tool in (
            diagnose_longitudinal_graph,
            get_trajectory_archetypes,
            get_cross_time_neighbors,
        ):
            response = json.loads(asyncio.run(tool("lng_missing")))
            assert response["error_code"] == "LONGITUDINAL_ID_UNKNOWN"
            assert response["agent_action"]

    def test_omitted_handle_falls_back_to_latest_panel(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(diagnose_longitudinal_graph(response_format="json"))
        )
        assert response["longitudinal_id"] == built["longitudinal_id"]

    def test_building_a_panel_preserves_static_session_state(self, tmp_path):
        """A panel is n*T observation rows and must not rebind session.data."""
        _sessions.clear()
        session = _get_session(None)
        sentinel = pd.DataFrame({"a": [1, 2, 3]})
        session.data = sentinel
        session.clusters = pd.Series([0, 1, 0])

        path = _write_long(tmp_path)
        dataset = json.loads(asyncio.run(ingest_dataset(path)))
        asyncio.run(
            build_longitudinal_graph(
                dataset_id=dataset["dataset_id"],
                entity_column="patient_id",
                time_column="hour",
                config_yaml=_config_yaml(),
            )
        )

        assert session.data is sentinel
        assert session.clusters is not None


class TestClassifyTrajectories:
    def test_classify_complexity(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                classify_trajectories(
                    built["longitudinal_id"],
                    method="complexity",
                    threshold=0.25,
                    response_format="json",
                )
            )
        )
        assert response["n_entities"] == 30
        assert response["method"] == "complexity"
        assert "Stable (0 entropy)" in response["classes_summary"]
        assert "Volatile / Refractory" in response["classes_summary"]

    def test_classify_transition(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                classify_trajectories(
                    built["longitudinal_id"],
                    method="transition",
                    threshold=0.25,
                    response_format="json",
                )
            )
        )
        assert response["n_entities"] == 30
        assert response["method"] == "transition"
        assert "Highly Stable (>=80% retention)" in response["classes_summary"]

    def test_classify_sequence(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                classify_trajectories(
                    built["longitudinal_id"],
                    method="sequence",
                    threshold=0.25,
                    response_format="json",
                )
            )
        )
        assert response["n_entities"] == 30
        assert response["method"] == "sequence"

    def test_classify_levenshtein(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                classify_trajectories(
                    built["longitudinal_id"],
                    method="levenshtein",
                    threshold=0.25,
                    response_format="json",
                )
            )
        )
        assert response["n_entities"] == 30
        assert response["method"] == "levenshtein"
        assert len(response["classes_summary"]) > 0

    def test_classify_dtw(self, tmp_path):
        built = _build(tmp_path)
        response = json.loads(
            asyncio.run(
                classify_trajectories(
                    built["longitudinal_id"],
                    method="dtw",
                    threshold=0.25,
                    response_format="json",
                )
            )
        )
        assert response["n_entities"] == 30
        assert response["method"] == "dtw"
        assert len(response["classes_summary"]) > 0


def test_longitudinal_tools_are_registered():
    names = {tool.__name__ for tool in ALL_TOOLS_LIST}
    assert {
        "build_longitudinal_graph",
        "diagnose_longitudinal_graph",
        "get_trajectory_archetypes",
        "get_cross_time_neighbors",
        "classify_trajectories",
    } <= names


def test_config_helper_is_loadable():
    """Guards the test fixture itself against config schema drift."""
    assert load_config(yaml.safe_load(_config_yaml())).ball_mapper.epsilons
