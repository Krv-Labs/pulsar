"""Longitudinal preprocessing follows configured numeric rules and preserves keys."""

import asyncio
import json

import pandas as pd
import pytest
import yaml

from pulsar.config import load_config
from pulsar.mcp.longitudinal import PanelError, prepare_panel_frame, pivot_panel
from pulsar.mcp.session import _get_session, _sessions
from pulsar.mcp.tools.ingestion import ingest_dataset
from pulsar.mcp.tools.longitudinal import build_longitudinal_graph


def _frame() -> pd.DataFrame:
    rows = []
    for time in range(3):
        for entity in range(4):
            rows.append(
                {
                    "entity": f"e{entity}",
                    "time": time,
                    "x": None if entity == 0 and time == 1 else entity + time,
                    "y": entity * 2 + time * 0.5,
                    "discard_me": entity * 10 + time,
                    "label": "case" if entity % 2 else "control",
                }
            )
    return pd.DataFrame(rows)


def _config(*, impute=None, encode=None, drop_columns=None) -> str:
    return yaml.safe_dump(
        {
            "run": {"name": "longitudinal_preprocessing_test"},
            "preprocessing": {
                "drop_columns": drop_columns or [],
                "impute": impute or {},
                "encode": encode or {},
            },
            "sweep": {
                "projection": {
                    "method": "jl",
                    "dimensions": {"values": [2]},
                    "seed": {"values": [42]},
                },
                "ball_mapper": {"epsilon": {"values": [0.05, 0.1]}},
            },
            "cosmic_graph": {
                "construction_threshold": 0.0,
                "construction": "exact",
            },
            "output": {"n_reps": 1},
        }
    )


def _build(
    tmp_path, *, config_yaml: str, feature_columns=None, on_missing="drop_entity"
) -> dict:
    _sessions.clear()
    _get_session(None)
    path = tmp_path / "panel.csv"
    _frame().to_csv(path, index=False)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))
    return json.loads(
        asyncio.run(
            build_longitudinal_graph(
                dataset_id=dataset["dataset_id"],
                entity_column="entity",
                time_column="time",
                feature_columns=feature_columns,
                on_missing=on_missing,
                config_yaml=config_yaml,
                representation="trajectory",
                response_format="json",
            )
        )
    )


@pytest.mark.parametrize("on_missing", ["drop_entity", "allow_ragged"])
def test_configured_imputation_and_drops_apply_without_changing_panel_keys(
    tmp_path, on_missing
):
    result = _build(
        tmp_path,
        config_yaml=_config(
            impute={"x": {"method": "fill_mean"}},
            drop_columns=["entity", "time", "discard_me"],
        ),
        on_missing=on_missing,
    )

    assert result["status"] == "ok"
    assert result["panel"]["feature_columns_preview"] == ["x", "y"]
    assert result["panel"]["n_entities_kept"] == 4
    artifact = _get_session(None).longitudinal[result["longitudinal_id"]]
    assert set(artifact.trajectory.obs["entity_id"]) == {
        "e0",
        "e1",
        "e2",
        "e3",
    }
    assert set(artifact.trajectory.obs["timestamp"]) == {0, 1, 2}


def test_unhandled_feature_nan_returns_actionable_structured_error(tmp_path):
    result = _build(tmp_path, config_yaml=_config(), on_missing="allow_ragged")

    assert result["error_code"] == "PANEL_NAN_REMAINS"
    assert result["details"]["missing_by_column"] == {"x": 1}
    assert "preprocessing.impute" in result["agent_action"]


def test_explicit_feature_selection_cannot_reintroduce_dropped_column(tmp_path):
    result = _build(
        tmp_path,
        config_yaml=_config(drop_columns=["discard_me"]),
        feature_columns=["x", "y", "discard_me"],
    )

    assert result["error_code"] == "PANEL_FEATURE_DROPPED"
    assert result["details"]["columns"] == ["discard_me"]


def test_unselected_preprocessing_rules_do_not_block_explicit_features(tmp_path):
    config_yaml = _config(
        impute={"label": {"method": "fill_mean"}},
        encode={"discard_me": {"method": "one_hot"}},
    )

    result = _build(
        tmp_path,
        config_yaml=config_yaml,
        feature_columns=["x", "y"],
    )

    assert result["status"] == "ok"
    assert result["panel"]["feature_columns_preview"] == ["x", "y"]


def test_redaction_text_is_not_coerced_and_imputed(tmp_path):
    frame = _frame()
    frame["x"] = frame["x"].astype(object)
    frame.loc[frame["entity"] == "e0", "x"] = "REDACTED"
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    _sessions.clear()
    _get_session(None)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    result = json.loads(
        asyncio.run(
            build_longitudinal_graph(
                dataset_id=dataset["dataset_id"],
                entity_column="entity",
                time_column="time",
                feature_columns=["x", "y"],
                config_yaml=_config(impute={"x": {"method": "fill_mean"}}),
                response_format="json",
            )
        )
    )

    assert result["error_code"] == "PANEL_IMPUTE_NON_NUMERIC_VALUES"


def test_all_missing_feature_has_clear_structured_error(tmp_path):
    frame = _frame()
    frame["x"] = None
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    _sessions.clear()
    _get_session(None)
    dataset = json.loads(asyncio.run(ingest_dataset(str(path))))

    result = json.loads(
        asyncio.run(
            build_longitudinal_graph(
                dataset_id=dataset["dataset_id"],
                entity_column="entity",
                time_column="time",
                config_yaml=_config(impute={"x": {"method": "fill_mean"}}),
                response_format="json",
            )
        )
    )

    assert result["error_code"] == "PANEL_FEATURE_ALL_MISSING"


def test_missing_feature_values_still_follow_pivot_policy():
    frame = _frame()

    dropped = pivot_panel(frame, "entity", "time", on_missing="drop_entity")
    assert dropped.n_entities == 3
    assert "e0" in dropped.report["dropped_entities"]["preview"]

    with pytest.raises(PanelError) as ragged:
        pivot_panel(frame, "entity", "time", on_missing="allow_ragged")
    assert ragged.value.error_code == "PANEL_NAN_REMAINS"

    panel = pivot_panel(frame, "entity", "time", on_missing="forward_fill")
    assert panel.n_entities == 4
    assert panel.snapshots[1][panel.entity_ids.index("e0"), 0] == 0


def test_imputation_cannot_modify_panel_keys():
    frame = _frame()
    original = frame.copy(deep=True)
    config = load_config(
        yaml.safe_load(_config(impute={"entity": {"method": "fill_mean"}}))
    )

    with pytest.raises(PanelError) as error:
        prepare_panel_frame(frame, config, "entity", "time", None)

    assert error.value.error_code == "PANEL_PREPROCESSING_KEY_IMPUTE"
    pd.testing.assert_frame_equal(frame, original)
