"""
Tests for the progress_callback mechanism in ThemaRS.fit() and fit_multi().
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from pulsar.config import load_config
from pulsar.pipeline import ThemaRS


@pytest.fixture
def basic_config():
    return {
        "run": {"name": "test"},
        "preprocessing": {"drop_columns": [], "impute": {}},
        "sweep": {
            "pca": {"dimensions": {"values": [2]}, "seed": {"values": [42]}},
            "ball_mapper": {"epsilon": {"values": [0.5, 1.0]}},
        },
        "cosmic_graph": {"threshold": "0.0"},
        "output": {"n_reps": 1},
    }


@pytest.fixture
def test_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((50, 3)), columns=["x", "y", "z"])


def test_callback_fires_in_order(basic_config, test_data):
    """Callback fractions are monotonically increasing from 0 to 1."""
    cfg = load_config(basic_config)
    updates: list[tuple[str, float]] = []

    def cb(stage: str, fraction: float) -> None:
        updates.append((stage, fraction))

    ThemaRS(cfg).fit(data=test_data, progress_callback=cb)

    assert len(updates) > 0, "No callbacks fired"
    fractions = [f for _, f in updates]
    assert all(0.0 < f <= 1.0 for f in fractions), f"Fractions out of range: {fractions}"
    assert fractions == sorted(fractions), f"Fractions not monotonic: {fractions}"
    assert fractions[-1] == pytest.approx(1.0, abs=0.01), "Final fraction not ~1.0"


def test_callback_not_called_when_none(basic_config, test_data):
    """fit() with no callback completes without error."""
    cfg = load_config(basic_config)
    model = ThemaRS(cfg).fit(data=test_data, progress_callback=None)
    assert model.cosmic_graph is not None


def test_callback_exception_propagates(basic_config, test_data):
    """An exception raised in the callback propagates and aborts fit()."""
    cfg = load_config(basic_config)

    def bad_cb(stage: str, fraction: float) -> None:
        raise ValueError("intentional error")

    with pytest.raises(RuntimeError, match="progress_callback raised during"):
        ThemaRS(cfg).fit(data=test_data, progress_callback=bad_cb)


def test_cached_pca_stage_label(basic_config, test_data):
    """With cached embeddings, the pca stage name contains '(cached)'."""
    cfg = load_config(basic_config)
    model1 = ThemaRS(cfg).fit(data=test_data)

    labels: list[str] = []
    ThemaRS(cfg).fit(
        data=test_data,
        progress_callback=lambda stage, _: labels.append(stage),
        _precomputed_embeddings=model1._embeddings,
    )

    pca_labels = [s for s in labels if "pca" in s.lower()]
    assert pca_labels, "No PCA stage label found"
    assert any("cached" in s for s in pca_labels), (
        f"Expected '(cached)' in PCA label, got: {pca_labels}"
    )


def test_fit_with_progress_returns_model(basic_config, test_data, tmp_path):
    """fit_with_progress() returns a fitted ThemaRS model."""
    pytest.importorskip("rich")

    from pulsar.progress import fit_with_progress

    cfg = load_config(basic_config)
    model = fit_with_progress(ThemaRS(cfg), data=test_data)

    assert model.cosmic_graph is not None
    assert model.cosmic_graph.number_of_nodes() > 0


def test_fit_with_progress_import_error(basic_config, test_data, monkeypatch):
    """fit_with_progress() raises ImportError if rich is missing."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "rich.progress":
            raise ImportError("No module named 'rich'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from pulsar.progress import fit_with_progress
    cfg = load_config(basic_config)

    with pytest.raises(ImportError, match="rich"):
        fit_with_progress(ThemaRS(cfg), data=test_data)


def test_fit_multi_callback_fires(basic_config):
    """fit_multi() progress_callback fires with dataset prefix in stage names."""
    cfg = load_config(basic_config)
    rng = np.random.default_rng(1)
    ds1 = pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])
    ds2 = pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])

    updates: list[tuple[str, float]] = []
    ThemaRS(cfg).fit_multi([ds1, ds2], progress_callback=lambda s, f: updates.append((s, f)))

    stages = [s for s, _ in updates]
    assert any("Dataset 1/2" in s for s in stages), f"Missing dataset 1 label: {stages}"
    assert any("Dataset 2/2" in s for s in stages), f"Missing dataset 2 label: {stages}"

    fractions = [f for _, f in updates]
    assert fractions == sorted(fractions), f"fit_multi fractions not monotonic: {fractions}"
    assert fractions[-1] == pytest.approx(1.0, abs=0.01)
