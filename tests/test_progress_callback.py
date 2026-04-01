"""
Tests for the progress_callback mechanism in ThemaRS.fit() and fit_multi().
"""

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
def auto_config():
    return {
        "run": {"name": "test_auto"},
        "preprocessing": {"drop_columns": [], "impute": {}},
        "sweep": {
            "pca": {"dimensions": {"values": [2]}, "seed": {"values": [42]}},
            "ball_mapper": {"epsilon": {"values": [0.5, 1.0]}},
        },
        "cosmic_graph": {"threshold": "auto"},
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
    assert fractions[-1] == 1.0, "Final fraction must be exactly 1.0"


def test_callback_not_called_when_none(basic_config, test_data):
    """fit() with no callback completes without error."""
    cfg = load_config(basic_config)
    model = ThemaRS(cfg).fit(data=test_data, progress_callback=None)
    assert model.cosmic_graph is not None


def test_callback_exception_propagates(basic_config, test_data):
    """An exception raised in the callback propagates with its original type."""
    cfg = load_config(basic_config)

    def bad_cb(stage: str, fraction: float) -> None:
        raise ValueError("intentional error")

    with pytest.raises(ValueError, match="intentional error"):
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


def test_cached_pca_fractions_reach_one(basic_config, test_data):
    """With cached PCA (zero weight), fractions must still renormalize to 1.0."""
    cfg = load_config(basic_config)
    model1 = ThemaRS(cfg).fit(data=test_data)

    fractions: list[float] = []
    ThemaRS(cfg).fit(
        data=test_data,
        progress_callback=lambda _, f: fractions.append(f),
        _precomputed_embeddings=model1._embeddings,
    )

    assert fractions[-1] == 1.0, f"Final fraction with cached PCA is {fractions[-1]}, not 1.0"
    assert fractions == sorted(fractions), f"Fractions not monotonic: {fractions}"


def test_no_duplicate_fractions_at_one(basic_config, test_data):
    """No two consecutive callbacks should both report fraction 1.0."""
    cfg = load_config(basic_config)
    fractions: list[float] = []

    ThemaRS(cfg).fit(
        data=test_data,
        progress_callback=lambda _, f: fractions.append(f),
    )

    ones = [i for i, f in enumerate(fractions) if f == 1.0]
    assert len(ones) == 1, f"Expected exactly one 1.0 fraction, got {len(ones)} at indices {ones}"


def test_auto_threshold_reaches_one(auto_config, test_data):
    """With threshold='auto', progress still reaches exactly 1.0."""
    cfg = load_config(auto_config)
    updates: list[tuple[str, float]] = []

    ThemaRS(cfg).fit(
        data=test_data,
        progress_callback=lambda s, f: updates.append((s, f)),
    )

    fractions = [f for _, f in updates]
    assert fractions[-1] == 1.0, f"Auto-threshold final fraction is {fractions[-1]}"
    assert fractions == sorted(fractions), f"Fractions not monotonic: {fractions}"

    ones = [i for i, f in enumerate(fractions) if f == 1.0]
    assert len(ones) == 1, f"Duplicate 1.0 at indices {ones}"


def test_stage_count(basic_config, test_data):
    """fit() fires exactly 7 callbacks (one per stage)."""
    cfg = load_config(basic_config)
    stages: list[str] = []

    ThemaRS(cfg).fit(
        data=test_data,
        progress_callback=lambda s, _: stages.append(s),
    )

    assert len(stages) == 7, f"Expected 7 stages, got {len(stages)}: {stages}"


def test_fit_with_progress_returns_model(basic_config, test_data):
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


# ---------------------------------------------------------------------------
# fit_multi tests
# ---------------------------------------------------------------------------

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
    assert fractions[-1] == 1.0, f"fit_multi final fraction is {fractions[-1]}"


def test_fit_multi_stage_count(basic_config):
    """fit_multi with N=2 datasets fires exactly 5*2 + 2 = 12 callbacks."""
    cfg = load_config(basic_config)
    rng = np.random.default_rng(2)
    ds1 = pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])
    ds2 = pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])

    stages: list[str] = []
    ThemaRS(cfg).fit_multi([ds1, ds2], progress_callback=lambda s, _: stages.append(s))

    assert len(stages) == 12, f"Expected 12 callbacks for 2 datasets, got {len(stages)}: {stages}"


def test_fit_multi_load_stage_present(basic_config):
    """fit_multi includes 'load' in per-dataset stage names."""
    cfg = load_config(basic_config)
    rng = np.random.default_rng(3)
    ds1 = pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])

    stages: list[str] = []
    ThemaRS(cfg).fit_multi([ds1], progress_callback=lambda s, _: stages.append(s))

    load_stages = [s for s in stages if "load" in s]
    assert load_stages, f"No 'load' stage found in fit_multi: {stages}"


def test_fit_multi_no_duplicate_one(basic_config):
    """fit_multi must emit exactly one callback at fraction 1.0."""
    cfg = load_config(basic_config)
    rng = np.random.default_rng(4)
    ds1 = pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])
    ds2 = pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])

    fractions: list[float] = []
    ThemaRS(cfg).fit_multi([ds1, ds2], progress_callback=lambda _, f: fractions.append(f))

    ones = [i for i, f in enumerate(fractions) if f == 1.0]
    assert len(ones) == 1, f"Expected exactly one 1.0, got {len(ones)} at indices {ones}"
