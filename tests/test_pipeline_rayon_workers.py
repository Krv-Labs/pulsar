"""rayon_workers is accepted for compatibility but has no effect; it must say so."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pulsar.config import load_config
from pulsar.pipeline import ThemaRS


def _model():
    return ThemaRS(
        load_config(
            {
                "run": {"name": "rayon"},
                "preprocessing": {"drop_columns": [], "impute": {}},
                "sweep": {
                    "pca": {"dimensions": {"values": [2]}, "seed": {"values": [42]}},
                    "ball_mapper": {"epsilon": {"values": [0.5, 1.0]}},
                },
                "cosmic_graph": {"construction_threshold": "0.1"},
                "output": {"n_reps": 1},
            }
        )
    )


def _data():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])


def test_fit_multi_rayon_workers_warns_and_still_fits():
    model = _model()
    with pytest.warns(RuntimeWarning, match="rayon_workers has no effect") as rec:
        model.fit_multi([_data()], rayon_workers=2)
    assert rec[0].filename == __file__  # stacklevel points at the caller
    assert model.cosmic_graph is not None


def test_fit_multi_default_rayon_workers_does_not_warn():
    model = _model()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="rayon_workers")
        model.fit_multi([_data()])
    assert model.cosmic_graph is not None


def test_fit_multi_rayon_workers_still_validated():
    with pytest.raises(ValueError, match="rayon_workers must be positive"):
        _model().fit_multi([_data()], rayon_workers=0)
