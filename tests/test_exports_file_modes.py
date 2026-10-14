"""Atomic export writes must keep plain-write permission bits (not 0600/0700)."""

import os
import stat

import numpy as np
import pandas as pd
import pytest

from pulsar.config import load_config
from pulsar.exports import _atomic_write_file
from pulsar.mcp.tools.reporting import _atomic_write_text_or_df
from pulsar.pipeline import ThemaRS


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _control_file_mode(directory) -> int:
    control = directory / "control.txt"
    with open(control, "w") as fh:
        fh.write("x")
    return _mode(control)


def _control_dir_mode(directory) -> int:
    control = directory / "control_dir"
    os.mkdir(control)
    return _mode(control)


def _write_text(path):
    path.write_text("hello")


@pytest.mark.parametrize("writer", [_atomic_write_file, _atomic_write_text_or_df])
def test_atomic_write_matches_plain_write_mode(tmp_path, writer):
    dest = tmp_path / "out.csv"
    writer(dest, _write_text)
    assert dest.read_text() == "hello"
    assert _mode(dest) == _control_file_mode(tmp_path)
    # Must not be the NamedTemporaryFile 0600 default unless umask demands it.
    assert not list(tmp_path.glob(".out.csv.*.tmp"))


@pytest.mark.parametrize("writer", [_atomic_write_file, _atomic_write_text_or_df])
def test_atomic_write_overwrite_matches_plain_write_mode(tmp_path, writer):
    dest = tmp_path / "out.csv"
    dest.write_text("old")
    os.chmod(dest, 0o600)
    writer(dest, _write_text)
    assert dest.read_text() == "hello"
    assert _mode(dest) == _control_file_mode(tmp_path)


def test_export_bundle_modes_match_plain_creation(tmp_path):
    cfg = load_config(
        {
            "run": {"name": "modes"},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "pca": {"dimensions": {"values": [2]}, "seed": {"values": [42]}},
                "ball_mapper": {"epsilon": {"values": [0.5, 1.0]}},
            },
            "cosmic_graph": {"construction_threshold": "0.1"},
            "output": {"n_reps": 1},
        }
    )
    rng = np.random.default_rng(42)
    model = ThemaRS(cfg).fit(
        data=pd.DataFrame(rng.standard_normal((30, 3)), columns=["x", "y", "z"])
    )
    labels = np.array([0] * 15 + [1] * 15)
    expected_file = _control_file_mode(tmp_path)
    expected_dir = _control_dir_mode(tmp_path)
    for _ in range(2):  # fresh promote, then replace-existing promote
        model.export_dataset_bundle(
            output_dir=str(tmp_path), slug="bundle", cluster_labels=labels
        )
        bundle = tmp_path / "bundle"
        assert _mode(bundle) == expected_dir
        files = [p for p in bundle.rglob("*") if p.is_file()]
        assert files
        for f in files:
            assert _mode(f) == expected_file, f
        for d in (p for p in bundle.rglob("*") if p.is_dir()):
            assert _mode(d) == expected_dir, d
