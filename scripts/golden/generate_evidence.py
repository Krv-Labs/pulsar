"""Freeze preprocessing-evidence reference outputs from the Python v0.5 code.

Writes only new artifacts (existing golden fixtures are never touched):

* ``tests/fixtures/golden/evidence/input.csv``: a synthetic table built to
  exercise every evidence branch (duplicate / sum_of / correlated columns, a
  shared missing group, a mask dependency, numeric text, dates, booleans, an
  all-missing and a constant column).
* ``tests/fixtures/golden/evidence/evidence.json`` and
  ``tests/fixtures/golden/penguins/evidence.json``: the Python
  ``measure_preprocessing_evidence`` result (``cost.elapsed_ms`` omitted).
* ``tests/fixtures/golden/evidence/overlap.json``: ``neighborhood_overlap``
  known answers.

Run from the repo root: ``uv run python scripts/golden/generate_evidence.py``.
The Rust parity test (``crates/pulsar-parity/tests/evidence.rs``) reads these.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulsar.analysis.preprocessing_evidence import (
    measure_preprocessing_evidence,
    neighborhood_overlap,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "golden"
N = 60


def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(20260101)
    a = rng.normal(10.0, 3.0, N)
    b = rng.normal(5.0, 2.0, N)
    miss = np.zeros(N, dtype=bool)
    miss[[3, 9, 17, 25, 40]] = True
    flag = (rng.random(N) > 0.4).astype(int)
    flag[0], flag[1] = 0, 1

    f = rng.uniform(1.0, 9.0, N)
    g = rng.uniform(2.0, 8.0, N)
    f[miss], g[miss] = np.nan, np.nan
    m1 = rng.uniform(1.0, 4.0, N)
    m1[flag == 0] = np.nan

    money = [f"{int(v):,}" for v in rng.integers(1000, 90000, N)]
    money[5], money[12] = "tbd", "?"
    dates = pd.date_range("2020-01-01", periods=N, freq="7D").strftime("%Y-%m-%d")
    cat3 = rng.choice(["x", "y", "z"], N).astype(object)
    cat3[7] = None
    bools = rng.random(N) > 0.5
    bools_obj = pd.array(bools, dtype="object")
    bools_obj[11] = None

    df = pd.DataFrame(
        {
            "id": np.arange(N),
            "a": a,
            "b": b,
            "c_sum": a + b,
            "d_dup": a.copy(),
            "e_corr": 0.8 * a + rng.normal(0, 0.5, N),
            "f": f,
            "g": g,
            "h": rng.normal(0, 1, N),
            "flag": flag,
            "m1": m1,
            "cnt": rng.integers(0, 6, N),
            "const": np.full(N, 7.0),
            "cat2": rng.choice(["yes", "no"], N, p=[0.8, 0.2]),
            "cat3": cat3,
            "money": money,
            "when": dates,
            "is_ok": bools_obj,
            "allnan": np.full(N, np.nan),
            "neg": rng.integers(-3, 4, N).astype(float),
        }
    )
    return df


def to_json(ev: Any) -> dict[str, Any]:
    out = dataclasses.asdict(ev)
    out["cost"].pop("elapsed_ms", None)
    return out


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print("wrote", path.relative_to(ROOT))


def main() -> None:
    out = GOLDEN / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    df = synthetic()
    csv = out / "input.csv"
    df.to_csv(csv, index=False)
    # Evidence is measured on what read_csv gives back, as the MCP tools do.
    dump(out / "evidence.json", to_json(measure_preprocessing_evidence(pd.read_csv(csv))))

    penguins = pd.read_csv(GOLDEN / "penguins" / "input.csv")
    dump(
        GOLDEN / "penguins" / "evidence.json",
        to_json(measure_preprocessing_evidence(penguins)),
    )

    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 4))
    y = x + rng.normal(scale=0.3, size=x.shape)
    dump(
        out / "overlap.json",
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "k": 5,
            "overlap": neighborhood_overlap(x, y, k=5),
            "self_overlap": neighborhood_overlap(x, x, k=5),
        },
    )


if __name__ == "__main__":
    main()
