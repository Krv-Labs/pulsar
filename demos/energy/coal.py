"""
Pulsar demo — US Coal Plants dataset
=====================================
Downloads the dataset (once), runs a large grid search through the full
ThemaRS pipeline, and prints per-stage wall-clock timings.

Usage (from repo root):
    uv run python demos/energy/coal.py
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pulsar._pulsar import (
    CosmicGraph,
    StandardScaler,
    accumulate_pseudo_laplacians,
    ball_mapper_grid,
    find_stable_thresholds,
    impute_column,
    pca_grid,
)

from pulsar.config import load_config
from pulsar.hooks import cosmic_to_networkx
from pulsar.pipeline import ThemaRS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demos"
CSV_PATH = DEMO_DIR / "us_coal_plants_dataset.csv"
PARAMS_PATH = DEMO_DIR / "coal_params.yaml"
DATA_URL = (
    "https://raw.githubusercontent.com/Krv-Labs/retire/main/"
    "retire/resources/us_coal_plants_dataset.csv"
)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def ensure_data() -> None:
    if CSV_PATH.exists():
        print(f"[data] found {CSV_PATH.name}")
        return
    print(f"[data] downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, CSV_PATH)
    print(f"[data] saved to {CSV_PATH}")


# ---------------------------------------------------------------------------
# Auto-detect preprocessing from the raw CSV
# ---------------------------------------------------------------------------


def build_preprocessing(df: pd.DataFrame) -> tuple[list[str], dict]:
    """
    Return (drop_columns, impute_dict).

    Strategy:
    - Categorical (non-numeric) columns: keep, label-encode in fit(), impute
      any NaN values with sample_categorical.
    - Numeric columns with NaN: impute with sample_normal (Gaussian sampling).
    - drop_columns is always empty — we retain all columns.
    """
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    numeric_df = df.drop(columns=cat_cols)

    impute: dict = {}
    # categorical NaN columns
    for col in cat_cols:
        if df[col].isna().any():
            impute[col] = {"method": "sample_categorical", "seed": 42}
    # numeric NaN columns
    for col in numeric_df.columns:
        if numeric_df[col].isna().any():
            impute[col] = {"method": "sample_normal", "seed": 42}

    return [], impute


# ---------------------------------------------------------------------------
# Timed pipeline
# ---------------------------------------------------------------------------


class TimedThemaRS(ThemaRS):
    """ThemaRS with per-stage wall-clock timing."""

    def __init__(self, config):
        super().__init__(config)
        self.timings: dict[str, float] = {}

    def fit(self, data: pd.DataFrame | None = None) -> "TimedThemaRS":
        cfg = self.config

        # ── 1. Load ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        if data is None:
            if not cfg.data:
                raise ValueError("No data path in config and no DataFrame provided")
            if cfg.data.endswith(".parquet"):
                data = pd.read_parquet(cfg.data)
            else:
                data = pd.read_csv(cfg.data)
        self._data = data.copy()
        self.timings["load"] = time.perf_counter() - t0

        # ── 2. Preprocess (encode + drop + impute + dropna) ─────────────────
        t0 = time.perf_counter()
        drop = [c for c in cfg.drop_columns if c in data.columns]
        df = data.drop(columns=drop)

        # Label-encode all non-numeric columns: string → integer code (float64).
        # pandas Categorical codes use -1 for NaN, which we convert to np.nan so
        # the imputer (sample_categorical) can handle missing values normally.
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        for col in cat_cols:
            codes = pd.Categorical(df[col]).codes.astype(np.float64)
            df[col] = np.where(codes == -1, np.nan, codes)

        flag_cols = {
            f"{col}_was_missing": df[col].isna().astype(np.float64)
            for col in cfg.impute
            if col in df.columns
        }
        if flag_cols:
            df = pd.concat([df, pd.DataFrame(flag_cols, index=df.index)], axis=1)

        for col, spec in cfg.impute.items():
            if col not in df.columns:
                continue
            arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            imputed = impute_column(arr, spec.method, spec.seed)
            df[col] = imputed

        df = df.dropna(axis=0)
        X_raw = df.to_numpy(dtype=np.float64)
        n = X_raw.shape[0]
        self.timings["preprocess"] = time.perf_counter() - t0

        # ── 3. Scale ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        scaler = StandardScaler()
        X_scaled = np.array(scaler.fit_transform(X_raw))
        self.timings["scale"] = time.perf_counter() - t0

        # ── 4. PCA grid (randomized SVD, parallel across seeds) ─────────────
        t0 = time.perf_counter()
        embeddings = [
            np.ascontiguousarray(emb)
            for emb in pca_grid(X_scaled, cfg.pca.dimensions, cfg.pca.seeds)
        ]
        self.timings["pca_grid"] = time.perf_counter() - t0

        # ── 5. BallMapper grid ───────────────────────────────────────────────
        t0 = time.perf_counter()
        self._ball_maps = ball_mapper_grid(embeddings, cfg.ball_mapper.epsilons)
        self.timings["ball_mapper_grid"] = time.perf_counter() - t0

        # ── 6. Accumulate pseudo-Laplacians (Rust parallel) ──────────────────
        t0 = time.perf_counter()
        galactic_L = np.array(accumulate_pseudo_laplacians(self._ball_maps, n))
        self.timings["laplacian"] = time.perf_counter() - t0

        # ── 7. CosmicGraph (initial, threshold=0 to get weighted adj) ────────
        t0 = time.perf_counter()
        cg_temp = CosmicGraph.from_pseudo_laplacian(galactic_L, 0.0)
        self._weighted_adjacency = np.array(cg_temp.weighted_adj)
        self.timings["cosmic_graph_init"] = time.perf_counter() - t0

        # ── 8. Threshold stability analysis (if auto) ────────────────────────
        threshold = cfg.cosmic_graph.threshold
        if threshold == "auto":
            t0 = time.perf_counter()
            self._stability_result = find_stable_thresholds(self._weighted_adjacency)
            threshold = self._stability_result.optimal_threshold
            self.timings["stability_analysis"] = time.perf_counter() - t0
        else:
            self._stability_result = None
        self._resolved_threshold = float(threshold)

        # ── 9. CosmicGraph (final, with resolved threshold) ───────────────────
        t0 = time.perf_counter()
        self._cosmic_rust = CosmicGraph.from_pseudo_laplacian(
            galactic_L, self._resolved_threshold
        )
        self._cosmic_graph = cosmic_to_networkx(self._cosmic_rust)
        self.timings["cosmic_graph_final"] = time.perf_counter() - t0

        return self

    def print_report(self) -> None:
        total = sum(self.timings.values())
        col_w = 22

        header = f"{'Stage':<{col_w}}  {'Time (s)':>10}  {'% of total':>10}"
        rule = "─" * len(header)
        print()
        print(header)
        print(rule)
        for stage, secs in self.timings.items():
            pct = 100.0 * secs / total if total else 0.0
            print(f"{stage:<{col_w}}  {secs:>10.4f}  {pct:>9.1f}%")
        print(rule)
        print(f"{'TOTAL':<{col_w}}  {total:>10.4f}  {'100.0%':>10}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_data()

    # Load raw CSV to detect preprocessing requirements
    print("[config] detecting column types and NaN patterns ...")
    df_raw = pd.read_csv(CSV_PATH)
    print(f"[config] dataset shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    drop_columns, impute_specs = build_preprocessing(df_raw)
    cat_cols = df_raw.select_dtypes(exclude="number").columns.tolist()
    cat_nan = sum(1 for c in cat_cols if df_raw[c].isna().any())
    num_nan = sum(1 for c, s in impute_specs.items() if s["method"] == "sample_normal")
    print(f"[config] label-encoding {len(cat_cols)} categorical columns")
    print(f"[config] imputing {cat_nan} categorical NaN cols (sample_categorical)")
    print(f"[config] imputing {num_nan} numeric NaN cols (sample_normal/Gaussian)")

    # Load YAML and inject dynamic preprocessing
    with open(PARAMS_PATH) as f:
        raw_cfg = yaml.safe_load(f)

    raw_cfg.setdefault("preprocessing", {})
    raw_cfg["preprocessing"]["drop_columns"] = drop_columns
    raw_cfg["preprocessing"]["impute"] = impute_specs
    # Override data path to absolute so script works from any cwd
    raw_cfg["run"]["data"] = str(CSV_PATH)

    cfg = load_config(raw_cfg)
    n_pca = len(cfg.pca.dimensions) * len(cfg.pca.seeds)
    n_maps = n_pca * len(cfg.ball_mapper.epsilons)
    print(
        f"[grid]   {len(cfg.pca.dimensions)} dims × {len(cfg.pca.seeds)} seeds"
        f" × {len(cfg.ball_mapper.epsilons)} epsilons"
        f" = {n_maps} ball maps"
    )
    print()
    print("[run]    starting pipeline ...")

    model = TimedThemaRS(cfg)
    t_wall_start = time.perf_counter()
    model.fit()
    t_wall_total = time.perf_counter() - t_wall_start

    print(f"[run]    done  (wall clock: {t_wall_total:.3f}s)")

    # Print stability analysis results if auto threshold was used
    if model._stability_result is not None:
        sr = model._stability_result
        best_plateau = sr.plateaus[0] if sr.plateaus else None
        print(f"[stability] optimal threshold: {sr.optimal_threshold:.4f}")
        if best_plateau:
            print(
                f"[stability] longest plateau: {best_plateau.start_threshold:.3f} → "
                f"{best_plateau.end_threshold:.3f} ({best_plateau.component_count} components)"
            )
    else:
        print(f"[threshold] using manual threshold: {model._resolved_threshold:.4f}")

    print(
        f"[run]    cosmic graph: {model.cosmic_graph.number_of_nodes()} nodes,"
        f" {model.cosmic_graph.number_of_edges()} edges"
    )

    model.print_report()


if __name__ == "__main__":
    main()
