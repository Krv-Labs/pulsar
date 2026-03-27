"""
Pulsar demo — PhysioNet EHR dataset (eICU Collaborative Research Database)
===========================================================================
Uses the eICU patient table aggregated to per-patient static features.
This avoids time-series complexity by using summary statistics per patient.

The dataset requires PhysioNet credentialed access. This demo expects a
local CSV export of the patient table with selected features.

If you don't have eICU access, the demo includes a synthetic generator that
mimics the statistical properties of typical ICU patient cohorts.

Usage (from repo root):
    uv run python demo/physionet.py
    uv run python demo/physionet.py --synthetic  # use synthetic data
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
DEMO_DIR = REPO_ROOT / "demo"
CSV_PATH = DEMO_DIR / "eicu_patient_static.csv"
PARAMS_PATH = DEMO_DIR / "physionet_params.yaml"


# ---------------------------------------------------------------------------
# Synthetic EHR data generator (mimics eICU patient table statistics)
# ---------------------------------------------------------------------------


def generate_synthetic_ehr(n_patients: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic EHR data mimicking eICU patient table structure.

    Features are based on typical ICU patient characteristics:
    - Demographics: age, gender, ethnicity, BMI
    - Admission info: admission source, unit type, hospital stay length
    - Vitals (admission): heart rate, MAP, temperature, SpO2, respiratory rate
    - Labs (admission): creatinine, BUN, glucose, WBC, hemoglobin, platelets
    - Comorbidities: binary flags for common conditions
    - Severity scores: APACHE IV, predicted mortality
    - Outcomes: ICU LOS, hospital mortality
    """
    rng = np.random.default_rng(seed)

    # Demographics
    age = rng.normal(65, 15, n_patients).clip(18, 100)
    gender = rng.choice([0, 1], n_patients)  # 0=F, 1=M
    ethnicity = rng.choice(
        [0, 1, 2, 3, 4], n_patients, p=[0.65, 0.15, 0.10, 0.05, 0.05]
    )
    bmi = rng.normal(28, 6, n_patients).clip(15, 60)

    # Admission info
    admission_source = rng.choice([0, 1, 2, 3], n_patients, p=[0.4, 0.3, 0.2, 0.1])
    unit_type = rng.choice(
        [0, 1, 2, 3, 4], n_patients, p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )

    # Vitals at admission (with physiological correlations)
    heart_rate = rng.normal(88, 20, n_patients).clip(40, 180)
    map_bp = rng.normal(75, 15, n_patients).clip(40, 140)
    temperature = rng.normal(37.2, 0.8, n_patients).clip(34, 42)
    spo2 = rng.beta(20, 1, n_patients) * 15 + 85  # skewed toward high values
    respiratory_rate = rng.normal(20, 6, n_patients).clip(8, 45)

    # Labs at admission
    creatinine = rng.lognormal(0.5, 0.8, n_patients).clip(0.3, 15)
    bun = rng.lognormal(2.8, 0.6, n_patients).clip(5, 150)
    glucose = rng.lognormal(4.8, 0.4, n_patients).clip(40, 600)
    wbc = rng.lognormal(2.3, 0.5, n_patients).clip(1, 50)
    hemoglobin = rng.normal(11, 2.5, n_patients).clip(5, 18)
    platelets = rng.lognormal(5.2, 0.5, n_patients).clip(20, 800)
    lactate = rng.lognormal(0.7, 0.7, n_patients).clip(0.5, 20)

    # Comorbidities (binary, with realistic prevalence)
    hypertension = rng.binomial(1, 0.55, n_patients)
    diabetes = rng.binomial(1, 0.30, n_patients)
    copd = rng.binomial(1, 0.15, n_patients)
    chf = rng.binomial(1, 0.20, n_patients)
    ckd = rng.binomial(1, 0.18, n_patients)
    liver_disease = rng.binomial(1, 0.08, n_patients)
    cancer = rng.binomial(1, 0.12, n_patients)

    # Severity scores (correlated with vitals/labs)
    apache_base = (
        0.02 * age
        + 0.01 * np.abs(heart_rate - 80)
        + 0.02 * np.abs(map_bp - 70)
        + 0.5 * np.log1p(creatinine)
        + 0.3 * np.log1p(lactate)
        + 0.1 * (100 - spo2)
        + 2 * chf
        + 1.5 * ckd
    )
    apache_score = (
        (apache_base * 3 + rng.normal(0, 5, n_patients)).clip(0, 150).astype(int)
    )

    # Predicted mortality (logistic based on severity)
    logit = -4 + 0.05 * apache_score + 0.02 * age - 0.01 * map_bp
    predicted_mortality = 1 / (1 + np.exp(-logit))

    # Outcomes
    icu_los_days = rng.lognormal(1.0, 0.8, n_patients).clip(0.5, 60)
    hospital_mortality = (rng.random(n_patients) < predicted_mortality).astype(int)

    # Introduce missing values (realistic patterns)
    missing_mask = rng.random(n_patients) < 0.05
    bmi_with_nan = np.where(missing_mask, np.nan, bmi)

    missing_mask = rng.random(n_patients) < 0.08
    lactate_with_nan = np.where(missing_mask, np.nan, lactate)

    missing_mask = rng.random(n_patients) < 0.03
    ethnicity_with_nan = np.where(missing_mask, np.nan, ethnicity)

    df = pd.DataFrame(
        {
            "patient_id": np.arange(n_patients),
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity_with_nan,
            "bmi": bmi_with_nan,
            "admission_source": admission_source,
            "unit_type": unit_type,
            "heart_rate": heart_rate,
            "mean_arterial_pressure": map_bp,
            "temperature": temperature,
            "spo2": spo2,
            "respiratory_rate": respiratory_rate,
            "creatinine": creatinine,
            "bun": bun,
            "glucose": glucose,
            "wbc": wbc,
            "hemoglobin": hemoglobin,
            "platelets": platelets,
            "lactate": lactate_with_nan,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "copd": copd,
            "chf": chf,
            "ckd": ckd,
            "liver_disease": liver_disease,
            "cancer": cancer,
            "apache_score": apache_score,
            "predicted_mortality": predicted_mortality,
            "icu_los_days": icu_los_days,
            "hospital_mortality": hospital_mortality,
        }
    )

    return df


# ---------------------------------------------------------------------------
# Auto-detect preprocessing from the raw data
# ---------------------------------------------------------------------------


def build_preprocessing(
    df: pd.DataFrame, id_col: str = "patient_id"
) -> tuple[list[str], dict]:
    """
    Return (drop_columns, impute_dict).

    Strategy:
    - Drop ID columns (not useful for clustering)
    - Categorical (non-numeric) columns: keep, impute with sample_categorical
    - Numeric columns with NaN: impute with sample_normal
    """
    drop_cols = [id_col] if id_col in df.columns else []

    df_work = df.drop(columns=drop_cols, errors="ignore")
    cat_cols = df_work.select_dtypes(exclude="number").columns.tolist()
    numeric_df = df_work.drop(columns=cat_cols, errors="ignore")

    impute: dict = {}

    # Categorical NaN columns
    for col in cat_cols:
        if df_work[col].isna().any():
            impute[col] = {"method": "sample_categorical", "seed": 42}

    # Numeric NaN columns
    for col in numeric_df.columns:
        if numeric_df[col].isna().any():
            impute[col] = {"method": "sample_normal", "seed": 42}

    return drop_cols, impute


# ---------------------------------------------------------------------------
# Timed pipeline (reused from coal.py)
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
    parser = argparse.ArgumentParser(description="PhysioNet EHR demo for Pulsar")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic EHR data instead of real PhysioNet data",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=2000,
        help="Number of patients for synthetic data (default: 2000)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to real EHR CSV file (eICU patient table export)",
    )
    args = parser.parse_args()

    # Load or generate data
    if args.synthetic or not (args.data or CSV_PATH.exists()):
        print(f"[data] generating synthetic EHR data ({args.n_patients} patients) ...")
        df_raw = generate_synthetic_ehr(n_patients=args.n_patients)
        print(
            f"[data] synthetic dataset shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols"
        )
        data_source = "synthetic"
    else:
        data_path = Path(args.data) if args.data else CSV_PATH
        if not data_path.exists():
            print(f"[error] Data file not found: {data_path}")
            print("[error] Use --synthetic flag to generate synthetic data, or")
            print("[error] provide a path to eICU patient table CSV with --data")
            return
        print(f"[data] loading {data_path} ...")
        df_raw = pd.read_csv(data_path)
        print(f"[data] dataset shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
        data_source = str(data_path)

    # Detect preprocessing requirements
    print("[config] detecting column types and NaN patterns ...")
    drop_columns, impute_specs = build_preprocessing(df_raw)

    cat_cols = df_raw.select_dtypes(exclude="number").columns.tolist()
    cat_nan = sum(1 for c in cat_cols if df_raw[c].isna().any())
    num_nan = sum(1 for c, s in impute_specs.items() if s["method"] == "sample_normal")

    print(f"[config] dropping {len(drop_columns)} ID columns: {drop_columns}")
    print(f"[config] imputing {cat_nan} categorical NaN cols (sample_categorical)")
    print(f"[config] imputing {num_nan} numeric NaN cols (sample_normal/Gaussian)")

    # Build config
    raw_cfg = {
        "run": {
            "name": "physionet_ehr_demo",
            "data": data_source,
        },
        "preprocessing": {
            "drop_columns": drop_columns,
            "impute": impute_specs,
        },
        "sweep": {
            "pca": {
                "dimensions": {"values": [2, 3, 5, 8, 10, 12]},
                "seed": {"values": [42, 7, 13, 99, 123, 456]},
            },
            "ball_mapper": {
                "epsilon": {"range": {"min": 0.4, "max": 2.5, "steps": 40}},
            },
        },
        "cosmic_graph": {
            "threshold": 0.0,
            "neighborhood": "node",
        },
        "output": {
            "n_reps": 5,
        },
    }

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
    model.fit(df_raw)
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

    # Print some EHR-specific analysis
    print()
    print("=" * 60)
    print("EHR Analysis Summary")
    print("=" * 60)

    n_nodes = model.cosmic_graph.number_of_nodes()
    n_edges = model.cosmic_graph.number_of_edges()

    print(f"Patients analyzed:        {df_raw.shape[0]}")
    print(f"Features used:            {df_raw.shape[1] - len(drop_columns)}")
    print(f"Cosmic graph nodes:       {n_nodes}")
    print(f"Cosmic graph edges:       {n_edges}")

    # Edge weight distribution
    weights = np.array([d["weight"] for _, _, d in model.cosmic_graph.edges(data=True)])
    if len(weights) > 0:
        print()
        print("Edge Weight Distribution:")
        print(f"  Min:                    {weights.min():.4f}")
        print(f"  25th percentile:        {np.percentile(weights, 25):.4f}")
        print(f"  Median:                 {np.median(weights):.4f}")
        print(f"  75th percentile:        {np.percentile(weights, 75):.4f}")
        print(f"  Max:                    {weights.max():.4f}")
        print(f"  Mean:                   {weights.mean():.4f}")
        print(f"  Std dev:                {weights.std():.4f}")


if __name__ == "__main__":
    main()
