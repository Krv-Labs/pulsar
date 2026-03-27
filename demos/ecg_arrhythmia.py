"""
Pulsar demo — ECG Arrhythmia Classification (Feature Summarization Approach)
=============================================================================

This demo shows how to handle time-series data by summarizing each recording
into per-patient feature vectors, then using the standard Pulsar pipeline.

Dataset: PhysioNet ECG Arrhythmia Database
- 45,152 patients with 10-second, 12-lead ECG recordings at 500 Hz
- Each recording: 12 leads × 5,000 samples
- Labels: SNOMED-CT coded arrhythmia diagnoses

Approach:
1. Extract summary features from each ECG (collapsing 5000 samples → ~80 features)
2. Run standard ThemaRS pipeline
3. Compare discovered clusters against known arrhythmia labels

This demonstrates the "summarize time-series" approach — useful when:
- The overall signal shape matters more than moment-to-moment dynamics
- You want to leverage existing non-temporal pipelines
- Computational efficiency is important (no 3D tensor)

Usage:
    uv run python demos/ecg_arrhythmia.py --synthetic
    uv run python demos/ecg_arrhythmia.py --data path/to/ecg_data/
"""

from __future__ import annotations

import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd
from pulsar._pulsar import (
    CosmicGraph,
    StandardScaler,
    accumulate_pseudo_laplacians,
    ball_mapper_grid,
    pca_grid,
)
from scipy import signal
from scipy.stats import kurtosis, skew

from pulsar.config import load_config
from pulsar.hooks import cosmic_to_networkx

# ---------------------------------------------------------------------------
# ECG Feature Extraction
# ---------------------------------------------------------------------------

ECG_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def detect_r_peaks(ecg_signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """
    Simple R-peak detection using bandpass filter + threshold.

    This is a simplified detector suitable for demonstration. For clinical
    use, consider more robust algorithms (Pan-Tompkins, wavelet-based, etc.).
    """
    # Bandpass filter 5-15 Hz to isolate QRS complex
    nyq = fs / 2
    low, high = 5 / nyq, 15 / nyq
    b, a = signal.butter(2, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, ecg_signal)

    # Square to emphasize peaks
    squared = filtered**2

    # Moving average for adaptive threshold
    window = int(0.15 * fs)  # 150ms window
    if window < 1:
        window = 1
    ma = np.convolve(squared, np.ones(window) / window, mode="same")

    # Find peaks above threshold
    threshold = np.mean(ma) + 0.5 * np.std(ma)
    peaks, _ = signal.find_peaks(squared, height=threshold, distance=int(0.3 * fs))

    return peaks


def extract_ecg_features(ecg: np.ndarray, fs: int = 500) -> dict:
    """
    Extract summary features from a 12-lead ECG recording.

    Parameters
    ----------
    ecg : np.ndarray
        Shape (12, n_samples) — 12 leads, typically 5000 samples at 500 Hz.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Feature dictionary with ~80 features.
    """
    n_leads, n_samples = ecg.shape
    features = {}

    # Per-lead statistical features
    for i, lead in enumerate(ECG_LEADS):
        sig = ecg[i]

        # Basic statistics
        features[f"{lead}_mean"] = np.mean(sig)
        features[f"{lead}_std"] = np.std(sig)
        features[f"{lead}_min"] = np.min(sig)
        features[f"{lead}_max"] = np.max(sig)
        features[f"{lead}_range"] = np.ptp(sig)
        features[f"{lead}_skew"] = skew(sig)
        features[f"{lead}_kurtosis"] = kurtosis(sig)

        # Signal energy
        features[f"{lead}_energy"] = np.sum(sig**2) / n_samples

        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(sig))) > 0)
        features[f"{lead}_zcr"] = zero_crossings / n_samples

    # R-peak based features (use lead II as primary)
    lead_ii = ecg[1]
    r_peaks = detect_r_peaks(lead_ii, fs)

    if len(r_peaks) >= 2:
        rr_intervals = np.diff(r_peaks) / fs * 1000  # in ms

        features["rr_mean"] = np.mean(rr_intervals)
        features["rr_std"] = np.std(rr_intervals)
        features["rr_min"] = np.min(rr_intervals)
        features["rr_max"] = np.max(rr_intervals)
        features["rr_range"] = np.ptp(rr_intervals)

        # Heart rate
        features["hr_mean"] = 60000 / np.mean(rr_intervals)
        features["hr_std"] = (
            np.std(60000 / rr_intervals) if len(rr_intervals) > 1 else 0
        )

        # HRV metrics (simplified)
        if len(rr_intervals) >= 3:
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            features["hrv_rmssd"] = rmssd
            features["hrv_sdnn"] = np.std(rr_intervals)
        else:
            features["hrv_rmssd"] = 0
            features["hrv_sdnn"] = 0

        features["n_beats"] = len(r_peaks)
    else:
        # Unable to detect sufficient R-peaks
        features["rr_mean"] = np.nan
        features["rr_std"] = np.nan
        features["rr_min"] = np.nan
        features["rr_max"] = np.nan
        features["rr_range"] = np.nan
        features["hr_mean"] = np.nan
        features["hr_std"] = np.nan
        features["hrv_rmssd"] = np.nan
        features["hrv_sdnn"] = np.nan
        features["n_beats"] = len(r_peaks)

    return features


# ---------------------------------------------------------------------------
# Synthetic ECG Generator
# ---------------------------------------------------------------------------


def generate_synthetic_ecg(
    n_patients: int = 1000,
    n_samples: int = 5000,
    fs: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate synthetic ECG recordings with different rhythm types.

    Returns
    -------
    ecgs : np.ndarray
        Shape (n_patients, 12, n_samples)
    labels : pd.DataFrame
        Patient metadata including rhythm type
    """
    rng = np.random.default_rng(seed)

    # Rhythm types with relative frequencies
    rhythm_types = {
        "NSR": 0.50,  # Normal Sinus Rhythm
        "AFIB": 0.15,  # Atrial Fibrillation
        "SB": 0.10,  # Sinus Bradycardia
        "ST": 0.08,  # Sinus Tachycardia
        "PAC": 0.07,  # Premature Atrial Contraction
        "PVC": 0.05,  # Premature Ventricular Contraction
        "RBBB": 0.03,  # Right Bundle Branch Block
        "LBBB": 0.02,  # Left Bundle Branch Block
    }

    rhythms = list(rhythm_types.keys())
    probs = list(rhythm_types.values())

    ecgs = np.zeros((n_patients, 12, n_samples))
    patient_rhythms = rng.choice(rhythms, size=n_patients, p=probs)

    t = np.linspace(0, n_samples / fs, n_samples)

    for i in range(n_patients):
        rhythm = patient_rhythms[i]

        # Base heart rate varies by rhythm
        if rhythm == "SB":
            hr = rng.uniform(40, 60)
        elif rhythm == "ST":
            hr = rng.uniform(100, 150)
        elif rhythm == "AFIB":
            hr = rng.uniform(80, 160)
        else:
            hr = rng.uniform(60, 100)

        # Generate synthetic heartbeats
        beat_interval = 60 / hr
        n_beats = int(n_samples / fs / beat_interval) + 2

        for lead_idx in range(12):
            sig = np.zeros(n_samples)

            # Lead-specific amplitude scaling
            lead_scale = 1.0 + 0.3 * rng.standard_normal()

            beat_times = []
            current_time = rng.uniform(0, beat_interval)

            for _ in range(n_beats):
                if rhythm == "AFIB":
                    # Irregular RR intervals
                    interval = beat_interval * rng.uniform(0.6, 1.4)
                elif rhythm in ("PAC", "PVC") and rng.random() < 0.15:
                    # Occasional premature beat
                    interval = beat_interval * rng.uniform(0.5, 0.7)
                else:
                    # Regular with small variability
                    interval = beat_interval * rng.uniform(0.95, 1.05)

                beat_times.append(current_time)
                current_time += interval

            # Generate QRS-like waveform at each beat
            for bt in beat_times:
                sample_idx = int(bt * fs)
                if sample_idx >= n_samples - 50:
                    continue

                # Simplified QRS complex
                qrs_width = int(0.08 * fs)  # 80ms
                if rhythm in ("RBBB", "LBBB"):
                    qrs_width = int(0.12 * fs)  # Widened QRS

                if sample_idx + qrs_width < n_samples:
                    # R wave (positive deflection)
                    r_amp = lead_scale * (1.0 + 0.2 * rng.standard_normal())
                    if rhythm == "PVC" and rng.random() < 0.15:
                        r_amp *= 1.5  # Larger amplitude for PVC

                    qrs = r_amp * signal.windows.gaussian(qrs_width, qrs_width / 4)
                    sig[sample_idx : sample_idx + qrs_width] += qrs[
                        : n_samples - sample_idx
                    ]

                    # T wave
                    t_delay = int(0.2 * fs)
                    t_width = int(0.15 * fs)
                    t_start = sample_idx + t_delay
                    if t_start + t_width < n_samples:
                        t_amp = lead_scale * 0.3 * (1.0 + 0.3 * rng.standard_normal())
                        t_wave = t_amp * signal.windows.gaussian(t_width, t_width / 3)
                        sig[t_start : t_start + t_width] += t_wave

            # Add baseline wander and noise
            baseline = 0.1 * np.sin(2 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi))
            noise = 0.02 * rng.standard_normal(n_samples)

            ecgs[i, lead_idx] = sig + baseline + noise

    # Create labels dataframe
    labels = pd.DataFrame(
        {
            "patient_id": np.arange(n_patients),
            "rhythm": patient_rhythms,
            "age": rng.normal(60, 15, n_patients).clip(18, 95).astype(int),
            "sex": rng.choice(["M", "F"], n_patients),
        }
    )

    return ecgs, labels


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def extract_features_from_dataset(
    ecgs: np.ndarray,
    fs: int = 500,
) -> pd.DataFrame:
    """
    Extract features from all ECG recordings.

    Parameters
    ----------
    ecgs : np.ndarray
        Shape (n_patients, 12, n_samples)
    fs : int
        Sampling frequency

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per patient
    """
    n_patients = ecgs.shape[0]
    all_features = []

    for i in range(n_patients):
        features = extract_ecg_features(ecgs[i], fs)
        features["patient_id"] = i
        all_features.append(features)

    df = pd.DataFrame(all_features)
    df = df.set_index("patient_id")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ECG Arrhythmia demo - summarize time-series approach"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic ECG data",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=1000,
        help="Number of patients for synthetic data (default: 1000)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to real ECG data directory (WFDB format)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ECG Arrhythmia Demo — Summarize Time-Series Approach")
    print("=" * 70)
    print()

    # Generate or load data
    if args.synthetic or args.data is None:
        print(f"[data] Generating synthetic ECG data ({args.n_patients} patients)...")
        t0 = time.perf_counter()
        ecgs, labels = generate_synthetic_ecg(n_patients=args.n_patients)
        print(f"[data] Generated in {time.perf_counter() - t0:.2f}s")
        print(f"[data] ECG shape: {ecgs.shape} (patients × leads × samples)")
        print("[data] Rhythm distribution:")
        for rhythm, count in labels["rhythm"].value_counts().items():
            print(f"       {rhythm}: {count} ({100 * count / len(labels):.1f}%)")
    else:
        print(f"[data] Loading real ECG data from {args.data}...")
        print("[error] Real data loading not implemented in this demo.")
        print("[error] Use --synthetic flag or implement WFDB loading.")
        return

    print()

    # Extract features
    print("[features] Extracting summary features from ECG signals...")
    t0 = time.perf_counter()
    features_df = extract_features_from_dataset(ecgs)
    print(
        f"[features] Extracted {features_df.shape[1]} features in {time.perf_counter() - t0:.2f}s"
    )

    # Handle NaN values (from failed R-peak detection)
    n_nan_rows = features_df.isna().any(axis=1).sum()
    if n_nan_rows > 0:
        print(f"[features] Dropping {n_nan_rows} patients with missing features")
        features_df = features_df.dropna()
        labels = labels.loc[features_df.index]

    print(
        f"[features] Final dataset: {len(features_df)} patients × {features_df.shape[1]} features"
    )
    print()

    # Build Pulsar config
    config = load_config(
        {
            "run": {"name": "ecg_arrhythmia_demo"},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "pca": {
                    "dimensions": {"values": [3, 5, 8, 12, 15]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.5, "max": 3.0, "steps": 30}},
                },
            },
            "cosmic_graph": {"threshold": 0.0},
            "output": {"n_reps": 5},
        }
    )

    n_maps = (
        len(config.pca.dimensions)
        * len(config.pca.seeds)
        * len(config.ball_mapper.epsilons)
    )
    print(
        f"[grid] {len(config.pca.dimensions)} dims × {len(config.pca.seeds)} seeds × {len(config.ball_mapper.epsilons)} epsilons = {n_maps} ball maps"
    )
    print()

    # Run pipeline
    print("[pipeline] Running Pulsar pipeline...")
    t_start = time.perf_counter()

    # Prepare data
    X = features_df.to_numpy(dtype=np.float64)
    n = X.shape[0]

    # Scale
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_scaled = np.array(scaler.fit_transform(X))
    t_scale = time.perf_counter() - t0

    # PCA grid
    t0 = time.perf_counter()
    embeddings = [
        np.ascontiguousarray(emb)
        for emb in pca_grid(X_scaled, config.pca.dimensions, config.pca.seeds)
    ]
    t_pca = time.perf_counter() - t0

    # Ball Mapper grid
    t0 = time.perf_counter()
    ball_maps = ball_mapper_grid(embeddings, config.ball_mapper.epsilons)
    t_bm = time.perf_counter() - t0

    # Accumulate pseudo-Laplacians
    t0 = time.perf_counter()
    galactic_L = np.array(accumulate_pseudo_laplacians(ball_maps, n))
    t_lap = time.perf_counter() - t0

    # Build CosmicGraph
    t0 = time.perf_counter()
    cosmic = CosmicGraph.from_pseudo_laplacian(
        galactic_L, config.cosmic_graph.threshold
    )
    G = cosmic_to_networkx(cosmic)
    t_cosmic = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    print(f"[pipeline] Done in {t_total:.2f}s")
    print()

    # Timing report
    print("Timing Breakdown:")
    print("-" * 40)
    print(f"  Scale:          {t_scale:>8.3f}s  ({100 * t_scale / t_total:>5.1f}%)")
    print(f"  PCA grid:       {t_pca:>8.3f}s  ({100 * t_pca / t_total:>5.1f}%)")
    print(f"  BallMapper:     {t_bm:>8.3f}s  ({100 * t_bm / t_total:>5.1f}%)")
    print(f"  Laplacian:      {t_lap:>8.3f}s  ({100 * t_lap / t_total:>5.1f}%)")
    print(f"  CosmicGraph:    {t_cosmic:>8.3f}s  ({100 * t_cosmic / t_total:>5.1f}%)")
    print("-" * 40)
    print(f"  TOTAL:          {t_total:>8.3f}s")
    print()

    # Results
    print("Cosmic Graph Results:")
    print("-" * 40)
    print(f"  Nodes:          {G.number_of_nodes()}")
    print(f"  Edges:          {G.number_of_edges()}")

    if G.number_of_edges() > 0:
        weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        print(
            f"  Edge weights:   min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}"
        )

    # Cluster analysis by rhythm type
    print()
    print("Cluster Analysis by Rhythm Type:")
    print("-" * 40)

    # Get connected components as clusters
    components = list(nx.connected_components(G))
    n_clusters = len(components)
    print(f"  Connected components: {n_clusters}")

    if n_clusters > 1 and n_clusters < 50:
        # Assign cluster IDs to patients
        cluster_labels = np.zeros(n, dtype=int)
        for cluster_id, component in enumerate(components):
            for node in component:
                cluster_labels[node] = cluster_id

        # Map back to original patient IDs
        patient_ids = features_df.index.to_numpy()

        # Cross-tabulate rhythm vs cluster
        cluster_df = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "cluster": cluster_labels,
                "rhythm": labels.loc[patient_ids, "rhythm"].values,
            }
        )

        print()
        print("  Rhythm distribution per cluster (top 5 clusters by size):")
        cluster_sizes = (
            cluster_df.groupby("cluster").size().sort_values(ascending=False)
        )

        for cluster_id in cluster_sizes.head(5).index:
            cluster_rhythms = cluster_df[cluster_df["cluster"] == cluster_id]["rhythm"]
            size = len(cluster_rhythms)
            dominant = cluster_rhythms.value_counts().head(3)
            rhythm_str = ", ".join([f"{r}:{c}" for r, c in dominant.items()])
            print(f"    Cluster {cluster_id}: {size} patients — {rhythm_str}")

    print()
    print("=" * 70)
    print("Demo complete.")
    print()
    print("Key insight: This 'summarize time-series' approach collapses the")
    print("5000-sample ECG into ~80 features per patient, enabling use of")
    print("the standard Pulsar pipeline. Good for classification tasks where")
    print("overall signal characteristics matter more than temporal dynamics.")
    print("=" * 70)


if __name__ == "__main__":
    main()
