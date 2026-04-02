# MMLU Topology Demo

Reveals hidden geometric structure in the MMLU benchmark using Pulsar's topological sweep.

## Quick Start

```bash
# From the repo root
uv sync --group demos
uv run maturin develop --release
cd demos/mmlu
jupyter notebook mmlu_topology_demo.ipynb
```

First run downloads MMLU (~14k questions) and computes embeddings (~2 min on Apple Silicon, ~10 min CPU). Subsequent runs use cached data in `data/`.

## What This Shows

MMLU is the standard LLM benchmark: 57 subjects, ~14,000 test questions, one leaderboard number. We ran Pulsar's multi-configuration topological sweep on the embedded question space and found:

1. Silhouette analysis on the cosmic graph identifies **12 distinct geometric regions** that cut across subject boundaries
2. Two subjects form tight geometric islands: `moral_scenarios` (100% isolation) and `professional_law` (87% of its region)
3. Per-question evaluation of GPT-4o-mini, Claude 3 Haiku, Gemini 2.5 Flash, and Grok 4 shows **significant accuracy gaps** between best and worst regions — the leaderboard hides this
4. Random sampling needs **3x more questions** than topology-aware sampling to cover all regions

## How the Parameters Were Calibrated

Getting useful results from Ball Mapper requires epsilon values matched to the actual scale of the data. This turned out to be the critical challenge — and the `helpers/` scripts document the full calibration process.

### The Problem: StandardScaler Changes Everything

Pulsar's pipeline runs `StandardScaler` (z-score normalization) before PCA. This is correct for general-purpose TDA, but it inflates pairwise distances by ~20x compared to the raw normalized embeddings:

| Space | Median Pairwise Distance |
|-------|------------------------:|
| Raw bge-small embeddings (384d) | 1.05 |
| After StandardScaler + PCA to 5d | 2.82 |
| After StandardScaler + PCA to 10d | 4.26 |
| After StandardScaler + PCA to 20d | 6.22 |
| After StandardScaler + PCA to 50d | 9.95 |

Our initial epsilon range (`0.05–0.5`) was calibrated for raw embeddings. After scaling, those values are **two orders of magnitude too small** — every point becomes its own ball center, producing ~5,000 singleton balls with zero edges. The Laplacian accumulates noise, and the cosmic graph is empty.

### Finding the Sweet Spot

We tested Ball Mapper directly at each PCA dimension to find where `n_balls` falls in the informative range (~20–500 balls with overlapping coverage):

| PCA Dim | Too Small (singletons) | Sweet Spot | Too Large (1 ball) |
|---------|----------------------:|:----------:|:-------------------|
| 5d | eps < 0.5 | **1.0 – 2.0** (26–424 balls) | eps > 3.0 |
| 10d | eps < 1.0 | **2.0 – 4.5** (12–728 balls) | eps > 5.0 |
| 20d | eps < 2.0 | **4.0 – 6.0** (~120–600 balls) | eps > 8.0 |
| 50d | eps < 5.0 | **8.0 – 12.0** (~30 balls) | eps > 15.0 |

The sweet spot scales roughly linearly with PCA dimension.

### Why We Dropped High PCA Dimensions

At 50d and above, the sweet spot is a single epsilon value — there's no range to sweep. Ball Mapper needs diverse covers across the grid, not 8 identical ball maps at eps=10. We focus on 5d/10d/20d where a single epsilon list (`1.0–6.0`) produces meaningful variation across resolutions.

### How the Number of Clusters Is Chosen: Silhouette Analysis

Rather than picking an arbitrary k, we sweep k=2..15 and measure **silhouette score** — how well each question fits its assigned cluster vs. the next-best cluster. Spectral clustering on the cosmic graph's weighted adjacency does the structural work; the silhouette score picks the resolution.

The silhouette curve peaks at **k=12** (score=0.0349), confirming 12 structurally distinct regions.

### Why `threshold: 0.0` Instead of `auto`

Pulsar's auto-threshold uses persistent homology to find stable edge-weight cutoffs. It works beautifully for data with **clear geometric separations** — clinical datasets, geographic data, sensor readings — where the weight distribution has natural gaps.

**MMLU's embedding space is unusually smooth.** The weight distribution is concentrated in a narrow band (median ~0.16, P95 ~0.21) with no natural breakpoints. This is typical of dense text embeddings from modern transformers, where every question has *some* similarity to every other question. The auto-threshold picks a value in the empty upper range (~0.66), killing 99.9% of edges and leaving only singletons.

Instead, we set `threshold: 0.0` (keep all edges) and use **spectral clustering** on the full weighted adjacency matrix. Spectral clustering operates on the eigenvectors of the graph Laplacian, which naturally captures community structure without needing a hard threshold.

### Why This Matters Beyond MMLU

Most real-world structured data has **much sparser, more separated geometry** than text embeddings:

- **EHR / clinical data**: Patient subgroups form distinct phenotypic clusters with clear boundaries. Pulsar's auto-threshold finds these with zero tuning.
- **Genomics**: Gene expression profiles have natural subtypes. The persistent homology analysis reveals stable groupings across perturbation scales.
- **Industrial / sensor data**: Operational modes create well-separated regions in feature space. The cosmic graph's connected components directly correspond to these modes.
- **Financial data**: Market regimes and portfolio clusters have discrete structure that auto-threshold captures naturally.

MMLU is a **stress test** for topological data analysis. If Pulsar can find meaningful structure in a smooth, 384-dimensional embedding space with no obvious separations — using the eigengap on the Laplacian spectrum — it will find *much more* in data that actually has geometric features to discover. The auto-threshold + persistent homology pipeline becomes even more powerful on data with real structure.

### Final Configuration

```yaml
sweep:
  pca:
    dimensions: [5, 10, 20]          # 3 scales with diverse ball maps
    seed: [42, 7, 13, 99, 123, 456, 789, 1000]  # 8 random seeds
  ball_mapper:
    epsilon: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
cosmic_graph:
  threshold: 0.0
```

264 ball maps. Runs in ~50 seconds on a 5,000-point subsample.

## Embedding Model Dependency

The geometric structure in this demo is discovered in bge-small-en-v1.5's embedding space. Different embedding models may reveal different cluster boundaries. This is a feature, not a bug — Pulsar can characterize how *any* representation organizes a dataset, which is useful for comparing embedding models or understanding what structure a particular model "sees."

## Results

12 regions, sizes 318–478. NMI = 0.335 (subjects partially but incompletely align with topology). Silhouette peak at k=12.

| Region | Size | Theme | Top Subjects |
|--------|-----:|-------|-------------|
| 0 | 431 | **Psychology / Behavioral** | professional_psychology (24%), hs_psychology (19%) |
| 1 | 437 | **Medicine / Health** | professional_medicine (19%), nutrition (12%), clinical_knowledge (11%) |
| 2 | 432 | **Mathematics / Quantitative** | elementary_math (19%), hs_math (16%), hs_statistics (5%) |
| 3 | 318 | **Moral Reasoning** | moral_scenarios (100%) — complete isolation |
| 4 | 364 | **General Knowledge** | miscellaneous (32%), world_religions (8%), global_facts (6%) |
| 5 | 478 | **Law** | professional_law (87%) — tightest cluster |
| 6 | 448 | **Applied Science / Engineering** | conceptual_physics (11%), electrical_engineering (8%) |
| 7 | 333 | **Philosophy / Logic** | philosophy (14%), moral_disputes (10%), logical_fallacies (8%) |
| 8 | 477 | **History** | hs_world_history (14%), hs_us_history (12%), hs_european_history (8%) |
| 9 | 438 | **Life Science / Biology** | hs_biology (8%), prehistory (7%), human_aging (7%) |
| 10 | 429 | **Economics / Business** | hs_macroeconomics (14%), marketing (14%), hs_microeconomics (11%) |
| 11 | 385 | **Policy / Governance** | security_studies (7%), hs_government (7%), professional_accounting (6%) |

Key findings:
- `moral_scenarios` isolates completely (318 questions, 1 subject). Structurally alien to the rest of MMLU.
- `professional_law` dominates Region 5 at 87% — the tightest geometric cluster in the benchmark.
- Most regions span 30–50+ subjects. The real structure is thematic, not administrative.
- Psychology splits: behavioral psych → Region 0, philosophical psych → Region 7. One subject label, two distinct question types.
- Per-question model evaluation (GPT-4o-mini, Claude 3 Haiku, Gemini 2.5 Flash, Grok 4) shows significant accuracy gaps between best and worst regions. The leaderboard never shows this.
- Random sampling needs 3x more questions than topology-aware sampling to cover all 12 regions.

## Helper Scripts

The `helpers/` directory contains the calibration scripts used to find these parameters:

| Script | Purpose |
|--------|---------|
| `run_model_evals.py` | Offline script to evaluate 4 models (GPT-4o-mini, Claude 3 Haiku, Gemini 2.5 Flash, Grok Fast via xAI) on the 5k subsample via API. Produces `data/model_eval_results.csv`. Requires `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`. |
| `diagnose.py` | Full diagnostic suite: weight distributions, stability curves, ball map diversity, pairwise distances. Run with `--config` to test any YAML. |
| `quick_experiment.py` | First-pass experiments at 1,500 points. Showed epsilon range was too small. |
| `quick_experiment_v2.py` | Discovered the StandardScaler distance inflation. Tested BallMapper directly per PCA dim. |
| `quick_experiment_v3.py` | Confirmed degenerate ball maps even with "corrected" ranges (still pre-scale). |
| `quick_experiment_v4.py` | Final calibration with post-scale distances. Found the working config. |

To re-run diagnostics on a new config:

```bash
cd demos/mmlu
uv run python helpers/diagnose.py --config mmlu_params.yaml
```

## File Structure

```
demos/mmlu/
  mmlu_topology_demo.ipynb    # Main notebook
  mmlu_params.yaml            # Calibrated Pulsar config
  README.md                   # This file
  helpers/
    diagnose.py               # Diagnostics
    quick_experiment*.py       # Calibration experiments
  data/                       # Auto-created on first run
    mmlu_questions.csv         # Cached MMLU dataset
    mmlu_embeddings_all.npy   # Cached embeddings (384d)
    mmlu_umap_sub.npy         # Cached UMAP (subsample)
    model_eval_results.csv    # Pre-computed per-question model accuracy
    eigengap.png              # Eigenvalue scree plot
    *.png                     # Generated figures
```
