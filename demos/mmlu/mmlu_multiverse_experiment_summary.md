# MMLU Multiverse Cosmic Graph Experiment: Setup and Findings

## Executive Summary

This experiment analyzes whether a topology-first view of MMLU questions reveals stable semantic regions that cut across model families and subject labels.

A single fused cosmic graph was built from 100 representations (10 models x 10 embedding variants) over 4,970 MMLU questions spanning 57 subjects. Spectral clustering on the graph selected 15 topological regions as the best partition by silhouette.

The main finding is that the topology is neither random nor equivalent to raw subject taxonomy:

- Subject-topology alignment is moderate (NMI = 0.470, ARI = 0.208).
- Some regions are highly mixed, while others are near-pure "islands" (up to 100% dominant subject).
- Models show consistent region-specific gains and failures relative to their own average accuracy, indicating shared geometric difficulty regimes.

## Experiment Setup

### Goal

Characterize the global geometry of MMLU by fusing many model-derived representations into one graph, then evaluate:

1. How many topological regions are supported by the fused graph.
2. How those regions relate to MMLU subject labels.
3. How model accuracy shifts by region.

### Data and Inputs

- Dataset cache: `demos/mmlu/data/mmlu_questions.csv`
- Cosmic fit summary: `demos/mmlu/data/scalability_all_100/mmlu_cosmic_fit_summary.json`
- Weighted adjacency: `demos/mmlu/data/scalability_all_100/mmlu_cosmic_weighted_adjacency.npz`
- Model outcomes: `demos/mmlu/data/model_eval_results.csv`

### Run Metadata

- Created at: 2026-04-02T11:44:26.567296+00:00
- Representations fused: 100
- Total ball maps accumulated: 10,500
- Run mode: subsample
- Rows analyzed: 4,970
- Subjects represented: 57
- Resolved graph threshold: 0.0

### Analysis Procedure

1. Load and sanitize the weighted adjacency matrix (symmetrize, clamp to [0, 1], set diagonal to 1).
2. Convert to precomputed distances (`1 - W`).
3. Sweep spectral clustering for k = 2..20 and score with silhouette.
4. Select best k and compute subject-vs-topology agreement (ARI, NMI).
5. Interpret each region via dominant subject share and subject diversity.
6. Produce UMAP visualizations of topology regions versus subject labels.
7. Join region labels with per-question model correctness and compute per-model, per-region deltas from each model's overall mean accuracy.

## Quantitative Findings

### 1) Optimal Number of Regions

Silhouette increased steadily from k=2 to k=15, then dropped sharply at k=16.

- Best k: 15
- Best silhouette: 0.016274

Selected silhouette values:

- k=10: 0.014913
- k=12: 0.015013
- k=14: 0.016130
- k=15: 0.016274 (peak)
- k=16: 0.008926 (large drop)

Interpretation: the fused topology supports a 15-region partition before over-fragmentation begins.

### 2) Subject Labels vs Topological Regions

- NMI(subject, topology): 0.470437
- ARI(subject, topology): 0.207861

Interpretation: subject labels explain part of the geometry, but topology captures additional structure beyond subject taxonomy.

### 3) Region Composition

Cluster sizes are uneven:

- Largest region size: 964
- Smallest region size: 5
- Full size vector: [421, 237, 209, 964, 226, 637, 5, 387, 318, 407, 339, 257, 7, 459, 97]

High-purity regions (>=95% dominant subject):

- Region 4: 226 questions, 99.6% Professional Law
- Region 8: 318 questions, 100.0% Moral Scenarios
- Region 11: 257 questions, 98.8% Professional Law

Largest mixed regions (top by size) show broad cross-subject blending:

- Region 3 (n=964): 51 subjects, dominant subject share 16.2%
- Region 5 (n=637): 32 subjects, dominant subject share 14.4%
- Region 13 (n=459): 46 subjects, dominant subject share 11.3%

Interpretation: the graph contains both broad manifold regions and sharp, semantically focused islands.

### 4) Model Performance by Region

Overall model accuracies (after joining with region labels):

1. gemini-3.1-flash-lite: 88.97%
2. grok-fast: 81.75%
3. gpt-4.1-mini: 81.49%
4. gpt-4o-mini: 76.24%
5. claude-3-haiku: 65.63%
6. claude-haiku-4-5: 61.31%
7. gemini-2.5-flash: 59.20%

Each model exhibits strong region-specific deviations from its own mean:

- gemini-3.1-flash-lite: best Region 7 (+4.57 pts), worst Region 12 (-17.55 pts)
- grok-fast: best Region 6 (+18.25 pts), worst Region 8 (-18.23 pts)
- gpt-4.1-mini: best Region 2 (+8.46 pts), worst Region 8 (-18.28 pts)
- gpt-4o-mini: best Region 6 (+23.76 pts), worst Region 8 (-23.41 pts)
- claude-3-haiku: best Region 6 (+34.37 pts), worst Region 8 (-34.50 pts)
- claude-haiku-4-5: best Region 5 (+22.99 pts), worst Region 14 (-46.87 pts)
- gemini-2.5-flash: best Region 2 (+19.75 pts), worst Region 14 (-16.93 pts)

Interpretation: topological region is a strong explanatory axis for model behavior; failure modes are clustered, not uniformly distributed.

## Visual Story (What the Figures Show)

The notebook generates three figure groups:

1. Silhouette sweep plot: smooth gain up to k=15, then a discontinuous drop.
2. UMAP overlays:
   - Topology-region coloring yields coherent geometric blocks.
   - Subject coloring partially overlaps those blocks, but not perfectly.
3. Region delta heatmap:
   - Clear red/blue bands by region and model indicate localized capability strengths/weaknesses.

## Key Takeaways for a Blog Narrative

- Fusing many representation views (100 total) yields a stable, interpretable topological map of MMLU.
- The resulting regions are not just subject labels in disguise; they encode cross-subject structure.
- A few nearly pure semantic islands coexist with broad mixed regions.
- Region-conditioned accuracy reveals model-specific strengths and brittle zones that are hidden by aggregate benchmarks.
- Topological analysis can complement leaderboard metrics by exposing where performance succeeds or fails in the manifold.

## Reproducibility Notes

Notebook source:

- `demos/mmlu/mmlu_mulitverse_demo.ipynb`

Core fixed parameters used in this run:

- `K_RANGE = 2..20`
- Spectral clustering with precomputed affinity
- `random_state = 42`
- `n_init = 10`
- Subsample target = 5,000 (realized 4,970 rows)

If re-running, ensure the same cached inputs are present under `demos/mmlu/data/` to match these exact numbers.
