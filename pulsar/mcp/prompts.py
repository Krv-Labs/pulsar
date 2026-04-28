"""
Opt-in workflow prompts for Pulsar MCP agents.

These are not injected into `FastMCP(instructions=...)`. Agents request them
explicitly via the `get_workflow_guide` tool so clients that do not want the
opinionated workflow do not pay for it on every session.
"""

from __future__ import annotations


WORKFLOW_PROMPT = """\
# Pulsar Topological Analysis Workflow

Reveal the dataset's topology; do not force convenient clusters.

## PHASE I: INGEST & CALIBRATE
1. Ingest: Use `ingest_dataset(path)` handles. Prefer `dataset_id` everywhere.
2. Characterize: `characterize_dataset(dataset_id)` returns a SPARSE schema —
   dtype, n_unique, missingness — for ALL columns. Numeric stats and
   top_values are intentionally omitted to keep payload small for wide
   datasets. Use `probe_columns(dataset_id, ['col_name'])` for deep
   per-column inspection (sample values, distributions). Max 20 columns
   per probe call.
3. Calibrate: `create_config(dataset_id)` is mandatory. It returns the
   [p5, p95] epsilon domain. EPSILON OUTSIDE THIS RANGE PRODUCES
   DEGENERATE GRAPHS.
4. Validate Config: `validate_config(config_yaml, dataset_id)`.

## PHASE II: EXECUTE & VALIDATE
5. Run: `run_topological_sweep`.
6. Validate: `diagnose_cosmic_graph`.
   - GATE: If density > 0.8 or < 0.1, STOP. Refine config (Step 2).
   - GATE: component_count=1 is normal; do not force separation by
     narrowing epsilon.

## THRESHOLD MECHANICS
Two independent levers — do not confuse them:
- **Accumulation threshold** (`cosmic_graph.threshold` in config): set BEFORE the
  sweep. Controls how often two points must co-occur across ball maps to share an
  edge. Too high → singletons. Too low → over-connected hairball. Default "auto"
  is correct for most datasets; only override if `diagnose_cosmic_graph` shows
  density < 0.1 (raise it) or > 0.8 (lower it), then re-run the sweep.
- **Edge weight threshold** (`edge_weight_threshold` in `generate_cluster_dossier`):
  applied AFTER the graph is built. Use `get_threshold_stability_curve` and
  `diagnose_cosmic_graph` weight percentiles (weight_p25–p75) to choose a value.
  This tunes cluster count and sharpness, not graph connectivity.

## PHASE III: CONTRASTIVE INTERPRETATION
7. Cluster: `generate_cluster_dossier`.
8. Contrast: Perform comparative analysis. Identify the 'Pivot Feature' —
   the variable that most cleanly separates Cluster A from its topological
   neighbors. Do not name in isolation.
9. Report: `export_html_report`. CRITICAL: YOU MUST pass synthesized,
   highly informative `cluster_names` based on Step 8. Names must be
   descriptive (e.g., 'Male Gentoos w/ Large Flippers'). Passing
   `cluster_names` is the difference between a raw Data Dump and a
   high-impact Research Paper.

## PHILOSOPHY
- Pulsar is a multi-scale aggregator, not a tuner. More grid points =
  more topological evidence. ALL ball maps are fused into ONE cosmic
  graph.
- Wide PCA arrays and epsilon ranges are always superior to single
  points.
- The cosmic graph is evidence, not a score to maximize.

## PATH VISIBILITY (Claude Desktop)
Claude Desktop sandboxes are isolated. DO NOT use chunked/base64 uploads
for local files. Use the 'Cache-Bridge' pattern:
1. Call `get_runtime_context` to find `cache_dir`.
2. Use a shell command to `cp` your file into that `cache_dir`.
3. Call `ingest_dataset(path)` on the new path.
This is 100x faster and avoids protocol overhead.
"""
