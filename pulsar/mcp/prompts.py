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
2. Characterize: `characterize_dataset(dataset_id)` returns a compact,
   summary-first map: raw numeric geometry, schema counts, and a capped preview
   of interesting columns. Full per-column profiles are intentionally omitted
   to keep wide datasets out of agent context. Use
   `probe_columns(dataset_id, ['col_name'])` for deep per-column inspection
   (sample values, distributions). Call `characterize_dataset` once per dataset;
   do not repeat it casually on wide tables. Max 20 columns per probe call.
3. Calibrate: `create_config(dataset_id)` is mandatory. It returns the
   baseline config, processed-space distance percentiles, and a
   `sweep_strategy` block. Treat this as a broad first pass, not a final
   answer.
4. Validate Config: `validate_config(config_yaml, dataset_id)`.

## PHASE II: EXECUTE & VALIDATE
5. Run: `run_topological_sweep`.
6. Validate: `diagnose_cosmic_graph`.
   - GATE: If `grid_adequacy_status` is `under_sampled` or `thin_grid`,
     widen the grid before interpreting clusters.
   - GATE: Inspect the `advisories` list. `EMPTY_GRAPH` and `HIGH_SINGLETONS`
     require lowering the construction threshold or shifting the PCA grid
     before proceeding. If `finalization_gate.status` is `blocked`, run the
     suggested targeted resolution sweep or explicitly justify why the dominant
     component is clinically expected.
   - GATE: If density > 0.8 or < 0.1, STOP. Refine config (Step 2).
   - GATE: component_count=1 is normal; do not force separation by
     narrowing epsilon.
7. Compare: Use `get_experiment_history`, `compare_sweeps`, and
   `summarize_sweep_history` after each refinement. `summarize_sweep_history`
   distills patterns (e.g., which PCA dims caused hairballs) across the
   session; you still own the next-config decision. Prefer 2-3 deliberate
   sweeps over one brittle perfect-looking run.
8. Inspect payloads summary-first. Use `generate_cluster_dossier(detail="summary")`
   for routine exploration. Treat `detail="standard"` and `detail="full"` as
   late-stage dossier modes. For graph structure, use
   `get_topological_skeleton(detail="nodes")` for the compact hub/bridge summary,
   `detail="edges"` for capped raw edges, `detail="full_nodes"` for capped raw
   nodes, and `detail="full"` only when both raw graph arrays and config YAML are
   needed.

## GRID GEOMETRY & EXPERT AGENT AUTONOMY
The default config returned by `create_config` is ONLY a baseline starting guess. As the expert in the room, you have full autonomy—and the duty—to actively override, widen, and shift these parameters based on diagnostic feedback.

- **Broad First, Then Concentrate**: Start with the broad baseline grid from
  `create_config`. On high-dimensional processed data this may include a wide
  projection tail such as `[2, 5, 10, 15, 16]`. After the first run, compare sweeps and
  concentrate around the stable region rather than keeping every tail value
  forever.
- **Stay in the KD-tree Envelope by Default**: Unless the user explicitly asks
  for higher-dimensional projections, keep JL/PCA projection dimensions at
  `16` or less so Ball Mapper can use the KD-tree radius-query acceleration.
- **"Widen and Shift" over "Narrow Down"**: Do not optimize by narrowing down to a single PCA dimension. To find stable structures, maintain a wide multi-scale grid, but *shift* it away from degenerate areas. For example, if a baseline sweep of `[2, 3, 4, 5]` results in a dense hairball, do not narrow to `[5]`. Instead, shift the grid upwards (e.g., to `[4, 6, 8, 10]` or `[5, 8, 12]`) to drop low-dimensional flat noise while preserving multi-scale consensus.
- **The PCA Floor**: Never include PCA dimensions below the elbow of the cumulative variance curve. 1D, 2D, or 3D projections of complex, high-dimensional datasets collapse structures and inject spurious consensus edges that create false "hairballs." If the variance elbow is at 5D, your grid floor must be at least 4D or 5D (e.g., sweep `[5, 8, 12]`).
- **The Epsilon Gates**: Keep epsilon inside the returned k-NN distance
  domain unless you have a diagnostic reason to test a boundary. If the graph is
  shattered, raise the upper epsilon bound. If it is a dense hairball, lower the
  upper bound or shift the PCA grid upward.
- **Use the Tools**: `refine_config` supports quick PCA/epsilon edits,
  `run_topological_sweep` reports metric diffs, and `compare_sweeps` compares
  run IDs. Use them to iteratively find a representative grid.

## NO HILL-CLIMBING OPTIMIZATION
The cosmic graph is a stable representation, not a score to maximize.
- **Do Not Optimize Graph Metrics**: Do not treat density, component count, or singleton fraction as quality metrics to optimize. They are descriptive indicators of scale, not success scores. 
- **Seek Structural Persistence**: A topological pattern is "real" if it persists across many combinations of epsilon, PCA dimensions, and seeds. Look for structural invariants across sweeps rather than "tuning" for a single perfect-looking graph.

## THRESHOLD MECHANICS
Two independent levers operate on the same underlying weighted matrix at
different stages. Do not confuse them.

- **Construction threshold** (`cosmic_graph.construction_threshold` in config):
  set BEFORE the sweep. Gates the persisted binary cosmic graph used for
  graph-connectivity diagnostics and visualization. The full weighted matrix
  is still retained. Too high → singletons. Too low → over-connected hairball.
  Default `"auto"` runs persistent homology stability analysis
  (`threshold_stability_summary` in the sweep response) and picks the longest
  stable plateau. Override only if `diagnose_cosmic_graph` returns an
  `EMPTY_GRAPH` / `HIGH_SINGLETONS` advisory (lower it) or density > 0.8
  (raise it), then re-run the sweep.
  Use `get_threshold_stability_curve` for threshold options. Its
  `threshold_candidate_policy` is an interpretation lens, not an optimization
  target:
  - `balanced`: default MCP choice for general analysis.
  - `report_ready`: use before final cohort naming or HTML report claims.
  - `detail_seeking`: use when the user asks for more detailed clusters or
    rare archetypes; singleton-heavy outputs are exploratory.
  - `outlier_mining`: use only when the user asks for anomalies, frontier
    cases, or singleton residuals.
  Do not loop over policies to maximize a score; pick the policy from the
  user's intent and validate with feature evidence.

- **Interpretation threshold** (`interpretation_edge_weight_threshold` in
  `generate_cluster_dossier`): applied AFTER the graph is built. Slices the
  retained weighted matrix for clustering and reporting. For `method="auto"`
  and `method="components"`, the default inherits the construction threshold
  so clustering operates on the same surface you diagnosed. For explicit
  `method="spectral"`, the default is `0.0` so spectral uses the full weighted
  affinity matrix unless you deliberately sparsify it. Pass an explicit value
  to deliberately diverge. Use `diagnose_cosmic_graph` weight percentiles
  (weight_p25–p95) to pick a value when overriding. This changes
  interpretation-time connectivity and cluster fragmentation; it does not
  rebuild the persisted cosmic graph.

When divergence is intentional, the JSON dossier response carries
`construction_threshold`, `interpretation_edge_weight_threshold`, and
`threshold_inherited`, plus `threshold_surface` guidance so you can see exactly
which surface clustering used.

## PHASE III: CONTRASTIVE INTERPRETATION
7. Cluster: `generate_cluster_dossier`.
8. Contrast: Perform comparative analysis. Identify the 'Pivot Feature' —
   the variable that most cleanly separates Cluster A from its topological
   neighbors. Do not name in isolation.
9. Report: `export_html_report`. CRITICAL: YOU MUST pass synthesized,
   highly informative `cluster_names` based on Step 8. Names must be
   descriptive (e.g., 'Male Gentoos w/ Large Flippers'). Passing
   `cluster_names` as a flat JSON object like
   `{"0": "Male Gentoos w/ Large Flippers"}` is the difference between a raw
   Data Dump and a high-impact Research Paper.

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
