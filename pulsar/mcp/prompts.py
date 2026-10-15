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
   For a quick longitudinal run, use `sample_longitudinal_entities(dataset_id, entity_column)`
   after ingestion; it returns a derived dataset handle without exposing rows.
   `min_observations` counts rows. For a distinct-timepoint minimum, also pass
   `time_column` and `min_time_points`; this filters entities but does not ensure
   `drop_entity` retains complete coverage across every timepoint in the dataset.
2. Characterize: `characterize_dataset(dataset_id)` returns a compact,
   summary-first map: raw numeric geometry, schema counts, and a capped preview
   of interesting columns. Full per-column profiles are intentionally omitted
   to keep wide datasets out of agent context. Use
   `probe_columns(dataset_id, ['col_name'])` for deep per-column inspection
   (sample values, distributions). Call `characterize_dataset` once per dataset;
   do not repeat it casually on wide tables. Max 20 columns per probe call.
3. Preprocess: `draft_preprocessing_plan(dataset_id, target_columns=..., entity_column=..., time_column=...)`
   measures column evidence (shared missing masks, conditional columns, exact
   duplicates/sums, date/text kinds) and drafts a versioned plan (`plan_id`,
   `revision`) of labeled defaults (`decided_by="default"`) plus open questions
   (`blank_meaning`, `conditional_column`, `date_columns`, `duplicate_signal`,
   `outcome_columns`). The plan is evidence + labeled defaults, not a verdict;
   `stakes` (`neighbors_kept`, `feature_count_delta`) measure how much the map's
   k-NN neighborhoods move between options, not a score to optimize.
   - Read the questions and evidence first. Record choices you can infer from
     context or schema via `edit_preprocessing_plan(plan_id, base_revision, answers=[...], edits=[...], source="agent")`.
   - Ask the user ONLY questions whose answer cannot be inferred and whose
     stakes are material. Never present the server's question text verbatim to a
     non-technical user; translate it into their domain vocabulary.
   - Example (clinician, `blank_meaning` on `n_inpatient_claims` with no 0s):
     ask *"If a patient has no recorded hospital stays, does that mean they had
     none, or that we don't know?"* → if they say "none", call
     `edit_preprocessing_plan(plan_id, base_revision=1, answers=[{"question_id": "q1", "option_id": "none"}], source="user")`.
4. Calibrate & Validate: `create_config(dataset_id, plan_id=plan_id)` consumes
   the plan (emitting readable group markers like `n_inpatient_claims_and_2_more_was_missing`,
   described in `marker_legend`) and returns the baseline config, processed-space
   distance percentiles, and `sweep_strategy` (treat as a broad first pass, not a final
   answer). Calling `create_config` without `plan_id` drafts baseline defaults
   automatically and emits a `PREPROCESSING_QUESTIONS_OPEN` warning when open
   questions remain. Use `validate_preprocessing_config` or
   `validate_config(config_yaml, dataset_id)` when checking manual YAML edits.

## PHASE II: EXECUTE & VALIDATE
5. Run: `run_topological_sweep`.
6. Validate: `diagnose_cosmic_graph`.
   - Read it as current graph-state measurement, not a directive. Use
     `scale`, `component_morphology`, `sweep_support`, `observed_patterns`, and
     `risk_factors` to judge the graph against the user's objective.
   - If `sweep_support.grid_adequacy_status` is `under_sampled` or `thin_grid`,
     treat structural claims as weak until a broader grid is checked.
   - Inspect `risk_factors` for regimes such as `empty_graph`,
     `singleton_heavy`, `dense_hairball`, or `dominant_component`.
   - SPARSIFICATION CONTEXT: `cosmic_graph.sparsify` builds a spectral
     sparsifier after the cosmic graph is already constructed. It is not a
     construction-time speedup and it is not an H0/component-topology cleanup
     tool. It preserves spectral/effective-resistance structure, not human-
     readable component topology. Consider it for repeated spectral analysis,
     compact graph handoff, approximation audits, memory-sensitive graph
     analytics, or Laplacian/preconditioned numerical workflows; otherwise
     diagnose the constructed graph directly.
     When it is enabled, density and edge counts describe the sparsified graph
     surface and are not directly comparable to non-sparsified runs.
   - COMPONENT CONTEXT: `component_count=1` can be a valid topology. Do not
     treat component count alone as a failure signal; compare it with density,
     giant fraction, threshold stability, feature evidence, and the user's
     objective.
7. Compare: Use `get_sweep_history` and `compare_sweeps` after each
   refinement. `get_sweep_history(detail="summary")` distills patterns (e.g.,
   which projection dims collapsed structure) across the session; you still own
   the next-config decision. Prefer 2-3 deliberate sweeps over one brittle
   perfect-looking run.
8. Inspect payloads summary-first. Use `generate_cluster_dossier(detail="summary")`
   for routine exploration. Treat `detail="standard"` and `detail="full"` as
   late-stage dossier modes. For graph structure, use
   `get_topological_skeleton(detail="nodes")` for the compact hub/bridge summary,
   `detail="edges"` for capped raw edges, `detail="full_nodes"` for capped raw
   nodes, and `detail="full"` only when both raw graph arrays and config YAML are
   needed.

## BACKGROUND TASK MODE (opt-in)

Server operators enable it with `PULSAR_MCP_ENABLE_TASKS=1` (needs
`fastmcp[tasks]`). Only `run_topological_sweep` is task-capable; every other tool
stays synchronous. The result is identical to the synchronous path. A
task-capable client polls transparently, and progress is still reported. The
durable state is the session handles (`run_id`, `dataset_id`), not the task
object: follow-up tools such as `diagnose_cosmic_graph` use them as usual.

## RESULTS AND ERRORS
Every result has a text view and structured content. Errors set the MCP error flag
and carry `error_code` plus `agent_action`: act on `agent_action` before retrying
(a blind retry rarely helps).

## RUN HANDLES
`run_topological_sweep` returns a `run_id`. Tools that read the fitted model accept
`run_id` to address an earlier cached run, or to work on stateless transports.
Omit it to use the session's latest run. Dossier and graph-artifact publishes only
accept the latest run (`SESSION_REVISION_CONFLICT` otherwise). The in-memory cache
holds only the last `PULSAR_MCP_MODEL_CACHE_SIZE` runs (default 3); an evicted or
unknown `run_id` returns `RUN_ID_UNKNOWN`, so re-run the sweep. Cluster labels stay
session-scoped; responses report `clusters_run_id` to show which run they describe.

## TRANSPORT
Session state (cluster labels, the latest-run pointer, longitudinal artifacts) is
guaranteed to persist between calls only on stdio and on legacy-session HTTP
(clients that receive an `mcp-session-id`). The 2026-07-28 protocol revision removed
protocol sessions, so on Streamable HTTP each request gets a fresh session. Fitted
models still survive there through the process-wide run cache: pass `run_id` (from
the sweep response) to model-reading tools instead of relying on "latest". Cluster
labels and longitudinal artifacts do not survive a fresh session, so regenerate
them in the same call chain. `dataset_id` handles live in the on-disk registry; model
state is in memory only and is lost on server restart. Check `get_runtime_context`
(`transport_assumption`, `model_continuity`, `model_cache`) if unsure.

## LONGITUDINAL DATA (PANELS)
Long-format data (one row per entity and time step) is the wrong input for the
static sweep, which treats every row as an independent point.

Longitudinal panels: ingest a long-format table (one row per entity and time step, discrete orderable time column, numeric features), optionally `sample_longitudinal_entities`, `create_config`, then `build_longitudinal_graph(dataset_id, entity_column, time_column, config_yaml)`. This returns `longitudinal_id`; omit it on later calls to use the most recent panel. Default `representation='trajectory'` (sparse, cross-time edges); 'temporal' builds a dense (n,n,T) tensor, so check `build_cost` first. Then `diagnose_longitudinal_graph` (measurements and the threshold ladder), `get_trajectory_archetypes` (identical cluster sequences), `get_cross_time_neighbors` (lookalikes at other times), `classify_trajectories` (rule- or distance-based classes). Thresholds on these tools are interpretation filters, not the construction threshold. Cluster counts move steeply with threshold, so read the ladder. These tools read session-only longitudinal artifacts, not the fitted cosmic model, and take no `run_id`.

Details:
- `representation="trajectory"` nodes are observations `(entity, t)`; `"temporal"`
  nodes are entities (edges inside one time slice, collapsed by aggregation) and is
  bounded by `PULSAR_MCP_MAX_TENSOR_BYTES`; `"both"` builds each. For temporal, the
  diagnose tool compares six aggregations with `why`/`best_for`/`avoid_for`; choose
  one deliberately.
- `get_cross_time_neighbors` takes `t` as an original time-column label; responses
  give it as `time_label` and the positional index as `t`.
- `classify_trajectories` returns a class summary and a bounded entity preview;
  pass `detail="full"` for every entity.
- `time_column` must already be discrete: bin timestamps before ingest. `on_missing`
  is `drop_entity` (default; required for temporal), `forward_fill`, or
  `allow_ragged` (trajectory only).
- The build refits the projection grid and may recalibrate epsilon to the panel's
  k-NN domain; read `config_adaptation`.
- Longitudinal artifacts do not feed the cluster dossier chain.

## GRID GEOMETRY & EXPERT AGENT AUTONOMY
The default config returned by `create_config` is ONLY a baseline starting guess. As the expert in the room, you have full autonomy—and the duty—to actively override, widen, and shift these parameters based on diagnostic feedback.

- **Broad First, Then Concentrate**: Start with the broad baseline grid from
  `create_config`. On high-dimensional processed data this may include a wide
  projection tail such as `[10, 15, 16]`. After the first run, compare sweeps and
  concentrate around the stable region rather than keeping every tail value
  forever.
- **Stay in the KD-tree Envelope by Default**: `create_config` caps the generated
  projection grid at `16` dims so Ball Mapper can use KD-tree radius-query
  acceleration. To go wider when the user explicitly requests it, set the dims
  via `refine_config` (`projection_dimensions`, uncapped); projections above 16
  fall back to a linear scan in Ball Mapper.
- **"Widen and Shift" over "Narrow Down"**: Do not optimize by narrowing down to a single projection dimension. To find stable structures, maintain a wide multi-scale grid, but *shift* it away from degenerate areas. For example, if a baseline sweep of `[2, 3, 4, 5]` collapses into one dominant component (high `giant_fraction`), do not narrow to `[5]`. Instead, shift the grid upwards (e.g., to `[4, 6, 8, 10]` or `[5, 8, 12]`) to drop low-dimensional flat noise while preserving multi-scale consensus.
- **The Projection Floor**: Avoid an ultra-low grid floor (1D/2D/3D) for complex high-dimensional data — it collapses structure and injects spurious consensus edges. Floor at or above the cumulative-variance elbow of the processed data (from `characterize_dataset`); e.g., if the elbow is at 5D, sweep `[5, 8, 12]`. This applies to both the JL default and the legacy `method: pca` path: the variance curve guides *which* dimensions to sweep, while the method only changes *how* points are projected onto them (JL preserves pairwise distances).
- **The Epsilon Gates**: Keep epsilon inside the returned k-NN distance
  domain unless you have a diagnostic reason to test a boundary. If the graph is
  shattered, raise the upper epsilon bound. If one component dominates (high
  `giant_fraction`), lower the upper bound or shift the projection grid upward.
- **Use the Tools**: `refine_config` supports quick projection/epsilon edits,
  `run_topological_sweep` reports metric diffs, and `compare_sweeps` compares
  run IDs. Use them to iteratively find a representative grid.

## NO HILL-CLIMBING OPTIMIZATION
The cosmic graph is a stable representation, not a score to maximize.
- **Do Not Optimize Graph Metrics**: Do not treat density, component count, or singleton fraction as quality metrics to optimize. They are descriptive indicators of scale, not success scores.
- **Seek Structural Persistence**: A topological pattern is "real" if it persists across many combinations of epsilon, projection dimensions, and seeds. Look for structural invariants across sweeps rather than "tuning" for a single perfect-looking graph.

## METHODOLOGY GUARDRAILS
- **Do Not Reintroduce Proxies of the Target**: If the question is "do features X distinguish outcome Y?", do not add columns that are near-proxies of Y (e.g. a geographic field that almost perfectly maps to the class). Clean clusters obtained that way are the proxy leaking the label, not the features distinguishing it — it silently abandons the experiment. Drop proxies along with the label.
- **Chaining vs. Clean H0 Slices**: Before using spectral clustering on a
  dominant component, audit `get_threshold_stability_curve`. If a stricter
  threshold candidate is `report_ready` or `balanced`, first run
  `generate_cluster_dossier(method="components",
  interpretation_edge_weight_threshold=<candidate>)` to read the natural H0
  components. Use `method="spectral"` when stricter threshold climbing only
  creates singleton/tiny-component dust, or when the question is latent
  structure inside a genuinely continuous dominant component.

## THRESHOLD MECHANICS
Two independent levers operate on the same underlying weighted matrix at
different stages. Do not confuse them.

- **Construction threshold** (`cosmic_graph.construction_threshold` in config):
  set BEFORE the sweep. Gates the persisted binary cosmic graph used for
  graph-connectivity diagnostics and visualization. The full weighted matrix
  is still retained. Too high → singletons. Too low → over-connected hairball.
  Default `"auto"` runs persistent homology stability analysis
  (`threshold_stability_summary` in the sweep response) and picks the longest
  stable plateau. Override only when `diagnose_cosmic_graph` measurements and
  the user's objective show that the current construction surface is the wrong
  scale, then re-run the sweep.
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
  affinity matrix. For repeated spectral analysis on a dense graph, use
  `diagnose_cosmic_graph` weight and scale fields to decide whether to estimate
  a spectral artifact. Pass an explicit value to deliberately diverge. Use
  `diagnose_cosmic_graph` weight percentiles
  (weight_p25–p95) to pick a value when overriding. This changes
  interpretation-time connectivity and cluster fragmentation; it does not
  rebuild the persisted cosmic graph.

When divergence is intentional, the JSON dossier response carries
`construction_threshold`, `interpretation_edge_weight_threshold`, and
`threshold_inherited`, plus `threshold_surface` guidance so you can see exactly
which surface clustering used.

## COSMIC GRAPH CONSTRUCTION (MINHASH)
The cosmic graph is built by an approximate, randomized recipe: each edge weight is
a MinHash estimate of the Jaccard similarity of two points' ball-sets, averaged over
`cosmic_graph.minhash_d` independent hash functions. This replaces the exact
O(Σ|B_c|²) co-occurrence count with an O(d·M) sketch. Keep the *construction* recipe
distinct from the *interpretation* layer (threshold/components/clustering above):
construction only has to faithfully approximate the weighted graph.

- Weights are unbiased with `Var = J(1−J)/d`, so accuracy depends only on `d` —
  **independent of dataset size**. Hoeffding: `P(|Ŵ−W|≥ε) ≤ 2e^{−2dε²}`.
- `minhash_d` is the only knob and rarely needs tuning. Default `256` gives a 95% CI
  of ±0.061 (worst case). Lower it on massive datasets to cut signature memory
  (`d·n·4` bytes) and construction time (linear in `d`); raise it for tighter weights.
- `characterize_dataset` surfaces a `minhash_advisory` (suggested `d`, with the
  memory/time win and CI cost) when `n` is large; `diagnose_cosmic_graph` reports the
  realized error (`minhash_profile`) for the `d` actually used. Set with
  `refine_config(..., cosmic_graph.minhash_d=<value>)`. Runs are reproducible via
  `cosmic_graph.minhash_seed`.
- For exact, bit-identical co-occurrence weights (at higher time/memory cost), set
  `cosmic_graph.construction="exact"` — it builds the sparse pseudo-Laplacian backbone
  directly with no estimation error (no `minhash_profile`). Default `"minhash"` is the
  right choice for routine exploration.

## PHASE III: CONTRASTIVE INTERPRETATION
10. Cluster: `generate_cluster_dossier`.
11. Contrast: Perform comparative analysis. Identify the 'Pivot Feature' —
   the variable that most cleanly separates Cluster A from its topological
   neighbors. Do not name in isolation.
12. Report: `export_html_report`. CRITICAL: YOU MUST pass synthesized,
   highly informative `cluster_names` based on Step 11. Names must be
   descriptive (e.g., 'Male Gentoos w/ Large Flippers'). Passing
   `cluster_names` as a flat JSON object like
   `{"0": "Male Gentoos w/ Large Flippers"}` is the difference between a raw
   Data Dump and a high-impact Research Paper.

## PHILOSOPHY
- Pulsar is a multi-scale aggregator, not a tuner. More grid points =
  more topological evidence. ALL ball maps are fused into ONE cosmic
  graph.
- Wide projection arrays and epsilon ranges are always superior to single
  points.
- The cosmic graph is evidence, not a score to maximize.

## PATH VISIBILITY (Claude Desktop)
Claude Desktop sandboxes are isolated from the server host. DO NOT use
chunked/base64 uploads, and do not use bash/cp as a bridge by default.
1. Ask the user for a host-visible absolute path, or call `get_runtime_context`
   to see `cwd` and `cache_dir`.
2. Only when the user has confirmed the server host can see `cache_dir`, copy
   the file there as the sandbox escape.
3. Call `ingest_dataset(path)` and use the returned `dataset_id`.

## SWEEP vs CLUSTER STATE
Any new `run_topological_sweep` discards the session's cluster labels and
feature-evidence cache. Cluster tools and exports must follow a fresh
`generate_cluster_dossier`. `run_id` on cluster tools is a provenance check,
not a way to recover old labels.
"""
