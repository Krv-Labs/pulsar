//! Opt-in workflow guide (`pulsar/mcp/prompts.py`). Not injected into the
//! server instructions: agents fetch it with `get_workflow_guide`.

/// The terse tool map every client sees in `initialize`.
pub const INSTRUCTIONS: &str = "Manifold discovery and topological data analysis for tabular datasets. Call \
`get_workflow_guide` once for the end-to-end procedure and tool map.\n\
Shared params across tools: `detail` ('summary' default; 'full' for audit/debug) and \
`response_format` ('json' for structured payloads; 'markdown' where a readable rendering is supported).\n\
Results carry structured content; errors set the MCP error flag with a JSON envelope \
(error_code, agent_action). Long sweeps: run_topological_sweep(execution='async') returns a run_id \
immediately; poll it with wait_job.";

pub const WORKFLOW_PROMPT: &str = r#"# Pulsar Topological Analysis Workflow

Reveal the dataset's topology; do not force convenient clusters.

## PHASE I: INGEST & CALIBRATE
1. Ingest: `ingest_dataset(path)` registers a host-visible CSV/Parquet file and returns a
   `dataset_id`. Prefer `dataset_id` everywhere; no other tool reads raw paths.
2. Calibrate: write the sweep config as raw YAML (see `pulsar schema config` for the format)
   and check it with `validate_config(config_yaml, dataset_id)`. It returns the normalized
   YAML and structured issues. The config is a broad first pass, not a final answer.

## PHASE II: EXECUTE & VALIDATE
3. Run: `run_topological_sweep(config_yaml, dataset_id)`. The response carries `run_id`, key
   metrics, the construction threshold, a threshold-stability summary and a finalization gate.
   - Default `execution='sync'` waits for the fit. On clients that support the MCP tasks
     extension the server returns a task handle instead and the client polls it transparently.
   - `execution='async'` returns `{status: "running", run_id}` at once; poll with
     `wait_job(run_id)` (blocks up to `timeout_s`, then reports progress) or `get_job(run_id)`.
     The completed sweep response is persisted, so `wait_job` returns the same payload the
     synchronous call would have.
4. Validate: `diagnose_cosmic_graph`. Read it as graph-state measurement, not a directive. Use
   `scale`, `component_morphology`, `sweep_support`, `observed_patterns` and `risk_factors`
   to judge the graph against the user's objective.
   - If `sweep_support.grid_adequacy_status` is `under_sampled` or `thin_grid`, treat
     structural claims as weak until a broader grid is checked.
   - COMPONENT CONTEXT: `component_count=1` can be a valid topology. Compare it with density,
     giant fraction, threshold stability, feature evidence and the user's objective.
   - SPARSIFICATION CONTEXT: `cosmic_graph.sparsify` builds a spectral sparsifier after the
     cosmic graph is constructed. It preserves spectral structure, not component topology;
     density and edge counts of sparsified runs are not comparable to others.
5. Compare: `get_sweep_history` and `compare_sweeps` after each refinement.
   `get_sweep_history(detail='summary')` distills patterns across persisted runs (which
   projection dimensions collapsed structure, where singletons spiked); you still own the
   next-config decision. Prefer 2-3 deliberate sweeps over one brittle perfect-looking run.
6. Inspect payloads summary-first. `get_topological_skeleton(detail='nodes')` gives the
   compact hub/bridge summary; `detail='edges'` capped raw edges; `detail='full'` nodes,
   edges and the config YAML.

## RESULTS AND ERRORS
Every result has a text view and structured content. Errors set the MCP error flag and
carry `error_code` plus `agent_action`: act on `agent_action` before retrying.

## RUN HANDLES AND PERSISTENCE
`run_topological_sweep` returns a `run_id`. Runs, their constructed graph and their full
weighted affinity are persisted in the store (`get_runtime_context` shows `cache_dir`), so
every model-reading tool accepts any completed `run_id`, on any transport, across server
restarts. Omit `run_id` to use the session's latest run. A run that is still `queued` or
`running` returns `RUN_NOT_COMPLETE` with its job state; an unknown id returns
`RUN_ID_UNKNOWN`. The in-memory model cache keeps the last `PULSAR_MCP_MODEL_CACHE_SIZE`
fitted models (default 3) for faster reads; evicted runs are reloaded from disk.

## GRID GEOMETRY & EXPERT AGENT AUTONOMY
The first config is only a baseline. Actively widen and shift the grid from diagnostic
feedback.
- **Broad first, then concentrate**: start wide across projection dimensions and epsilon;
  after the first run, compare sweeps and concentrate around the stable region.
- **"Widen and shift" over "narrow down"**: never optimise by collapsing to a single
  projection dimension. If a baseline grid collapses into one dominant component (high
  `giant_fraction`), shift the grid upwards (e.g. `[4, 6, 8, 10]`) rather than narrowing.
- **The projection floor**: avoid an ultra-low floor (1-3 dims) on complex high-dimensional
  data; it collapses structure and injects spurious consensus edges.
- **The epsilon gates**: keep epsilon inside the data's k-NN distance domain. Shattered
  graph: raise the upper bound. One dominant component: lower it or shift projections up.
- Projections above 16 dimensions fall back from the KD-tree to a linear scan in Ball
  Mapper; stay at or below 16 unless the user asks for more.

## NO HILL-CLIMBING OPTIMISATION
The cosmic graph is a stable representation, not a score to maximise. Density, component
count and singleton fraction are descriptive indicators of scale, not success scores. A
topological pattern is real if it persists across many combinations of epsilon, projection
dimensions and seeds.

## THRESHOLD MECHANICS
Two independent levers act on the same weighted matrix at different stages.
- **Construction threshold** (`cosmic_graph.construction_threshold`): set BEFORE the sweep.
  Gates the persisted cosmic graph used for connectivity diagnostics. The full weighted
  affinity is retained alongside it. Too high: singletons. Too low: hairball. Default
  `"auto"` runs H0 persistent-homology stability analysis and picks the longest stable
  plateau. Use `get_threshold_stability_curve` for threshold options; its
  `threshold_candidate_policy` is an interpretation lens, not an optimisation target:
  `balanced` (default), `report_ready` (before final cohort naming), `detail_seeking`
  (finer splits, exploratory), `outlier_mining` (anomalies and singleton tails only).
- **Interpretation threshold**: applied AFTER the graph is built when clustering or
  reporting slices the retained affinity. It never rebuilds the persisted cosmic graph.

## COSMIC GRAPH CONSTRUCTION (MINHASH)
Edge weights are MinHash estimates of the Jaccard similarity of two points' ball-sets over
`cosmic_graph.minhash_d` hash functions: unbiased with `Var = J(1-J)/d`, so accuracy depends
only on `d`, independent of dataset size. Default 256 gives a worst-case 95% CI of +-0.061.
Lower it on massive datasets to cut signature memory (`d*n*4` bytes) and time; raise it for
tighter weights. `diagnose_cosmic_graph` reports the realised error (`minhash_profile`).
`cosmic_graph.construction: exact` builds exact co-occurrence weights at higher cost.

## PATH VISIBILITY (sandboxed clients)
Sandboxes are isolated from the server host. Do not use bash/cp as a bridge by default:
ask the user for a host-visible absolute path, or call `get_runtime_context` to see `cwd`
and `cache_dir`. Only when the user confirms the server host can see `cache_dir`, copy the
file there, then `ingest_dataset(path)`.

## PHILOSOPHY
Pulsar is a multi-scale aggregator, not a tuner. More grid points mean more topological
evidence; all ball maps are fused into one cosmic graph. The cosmic graph is evidence, not
a score to maximise.
"#;
