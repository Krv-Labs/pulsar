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
   `dataset_id`. Prefer `dataset_id` everywhere; only `characterize_dataset` also accepts a raw
   `csv_path`.
1b. Look: `characterize_dataset(dataset_id)` returns row/feature counts, the PCA variance curve,
   k-NN scale and a ranked column preview (measurements only, plus a `minhash_advisory` above
   500,000 rows). `probe_columns(dataset_id, columns=[...])` drills into at most 20 named columns
   (sample values, missingness, range, top values).
2. Plan preprocessing: `draft_preprocessing_plan(dataset_id, target_columns=..., entity_column=...,
   time_column=...)` measures column evidence (shared missing masks, conditional columns, exact
   duplicates and sums, date/text kinds, and `evidence.target_proxies`, which measures whether
   missingness markers or category dummies encode a target or boolean event flag) and drafts a
   versioned plan (`plan_id`, `revision`) with labeled defaults (`decided_by="default"`) and open
   questions (`blank_meaning`, `code_columns`, `conditional_column`, `date_columns`,
   `duplicate_signal`, `mixed_values`, `mostly_blank`, `outcome_columns`). The plan is evidence plus
   labeled defaults, not a verdict; `stakes` (`neighbors_kept`, `feature_count_delta`) measure how
   much the map's k-NN neighborhoods move between options, not a score to optimize. Plans are
   persisted, so a `plan_id` keeps working across server restarts.
   - Read the questions, `evidence.target_proxies` and `warnings` (such as `TARGET_PROXY_COLUMNS`
     when `share_with_target == 1.0`) first. Record choices you can infer from context or schema via
     `edit_preprocessing_plan(plan_id, base_revision, answers=[...], edits=[...], source="agent")`.
   - Ask the user ONLY questions whose answer cannot be inferred and whose stakes are material.
     Never present the server's question text verbatim to a non-technical user; translate it into
     their domain vocabulary. Example (clinician, `blank_meaning` on `n_inpatient_claims` with no
     0s): ask "If a patient has no recorded hospital stays, does that mean they had none, or that we
     don't know?"; if they say "none", call `edit_preprocessing_plan(plan_id, base_revision=1,
     answers=[{"question_id": "<question_id>", "option_id": "<option_id>"}], source="user")` using
     the ids the payload lists.
   - To explore several plausible imputations, answer with `{"question_id": "<question_id>",
     "option_ids": ["<id1>", "<id2>"]}` or edit a column with `impute_method: ["fill_median",
     "sample_normal"]` / an `impute` candidate list (`method`, optional `value`, `seed`, boolean
     `indicator`). Each option keeps its indicator; selected options must agree on role, encoding
     and shared missingness groups. A top-level indicator edit overrides all candidates.
   - A stale `base_revision` returns `SESSION_REVISION_CONFLICT` naming the current revision.
   - To use the plan in a sweep, draft with `detail='full'` and copy `preprocessing_yaml` into the
     `preprocessing:` block of your config. Check it with
     `validate_preprocessing_config(config_yaml, dataset_id)`, which dry-runs only preprocessing.
3. Calibrate: `create_config(dataset_id, plan_id=..., intent=...)` writes a broad baseline sweep
   config. Epsilon and projection dimensions are calibrated in the processed feature space
   under the plan (`plan_id`; without one a default plan is drafted and stored, and a warning
   names any open questions or an edited plan it did not use). It returns the YAML, the plan
   provenance, the estimated ball-map count and the k-NN percentiles that bound a sensible
   epsilon, and stores the YAML as the session's active config. You can also write the config as
   raw YAML (see `pulsar schema config`) and check it with `validate_config(config_yaml,
   dataset_id)`, which returns the normalized YAML and structured issues. The config is a broad
   first pass, not a final answer: between sweeps, `refine_config(overrides=...)` shifts or
   concentrates the grid on the active config and returns a diff. Nested overrides set one leaf
   (`{"sweep": {"ball_mapper": {"epsilon": {"range": {"max": 1.0}}}}}`), short keys such as
   `projection_dimensions`, `projection_seeds`, `epsilon_values`, `n_reps` set a whole
   parameter, and `{"remove_keys": ["preprocessing.encode.species"]}` deletes entries. A refined
   config that would not validate is rejected and the active config is left unchanged.

## PHASE II: EXECUTE & VALIDATE
4. Run: `run_topological_sweep(config_yaml, dataset_id)`. The response carries `run_id`, key
   metrics, the construction threshold, a threshold-stability summary and a finalization gate.
   - Default `execution='sync'` waits for the fit. On clients that support the MCP tasks
     extension the server returns a task handle instead and the client polls it transparently.
   - `execution='async'` returns `{status: "running", run_id}` at once; poll with
     `wait_job(run_id)` (blocks up to `timeout_s`, then reports progress) or `get_job(run_id)`.
     The completed sweep response is persisted, so `wait_job` returns the same payload the
     synchronous call would have.
5. Validate: `diagnose_cosmic_graph`. Read it as graph-state measurement, not a directive. Use
   `scale`, `component_morphology`, `sweep_support`, `observed_patterns` and `risk_factors`
   to judge the graph against the user's objective.
   - If `sweep_support.grid_adequacy_status` is `under_sampled` or `thin_grid`, treat
     structural claims as weak until a broader grid is checked.
   - COMPONENT CONTEXT: `component_count=1` can be a valid topology. Compare it with density,
     giant fraction, threshold stability, feature evidence and the user's objective.
   - SPARSIFICATION CONTEXT: `cosmic_graph.sparsify` builds a spectral sparsifier after the
     cosmic graph is constructed. It preserves spectral structure, not component topology;
     density and edge counts of sparsified runs are not comparable to others.
6. Compare: `get_sweep_history` and `compare_sweeps` after each refinement.
   `get_sweep_history(detail='summary')` distills patterns across persisted runs (which
   projection dimensions collapsed structure, where singletons spiked); you still own the
   next-config decision. Prefer 2-3 deliberate sweeps over one brittle perfect-looking run.
7. Inspect payloads summary-first. `get_topological_skeleton(detail='nodes')` gives the
   compact hub/bridge summary; `detail='edges'` capped raw edges; `detail='full'` nodes,
   edges and the config YAML.

## PHASE III: INTERPRET
8. Cluster: `generate_cluster_dossier(detail='summary')` for routine exploration; treat
   `detail='standard'` and `detail='full'` as late-stage dossier modes. It clusters the latest
   run, persists a `cluster_assignment_id` (method, interpretation threshold, labels) and
   publishes the labels into the session. If a stricter threshold candidate from
   `get_threshold_stability_curve` is `report_ready` or `balanced`, first try
   `method='components', interpretation_edge_weight_threshold=<candidate>` to read the natural
   H0 components; use `method='spectral'` when stricter thresholds only shed singleton dust, or
   for latent structure inside one continuous dominant component.
   Any new `run_topological_sweep` discards the session's cluster labels: dossiers can only be
   published for the latest run (`SESSION_REVISION_CONFLICT` otherwise).
9. Contrast: read the published labels with `get_cluster_profile` (one cluster),
   `get_feature_signal` (chosen features across clusters), `get_cluster_signal_matrix`
   (all features at a glance) and `compare_clusters` (two clusters, per-feature tests). Identify
   the 'Pivot Feature' that most cleanly separates a cluster from its topological neighbours;
   do not name clusters in isolation. These tools read only the latest dossier: after a new
   sweep they return `CLUSTERS_NOT_BUILT`, and `run_id` is a provenance check
   (`CLUSTERS_STALE`), not a way to recover old labels.

## LONGITUDINAL DATA (PANELS)
Long-format data (one row per entity and time step) is the wrong input for the static sweep,
which treats every row as an independent point. Ingest the long-format table, optionally
`sample_longitudinal_entities` (a deterministic entity sample, registered as a new Parquet
`dataset_id`), `create_config`, then `build_longitudinal_graph(dataset_id, entity_column,
time_column, config_yaml)`. It returns a `longitudinal_id`; omit it on later calls to use the
most recent panel. `representation='trajectory'` (default) builds a sparse graph over
`(entity, t)` observations whose edges may cross time steps; `'temporal'` builds a dense
`(n, n, T)` entity tensor bounded by `PULSAR_MCP_MAX_TENSOR_BYTES`, so check `build_cost`
first; `'both'` builds each. The build refits the projection grid and may recalibrate epsilon
to the panel's k-NN domain: read `config_adaptation`. `time_column` must already be discrete;
`on_missing` is `drop_entity` (default; required for temporal), `forward_fill` or
`allow_ragged` (trajectory only).
Then `diagnose_longitudinal_graph` (measurements and the threshold ladder; for temporal, six
aggregations side by side with `why`/`best_for`/`avoid_for`), `get_trajectory_archetypes`
(identical cluster sequences), `get_cross_time_neighbors` (lookalikes at other times; `t` is
an original time label, returned as `time_label`) and `classify_trajectories` (rule- or
distance-based classes; `detail='full'` for every entity). Thresholds on these tools are
interpretation filters, not the construction threshold, and cluster counts move steeply with
them: read the ladder. The session keeps the four newest panels in memory; an older handle
returns `LONGITUDINAL_NOT_LOADED` with what to rebuild. Longitudinal panels do not feed the
cluster dossier chain and take no `run_id`.

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
- Projection dimension is a sweep axis: each dimension is a different lens. JL scales
  every dimension to the same expected distances, so one epsilon list serves the grid.
  JL draws its matrix row by row at the grid's largest dimension, so raising the grid's
  largest dimension changes the draws at every other dimension for the same seed; adding
  a dimension below it leaves existing embeddings unchanged. Above 16 dimensions Ball
  Mapper swaps its KD-tree for a linear scan: same membership, different speed.

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
  In `generate_cluster_dossier`, `method='auto'` and `'components'` inherit the construction
  threshold by default, so clustering reads the surface you diagnosed; explicit
  `method='spectral'` defaults to 0.0, the full weighted affinity.

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
