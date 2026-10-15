.. _mcp:

==================
MCP Server
==================

**No Code. Just Data and Insight.**

The Pulsar MCP (Model Context Protocol) server lets AI clients—Claude, Gemini, Cursor, and others—analyze your data autonomously. You don't need to write code, tune parameters, or understand Ball Mapper. Just point the AI at your CSV and ask for the story.

This guide is for **domain experts** who know what their data *means* but don't want to write scikit-learn pipelines.

Workflow Comparison
-------------------

.. list-table::
   :widths: 30 25 25 20
   :header-rows: 1

   * - Approach
     - You Do
     - AI Does
     - Speed
   * - **YAML-Driven** (manual)
     - Write YAML, run pipeline
     - (nothing)
     - Depends on grid size
   * - **Programmatic** (Python)
     - Write Python, orchestrate
     - (nothing)
     - Depends on grid size
   * - **MCP + Claude** (recommended)
     - Point AI at CSV, ask question
     - Entire analysis workflow
     - ~2–30s (automated tuning)

MCP is the fastest path from "I have a CSV" to "Here's what it means."

The Value Prop
--------------

**Traditional clustering** (K-means, DBSCAN):
- You guess the number of clusters
- Algorithm forces your data into spheres
- You get a silhouette plot and hope for the best

**Pulsar with Claude**:
- Claude probes your data's geometry automatically
- Finds true topological structure (manifolds, voids, networks)
- Generates a statistical dossier (z-scores, trait profiles, separation metrics)
- You read the story, not a confusion matrix

Claude handles the heavy lifting: measuring column evidence, drafting a versioned preprocessing plan (asking you plain-language questions only when domain meaning is ambiguous), calibrating parameters, and iterating on the sweep.

Setup
-----

Easy install (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then register Pulsar with every agent tool you use:

.. code-block:: bash

   uvx --from thema-pulsar pulsar install

The installer detects Claude Code, Claude Desktop, Codex CLI, Gemini CLI, GitHub Copilot CLI, Cursor, VS Code, and Google Antigravity. It writes the ``pulsar`` MCP entry each client expects, using an absolute path to ``uvx`` so GUI-launched apps can spawn the server. Inspect registrations with ``uvx --from thema-pulsar pulsar status``; remove them cleanly with ``uvx --from thema-pulsar pulsar uninstall``.

The prompt talks to your terminal directly, so it still appears when output is
redirected (``pulsar install | tee install.log``). Where escape codes will not
render (``TERM=dumb``) it falls back to a numbered list. Where there is no
terminal at all — CI, a systemd unit — pass explicit harness ids (``claude``,
``claude-desktop``, ``codex``, ``gemini``, ``copilot``, ``cursor``, ``vscode``,
``antigravity``) or ``--all``:

.. code-block:: bash

   uvx --from thema-pulsar pulsar install --all
   uvx --from thema-pulsar pulsar uninstall --all --yes

Manual per-client setup
~~~~~~~~~~~~~~~~~~~~~~~

Pulsar ships an MCP server entry point (``pulsar-mcp``) via the ``mcp`` extra of the published ``thema-pulsar`` package. You do **not** need to clone the repo or create a Python environment. If you prefer to configure clients by hand, choose your client below. ``uvx`` downloads the released package from PyPI and runs it in an isolated environment.

.. note::
  Pulsar works with any MCP-capable client, including Cursor and Gemini CLI, where you can add Pulsar as an MCP server/tool.

.. note::
  **The first launch can take a little while.** Pulsar is a substantial scientific server, and ``uvx`` may need to download the package and initialize its numerical dependencies before MCP tools appear. Wait for it to finish before retrying; later launches normally reuse uv's cache.

  The Gemini command below sets ``--timeout 60000`` (60,000 milliseconds, or 60 seconds) so this first boot can complete. Clients that expose no timeout setting may give up before the download finishes — for those, either run the ``uvx`` command once in a terminal to warm uv's cache before starting the client, or use a persistent install (see `Alternative install methods`_) so there is nothing to download at launch.

.. tab-set::

   .. tab-item:: Claude Desktop

      Open ``~/Library/Application Support/Claude/claude_desktop_config.json`` (macOS) or ``%APPDATA%\Claude\claude_desktop_config.json`` (Windows) and add:

      .. code-block:: json

         {
           "mcpServers": {
             "pulsar": {
               "command": "uvx",
               "args": ["--from", "thema-pulsar[mcp]", "pulsar-mcp"]
             }
           }
         }

      Restart Claude Desktop. A hammer icon in new chats confirms the tools loaded.

      .. note::
         GUI-launched apps on macOS often don't inherit your shell ``PATH``. If Claude can't find ``uvx``, replace ``"command": "uvx"`` with its absolute path (find it with ``which uvx``, e.g. ``/Users/yourname/.local/bin/uvx``).

   .. tab-item:: Gemini CLI

      .. code-block:: bash

         gemini mcp add --scope user --timeout 60000 pulsar uvx -- --from "thema-pulsar[mcp]" pulsar-mcp

      The ``--`` matters: it stops Gemini's own argument parser from consuming ``--from``. This user-scoped registration works from any directory and makes Pulsar available in every *trusted* Gemini workspace (see the trust note below). Use ``--scope project`` instead to limit it to the current project. The longer timeout gives ``uvx`` time to download and initialize Pulsar on its first launch.

      Launch Gemini from the workspace containing your data, then run ``/mcp list`` to confirm Pulsar is connected.

      .. note::
         **Workspace trust gates stdio MCP servers.** As of Gemini CLI 0.53, Gemini does not start stdio MCP servers in an untrusted workspace, so a user-scoped Pulsar registration still stays dormant until you trust the folder. Choose **Trust folder** when first prompted; if you previously declined, run ``/permissions trust`` inside Gemini and select **Trust folder**. Let Gemini relaunch — running sessions do not reload external settings changes, so if it does not relaunch on its own, fully exit and start it again.

         See `Gemini CLI folder trust <https://google-gemini.github.io/gemini-cli/docs/cli/trusted-folders.html>`_ for what untrusted mode restricts and how to review your trust decisions. Note that on 0.53 the trust prompt appears without any opt-in, so ignore that page's claim that the feature is disabled by default.

   .. tab-item:: Claude Code

      .. code-block:: bash

         claude mcp add pulsar -- uvx --from "thema-pulsar[mcp]" pulsar-mcp

   .. tab-item:: Cursor / Windsurf

      Open **Settings → Features → MCP → Add new MCP server**:

      - Name: ``pulsar``
      - Type: ``command``
      - Command: ``uvx --from "thema-pulsar[mcp]" pulsar-mcp``

Alternative install methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer a persistent install over ephemeral ``uv`` invocations:

.. code-block:: bash

   pipx install "thema-pulsar[mcp]"
   pulsar install --mode pipx

Or ``pip install "thema-pulsar[mcp]"`` in any venv, then point clients at ``pulsar-mcp`` directly (or run ``pulsar install --mode pipx`` from that environment).

Developing against a local clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributors working on the Pulsar source can launch the server from a checkout instead:

.. code-block:: bash

   uv sync --extra mcp
   uv run pulsar-mcp

   # or, equivalently, using the dev dependency-group:
   uv run --group mcp pulsar-mcp

Point your MCP client at ``uv run --group mcp pulsar-mcp`` (with ``cwd`` set to the clone) for live-edit development.

Workflow
--------

Once connected, give the AI a goal rather than instructions. The AI already knows the technical steps.

**The recommended prompt:**

   *"I have a dataset at* ``path/to/data.csv``\ *. Use Pulsar to find the hidden structure and tell me the story of this data. I'm looking for meaningful subgroups and the specific traits that define them."*

Under the hood the AI will:

1. **Characterize geometry** — probe k-NN distances and projection dimensions to ground parameter choices; on large datasets, may include a ``minhash_advisory`` with suggested ``cosmic_graph.minhash_d`` that agents can apply via ``refine_config`` before expensive sweeps
2. **Draft a preprocessing plan** — measure column evidence, apply labeled defaults, and surface plain-language questions ranked by how much each choice moves the map's neighborhoods
3. **Resolve open preprocessing questions** — record what can be inferred from context (``source="agent"``) and ask you in plain language only when domain meaning is genuinely ambiguous (``source="user"``)
4. **Calibrate and run a topological sweep** — build a calibrated baseline config from the plan and find the most stable version of the data's shape
5. **Iterate on the sweep grid** — diagnose graph morphology and tune projection dimensions or epsilon when results are degenerate
6. **Generate a Dossier** — statistical profiles of each discovered subpopulation

Available MCP Tools
-------------------

The server exposes these tools to the AI client, which chains them together automatically. Call ``get_workflow_guide`` once at the start of a session for the full opinionated procedure — the tables below are a reference, not a sequence.

Core loop
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **get_workflow_guide**
     - Returns the full end-to-end procedure as markdown. Opt-in — not injected into every session automatically.
   * - **ingest_dataset**
     - Registers a host-visible CSV/Parquet path and returns the stable ``dataset_id`` every other tool keys off of.
   * - **characterize_dataset**
     - Quick exploratory summary: k-NN distances (is your data sparse or dense?), projection dimensions, missing value patterns. On large datasets (``n`` above the advisor threshold), may include ``minhash_advisory`` with suggested ``minhash_d``, memory estimate, and error bounds.
   * - **create_config**
     - Builds a baseline sweep config calibrated against the *processed* feature space (after preprocessing + scaling) for a given ``dataset_id`` and optional ``plan_id`` from ``draft_preprocessing_plan`` (or drafts baseline defaults when ``plan_id`` is omitted, emitting a ``PREPROCESSING_QUESTIONS_OPEN`` warning if open questions remain). The mandatory first config-generation step — never let the agent guess parameters.
   * - **run_topological_sweep**
     - Execute the full Pulsar pipeline: imputation → projection → Ball Mapper → cosmic graph, from inline YAML or a session's active config. Returns structured JSON with metrics and experiment diff. Config persistence is opt-in. Results cached per session.
   * - **diagnose_cosmic_graph**
     - Current graph-state observables: scale, component morphology, weight distribution, sweep support, observed patterns, and risk factors. When ``construction: minhash``, includes ``minhash_profile`` (realized weight accuracy and confidence interval at the configured signature depth). The agent interprets these measurements against the user's objective.
   * - **refine_config**
     - Applies targeted dotted-path overrides (or key deletions) to a config instead of regenerating the whole YAML — the normal way to iterate after a diagnosis.
   * - **generate_cluster_dossier**
     - Computes clusters on the interpretation-threshold slice and returns per-cluster statistical evidence: trait profiles, homogeneity scores, separation metrics, concentration measures, plus clustering method metadata (method used, silhouette score). Persists a ``cluster_assignment_id``.
   * - **compare_clusters**
     - Pairwise statistical tests (Welch's t-test, Kolmogorov-Smirnov, Cohen's d, effect sizes) between two cached clusters. Answers "Are these really different, or just noise?"
   * - **export_labeled_data**
     - Write a CSV of your original dataframe with labels (and the cluster names you supply) from a specific ``cluster_assignment_id`` emitted by ``generate_cluster_dossier``. The CSV is ready for downstream analysis, visualization, or handoff to domain experts.

Preprocessing helpers
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **probe_columns**
     - Per-column drilldown (≤20 columns per call): sample values, missingness, distributions. Use after ``characterize_dataset`` when a specific column needs a closer look.
   * - **draft_preprocessing_plan**
     - Measures column-level evidence (shared missing masks, conditional missingness, exact duplicates and sums, date/categorical/ID kinds) and stores a versioned ``PreprocessingPlan`` (``plan_id``, ``revision=1``) in the session with labeled defaults and open plain-language questions ranked by measured ``stakes``.
   * - **edit_preprocessing_plan**
     - Applies question ``answers`` and/or per-column ``edits`` to an existing plan at ``base_revision``, records who decided each change (``source="agent"`` or ``"user"``), dry-runs ``preprocess_dataframe`` to validate the schema, and returns the updated revision and ``preprocessing_yaml``.
   * - **validate_preprocessing_config**
     - Dry-run only the preprocessing stage against the session data — no projection, no sweep cost. Returns PASS with a schema summary, or a structured error pointing to ``edit_preprocessing_plan`` or YAML edits.
   * - **validate_config**
     - Validates and normalizes a complete config into canonical YAML, independent of running a sweep.

**Tool contracts and payload headlines**

- **draft_preprocessing_plan**:

  - *Purpose*: Measures dataset evidence without hard-coded domain rules and drafts a versioned preprocessing plan where every choice is labeled with ``decided_by`` (``"default"``, ``"agent"``, or ``"user"``) and ``status`` (``"resolved"`` or ``"open"``). Open choices carry plain-language question cards across five topics (``blank_meaning``, ``conditional_column``, ``date_columns``, ``duplicate_signal``, ``outcome_columns``) sorted by measured ``stakes`` (``neighbors_kept``, the k-NN neighborhood Jaccard overlap on a deterministic row sample vs. the default option, plus ``feature_count_delta``).
  - *Parameters*: ``dataset_id``, ``target_columns=[]``, ``entity_column=""``, ``time_column=""``, ``stakes=True``, ``detail="summary" | "full"``, ``question_limit=8``, ``evidence_limit=12``, ``response_format="markdown" | "json"``.
  - *Payload headline*: ``plan_id``, ``dataset_id``, ``revision``, ``plan_fingerprint``, ``counts`` (``columns``, ``features``, ``dropped``, ``open_entries``, ``open_questions``, ``decided_by``), ``processed_space`` (``feature_count``, ``missing_marker_count``, ``dummy_column_count``), capped ``questions`` (with ``options``, ``default_option_id``, and ``stakes``), ``evidence`` (``missing_groups``, ``mask_dependencies``, ``relations``, ``rare_binary_leverage``, ``constant_columns_preview``, ``datetime_columns_preview``, ``integer_coded_preview``), ``weak_hints`` (``event_partner_columns``), ``preprocessing_yaml``, ``cost``, ``how_to_use``, and ``next_tools`` (``["edit_preprocessing_plan", "create_config", "probe_columns"]``). ``detail="full"`` adds per-column ``entries`` and uncapped lists.

- **edit_preprocessing_plan**:

  - *Purpose*: Updates a stored plan at ``base_revision`` with question ``answers`` or direct column ``edits`` while tracking decision provenance (``source="agent"`` when inferred from context or schema; ``source="user"`` when relaying a human answer). Stale ``base_revision`` values fail with ``SESSION_REVISION_CONFLICT``; invalid dry-run schemas return ``validation: {valid: false, error: ...}`` so the agent can adjust before running a sweep.
  - *Parameters*: ``plan_id``, ``base_revision``, ``answers=[{question_id, option_id}]``, ``edits=[{column, role?, impute_method?, impute_value?, indicator?, encode?}]``, ``source="agent" | "user"``, ``response_format="markdown" | "json"``.
  - *Payload headline*: ``plan_id``, ``revision``, ``plan_fingerprint``, ``changes`` (capped diff of ``{column, field, was, now}``), ``counts``, ``processed_space``, ``validation`` (``{valid, error}``), ``open_questions_preview``, ``preprocessing_yaml``, and ``next_tools``.

**Example question/answer flow**

The plan is evidence plus labeled defaults, not a verdict, and ``stakes`` measure how much the map moves rather than a score to optimize. Agents resolve what they can infer from context (``source="agent"``) and ask the user only high-stakes questions whose answer cannot be inferred, rephrased in the user's own domain language:

1. ``draft_preprocessing_plan(dataset_id="ds_1")`` reports missing group ``mg1`` (``n_inpatient_claims``, ``inpatient_days``, ``inpatient_payment``) where all three columns are blank on the exact same rows and never record a ``0``. It opens question ``q1`` (topic ``blank_meaning``, ``stakes.neighbors_kept: 0.41``) with options ``none`` (fill with ``0``, no missingness marker) and ``unknown`` (default: ``fill_median`` plus shared marker ``n_inpatient_claims_and_2_more_was_missing``).
2. Rather than repeating server wording verbatim, the agent asks the clinician in domain terms: *"If a patient has no recorded hospital stays, does that mean they had none, or that we don't know?"*
3. The clinician replies that blanks mean zero stays. The agent records the answer with ``source="user"``:

   .. code-block:: python

      edit_preprocessing_plan(
          plan_id="pp_9f2c1a04b8",
          base_revision=1,
          answers=[{"question_id": "q1", "option_id": "none"}],
          source="user",
      )

4. The agent passes the updated ``plan_id`` to ``create_config(dataset_id="ds_1", plan_id="pp_9f2c1a04b8")``, which embeds ``# preprocessing plan pp_9f2c1a04b8 revision 2 (<fingerprint>)`` and the resolved ``preprocessing:`` block into the sweep config. (Calling ``create_config`` without ``plan_id`` drafts baseline defaults automatically and emits a ``PREPROCESSING_QUESTIONS_OPEN`` warning when open questions remain.)

**Preprocessing config reference**

In addition to ``fill_mean``, ``fill_median``, ``fill_mode``, ``sample_normal``, ``sample_categorical``, and ``one_hot``, the ``preprocessing:`` YAML block supports:

- ``impute.<column>.method: fill_value`` with ``value: <number | string>`` — replaces missing entries with a finite numeric constant (for numeric columns, e.g. ``{method: fill_value, value: 0}``) or a literal category label (for string/categorical columns before ``one_hot``, e.g. ``{method: fill_value, value: "(blank)"}``). Works even when a column has 100% missing values.
- ``impute.<column>.indicator: true | false`` (default ``true``) — controls whether ``preprocess_dataframe`` emits a per-column ``<column>_was_missing`` binary indicator. Per-column indicators are emitted only when ``indicator: true`` and the column does not belong to any group in ``missing_indicator_groups``.
- ``missing_indicator_groups: {<group_name>: [<col1>, <col2>, ...]}`` — emits a single shared ``<group_name>_was_missing`` (``0.0`` / ``1.0``) column indicating whether any listed column was missing in the input data. Evaluated *before* ``drop_columns`` so dropped columns (such as raw dates) can still contribute a presence/absence flag without duplicating identical per-column markers across a block.

.. code-block:: yaml

   # preprocessing plan pp_9f2c1a04b8 revision 2 (a1b2c3d4e5f6)
   preprocessing:
     drop_columns: ["patient_id", "death_date"]
     missing_indicator_groups:
       outpatient_visits_and_1_more: ["outpatient_visits", "outpatient_cost"]
     impute:
       n_inpatient_claims: {method: fill_value, value: 0, indicator: false}
       inpatient_days: {method: fill_value, value: 0, indicator: false}
       outpatient_visits: {method: fill_median, indicator: false}
       outpatient_cost: {method: fill_median, indicator: false}
       race: {method: fill_value, value: "(blank)", indicator: false}
     encode:
       race: {method: one_hot}

Sweep history and graph structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **get_sweep_history**
     - The session's sweep table plus pattern synthesis across runs — what's been tried and how outcomes shifted.
   * - **compare_sweeps**
     - Markdown diff of two persisted runs' config and metrics.
   * - **get_threshold_stability_curve**
     - The H0 persistence-stability curve, structural breakpoints, and candidate threshold "lenses" (balanced / report-ready / detail-seeking / outlier-mining) for choosing ``construction_threshold`` deliberately instead of guessing.
   * - **get_topological_skeleton**
     - Graph connectivity structure — hub/bridge summary, capped node/edge preview — for the live run or a stored artifact.
   * - **create_graph_artifact**
     - Estimates or builds a read-only spectral-sparsifier artifact from the current live run. Costly and opt-in; defaults to ``estimate_only=True`` so the agent can see the cost before paying it.

Cluster interpretation
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **get_cluster_profile**
     - Targeted evidence for a single cluster from the cached dossier state.
   * - **get_feature_signal**
     - Cross-cluster evidence for specific named features (numeric or categorical).
   * - **get_cluster_signal_matrix**
     - Cross-cluster signal matrix ranked by discriminative power — which features actually separate the clusters.

Export
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **export_html_report**
     - Renders a full standalone HTML report from the cached dossier. Pass ``cluster_names`` for readable output; falls back to auto-generated names otherwise.
   * - **export_dataset_bundle**
     - Writes a Parquet bundle (raw/clean tabular data, graph nodes/edges/cosmic/groups) plus an ``export_manifest.json`` for downstream pipelines.

Longitudinal and temporal panels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For entity × time data (patients over visits, sensors over readings). Requires ``create_config`` first for calibration.

Longitudinal builds apply configured numeric imputation and ``drop_columns``
before constructing the panel. Entity and time columns remain panel keys even
when listed in ``drop_columns``; imputation must not target either key. Explicit
``feature_columns`` must not include a dropped column. Text redaction markers
are not converted to missing values for imputation.

``on_missing`` still determines how incomplete histories are handled. With
``allow_ragged``, any remaining missing feature values produce a structured
error before calibration. Numeric longitudinal features are supported;
categorical encoding is not applied to them.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **sample_longitudinal_entities**
     - Selects a deterministic subset of entities while retaining all their observations. Returns a new ``dataset_id`` backed by Parquet, preserving column types. Generate a config for that sample before building its graph.
   * - **build_longitudinal_graph**
     - Pivots a long-format table into a panel and builds a ``trajectory`` (sparse observation-node) and/or ``temporal`` (dense entity × feature × time tensor) representation. Returns a ``longitudinal_id``.
   * - **diagnose_longitudinal_graph**
     - Measurements for whichever representation was built: trajectory similarity/cross-time stats, or the temporal tensor's aggregation modes side by side.
   * - **get_trajectory_archetypes**
     - Groups entities by the cluster-sequence they trace over time (trajectory surface only), with a threshold ladder.
   * - **get_cross_time_neighbors**
     - Finds observations at other time steps resembling a given ``(entity, time)`` observation.
   * - **classify_trajectories**
     - Classifies entity trajectories into cohorts via complexity (Shannon entropy), transition (Markov self-retention), sequence (structural path type), Levenshtein, or DTW distance. Returns a per-class summary and a 25-entity preview; pass ``detail="full"`` for every entity.

Meta
~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **get_runtime_context**
     - Server cwd, cache directory, session state, which runs the in-memory model cache holds, and path-visibility guidance — useful for sandboxed clients that need to know what the server can actually see on disk.

Result Shape and Errors
-----------------------

Every tool returns **two views of the same result**: a text view (markdown or JSON, per the tool's ``response_format``) for the model to read, and ``structured_content``, a JSON object validated against the tool's declared output schema. Clients that parse results programmatically should read the structured content; the text is identical in substance.

Every tool also advertises MCP annotations (read-only, idempotent, destructive, and ``openWorldHint: false`` because the server only touches local files), an output schema, and a description for every parameter.

Cluster state is tied to the latest sweep: any new ``run_topological_sweep`` discards the session's cluster labels and feature-evidence cache, so cluster tools and exports must follow a fresh ``generate_cluster_dossier``. ``run_id`` on cluster tools is a provenance check, not a way to recover old labels.

Responses larger than ``PULSAR_MCP_MAX_RESPONSE_BYTES`` are turned into a ``RESPONSE_TOO_LARGE`` error envelope (see below) rather than truncated, so output schemas stay advertised and every payload stays parseable.

Failures set the MCP **error flag** (``isError``) on the result and carry a JSON envelope in both views:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``status``
     - Always ``"error"``.
   * - ``tool``
     - The tool that failed.
   * - ``reason``
     - Human-readable cause.
   * - ``error_code``
     - Stable machine-readable code, for example ``RUN_ID_UNKNOWN``, ``CLUSTERS_STALE``, ``SESSION_REVISION_CONFLICT``, ``MODEL_NOT_LOADED``, or ``INVALID_ARGUMENT``.
   * - ``agent_action``
     - A concrete next step. Act on it before retrying.
   * - ``details`` / ``metrics``
     - Handle, path, and measurement context relevant to the failure.

Run Handles and the Model Cache
-------------------------------

``run_topological_sweep`` returns a ``run_id``. Tools that read the fitted model (diagnostics, threshold curve, cluster profiles, exports) accept an optional ``run_id``:

- Omit it to use the session's latest run.
- Pass it to read an **earlier run**, for example to compare a graph before and after a refinement.
- Pass it on **stateless transports**, where "latest" cannot be remembered between requests.

Fitted models are kept in a process-wide LRU cache holding the last ``PULSAR_MCP_MODEL_CACHE_SIZE`` runs (default 3). A run that has been evicted, or never existed, returns ``RUN_ID_UNKNOWN``; re-run the sweep. ``get_runtime_context`` lists the cached ``run_id`` values. A cached run holds its model and data in memory until it is evicted, so raise the cache size deliberately on large datasets.

Tools that **publish** session state (``generate_cluster_dossier``, ``create_graph_artifact``) accept only the latest run and refuse an older one with ``SESSION_REVISION_CONFLICT``. Cluster labels remain session-scoped; cluster tools report ``clusters_run_id`` so you can see which run the labels describe. Longitudinal tools read session-only longitudinal artifacts (``longitudinal_id``), not the fitted model, and take no ``run_id``.

Transport Continuity
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Transport
     - What persists between calls
   * - **stdio** (default)
     - One client, one session: cluster labels, the latest-run pointer, longitudinal artifacts, and fitted models all persist.
   * - **Streamable HTTP with sessions** (legacy clients that receive an ``mcp-session-id``)
     - The same as stdio, per session.
   * - **Streamable HTTP, stateless** (protocol revision 2026-07-28)
     - Each request gets a fresh session. Fitted models survive through the model cache, so pass ``run_id``. Cluster labels and longitudinal artifacts do not; rebuild them in the same chain of calls. ``dataset_id`` handles live in the on-disk registry and always work.

Check ``get_runtime_context`` (``transport_assumption``, ``model_continuity``, ``model_cache``) if unsure. Model state is in memory only and is lost when the server restarts.

Environment Variables
---------------------

Server operators tune the MCP server through environment variables, set in the MCP client's ``env`` block or the shell that launches ``pulsar-mcp``.

.. list-table::
   :header-rows: 1
   :widths: 38 14 48

   * - Variable
     - Default
     - Effect
   * - ``PULSAR_MCP_TOOL_TIMEOUT_S``
     - ``1800``
     - Per-call timeout in seconds on long tools (sweep, longitudinal build, dossier, characterize, HTML report, dataset bundle).
   * - ``PULSAR_MCP_MAX_RESPONSE_BYTES``
     - ``100000``
     - Backstop on a single response's size, measured as the text view plus the serialized structured content together (so roughly 2x the text an agent reads; about 25k tokens combined). ``0`` or a negative value disables it. A larger result is replaced, never truncated, by an ``is_error`` envelope with ``error_code: RESPONSE_TOO_LARGE`` (``details.size_bytes`` / ``details.limit_bytes``); retry with ``detail="summary"``, lower ``max_*`` / preview limits, or a narrower tool.
   * - ``PULSAR_MCP_MODEL_CACHE_SIZE``
     - ``3``
     - Number of fitted runs kept addressable by ``run_id`` (minimum 1).
   * - ``PULSAR_MCP_MAX_CONCURRENT_HEAVY``
     - ``1``
     - Concurrent heavy computations (sweeps, builds) admitted at once; others queue.
   * - ``PULSAR_MAX_SESSIONS``
     - ``3``
     - Maximum live client sessions held in memory; the least recently used is evicted beyond this.
   * - ``PULSAR_MCP_ENABLE_TASKS``
     - off
     - Set to ``1``, ``true``, or ``yes`` to let task-capable clients run ``run_topological_sweep`` in the background (requires ``fastmcp[tasks]``).
   * - ``PULSAR_MCP_ENABLE_UPLOAD``
     - off
     - Set to ``1`` to expose the chunked upload tools (``begin_dataset_upload``, ``append_dataset_chunk``, ``finalize_dataset_upload``) for sandboxed clients that cannot share a path.
   * - ``PULSAR_MCP_MAX_TENSOR_BYTES``
     - ``2 GiB``
     - Peak working-set ceiling for the dense ``temporal`` longitudinal representation.

Example: Palmer Penguins
------------------------

The Palmer Penguins dataset (344 birds, 3 species, 3 islands) is a useful benchmark because the correct answer is known. Running an unsupervised sweep—dropping species labels entirely—recovers the biology.

**Getting the data**

The dataset is not bundled with Pulsar. Export it to CSV with either of these one-liners:

.. code-block:: python

   # Option A: palmerpenguins package
   # pip install palmerpenguins
   import palmerpenguins
   palmerpenguins.load_penguins().to_csv("demos/penguins/penguins.csv", index=False)

.. code-block:: python

   # Option B: seaborn (no extra install if already present)
   import seaborn as sns
   sns.load_dataset("penguins").to_csv("demos/penguins/penguins.csv", index=False)

.. code-block:: yaml

   run:
     name: penguin_species_recovery_dim5
     data: "demos/penguins/penguins.csv"
   preprocessing:
     drop_columns: ["species", "rowid", "year"]
     encode:
       island: {method: one_hot}
       sex: {method: one_hot}
     impute:
       bill_length_mm: {method: fill_mean}
       bill_depth_mm: {method: fill_mean}
       flipper_length_mm: {method: fill_mean}
       body_mass_g: {method: fill_mean}
   sweep:
     projection:
       method: jl
       dimensions:
         values: [5]
       seed:
         values: [42]
       center: true
     ball_mapper:
       epsilon:
         range: {min: 0.80, max: 1.50, steps: 15}
   cosmic_graph:
     construction: minhash
     construction_threshold: auto

The resulting graph shattered into components along **island** and **sex** boundaries, not just species—revealing that habitat and morphological sex are geometrically dominant. Chinstraps on Dream Island were structurally indistinguishable from Adelies on the same island: the math reflected the biology.

Bringing Your Own Data
----------------------

1. Ensure your CSV is accessible on the machine running the MCP server.
2. Connect the server using the setup steps above.
3. Ask: *"Look at* ``my_data.csv`` *using Pulsar. Are there hidden structural groups?"*

The AI handles imputation, categorical encoding, and parameter scaling. Your job is to interpret the Dossier using domain knowledge.

.. seealso::

   - :ref:`Configuration <configuration>` — full YAML schema reference
