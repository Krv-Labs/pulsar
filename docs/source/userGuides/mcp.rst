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

Claude handles all the messy parts: imputation, categorical encoding, parameter selection, and iterating when the results look wrong. Dedicated tools let Claude fix preprocessing errors in ≤2 tool calls before re-running the sweep.

Setup
-----

Pulsar ships an MCP server entry point (``pulsar-mcp``) via the ``mcp`` extra of the published ``thema-pulsar`` package. You do **not** need to clone the repo or create a Python environment. Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then choose your client below. ``uvx`` downloads the released package from PyPI and runs it in an isolated environment.

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
         **Workspace trust gates stdio MCP servers.** As of Gemini CLI 0.53, Gemini does not start stdio MCP servers in an untrusted workspace, so a user-scoped Pulsar registration still stays dormant until you trust the folder. Choose **Trust folder** when first prompted; if you previously declined, run ``/permissions trust`` inside Gemini and select **Trust folder**. Let Gemini relaunch — running sessions do not reload external settings changes, so if it does not relaunch on its own, fully exit and start it again. See `Gemini CLI folder trust <https://google-gemini.github.io/gemini-cli/docs/cli/trusted-folders.html>`_ for current behavior.

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

   pipx install "thema-pulsar[mcp]"   # then use command: pulsar-mcp
   # or
   pip install "thema-pulsar[mcp]"    # in any venv

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
2. **Generate a preprocessing config** — recommend impute/encode rules for every column with rationale
3. **Validate preprocessing** — dry-run the preprocessing stage before committing to a full sweep
4. **Run a topological sweep** — find the most stable version of the data's shape
5. **Iterate automatically** — repair preprocessing errors and tune epsilon if results are degenerate
6. **Generate a Dossier** — statistical profiles of each discovered subpopulation

Available MCP Tools
-------------------

The server exposes these tools to the AI client. Claude automatically chains them together:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **characterize_dataset**
     - Quick exploratory summary: k-NN distances (is your data sparse or dense?), projection dimensions, missing value patterns. On large datasets (``n`` above the advisor threshold), may include ``minhash_advisory`` with suggested ``minhash_d``, memory estimate, and error bounds. Claude uses this to make smart initial parameter guesses instead of random choices.
   * - **run_topological_sweep**
     - Execute the full Pulsar pipeline: imputation → projection → Ball Mapper → cosmic graph, all from inline YAML config. Returns structured JSON with metrics and experiment diff. Config persistence is opt-in. Results cached per session.
   * - **generate_cluster_dossier**
     - Deep statistical report per discovered cluster: trait profiles, homogeneity scores, separation metrics, concentration measures. Answers "What makes this cluster distinct?" and "How confident are we in the boundaries?"
   * - **compare_clusters**
     - Pairwise statistical tests (Welch's t-test, Kolmogorov-Smirnov, Cohen's d, effect sizes) between clusters. Answers "Are these really different, or just noise?"
   * - **export_labeled_data**
     - Return your original dataframe with labels from a specific ``cluster_assignment_id`` emitted by ``generate_cluster_dossier``. Ready for downstream analysis, visualization, or handoff to domain experts.
   * - **diagnose_cosmic_graph**
     - Current graph-state observables: scale, component morphology, weight distribution, sweep support, observed patterns, and risk factors. When ``construction: minhash``, includes ``minhash_profile`` (realized weight accuracy and confidence interval at the configured signature depth). The agent interprets these measurements against the user's objective.
   * - **recommend_preprocessing**
     - Analyze column profiles and return a complete ``preprocessing:`` YAML block with per-column rationale. Call this before the first sweep to avoid hand-writing impute/encode rules from raw stats.
   * - **repair_preprocessing_config**
     - Parse a preprocessing error from ``run_topological_sweep``, look up the offending column in the dataset profile, and return a patched config with a change log. Fixes most errors in one call.
   * - **validate_preprocessing_config**
     - Dry-run only the preprocessing stage against the session data — no projection, no sweep cost. Returns PASS with a schema summary, or a structured error ready to pass to ``repair_preprocessing_config``.

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
