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

Claude handles all the messy parts: imputation, scaling, categorical encoding, parameter selection, iteration if the results look wrong.

Setup
-----

Pulsar ships an MCP server entry point (``pulsar-mcp``) in the ``mcp`` dependency group. Wire it into your AI client of choice:

.. tab-set::

   .. tab-item:: Claude Desktop

      Open ``~/Library/Application Support/Claude/claude_desktop_config.json`` (macOS) or ``%APPDATA%\Claude\claude_desktop_config.json`` (Windows) and add:

      .. code-block:: json

         {
           "mcpServers": {
             "pulsar": {
               "command": "uv",
               "args": ["run", "--group", "mcp", "pulsar-mcp"]
             }
           }
         }

      Restart Claude Desktop. A hammer icon in new chats confirms the tools loaded.

      .. note::
         If Claude can't find ``uv``, replace ``"command": "uv"`` with the absolute path (e.g. ``/Users/yourname/.local/bin/uv``).

   .. tab-item:: Gemini CLI

      .. code-block:: bash

         gemini mcp add pulsar uv run --group mcp pulsar-mcp

   .. tab-item:: Claude Code

      .. code-block:: bash

         claude mcp add pulsar -- uv run --group mcp pulsar-mcp

   .. tab-item:: Cursor / Windsurf

      Open **Settings → Features → MCP → Add new MCP server**:

      - Name: ``pulsar``
      - Type: ``command``
      - Command: ``uv run --group mcp pulsar-mcp``

Workflow
--------

Once connected, give the AI a goal rather than instructions. The AI already knows the technical steps.

**The recommended prompt:**

   *"I have a dataset at* ``path/to/data.csv``\ *. Use Pulsar to find the hidden structure and tell me the story of this data. I'm looking for meaningful subgroups and the specific traits that define them."*

Under the hood the AI will:

1. **Characterize geometry** — probe k-NN distances and PCA variance to ground parameter choices
2. **Run a topological sweep** — find the most stable version of the data's shape
3. **Iterate automatically** — tune epsilon if the result is a structureless blob or a shattered mess
4. **Generate a Dossier** — statistical profiles of each discovered subpopulation

Available MCP Tools
-------------------

The server exposes these tools to the AI client. Claude automatically chains them together:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - What It Does
   * - **characterize_dataset**
     - Quick exploratory summary: k-NN distances (is your data sparse or dense?), PCA variance (how many dimensions matter?), missing value patterns. Claude uses this to make smart initial parameter guesses instead of random choices.
   * - **run_topological_sweep**
     - Execute the full Pulsar pipeline: imputation → PCA → Ball Mapper → cosmic graph, all from inline YAML config. No disk I/O. Results cached per session.
   * - **generate_cluster_dossier**
     - Deep statistical report per discovered cluster: trait profiles, homogeneity scores, separation metrics, concentration measures. Answers "What makes this cluster distinct?" and "How confident are we in the boundaries?"
   * - **compare_clusters_tool**
     - Pairwise statistical tests (Welch's t-test, Kolmogorov-Smirnov, Cohen's d, effect sizes) between clusters. Answers "Are these really different, or just noise?"
   * - **export_labeled_data**
     - Return your original dataframe with discovered cluster labels attached. Ready for downstream analysis, visualization, or handoff to domain experts.
   * - **diagnose_cosmic_graph**
     - Health metrics on the graph structure: connected components, density, sparsity. Detects degenerate results (blob or shattered) and suggests YAML fixes automatically.

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
     pca:
       dimensions:
         values: [5]
     ball_mapper:
       epsilon:
         range: {min: 0.80, max: 1.50, steps: 15}
   cosmic_graph:
     threshold: auto

The resulting graph shattered into components along **island** and **sex** boundaries, not just species—revealing that habitat and morphological sex are geometrically dominant. Chinstraps on Dream Island were structurally indistinguishable from Adelies on the same island: the math reflected the biology.

Bringing Your Own Data
----------------------

1. Ensure your CSV is accessible on the machine running the MCP server.
2. Connect the server using the setup steps above.
3. Ask: *"Look at* ``my_data.csv`` *using Pulsar. Are there hidden structural groups?"*

The AI handles imputation, categorical encoding, and parameter scaling. Your job is to interpret the Dossier using domain knowledge.

.. seealso::

   - :ref:`Configuration <configuration>` — full YAML schema reference

