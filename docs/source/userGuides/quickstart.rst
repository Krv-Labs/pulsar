.. _quickstart:

==========
Quickstart
==========

Get from zero to insights in under 10 minutes.

Prerequisites
-------------

- For the no-code MCP path: `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ and an MCP client (Claude Code, Claude Desktop, Codex CLI, Gemini CLI, GitHub Copilot CLI, Cursor, VS Code, or Google Antigravity)
- For the Python API: Python 3.10+
- For development: Rust toolchain

Option 1: Use an AI Client (No Code, No Clone)
----------------------------------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then register Pulsar with your agent tools:

.. code-block:: bash

   uvx --from thema-pulsar pulsar install

Pick the clients you use from the interactive menu (or pass harness ids / ``--all`` in CI). Restart the client, then ask: *"Use Pulsar to find the hidden structure in* ``data.csv`` *and explain the meaningful subgroups."*

If you use Gemini CLI, choose **Trust folder** when prompted — Gemini does not start stdio MCP servers in an untrusted workspace. Run ``/mcp list`` to confirm Pulsar is connected. See :ref:`mcp` for status/uninstall, pipx mode, and manual per-client setup.

You do not need this repository or a local Python project. ``uvx`` downloads ``thema-pulsar`` from PyPI on first launch.

Option 2: Use a Pre-Built Demo
------------------------------

To run the source demo locally:

.. code-block:: bash

   # Run the penguins demo (no data download needed)
   git clone https://github.com/Krv-Labs/pulsar.git
   cd pulsar
   uv sync
   uv run maturin develop --release
   python -c "
   from pulsar.pipeline import ThemaRS
   config = {'run': {'name': 'penguins', 'data': 'demos/penguins/penguins.csv'}}
   model = ThemaRS(config)
   model.fit()
   print(f'Cosmic graph: {len(model.cosmic_graph.nodes())} nodes, {len(model.cosmic_graph.edges())} edges')
   "

Done! You've discovered penguin species structure without looking at species labels.

For all demos: :ref:`demos`

Option 3: YAML-Driven Workflow (Recommended for Reproducibility)
----------------------------------------------------------------

Use YAML configuration for transparent, reproducible pipelines.

**Step 1: Create a configuration file**

Create ``params.yaml``:

.. code-block:: yaml

   data:
     path: "data.csv"

   preprocessing:
     drop_columns: [id]
     impute:
       age:      {method: fill_mean}
       salary:   {method: fill_median}
       category: {method: sample_categorical, seed: 42}
     encode:
       category: {method: one_hot}

   sweep:
     projection:
       method: jl
       dimensions: {values: [2, 5, 10]}
       seed: {values: [42, 7, 13]}
       center: true
     ball_mapper:
       epsilon: {range: {min: 0.1, max: 0.5, steps: 5}}

   cosmic_graph:
     construction: minhash
     construction_threshold: "auto"

**Step 2: Run the pipeline**

.. code-block:: python

   from pulsar import ThemaRS

   model = ThemaRS("params.yaml")
   model.fit()

   # Access the final graph
   graph = model.cosmic_graph
   print(f"Nodes: {graph.number_of_nodes()}")
   print(f"Edges: {graph.number_of_edges()}")

**Step 3: Select representatives**

.. code-block:: python

   # Get the top 3 representative configurations
   reps = model.select_representatives(n_reps=3)
   for i, rep in enumerate(reps):
       print(f"Representative {i+1}: {rep}")

Option 4: Programmatic Configuration (Full Control)
---------------------------------------------------

For maximum control, build the config as a Python dict instead of a YAML file — see :doc:`programmatic` for the full pattern:

.. code-block:: python

   from pulsar import ThemaRS

   model = ThemaRS({
       "run": {"name": "example"},
       "sweep": {
           "projection": {"method": "jl", "dimensions": {"values": [2, 5, 10]}, "seed": {"values": [42]}},
           "ball_mapper": {"epsilon": {"range": {"min": 0.1, "max": 0.5, "steps": 5}}},
       },
   })
   model.fit(data="data.csv")

Understanding the Pipeline
--------------------------

Pulsar executes these stages:

1. **Impute**: Fill missing values in specified columns
2. **Scale**: StandardScaler normalization
3. **Projection sweep**: Project data to multiple dimensions with JL by default, or PCA when configured explicitly
4. **Ball Mapper sweep**: Build neighborhood graphs at multiple epsilon values
5. **Cosmic graph construction**: Fuse Ball Mapper outputs via MinHash (default) or exact sparse pseudo-Laplacian accumulation
6. **Threshold & assembly**: Apply ``construction_threshold`` to produce a sparse weighted similarity graph
7. **Selection**: Choose representative configurations via graph distances

.. code-block:: python

   # Access intermediate results
   print(f"Ball Mapper graphs: {len(model.ball_maps)}")
   print(f"Weighted adjacency shape: {model.weighted_adjacency.shape}")

Performance Tips
----------------

Pulsar's Rust core provides significant speedups. For large datasets, reduce sweep
resolution for faster iteration — fewer projection dimensions/seeds and fewer
epsilon steps:

.. code-block:: yaml

   sweep:
     projection:
       dimensions: {values: [5]}
       seed: {values: [42]}
     ball_mapper:
       epsilon: {range: {min: 0.2, max: 0.4, steps: 3}}

Next Steps
----------

- :doc:`programmatic` - Full API control
- :doc:`intermediate` - Tuning sweep parameters
- :ref:`Configuration <configuration>` - YAML schema reference
