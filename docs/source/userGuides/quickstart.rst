.. _quickstart:

==========
Quickstart
==========

Get from zero to insights in under 10 minutes.

.. code-block:: bash

   uv pip install pulsar


Prerequisites
-------------

- Python 3.10+
- For development: Rust toolchain

Option 1: Use a Pre-Built Demo (Fastest)
-----------------------------------------

The fastest way to see Pulsar in action:

.. code-block:: bash

   # Run the penguins demo (no data download needed)
   cd /path/to/pulsar
   uv sync
   uv run maturin develop --release
   python -c "
   from pulsar.pipeline import ThemaRS
   config = {'run': {'name': 'penguins', 'data': 'demos/penguins/penguins.csv'}}
   model = ThemaRS.from_dict(config)
   model.fit()
   print(f'Cosmic graph: {len(model.cosmic_graph.nodes())} nodes, {len(model.cosmic_graph.edges())} edges')
   "

Done! You've discovered penguin species structure without looking at species labels.

For all demos: :ref:`demos`

Option 2: Use with Claude AI (No Code)
---------------------------------------

Let Claude handle the analysis:

1. Set up Pulsar MCP server (see :ref:`mcp`)
2. Open Claude Desktop
3. Paste: *"Analyze the file at ``demos/penguins/penguins.csv`` using Pulsar. Find the hidden structure."*

Claude will orchestrate parameter tuning and generate a statistical dossier.

Option 3: YAML-Driven Workflow (Recommended for Reproducibility)
-----------------------------------------------------------------

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
     pca:
       dimensions: {values: [2, 5, 10]}
       seed: {values: [42, 7, 13]}
     ball_mapper:
       epsilon: {range: {min: 0.1, max: 0.5, steps: 5}}

   cosmic_graph:
     threshold: "auto"

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
   reps = model.select_representatives(k=3)
   for i, rep in enumerate(reps):
       print(f"Representative {i+1}: {rep}")

Option 4: Programmatic Configuration (Full Control)
-----------------------------------------------------

For maximum control, configure directly in Python:

.. code-block:: python

   from pulsar import ThemaRS

   model = ThemaRS(
       data="data.csv",
       pca_dims=[2, 5, 10],
       epsilon_range=(0.1, 0.5, 5),
       random_state=42,
   )
   model.fit()

Understanding the Pipeline
--------------------------

Pulsar executes these stages:

1. **Impute**: Fill missing values in specified columns
2. **Scale**: StandardScaler normalization
3. **PCA sweep**: Project data to multiple dimensions
4. **Ball Mapper sweep**: Build neighborhood graphs at multiple epsilon values
5. **Pseudo-Laplacians**: Compute graph Laplacians for each configuration
6. **Cosmic graph**: Aggregate into a weighted similarity graph
7. **Selection**: Choose representative configurations via graph distances

.. code-block:: python

   # Access intermediate results
   print(f"PCA configurations: {len(model.pca_results_)}")
   print(f"Ball Mapper graphs: {len(model.ball_mapper_graphs_)}")
   print(f"Weighted adjacency shape: {model.weighted_adjacency_.shape}")

Performance Tips
----------------

Pulsar's Rust core provides significant speedups. For large datasets:

.. code-block:: python

   # Reduce sweep resolution for faster iteration
   model = ThemaRS(
       data="large_data.csv",
       pca_dims=[5],           # Single dimension
       epsilon_range=(0.2, 0.4, 3),  # Fewer epsilon steps
   )

Next Steps
----------

- :doc:`programmatic` - Full API control
- :doc:`intermediate` - Tuning sweep parameters
- :ref:`Configuration <configuration>` - YAML schema reference
