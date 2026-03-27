.. _quickstart:

==========
Quickstart
==========

This guide gets you from zero to a Cosmic graph in under 10 minutes.

Prerequisites
-------------

- Python 3.10+
- Pulsar installed (``pip install pulsar``)
- For development: Rust toolchain for native extensions

YAML-Driven Workflow
--------------------

The recommended way to use Pulsar is with a YAML configuration file. This makes pipelines reproducible and shareable.

**Step 1: Create a configuration file**

Create ``params.yaml``:

.. code-block:: yaml

   data:
     path: "data.csv"
     target: "label"

   impute:
     columns: ["age", "income"]
     method: "median"
     seed: 42

   pca:
     dimensions: [2, 5, 10]
     seeds: [0, 1, 2]

   ball_mapper:
     epsilon_range: [0.1, 0.5, 5]  # min, max, steps

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

Programmatic Configuration
--------------------------

For more control, configure Pulsar directly in Python:

.. code-block:: python

   from pulsar import ThemaRS

   model = ThemaRS(
       data="data.csv",
       impute_columns=["age", "income"],
       impute_method="median",
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
- :doc:`advanced` - Custom Cosmic graph thresholds
- :ref:`Configuration <configuration>` - YAML schema reference
