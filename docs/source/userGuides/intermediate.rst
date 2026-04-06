.. _intermediate:

=============
Tuning Guide
=============

Pulsar sweeps over a grid of PCA dimensions and epsilon values. The quality of your results depends on how well that grid covers the structure of your data. This guide explains each parameter, what it controls, and how to tune it.

Parameter Specification Styles
--------------------------------

Every sweep parameter supports three styles:

.. code-block:: yaml

   # Explicit list — use exactly these values
   dimensions: {values: [2, 3, 5, 10]}

   # Linspace — evenly spaced between min and max
   epsilon: {range: {min: 0.1, max: 1.5, steps: 8}}

   # Single scalar — equivalent to values: [x]
   seed: 42

Mix styles freely within a config. The total grid size is
``len(dimensions) × len(seeds) × len(epsilons)``.

----

Preprocessing
-------------

Before any sweep runs, Pulsar preprocesses the input DataFrame. All preprocessing is declared in the ``preprocessing:`` block of your config.

``preprocessing.drop_columns``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Columns listed here are removed before any other step:

.. code-block:: yaml

   preprocessing:
     drop_columns: [id, timestamp, row_index]

``preprocessing.impute``
~~~~~~~~~~~~~~~~~~~~~~~~~

Numeric imputation is applied before scaling. Each column gets a method and an optional seed:

.. code-block:: yaml

   preprocessing:
     impute:
       age:      {method: fill_mean}          # low missingness (< 30%)
       income:   {method: fill_median}        # skewed distributions
       weight:   {method: sample_normal, seed: 42}  # high missingness (>= 30%)

Available methods: ``fill_mean``, ``fill_median``, ``fill_mode``, ``sample_normal``, ``sample_categorical``.

.. note::

   Columns **not** listed in ``impute`` or ``encode`` must arrive NaN-free. If any NaN values remain after preprocessing, Pulsar raises a ``ValueError`` naming the offending column(s) and row counts. There is no silent row-dropping.

``preprocessing.encode``
~~~~~~~~~~~~~~~~~~~~~~~~~

Categorical (string/object) columns must be encoded before they can be passed to the Rust pipeline. Only one-hot encoding is supported:

.. code-block:: yaml

   preprocessing:
     encode:
       island:    {method: one_hot}
       sex:       {method: one_hot}
       diagnosis: {method: one_hot, max_categories: 20}

**Cardinality rules:**

- If a column has more than 50 unique values, Pulsar emits a ``UserWarning`` — each category adds a dimension that dilates Euclidean distances after scaling.
- Use ``max_categories: N`` to turn this into a hard error, preventing accidental high-cardinality encodings.
- Columns with more than 100 unique values should generally be dropped, not encoded.

.. code-block:: yaml

   preprocessing:
     encode:
       # Hard limit — raises ValueError if more than 20 categories found
       icd_code: {method: one_hot, max_categories: 20}

**String imputation before encoding:**

If a categorical column has missing values, add an ``impute`` rule alongside the ``encode`` rule. Use ``sample_categorical`` for multi-class columns (preserves class proportions) or ``fill_mode`` for binary columns:

.. code-block:: yaml

   preprocessing:
     impute:
       sex: {method: fill_mode}         # binary — fill_mode is safe
       island: {method: sample_categorical, seed: 42}  # multi-class
     encode:
       sex:    {method: one_hot}
       island: {method: one_hot}

----

PCA Parameters
--------------

``sweep.pca.dimensions``
~~~~~~~~~~~~~~~~~~~~~~~~~

**What it is**: The number of principal components to retain before building Ball Mapper. Each dimension value produces a separate projection of the scaled data.

**What it controls**: Lower dimensions (2–5) emphasize dominant global structure. Higher dimensions (10–50) preserve more local variance and find finer-grained clusters. Pulsar sweeps all values and fuses the results, so you get multi-scale coverage in a single run.

**How to choose**:

- Start with ``[2, 5, 10]`` for exploration.
- For high-dimensional data (>100 features), add ``20`` or ``50``.
- Avoid dimensions larger than the number of features — Pulsar will error.
- There is no benefit to values above ~50 for most datasets; the randomized SVD approximation degrades on very high dimensions.

.. code-block:: yaml

   sweep:
     pca:
       dimensions: {values: [2, 5, 10, 20]}

``sweep.pca.seed``
~~~~~~~~~~~~~~~~~~~

**What it is**: Random seeds for the randomized SVD (Halko et al. 2011). Multiple seeds produce slightly different PCA projections of the same dimensionality.

**What it controls**: Seed variation captures rotational ambiguity in PCA — two seeds at ``dimensions: 5`` give different 5D projections, each revealing slightly different structure. The Pseudo-Laplacian accumulator fuses all of them, improving robustness.

**How to choose**:

- Use 3–5 seeds for production runs: ``{values: [42, 7, 13, 99, 0]}``.
- One seed is fine for quick iteration.
- More seeds increase runtime linearly but rarely improve results beyond ~5.

.. code-block:: yaml

   sweep:
     pca:
       seed: {values: [42, 7, 13]}

----

Ball Mapper Parameters
-----------------------

``sweep.ball_mapper.epsilon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it is**: The ball radius used to define neighborhoods in PCA space. A ball centered on each landmark point covers all data points within distance ``epsilon``.

**What it controls**: This is the most impactful parameter.

- **Small epsilon** → many small, tight balls → many nodes → fine-grained, poztentially fragmented graph
- **Large epsilon** → few large balls with heavy overlap → fewer nodes → coarse, over-connected graph

The right epsilon captures natural cluster density without merging distinct groups.

**How to choose**:

- Start with a range spanning roughly one order of magnitude: ``{range: {min: 0.1, max: 1.5, steps: 8}}``.
- After a first run, inspect ``model.resolved_threshold`` and the node/edge count of ``model.cosmic_graph``. If the graph has a single giant component, epsilon is too large. If it is entirely disconnected, epsilon is too small.
- For normalized data (post-StandardScaler), values between 0.2 and 2.0 are typical.
- Use ``threshold: "auto"`` (see below) — it is designed to work with a wide epsilon range.

.. code-block:: yaml

   sweep:
     ball_mapper:
       epsilon: {range: {min: 0.1, max: 1.5, steps: 10}}

----

Cosmic Graph Parameters
------------------------

``cosmic_graph.threshold``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it is**: A minimum edge-weight cutoff applied to the weighted adjacency matrix before converting to a NetworkX graph. Edges below the threshold are removed.

**What it controls**: Threshold trades sparsity against connectivity.

- **High threshold** → sparse graph, only the strongest topological relationships retained
- **Low threshold (0.0)** → dense graph, all co-memberships included; often produces one giant component with no useful structure
- **``"auto"``** → uses approximate H₀ persistent homology to find the threshold where the number of connected components stabilizes (a "plateau" in the component-count curve)

**How to choose**:

Always use ``threshold: "auto"`` unless you have a specific reason not to. It is the correct default for high-dimensional data. A fixed threshold of ``0.0`` will almost always produce a single, structureless component.

If you need a manual threshold (e.g. for reproducibility after exploration), read it from the fitted model:

.. code-block:: python

   model.fit()
   print(model.resolved_threshold)  # use this value in your YAML

.. code-block:: yaml

   cosmic_graph:
     threshold: "auto"      # recommended
     # threshold: 0.42      # manual override after inspection

``cosmic_graph.neighborhood``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it is**: The method used to compute normalized edge weights from the accumulated Pseudo-Laplacian. Currently ``"node"`` is the only supported value.

----

Output Parameters
------------------

``output.n_reps``
~~~~~~~~~~~~~~~~~~

**What it is**: The number of representative BallMapper configurations returned by ``model.select_representatives()``.

**What it controls**: After fitting, Pulsar has built hundreds or thousands of individual Ball Maps (one per ``dimension × seed × epsilon`` combination). ``select_representatives`` clusters them by structural similarity (node count, edge count, epsilon) and returns the ``n_reps`` maps closest to each cluster centroid — a diverse, compact summary of the full sweep.

**How to choose**:

- ``4`` is a reasonable default for visualization and reporting.
- Increase to ``8–12`` for a richer sample if you are building an ensemble or feeding results to a downstream model.
- Has no effect on ``model.cosmic_graph`` — that is always built from the full sweep.

.. code-block:: yaml

   output:
     n_reps: 4

----

Example: Full Tuned Config
---------------------------

.. code-block:: yaml

   run:
     name: my_experiment
     data: data.csv

   preprocessing:
     drop_columns: [id, timestamp]
     impute:
       age:      {method: fill_mean}
       income:   {method: sample_normal, seed: 42}
       sex:      {method: fill_mode}
     encode:
       sex:      {method: one_hot}
       category: {method: one_hot, max_categories: 10}

   sweep:
     pca:
       dimensions: {values: [2, 5, 10]}
       seed: {values: [42, 7, 13]}
     ball_mapper:
       epsilon: {range: {min: 0.1, max: 1.5, steps: 8}}

   cosmic_graph:
     threshold: "auto"

   output:
     n_reps: 4

This produces ``3 × 3 × 8 = 72`` Ball Maps, fused into a single Cosmic Graph with an automatically selected threshold.
