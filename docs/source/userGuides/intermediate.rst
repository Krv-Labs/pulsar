.. _intermediate:

Tuning Guide
============

Pulsar sweeps over projection dimensions, random seeds, and Ball Mapper
epsilon values. The quality of a run depends on how well that grid covers the
geometry of the data.

Parameter Specification Styles
------------------------------

Sweep parameters support explicit values, ranges, and scalars:

.. code-block:: yaml

   dimensions: {values: [2, 3, 5, 10]}
   epsilon: {range: {min: 0.1, max: 1.5, steps: 8}}
   seed: 42

The total grid size is ``len(dimensions) * len(seeds) * len(epsilons)``.

Preprocessing
-------------

``preprocessing.drop_columns``
   Columns listed here are removed before any other step.

   .. code-block:: yaml

      preprocessing:
        drop_columns: [id, timestamp, row_index]

``preprocessing.impute``
   Numeric imputation is applied before scaling. Available methods are
   ``fill_mean``, ``fill_median``, ``fill_mode``, ``sample_normal``, and
   ``sample_categorical``.

   .. code-block:: yaml

      preprocessing:
        impute:
          age: {method: fill_mean}
          income: {method: fill_median}
          weight: {method: sample_normal, seed: 42}

   Columns not listed in ``impute`` or ``encode`` must arrive NaN-free. If NaNs
   remain after preprocessing, Pulsar raises a ``ValueError`` naming the
   offending columns and row counts.

``preprocessing.encode``
   Categorical columns must be encoded before they can be passed to the Rust
   pipeline. Only one-hot encoding is currently supported.

   .. code-block:: yaml

      preprocessing:
        encode:
          island: {method: one_hot}
          sex: {method: one_hot}
          diagnosis: {method: one_hot, max_categories: 20}

   If a column has more than 50 unique values, Pulsar emits a ``UserWarning``.
   Use ``max_categories`` to make that limit a hard error. Columns with more
   than 100 unique values should usually be dropped instead of encoded.

Projection Parameters
---------------------

``sweep.projection.method``
   The projection backend used before Ball Mapper. ``jl`` is the default
   Johnson-Lindenstrauss random projection. ``pca`` selects the legacy
   randomized PCA implementation.

   .. code-block:: yaml

      sweep:
        projection:
          method: jl

   Use PCA explicitly only when you need variance-ordered axes or compatibility
   with older runs:

   .. code-block:: yaml

      sweep:
        projection:
          method: pca

``sweep.projection.dimensions``
   The number of projected dimensions to retain before Ball Mapper. Lower
   dimensions (2-5) emphasize global structure; higher dimensions (10-16)
   preserve more local variation.

   .. code-block:: yaml

      sweep:
        projection:
          method: jl
          dimensions: {values: [2, 5, 10, 16]}

   Start with ``[2, 5, 10]`` for exploration. For exceptionally high-dimensional data, add
   ``15`` or ``16``. Unless the user specifically asks for wider projections,
   keep both JL and PCA dimensions at 16 or below so Ball Mapper can use the
   KD-tree radius-query path. Dimensions above 16 still work, but use the
   compatibility linear scan path.

``sweep.projection.seed``
   Random seeds for projection generation. Multiple seeds produce different
   views of the same dimensionality, and Pulsar fuses them through the
   pseudo-Laplacian accumulator.

   .. code-block:: yaml

      sweep:
        projection:
          seed: {values: [42, 7, 13]}

``sweep.projection.center``
   Whether to subtract column means before JL projection. The default is
   ``true``.

   .. code-block:: yaml

      sweep:
        projection:
          center: true

.. note::

   Legacy configs that only contain ``sweep.pca`` are still accepted. They are
   treated as dimension/seed aliases for the default JL projection. To restore
   old PCA behavior, use ``sweep.projection.method: pca``.

Ball Mapper Parameters
----------------------

``sweep.ball_mapper.epsilon``
   The ball radius used to define neighborhoods in projection space.

   - Small epsilon: many small balls, fine-grained but possibly fragmented.
   - Large epsilon: fewer balls with heavy overlap, coarse and possibly
     over-connected.

   .. code-block:: yaml

      sweep:
        ball_mapper:
          epsilon: {range: {min: 0.1, max: 1.5, steps: 10}}

   After a first run, inspect ``model.resolved_construction_threshold`` and the
   node/edge count of ``model.cosmic_graph``. If the graph is a giant component,
   epsilon is too large. If it is mostly disconnected, epsilon is too small.

Cosmic Graph Parameters
-----------------------

``cosmic_graph.construction``
   How edge weights are computed when fusing Ball Mapper outputs into the Cosmic
   Graph.

   - ``minhash`` (default): approximate but unbiased Jaccard estimates of each
     point's ball-set overlap via MinHash signatures and LSH banding. Fast,
     sub-quadratic, and constant-memory — the recommended path for large sweeps
     and massive ``n``.
   - ``exact``: bit-identical sparse pseudo-Laplacian co-occurrence weights.
     Use when exact, reproducible weights matter more than speed or memory.

   .. code-block:: yaml

      cosmic_graph:
        construction: minhash

``cosmic_graph.minhash_d`` / ``cosmic_graph.minhash_seed``
   MinHash signature depth and seed (only used when ``construction: minhash``).

   - ``minhash_d`` (default ``256``): number of hash functions. Edge weights are
     unbiased Jaccard estimates with variance ``J(1−J)/d`` — error depends only
     on ``d``, independent of ``n`` or ball count. Lower ``d`` reduces signature
     memory (``d·n·4`` bytes) and construction time at the cost of wider
     confidence intervals.
   - ``minhash_seed`` (default ``42``): makes randomized construction
     reproducible.

   Defaults rarely need tuning. On large datasets, the MCP server may suggest a
   lower ``minhash_d`` via ``characterize_dataset`` (see :ref:`mcp`).

   .. code-block:: yaml

      cosmic_graph:
        construction: minhash
        minhash_d: 256
        minhash_seed: 42

``cosmic_graph.construction_threshold``
   Minimum edge-weight cutoff applied after graph construction. ``"auto"`` uses
   approximate H0 persistent homology to find a stable component-count plateau.

   .. code-block:: yaml

      cosmic_graph:
        construction_threshold: auto

``cosmic_graph.sparsify``
   Whether Pulsar should spectrally sparsify the unthresholded Cosmic Graph
   before selecting and applying ``construction_threshold``. The default is
   ``false``; leave it off for routine structural analysis.

   .. code-block:: yaml

      cosmic_graph:
        construction_threshold: auto
        sparsify: false
        sparsify_epsilon: 1.0
        sparsify_seed: 42
        sparsify_sketch_dim: null
        sparsify_sample_count: null

   With this default, ``model.cosmic_graph`` is a sparse ``networkx.Graph`` with
   ``weight`` edge attributes. Enable sparsification only when you explicitly
   need a compact, spectrum-preserving graph for downstream spectral algorithms;
   it is not a construction-time speedup or a graph-cleaning step.

``cosmic_graph.neighborhood``
   The method used to compute normalized edge weights from the accumulated
   pseudo-Laplacian. Currently ``"node"`` is the only supported value.

Public Python hooks
-------------------

.. code-block:: python

   graph = model.cosmic_graph      # sparse NetworkX graph, threshold applied
   edges = model.weighted_edges()  # thresholded sparse edge list
   dense = model.dense_cosmic_rust # original Rust graph, exposed for compatibility

   # Opt-in spectral sparsification and refresh model outputs.
   model.spectral_sparsify(epsilon=0.8, seed=7, update=True)

Output Parameters
-----------------

``output.n_reps``
   Number of representative BallMapper configurations returned by
   ``model.select_representatives()``. This does not affect
   ``model.cosmic_graph``, which is always built from the full sweep.

   .. code-block:: yaml

      output:
        n_reps: 4

Full Example
------------

.. code-block:: yaml

   run:
     name: my_experiment
     data: data.csv

   preprocessing:
     drop_columns: [id, timestamp]
     impute:
       age: {method: fill_mean}
       income: {method: sample_normal, seed: 42}
       sex: {method: fill_mode}
     encode:
       sex: {method: one_hot}
       category: {method: one_hot, max_categories: 10}

   sweep:
     projection:
       method: jl
       dimensions: {values: [2, 5, 10]}
       seed: {values: [42, 7, 13]}
       center: true
     ball_mapper:
       epsilon: {range: {min: 0.1, max: 1.5, steps: 8}}

   cosmic_graph:
     construction: minhash
     minhash_d: 256
     minhash_seed: 42
     construction_threshold: auto
     sparsify: false
     sparsify_epsilon: 1.0
     sparsify_seed: 42

   output:
     n_reps: 4

This produces ``3 * 3 * 8 = 72`` Ball Maps, fuses them into a Cosmic Graph via
MinHash (default), then selects/applies ``construction_threshold``.
