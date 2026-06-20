.. _overview:

========
Overview
========

Pulsar is a **Rust-accelerated topological pipeline** for exploring model spaces through systematic parameter sweeps. It transforms raw data into a Cosmic graph that reveals relationships between different model configurations.

The Problem
-----------

When analyzing data, you face many preprocessing choices:

- Which imputation strategy?
- Which projection method and dimensions?
- What neighborhood size for graph construction?

Each combination produces a different representation. Pulsar explores this space systematically and uses topological methods to identify representative configurations.

Architecture
------------

Pulsar combines Python ergonomics with Rust performance:

.. mermaid::

   graph TB
      subgraph "Python Layer"
         A["ThemaRS API (pipeline.py)"]
         B["Config parsing (config.py)"]
         C["NetworkX integration (analysis/hooks.py)"]
         P["Progress reporting (runtime/progress.py)"]
         T["Temporal graphs (representations/temporal.py)"]
      end

      subgraph "Rust Core (PyO3)"
         D["Imputation"]
         E["JL/PCA projection"]
         F["Ball Mapper"]
         G["Laplacian accumulation"]
      end

      subgraph "Output"
         H["Cosmic Graph"]
         I["Representatives"]
      end

      A --> B
      B --> D
      D --> E
      E --> F
      F --> G
      G --> H
      H --> C
      C --> I

      style A fill:#f9f9f9,stroke:#999
      style D fill:#FCF3CF,stroke:#D4AC0D,stroke-width:2px
      style E fill:#FCF3CF,stroke:#D4AC0D,stroke-width:2px
      style F fill:#FCF3CF,stroke:#D4AC0D,stroke-width:2px
      style G fill:#FCF3CF,stroke:#D4AC0D,stroke-width:2px
      style H fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

Pipeline Stages
---------------

**1. Data Loading & Imputation**

Load tabular data and fill missing values with configurable strategies (mean, median, or custom). Multiple imputation seeds generate diverse candidates.

**2. Scaling & Projection Sweep**

StandardScaler normalization followed by Johnson-Lindenstrauss (JL) random projection by default. Pulsar sweeps across multiple dimension settings and seeds to explore different embedding spaces. Set ``sweep.projection.method: pca`` to use the legacy randomized PCA path.

**3. Ball Mapper Graph Construction**

For each projection, build Ball Mapper graphs at multiple epsilon values. Low-dimensional embeddings (1-16 dimensions) use a KD-tree radius query for membership assignment; wider embeddings fall back to the linear scan path.

**4. Pseudo-Laplacian Accumulation**

Compute graph Laplacians for each Ball Mapper configuration and aggregate them into a summary representation.

**5. Cosmic Graph Assembly**

Combine pseudo-Laplacians into a weighted graph where edges represent similarity between configurations.

By default, Pulsar sparsifies the original unthresholded Cosmic Graph with effective-resistance sampling, then selects/applies the construction threshold on that sparse weighted graph. ``model.cosmic_graph`` is therefore a sparse ``networkx.Graph`` with ``weight`` attributes. Dense compatibility getters remain available when needed.

**6. Representative Selection**

Use graph distances (e.g., Forman-Ricci curvature) to identify the most central configurations.

Configuration Model
-------------------

Pulsar uses a hierarchical configuration:

.. code-block:: yaml

   run:
     name: my_experiment
     data: path/to/data.csv

   preprocessing:
     drop_columns: [id, timestamp]
     impute:
       age: {method: sample_normal, seed: 42}
       category: {method: sample_categorical, seed: 7}

   sweep:
     pca:
     projection:
       method: jl
       dimensions: {values: [2, 5, 10, 16]}
       seed: {values: [42, 7, 13]}
       center: true
     ball_mapper:
       epsilon: {range: {min: 0.1, max: 1.5, steps: 8}}

   cosmic_graph:
     construction_threshold: "auto"

Key Outputs
-----------

========================= ========================================================
Output                    Description
========================= ========================================================
``cosmic_graph``          NetworkX graph with weighted edges
``weighted_adjacency``    Dense similarity matrix
``_embeddings``           List of projection embeddings
``ball_mapper_graphs_``   List of Ball Mapper graphs
``stability_result``      Threshold selection diagnostics (if ``auto``)
========================= ========================================================

Performance
-----------

The Rust core provides significant speedups:

- **10-100x faster** Ball Mapper construction, with KD-tree acceleration for 1-16D embeddings
- **Parallel** JL/PCA projection computation across configurations
- **Memory efficient** Laplacian accumulation
- **Sparse** CosmicGraph spectral sparsification for smaller graph outputs

For large datasets (>10k rows) or extensive sweeps (>100 configurations), Pulsar's Rust implementation is essential.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Run your first pipeline
- :ref:`User Guide <user_guide>` - Configuration details
- :ref:`Configuration <configuration>` - YAML schema reference
