.. _overview:

========
Overview
========

Pulsar is a **Rust-accelerated topological pipeline** for exploring model spaces through systematic parameter sweeps. It transforms raw data into a Cosmic graph that reveals relationships between different model configurations.

The Problem
-----------

When analyzing data, you face many preprocessing choices:

- Which imputation strategy?
- How many PCA dimensions?
- What neighborhood size for graph construction?

Each combination produces a different representation. Pulsar explores this space systematically and uses topological methods to identify representative configurations.

Architecture
------------

Pulsar combines Python ergonomics with Rust performance:

.. mermaid::

   graph TB
      subgraph "Python Layer"
         A["ThemaRS API"]
         B["Configuration parsing"]
         C["NetworkX integration"]
      end

      subgraph "Rust Core (PyO3)"
         D["Imputation"]
         E["PCA computation"]
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

**2. Scaling & PCA Sweep**

StandardScaler normalization followed by PCA projection. Pulsar sweeps across multiple dimension settings to explore different embedding spaces.

**3. Ball Mapper Graph Construction**

For each PCA projection, build Ball Mapper graphs at multiple epsilon values. This captures local structure at different scales.

**4. Pseudo-Laplacian Accumulation**

Compute graph Laplacians for each Ball Mapper configuration and aggregate them into a summary representation.

**5. Cosmic Graph Assembly**

Combine pseudo-Laplacians into a weighted graph where edges represent similarity between configurations.

**6. Representative Selection**

Use graph distances (e.g., Forman-Ricci curvature) to identify the most central configurations.

Configuration Model
-------------------

Pulsar uses a hierarchical configuration:

.. code-block:: yaml

   data:
     path: "dataset.csv"
     target: "label"

   impute:
     columns: ["col1", "col2"]
     method: "median"
     seed: 42

   pca:
     dimensions: [2, 5, 10, 20]
     seeds: [0, 1, 2]

   ball_mapper:
     epsilon_range: [0.1, 0.5, 10]

   cosmic_graph:
     threshold: "auto"

Key Outputs
-----------

========================= ========================================================
Output                    Description
========================= ========================================================
``cosmic_graph``          NetworkX graph with weighted edges
``weighted_adjacency``    Dense similarity matrix
``pca_results_``          List of PCA projections
``ball_mapper_graphs_``   List of Ball Mapper graphs
``stability_result``      Threshold selection diagnostics (if ``auto``)
========================= ========================================================

Performance
-----------

The Rust core provides significant speedups:

- **10-100x faster** Ball Mapper construction
- **Parallel** PCA computation across configurations
- **Memory efficient** Laplacian accumulation

For large datasets (>10k rows) or extensive sweeps (>100 configurations), Pulsar's Rust implementation is essential.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Run your first pipeline
- :ref:`User Guide <user_guide>` - Configuration details
- :ref:`Configuration <configuration>` - YAML schema reference
