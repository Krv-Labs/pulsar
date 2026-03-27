.. _index:

======
Pulsar
======

**Rust-Accelerated Topological Model Pipeline**

Pulsar provides a high-performance topological workflow for data analysis: imputation, dimensionality reduction, Ball Mapper graph construction, and representative selection—all powered by a Rust core with a clean Python interface.

Quick Links
-----------

.. grid:: 1 2 3 3
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :octicon:`rocket` Quickstart
      :link: quickstart
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Run the full pipeline in minutes with YAML configuration.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Installation, configuration, and advanced workflows.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: api-reference
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Full API documentation for ``pulsar`` modules.

What is Pulsar?
---------------

Pulsar combines the ergonomics of Python with the speed of Rust to deliver fast topological analysis pipelines. Configure your workflow in YAML, and Pulsar handles:

- **Imputation** of numeric columns with configurable strategies
- **Scaling and projection** via systematic PCA parameter sweeps
- **Graph construction** using Ball Mapper with epsilon sweeps
- **Pseudo-Laplacian accumulation** across embedding configurations
- **Cosmic graph assembly** with weighted edge aggregation
- **Representative selection** using topological distances

Typical Workflow
----------------

.. mermaid::

   graph LR
      subgraph Input
         A[params.yaml + dataset]
      end

      subgraph "Stage 1: Preprocess"
         B["Impute columns"]
         C["StandardScaler"]
      end

      subgraph "Stage 2: Project"
         D["PCA grid sweep"]
         E1["dim=2"]
         E2["dim=5"]
         E3["dim=10"]
      end

      subgraph "Stage 3: Graph"
         F["Ball Mapper sweep"]
         G["Pseudo-Laplacians"]
      end

      subgraph "Stage 4: Select"
         H["Cosmic Graph"]
         I["Representatives"]
      end

      A --> B
      B --> C
      C --> D
      D --> E1
      D --> E2
      D --> E3
      E1 --> F
      E2 --> F
      E3 --> F
      F --> G
      G --> H
      H --> I

      style Input fill:#f9f9f9,stroke:#999
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style D fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style F fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style H fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

**YAML-Driven (Recommended)**

.. code-block:: python

   from pulsar import ThemaRS

   model = ThemaRS("params.yaml")
   model.fit()
   print(model.cosmic_graph.number_of_edges())

**Programmatic Configuration**

.. code-block:: python

   from pulsar import ThemaRS

   model = ThemaRS(
       data="data.csv",
       pca_dims=[2, 5, 10],
       epsilon_range=(0.1, 0.5, 5),
   )
   model.fit()
   representatives = model.select_representatives(k=3)

Key Features
------------

**Rust Performance**
   Core algorithms implemented in Rust via PyO3 for 10-100x speedups over pure Python.

**Grid-Based Exploration**
   Systematic sweeps over PCA dimensions, epsilon values, and imputation seeds.

**Ball Mapper Graphs**
   Local neighborhood graphs that preserve topological structure at multiple scales.

**Cosmic Graph Assembly**
   Aggregate pseudo-Laplacians into a single weighted graph for downstream analysis.

**YAML Configuration**
   Declarative configs for reproducible, shareable pipelines.

Installation
------------

.. code-block:: bash

   pip install pulsar

For development (requires Rust toolchain):

.. code-block:: bash

   git clone https://github.com/Krv-Analytics/pulsar.git
   cd pulsar
   uv sync --extra dev --extra docs
   uv run maturin develop --release

Supports Python 3.10, 3.11, 3.12.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Run your first pipeline
- :ref:`User Guide <user_guide>` - Installation and configuration details
- :ref:`API Reference <api-reference>` - Class and function documentation
- :ref:`Configuration <configuration>` - YAML schema reference

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   overview
   user_guide
   configuration

References
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
