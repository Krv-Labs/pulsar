.. _index:

======
Pulsar
======

**Discover Hidden Structure in Your Data**

Pulsar is a **topological data analysis tool** that finds the real shape of high-dimensional data. Instead of forcing your data into spheres (like K-means), Pulsar reveals manifolds, clusters, and intricate structure using Ball Mapper — a proven algorithm for topological discovery.

With an AI assistant (Claude, Gemini, Cursor), you don't even need code.

Get Started
-----------

.. grid:: 1 2 3 3
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :octicon:`zap` Try in 5 Minutes
      :link: demos
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      See Pulsar in action with real data: penguins, MMLU, clinical trajectories, and more.

   .. grid-item-card:: :octicon:`hubot` Use with Claude AI
      :link: mcp
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      No code required. Let Claude handle the analysis with Pulsar MCP tools.

   .. grid-item-card:: :octicon:`code` Python API
      :link: quickstart
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Write your own analysis with Pulsar's clean Python interface.

Why Pulsar?
-----------

**You discover structure that traditional clustering misses.**

K-means forces your data into spheres. DBSCAN gets confused by varying density. Pulsar finds the *true topology* — manifolds, voids, intricate networks.

Here's what you'll discover:

- **Penguins**: Topology recovers species perfectly *without* looking at species labels. And reveals that island and sex are structurally as important as species.
- **MMLU**: The standard LLM benchmark hides 12 distinct clusters within 57 subjects. Reveals leaderboard blind spots.
- **Clinical Data**: Patients with identical vital signs can have opposite trajectories. Topology-aware clustering catches this.
- **Infrastructure**: Coal plants cluster by operational region and capacity, revealing hidden grid structure.

See all demos: :ref:`demos`

What is Pulsar?
---------------

Pulsar is a **Rust-accelerated Python library** for topological data analysis:

- **Input**: CSV, Parquet, or Pandas DataFrame
- **Process**: Grid sweeps over PCA dimensions and epsilon values (Ball Mapper)
- **Output**: Weighted network graph (networkx.Graph) showing cluster structure

The workflow:

.. mermaid::

   graph LR
      A["Data<br/>(CSV)"] --> B["Preprocess"]
      B --> C["PCA Grid"]
      C --> D["Ball Mapper"]
      D --> E["Cosmic Graph"]
      E --> F["Clusters &<br/>Insights"]

      style A fill:#f0f0f0,stroke:#999
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style C fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style D fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style E fill:#DFF0D8,stroke:#3C763D,stroke-width:2px
      style F fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

**Driven by AI** (recommended for exploration)

Use Pulsar with Claude AI or Gemini. Point it at your CSV and ask for insights. No code.

See: :ref:`mcp`

**Programmatic** (for reproducibility and automation)

Configure YAML or Python, fit the model, extract the cosmic graph.

.. code-block:: python

   from pulsar import ThemaRS

   model = ThemaRS("params.yaml")
   model.fit()
   cosmic = model.cosmic_graph  # networkx.Graph

Key Capabilities
----------------

**Topological Discovery**
   Ball Mapper + grid sweeps reveal manifold structure, not just spherical clusters.

**Rust Performance**
   Core algorithms in Rust via PyO3. 10-100x speedups over pure Python implementations.

**Grid-Based Exploration**
   Sweep over PCA dimensions, epsilon values, and random seeds to find robust structure.

**Temporal Data**
   TemporalCosmicGraph for 3D tensors (patient × feature × time). Discover trajectory patterns.

**AI-Assisted Analysis**
   Use with Claude Desktop or Gemini. The AI orchestrates parameter tuning and generates statistical dossiers.

**YAML Configuration**
   Declarative, reproducible pipelines. Easy to version control and share.

**Python API**
   Clean interface: ``ThemaRS.fit()`` → ``networkx.Graph``. Integrate into any pipeline.

Installation
-------------

.. code-block:: bash

   pip install pulsar

For development (requires Rust toolchain):

.. code-block:: bash

   git clone https://github.com/Krv-Labs/pulsar.git
   cd pulsar
   uv sync
   uv run maturin develop --release

Supports Python 3.10, 3.11, 3.12.

Next Steps
----------

1. **See it in action**: :ref:`demos` — five real projects you can run in minutes
2. **Understand why**: :ref:`why_pulsar` — when to use topological analysis and when to use something else
3. **Use with AI**: :ref:`mcp` — let Claude or Gemini handle the analysis
4. **Go deeper**: :ref:`user_guide` — installation, configuration, and tuning
5. **API docs**: :ref:`api-reference` — full class and function reference

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   userGuides/quickstart
   userGuides/mcp
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   userGuides/demos
   userGuides/why_pulsar
   userGuides/installation
   userGuides/programmatic
   userGuides/intermediate

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   overview
   configuration
   api

References
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
