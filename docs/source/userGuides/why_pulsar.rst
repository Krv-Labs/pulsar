.. _why_pulsar:

===========
Why Pulsar?
===========

Pulsar solves a specific, hard problem: **finding real structure in high-dimensional data**.

Traditional clustering (K-means, hierarchical clustering, DBSCAN) assumes your data fits neat spheres or simple geometric shapes. But real data often has **manifolds, filaments, voids, and intricate topology** that these methods miss. Pulsar uses **topological data analysis** to recover the true shape.

---

The Problem: Why Traditional Clustering Breaks Down
====================================================

Imagine you have a dataset of patient health records (100 features), or text embeddings (384 dimensions), or sensor readings across time. You want to find meaningful subgroups.

**K-means will:**

- Force your data into *k* spheres, regardless of true structure
- Make you guess *k* in advance (or run it many times)
- Miss elongated clusters, holes, and manifold structure
- Treat all dimensions equally, even if some are noise

**Pulsar does something different:**

Pulsar finds the true *topological structure* — manifolds, voids, networks. Instead of forcing spheres, it respects the geometry of your data.

K-means says: "I see three groups." Pulsar says: "Your data is actually a network of 47 interconnected nodes with distinct communities, separated by natural density gaps."

---

The Pulsar Approach: Topological Data Analysis via Ball Mapper
==============================================================

Pulsar uses the **Ball Mapper algorithm** to recover the true topology:

1. **Sample-aware centers**: Pick centers greedily (largest gaps first), not randomly
2. **Overlapping balls**: Cover your data with overlapping hyperspheres of radius *epsilon*
3. **Connectivity graph**: Build a graph where nodes are balls, edges connect overlapping balls
4. **Weighted Laplacian**: Accumulate information across *many* epsilon values (grid sweep)
5. **Cosmic graph**: Normalize and threshold to get the final structure

**Why this works:**

- **Topology is intrinsic**: The shape you discover is independent of your choice of embedding or coordinate system
- **Grid sweeps find robustness**: Not relying on a single epsilon value; you see what persists across scales
- **Spectral clustering captures communities**: The Laplacian's eigenvectors reveal natural partitions without forcing spheres

---

Why It's Cool: Real-World Payoffs
=================================

**1. Biology Without Labels (Palmer Penguins)**

You have penguin measurements (bill length, flipper length, body mass, etc.). You don't tell Pulsar which species is which. It discovers three clusters that *perfectly recover the species* — plus revealing that island and sex are just as important structurally.

**Traditional clustering**: "I see three groups of similar penguins."
**Pulsar**: "I see three structurally distinct phenotypes. One is isolated on a specific island. Another splits by sex."

**2. Research Blind Spots (MMLU Benchmark)**

MMLU is the standard LLM benchmark: 57 subjects, one leaderboard number. Pulsar reveals:

- The true structure is **12 geometric clusters**, not 57 subjects
- `moral_scenarios` is completely isolated (different cognitive domain entirely)
- `professional_law` is the tightest cluster
- **The leaderboard hides regional accuracy gaps**: Different models do much better in some regions than others

**Traditional approach**: "GPT-4 gets 86.4%, Claude gets 84.2%."
**Pulsar**: "GPT-4 dominates in Mathematics (98%) and Law (95%) but struggles in Moral Reasoning (62%). Claude shows opposite strengths."

**3. Clinical Early Warning (PhysioNet Trajectories)**

Two patients have identical vital signs right now: HR 88, BP 120/80, SpO₂ 96%. But one is recovering from sepsis, the other is about to decline. You can't tell from the snapshot.

Pulsar's temporal analysis clusters patients by **trajectory archetype** (recovery vs. decline vs. stable). Early warning emerges from the trajectory, not from any single vital.

**Traditional approach**: "These two patients look the same now."
**Pulsar**: "Different trajectory clusters. Patient A is trending toward normal; Patient B is approaching a cliff."

**4. Infrastructure Insights (Coal Plants)**

You have 147 coal plants with location, capacity, age, emissions. Pulsar reveals:

- Plants cluster by **operational region and capacity tier**, not ownership
- Geographic structure aligns with electricity market zones
- Age/emissions profiles separate active vs. retiring cohorts

**Traditional approach**: "Here are the plants grouped by company."
**Pulsar**: "Here's the underlying grid topology and market structure hidden in the data."

---

When to Use Pulsar
==================

**Use Pulsar if you have:**

- **High-dimensional data** (>5 dimensions) with unknown structure
- **Complex topology** (not just sphere-like clusters)
- **Manifold or network structure** you want to visualize and understand
- **Multiple competing embeddings** (different models, different feature sets) and you want to compare what they "see"
- **Time-series or longitudinal data** (TemporalCosmicGraph for 3D tensors)
- **Real data** (not synthetic or perfectly separated)

**Don't use Pulsar if:**

- Your data is already cleanly separated (K-means works fine)
- You have fewer than ~20 points (not enough to estimate local topology)
- You need real-time inference (Pulsar is a discovery tool, not a live predictor)
- You're doing supervised classification (use a neural network instead)

**Decision Tree**

.. code-block:: text

   Do you know the structure of your data?
   ├─ YES (clear classes, known separations)
   │  └─ Use supervised learning (random forest, neural net)
   │
   └─ NO (unknown structure, high-dimensional)
      ├─ Is it time-series / longitudinal?
      │  └─ YES → Use TemporalCosmicGraph (3D tensors)
      │  └─ NO → Use standard ThemaRS (2D features)
      │
      ├─ Do you have 1000+ points?
      │  └─ YES → Pulsar is great (faster, more stable)
      │  └─ NO → Pulsar still works, but UMAP/t-SNE faster for viz
      │
      └─ Is the structure complex / non-convex?
         └─ YES → Use Pulsar (Ball Mapper + cosmic graph)
         └─ NO → K-means/GMM likely sufficient

---

Pulsar vs. Alternatives
=======================

.. list-table::
   :widths: 20 20 30 30
   :header-rows: 1

   * - Approach
     - Structure Type
     - Speed
     - Use Case
   * - **K-means**
     - Spherical clusters
     - Fast
     - Quick EDA, known k
   * - **DBSCAN**
     - Density-based
     - Fast
     - Outlier detection
   * - **UMAP**
     - Visualization
     - Very fast
     - 2D/3D projection (no clustering)
   * - **Spectral Clustering**
     - Graph-based
     - Moderate
     - If you have an adjacency matrix
   * - **Pulsar (Ball Mapper)**
     - **Topological, manifold-aware**
     - **Moderate–Slow (grid sweep)**
     - **Discovery, structure visualization, publication**

---

Architecture at a Glance
========================

Pulsar chains these stages:

.. code-block:: text

   Raw Data (CSV)
     ↓ [Preprocessing: impute missing, encode categorical]
     ↓ [Scale: standardize to z-scores]
     ↓ [PCA: reduce to k dimensions for noise control]
     ↓ [Ball Mapper: build overlapping covers at many epsilon values]
     ↓ [Accumulate Laplacian: pool information across grid]
     ↓ [Threshold: find connected components or spectral clusters]
     ↓ [Cosmic Graph: the final result (networkx.Graph)]

Each stage is optimized in Rust and parallelized with `rayon`. The Python layer orchestrates.

**Key insight**: The grid sweep (multiple epsilon values, multiple PCA dimensions, multiple random seeds) is *essential*. A single ball map can be misleading; the grid reveals robustness.

---

When Pulsar Shines: Real Examples
=================================

**Penguins**: "My data has unexpected structure."

Pulsar reveals that island and sex are as important as species, changing how you interpret the biology.

**MMLU**: "My benchmark has blind spots."

Pulsar uncovers that moral reasoning is a separate cognitive domain, and different models have opposite strengths in different regions.

**Clinical Data**: "I need early warning signals."

Pulsar clusters trajectories, not snapshots, revealing which patients are on a divergent path.

**Energy**: "I want to understand my infrastructure."

Pulsar reveals the underlying grid topology and market structure hidden in operational data.

---

Next Steps
==========

1. **See it in action**: Start with the :ref:`demos` section — each one is runnable in minutes.
2. **Use with AI**: Set up the :ref:`mcp` server and let Claude or Gemini handle the parameter tuning.
3. **Understand the parameters**: :ref:`intermediate` explains what PCA dimensions and epsilon do.
4. **Deep dive on theory**: The `Pulsar paper <https://www.nature.com/articles/s41567-024-02449-x>`_ (Nature Physics 2024) covers the mathematical foundations.
