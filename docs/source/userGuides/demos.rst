.. _demos:

=================================
Demos
=================================

Pulsar shines when you have real data and real questions. Below are five production demos that showcase different aspects of topological data analysis — from recovering hidden biology to revealing benchmark structure to analyzing clinical trajectories.

Each demo is self-contained and runnable in minutes. Pick one that matches your domain and see the insights Pulsar reveals.

---

1. Palmer Penguins: Recovering Biology Without Labels
=====================================================

.. raw:: html

   <div style="background-color: #f8f9fa; padding: 1em; border-left: 4px solid #0066cc; margin-bottom: 1.5em;">
   <strong>The Hook:</strong> Can topology rediscover penguin species without looking at species labels? Or discover that habitat and sex are equally important structurally?
   </div>

**The Data**

The `Palmer Penguins <https://allisonhorst.com/project/penguins/>`_ dataset contains 333 penguins from three species (Adelie, Chinstrap, Gentoo) with 8 morphological measurements: bill length, bill depth, flipper length, body mass, and more. It's the ideal educational dataset — real biology, no missing structure, universally understood.

**The Discovery**

After dropping species labels entirely and letting Pulsar discover structure in the remaining 5-dimensional feature space:

- **The Gentoos**: Completely isolated on Biscoe Island, then perfectly separated by sex. (They are chunky birds with distinctive morphology.)
- **The Adelies**: Fragmented by island of origin — the structural variation within the species is as important as the species itself.
- **The Chinstraps**: Indistinguishable from Dream Island Adelies. They share the same morphological envelope, so the math doesn't lie.

**Key Insight**: Topology reveals that **habitat and biological sex are as structurally important as species itself**. Traditional clustering (K-means) would force three spheres; topology shows the actual complexity.

**Try It Now**

This is the fastest way to see Pulsar in action. No dataset to download.

.. code-block:: bash

   # Option 1: Use Pulsar with Claude AI (recommended)
   # Install Pulsar MCP server in Claude Desktop (see :ref:`mcp` guide)
   # Then ask Claude: "Use Pulsar to analyze the penguin data at demos/penguins/penguins.csv"

   # Option 2: Run directly with Python
   cd /path/to/pulsar
   uv sync
   uv run maturin develop --release
   uv run python -c "
   from pulsar.pipeline import ThemaRS
   config = {'run': {'name': 'penguins', 'data': 'demos/penguins/penguins.csv'}}
   model = ThemaRS.from_dict(config)
   model.fit()
   print(f'Discovered {len(model.cosmic_graph.nodes())} nodes and {len(model.cosmic_graph.edges())} edges')
   "

**Deep Dive**

- Full walkthrough: `demos/penguins/README.md <https://github.com/Krv-Labs/pulsar/tree/main/demos/penguins>`_
- YAML configuration: `demos/penguins/params.yaml <https://github.com/Krv-Labs/pulsar/tree/main/demos/penguins>`_
- Notebook: The penguins example is also the starting point in the :ref:`mcp` guide

---

2. MMLU Benchmark Topology: 57 Subjects, 12 True Clusters
==========================================================

.. raw:: html

   <div style="background-color: #f8f9fa; padding: 1em; border-left: 4px solid #0066cc; margin-bottom: 1.5em;">
   <strong>The Hook:</strong> MMLU is the standard LLM benchmark: 57 subjects, one leaderboard number. What if the real structure doesn't match those labels?
   </div>

**The Data**

MMLU consists of ~14,000 test questions across 57 administrative subjects (professional medicine, history, chemistry, law, etc.). We embed all questions using `bge-small-en-v1.5` (384-dimensional sentence embeddings) and run Pulsar's topological sweep.

**The Discovery**

The geometric structure in embedding space reveals **12 distinct regions** that cut across subject boundaries:

.. list-table:: MMLU's Hidden Structure
   :widths: 10 50 40
   :header-rows: 1

   * - Region
     - Theme
     - Top Subjects
   * - 0
     - Psychology / Behavioral
     - professional_psychology, hs_psychology
   * - 1
     - Medicine / Health
     - professional_medicine, nutrition, clinical_knowledge
   * - 2
     - Mathematics / Quantitative
     - elementary_math, hs_math, hs_statistics
   * - 3
     - Moral Reasoning
     - **moral_scenarios (100% isolated)**
   * - 5
     - Law
     - **professional_law (87% of region)**
   * - 8
     - History
     - hs_world_history, hs_us_history

**Key Insights**:

- `moral_scenarios` forms a completely isolated island — structurally alien to the rest of MMLU
- `professional_law` is the tightest cluster (87% of Region 5)
- Psychology splits: behavioral questions in Region 0, philosophical in Region 7
- **Leaderboard blind spot**: Different LLMs have vastly different accuracy across regions. The single benchmark number hides this variation.
- Random sampling needs **3x more questions** than topology-aware sampling to cover all 12 regions

**Try It Now**

Jupyter notebook with full analysis and per-model evaluation:

.. code-block:: bash

   cd demos/mmlu
   uv sync --group demos
   uv run maturin develop --release
   jupyter notebook mmlu_topology_demo.ipynb

First run downloads and embeds ~14k questions (~2 min on Apple Silicon). Subsequent runs use cached data.

**Deep Dive**

- Full README with calibration details: `demos/mmlu/README.md <https://github.com/Krv-Labs/pulsar/tree/main/demos/mmlu>`_
- Jupyter notebook: `mmlu_topology_demo.ipynb <https://github.com/Krv-Labs/pulsar/tree/main/demos/mmlu>`_
- Configuration: `mmlu_params.yaml <https://github.com/Krv-Labs/pulsar/tree/main/demos/mmlu>`_

---

3. Clinical Trajectories: PhysioNet ICU Vitals Over Time
========================================================

.. raw:: html

   <div style="background-color: #f8f9fa; padding: 1em; border-left: 4px solid #0066cc; margin-bottom: 1.5em;">
   <strong>The Hook:</strong> Two patients with identical vital signs right now might have completely different futures. Can topology reveal their trajectory archetypes?
   </div>

**The Data**

The demo simulates 500 ICU patients over 72 hours with 8 vital signs: heart rate, systolic/diastolic BP, MAP, respiratory rate, temperature, SpO₂, lactate, glucose. Five distinct clinical archetypes are embedded in the synthetic trajectories (sepsis progression, recovery, decline, stable, recovery-plateau).

This demonstrates **TemporalCosmicGraph** — a 3D tensor approach (patient × feature × time) that captures patient-level temporal patterns, not just snapshots.

**The Discovery**

- Patients cluster by **trajectory type**, not current state. A recovering patient and a declining patient may have identical vitals right now but opposite futures.
- Multiple aggregation modes reveal different groupings:
  - **Persistence** → stable vs. volatile patients
  - **Trend** → improving vs. worsening trajectories
  - **Volatility** → high-risk vs. stable
  - **Change point** → when trajectory shifts occur
- Early warning signals emerge from trajectory clustering, not from any single vital.

**Try It Now**

With synthetic data (no real PHI):

.. code-block:: bash

   cd /path/to/pulsar
   uv sync
   uv run maturin develop --release
   uv run python demos/ehr/physionet.py --synthetic --n-patients 500

With real eICU data (if you have access via PhysioNet):

.. code-block:: bash

   # First download eICU from https://physionet.org
   uv run python demos/ehr/physionet.py --data /path/to/eicu.csv

**Deep Dive**

- Script: `demos/ehr/physionet.py <https://github.com/Krv-Labs/pulsar/tree/main/demos/ehr>`_
- Configuration: `physionet_params.yaml <https://github.com/Krv-Labs/pulsar/tree/main/demos/ehr>`_
- Configuration: `physionet_params.yaml <https://github.com/Krv-Labs/pulsar/tree/main/demos/ehr>`_

---

4. ECG Arrhythmia Classification via Feature Extraction
=======================================================

.. raw:: html

   <div style="background-color: #f8f9fa; padding: 1em; border-left: 4px solid #0066cc; margin-bottom: 1.5em;">
   <strong>The Hook:</strong> 60,000 raw ECG samples per patient. Can a compact feature vector capture enough to cluster arrhythmias?
   </div>

**The Data**

ECG (electrocardiogram) signals from the `PhysioNet Arrhythmia Database <https://physionet.org>`_: 12-lead recordings at 500 Hz, 10-second windows = 5,000 samples per lead, per patient. The demo extracts ~80 summary features per ECG:

- Statistical: mean, std, min, max, median, skewness, kurtosis
- Frequency: FFT peaks, power spectral density
- Morphological: zero crossings, rate of change statistics

**The Discovery**

- Topology reveals clusters that **align with SNOMED-CT arrhythmia diagnoses** better than K-means or traditional clustering
- Different leads emphasize different diagnostic features — combining all 12 leads captures the full arrhythmia signature
- Trade-off: Feature extraction is computationally efficient vs. true temporal modeling (TemporalCosmicGraph), with minimal loss in structure discovery

**Try It Now**

With synthetic ECG patterns:

.. code-block:: bash

   uv run python demos/ehr/ecg_arrhythmia.py --synthetic

With real PhysioNet data:

.. code-block:: bash

   # Download from https://physionet.org (requires registration)
   uv run python demos/ehr/ecg_arrhythmia.py --data /path/to/ecg_data

**Deep Dive**

- Script: `demos/ehr/ecg_arrhythmia.py <https://github.com/Krv-Labs/pulsar/tree/main/demos/ehr>`_
- Configuration: Hardcoded in the script; adjust PCA dimensions and epsilon range as needed

---

5. US Coal Plants: Production-Scale Grid Sweep
==============================================

.. raw:: html

   <div style="background-color: #f8f9fa; padding: 1em; border-left: 4px solid #0066cc; margin-bottom: 1.5em;">
   <strong>The Hook:</strong> Real infrastructure data at scale. How do operational coal plants cluster when you account for location, capacity, age, emissions, and status?
   </div>

**The Data**

147 US coal power plants with features: latitude, longitude, capacity (MW), age, emissions (CO₂, NOx, SO₂), operational status, retire year (if planned). Dataset is automatically downloaded from the `retire <https://github.com/Krv-Labs/retire>`_ project. Real-world, production-scale problem.

**The Discovery**

- Plants cluster by **operational region** and **capacity tier**, not administrative ownership
- Age and emissions profiles separate active vs. retiring cohorts
- Geographic clustering aligns with grid topology and energy markets
- The full sweep (4 PCA dims × 8 seeds × 50 epsilons = 4,000 ball maps) approximates the cosmic graph from the original `Pulsar Nature paper <https://www.nature.com/articles/s41567-024-02449-x>`_

**Try It Now**

Automatic dataset download, grid search, and timing report:

.. code-block:: bash

   uv run python demos/energy/coal.py

The demo prints per-stage wall-clock timings (preprocessing, PCA, Ball Mapper, graph accumulation, thresholding) and the final graph size. On a modern machine: ~2–5 seconds for the full 4,000-map sweep.

**Deep Dive**

- Script: `demos/energy/coal.py <https://github.com/Krv-Labs/pulsar/tree/main/demos/energy>`_
- Configuration: `coal_params.yaml <https://github.com/Krv-Labs/pulsar/tree/main/demos/energy>`_
- Data: automatically downloaded on first run from `retire project <https://github.com/Krv-Labs/retire>`_

---

Choosing Your Demo
==================

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Domain
     - Demo
     - Why Choose It
   * - Education / Getting Started
     - Palmer Penguins
     - Fastest, most intuitive
   * - Research / Benchmarks
     - MMLU
     - Reveals hidden structure
   * - Healthcare / Trajectories
     - PhysioNet (Clinical)
     - Time-series aware
   * - Healthcare / Signals
     - ECG Arrhythmia
     - Feature engineering
   * - Infrastructure / Scale
     - Coal Plants
     - Real-world, production-ready

---

Next Steps
==========

Once you've explored a demo:

1. **Use with Claude AI**: Set up the :ref:`mcp` server and point Claude at your own data. The AI will handle parameter tuning and generate statistical dossiers.
2. **Adapt for your data**: Copy the nearest demo's YAML config and adjust for your feature scales and desired PCA dimensions.
3. **Deep dive on parameters**: See :ref:`intermediate` for guidance on tuning epsilon ranges and dimension selection.
4. **Deploy to production**: The coal demo shows how to instrument timing and validation. See :ref:`intermediate` for configuration and parameter guidance.
