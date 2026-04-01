# Pulsar MCP: Penguins Discovery Demo

This demo walks through the use of the **Pulsar MCP Server** to perform unsupervised topological analysis on the classic Palmer Penguins dataset.

## 1. What is Pulsar & When to Use It
Pulsar is a geometric deep learning toolkit that discovers latent structure in complex, high-dimensional datasets. Unlike traditional clustering (K-Means), Pulsar uses **Topological Data Analysis (TDA)** to build a "Cosmic Graph" where edges represent shared geometric neighborhoods across a sweep of scales.

## 2. Prerequisites
- Python 3.10+
- `pulsar-tda` installed in your environment.
- An MCP-compatible LLM client (like Claude Desktop or Gemini).

## 3. Set Up & Start the MCP Server
If you are developing locally, you can start the server via:
```bash
python -m pulsar.mcp.server
```

## 4. Run Your First Analysis (Penguins Walkthrough)
In this demo, we perform a "blind" discovery to see if Pulsar can discover the taxonomic groups without explicit labels. 

We use the following optimal parameter configuration, which balances the continuous physical traits against the categorical variables:

```yaml
run:
  name: penguin_strict
  data: /Users/gathrid/Repos/pulsar/demos/MCP-penguins/penguins.csv
preprocessing:
  drop_columns: ['species', 'rowid', 'year']
  impute:
    bill_length_mm: {method: fill_mean, seed: 42}
    bill_depth_mm: {method: fill_mean, seed: 42}
    flipper_length_mm: {method: fill_mean, seed: 42}
    body_mass_g: {method: fill_mean, seed: 42}
  encode:
    island: {method: one_hot}
    sex: {method: one_hot}
sweep:
  pca:
    dimensions:
      values: [3]
    seed:
      values: [42]
  ball_mapper:
    epsilon:
      range:
        min: 0.5000
        max: 0.8000
        steps: 10
cosmic_graph:
  threshold: 0.4
output:
  n_reps: 4
```

## 5. Understanding Results
The topological graph perfectly partitions the penguins into **4 major groups**. Interestingly, because we kept `sex` and `island` alongside the physical traits, the topology reveals that **sexual dimorphism** (male vs. female size differences) is a driving structural factor in the population:

- **Cluster 1 (~32% of data):** The **Female Adelie/Chinstrap** group. Defined by lower body mass (~3421g) and short flippers (~188mm).
- **Cluster 0 (~31% of data):** The **Male Adelie/Chinstrap** group. Defined by higher mass (~4010g) and thicker bills (~19.1mm).
- **Cluster 6 (~17% of data):** The **Male Gentoo (Biscoe Island)** group. Characterized by massive flippers (~221mm) and high body mass (~5484g).
- **Cluster 5 (~17% of data):** The **Female Gentoo (Biscoe Island)** group. Characterized by medium/large flippers (~212mm) and medium mass (~4679g).

*(The remaining 5 nodes are singletons consisting of rows with heavily missing data or extreme outliers).*

The **Topological Analysis Dossier** helps prove these differences by surfacing:
- **Z-Scores**: How far a cluster's mean is from the global average.
- **Homogeneity**: How "tight" the cluster is (low variance).
