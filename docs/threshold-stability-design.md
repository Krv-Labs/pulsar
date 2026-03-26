# Threshold Stability Analysis for Cosmic Graphs

This document describes a principled approach to selecting edge weight thresholds for Cosmic Graphs using 0-dimensional homology (connected components) and stability analysis.

## Motivation

The Cosmic Graph produces a weighted adjacency matrix `W[i,j] ∈ [0,1]` representing the likelihood that two data points (e.g., patients) belong to the same topological cluster across the parameter sweep. To obtain a binary graph for downstream analysis, we threshold this matrix:

```
A[i,j] = 1  if W[i,j] > τ
         0  otherwise
```

Currently, τ is set manually (default 0.0). This raises natural questions:
- What threshold produces meaningful clusters?
- How sensitive are our conclusions to the choice of τ?
- Can we justify our choice to clinical collaborators?

## Approach: Stability Plateau Detection

### Core Idea

As we sweep τ from 0 to 1, the graph becomes increasingly sparse and the number of connected components changes. Some threshold ranges produce *stable* structure — small perturbations to τ don't change the component count. These plateaus correspond to robust topological features.

```
Components
    │
  8 │                              ████████████
    │                         █████
  4 │              ████████████
    │         █████
  2 │    █████
    │████
  1 └────────────────────────────────────────────► τ
    0                                            1
         ▲              ▲
         │              │
    stable plateau   stable plateau
```

**Key insight**: The longest plateau represents the threshold range where the graph structure is most robust to the similarity cutoff choice.

### Why This Is Defensible

1. **Clinical interpretability**: "We selected the threshold where patient groupings are most stable — small changes in our similarity cutoff don't change who clusters together."

2. **Topological grounding**: This is equivalent to reading off 0-dimensional persistent homology. The plateau lengths are the "lifespans" of connected components in the filtration.

3. **Established methodology**: Stability-based selection is well-established in clustering (consensus clustering, stability selection, gap statistic). We're applying the same principle to graph thresholding.

4. **Visual evidence**: The component-vs-threshold curve provides immediate visual justification that non-technical stakeholders can understand.

### Connection to Persistent Homology

Sweeping τ from 1 → 0 defines a *filtration* of nested graphs:

```
G(τ=1.0) ⊆ G(τ=0.9) ⊆ ... ⊆ G(τ=0.1) ⊆ G(τ=0.0)
```

The H₀ persistence diagram records when connected components are "born" (new isolated node appears as τ decreases) and when they "die" (merge with another component). Long-lived components — those persisting across a wide range of τ — represent stable topological features.

The plateau approach is a computationally efficient way to identify these stable features without computing full persistence diagrams.

## Algorithm

### Pseudocode

```python
def find_stable_thresholds(W: np.ndarray, top_k: int = 3):
    """
    Find threshold values that produce stable connected component structure.
    
    Returns the midpoints of the k longest plateaus in the 
    components-vs-threshold curve.
    """
    n = W.shape[0]
    
    # 1. Extract unique edge weights (candidate thresholds)
    triu_indices = np.triu_indices(n, k=1)
    weights = np.unique(W[triu_indices])
    weights = np.sort(weights)[::-1]  # descending order
    
    # 2. Sweep threshold, tracking component count
    #    Use incremental union-find for efficiency
    thresholds = []
    component_counts = []
    
    uf = UnionFind(n)
    current_idx = 0
    
    for τ in weights:
        # Add all edges with weight > τ
        while current_idx < len(edge_list) and edge_list[current_idx].weight > τ:
            i, j = edge_list[current_idx]
            uf.union(i, j)
            current_idx += 1
        
        thresholds.append(τ)
        component_counts.append(uf.num_components())
    
    # 3. Find plateaus (consecutive equal component counts)
    plateaus = identify_plateaus(thresholds, component_counts)
    
    # 4. Return midpoints of top-k longest plateaus
    plateaus.sort(key=lambda p: p.length, reverse=True)
    return [p.midpoint for p in plateaus[:top_k]]
```

### Complexity

| Operation | Time Complexity |
|-----------|-----------------|
| Extract unique weights | O(n²) |
| Sort weights | O(m log m) where m = unique weights |
| Union-find operations | O(m · α(n)) ≈ O(m) |
| Plateau detection | O(m) |
| **Total** | **O(n² + m log m)** |

For n = 10,000 patients with a dense weighted adjacency matrix, this completes in under a second.

## Output

### Structured Result

```python
@dataclass
class StabilityResult:
    # Recommended threshold (midpoint of longest plateau)
    optimal_threshold: float
    
    # All plateaus found, sorted by length
    plateaus: list[Plateau]
    
    # Full curve data for visualization
    thresholds: np.ndarray
    component_counts: np.ndarray
    
@dataclass  
class Plateau:
    start_threshold: float
    end_threshold: float
    component_count: int
    
    @property
    def length(self) -> float:
        return self.start_threshold - self.end_threshold
    
    @property
    def midpoint(self) -> float:
        return (self.start_threshold + self.end_threshold) / 2
```

### Visualization

```python
def plot_stability(result: StabilityResult):
    """
    Plot component count vs threshold with plateaus highlighted.
    """
    plt.step(result.thresholds, result.component_counts)
    
    # Highlight longest plateau
    best = result.plateaus[0]
    plt.axvspan(best.end_threshold, best.start_threshold, 
                alpha=0.3, label=f"Stable region (k={best.component_count})")
    
    plt.axvline(result.optimal_threshold, linestyle='--',
                label=f"Selected τ = {result.optimal_threshold:.3f}")
    
    plt.xlabel("Threshold (τ)")
    plt.ylabel("Connected Components")
    plt.title("Threshold Stability Analysis")
    plt.legend()
```

## API Design

### Standalone Function

```python
from pulsar import find_stable_thresholds

model = ThemaRS("params.yaml").fit()

# Analyze stability
result = find_stable_thresholds(model.weighted_adjacency, top_k=3)

print(f"Recommended threshold: {result.optimal_threshold}")
print(f"This produces {result.plateaus[0].component_count} stable clusters")

# Visualize
result.plot()

# Apply the optimal threshold
optimal_graph = model.cosmic_graph_at_threshold(result.optimal_threshold)
```

### Config Integration

```yaml
cosmic_graph:
  threshold: "auto"  # Use stability-based selection
  # OR
  threshold: 0.15    # Manual threshold
```

When `threshold: "auto"`, the pipeline automatically runs stability analysis and selects the optimal threshold, logging the result.

## Extension to Temporal Cosmic Graphs

For the 3D temporal tensor `W[i,j,t]`, stability analysis extends naturally:

### Per-Timestep Analysis

Run stability analysis independently at each time step to track how the "natural" threshold evolves:

```python
optimal_thresholds = [find_stable_thresholds(W[:,:,t]).optimal_threshold 
                      for t in range(T)]
```

This reveals whether the data's intrinsic cluster structure is tightening (increasing threshold) or loosening (decreasing threshold) over time.

### Aggregated Analysis

Apply stability analysis to the aggregated graphs (persistence, mean, recency-weighted) to find thresholds appropriate for each summary:

```python
persistence_threshold = find_stable_thresholds(tcg.persistence_matrix()).optimal_threshold
recency_threshold = find_stable_thresholds(tcg.recency_matrix(decay=0.9)).optimal_threshold
```

### Joint Stability

For advanced use cases, find thresholds that produce stable structure *across multiple time steps*:

```python
def find_temporally_stable_threshold(W_tensor):
    """Find τ where component count is stable across both τ and time."""
    # Compute component count surface: f(τ, t) → k
    # Find τ values where variance over t is minimized
    ...
```

## Implementation Notes

### Rust Acceleration

The core algorithm is a good candidate for Rust implementation:

1. **Union-find**: Already have `ndarray`, easy to add a simple union-find struct
2. **Parallel sweeps**: For temporal data, parallelize across time steps with rayon
3. **Memory efficiency**: Process edges in streaming fashion without materializing full edge list

### Edge Cases

- **Disconnected at τ=0**: If the graph has multiple components even with no threshold, all nodes in a component have `W[i,j] = 0` with nodes in other components. This is valid — report it.

- **Single component throughout**: If the graph stays connected for all τ, there are no plateaus. Return τ=max(W) as the threshold (just below which the graph fragments).

- **Ties**: Multiple plateaus of equal length. Return all of them; let the user choose based on desired cluster count.

## References

1. **Stability-based clustering**: Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-based validation of clustering solutions.

2. **Persistent homology**: Edelsbrunner, H., & Harer, J. (2010). Computational topology: an introduction.

3. **Consensus clustering**: Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003). Consensus clustering: a resampling-based method for class discovery.

## Next Steps

1. Implement `find_stable_thresholds()` in Python (prototype)
2. Add visualization utilities
3. Port core algorithm to Rust for performance
4. Integrate with `ThemaRS` config (`threshold: "auto"`)
5. Extend to temporal cosmic graphs
6. Write tests with synthetic data (known cluster structure)
7. Validate on clinical datasets
