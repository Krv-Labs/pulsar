# Python Code Simplifier - Pulsar Project Memory

## Key Issues in pulsar/mcp/ Integration (2026-03-31)

### Critical Pattern: Global State in FastMCP
- **Issue**: Module-level globals (_last_model, _last_data, _last_clusters) in server.py will fail under concurrent requests
- **Fix Strategy**: Extract SessionStore class per session_id from FastMCP context
- **Why**: FastMCP is designed for concurrent servers; globals violate this contract
- **Reference**: pulsar/mcp/server.py lines 19-23

### Error Handling Anti-Pattern
- **Issue**: Bare `except:` and `except Exception` clauses hide bugs, catch KeyboardInterrupt
- **Why**: Masks programming errors; makes production debugging impossible
- **Pattern to Apply**: Catch specific exceptions (FileNotFoundError, ValueError, nx.NetworkXError)
- **How to apply**: Use structured logging with context; let programming errors propagate

### Private Attribute Access
- **Issue**: Direct access to `model._data` breaks if ThemaRS refactors
- **Solution**: Add public property `@property def fitted_data()` to ThemaRS
- **Why**: Signals API contract; survives refactoring
- **Reference**: pulsar/mcp/server.py line 49

### Index/Data Alignment Risk
- **Issue**: build_dossier() assumes clusters.shape == data.shape but doesn't validate
- **Pattern**: Add `assert len(clusters) == len(data)` at function entry
- **Why**: Silent data corruption if shapes diverge (wrong cluster assigned to wrong row)
- **Reference**: pulsar/mcp/interpreter.py lines 104-106, 124-130

### Magic Numbers in resolve_clusters()
- **Issue**: Hardcoded thresholds (1 < n_comp < 50, singleton_count/n < 0.5, k range 2-15) lack documentation
- **Solution**: Extract to configurable ClusteringStrategy class with documented rationale
- **Why**: Current approach is unmaintainable; can't tune for different dataset sizes without code changes

## Pulsar Architecture Notes

- **ThemaRS**: Entry point at `pipeline.py`; stores _data (original after copy), cosmic_graph, weighted_adjacency, _resolved_threshold
- **Pipeline**: CSV/Parquet → Impute → Scale → PCA Grid → Ball Mapper Grid → Pseudo-Laplacian → Cosmic Graph
- **Rust/Python split**: Computation in Rust (_pulsar module); orchestration in Python
- **Hooks pattern**: pulsar/hooks.py provides label_points, membership_matrix, cosmic_clusters on graph outputs
- **Testing pattern**: Use conftest.py fixtures; validation tests in tests/correctness/; unit tests in tests/

## MCP Integration Notes

- **FastMCP**: Expects stateless tool functions; server.py initializes `mcp = FastMCP("Pulsar")`
- **Dependencies**: fastmcp>=0.1, mcp>=0.1 in pyproject.toml dependency group "mcp"
- **CLI script**: pulsar-mcp = "pulsar.mcp.server:main" in pyproject.toml
- **Three tools**: run_topological_sweep, generate_cluster_dossier, export_labeled_data
- **Current flow**: sweep → dossier → export (each tool depends on prior state)

## Recommendations Priority

**Priority 1**: Session manager (concurrency bug) → Input validation → Logging
**Priority 2**: ClusteringStrategy enum → Public API for ThemaRS → Type hint completeness
**Priority 3**: Remove unused parameter k → Optimize subgraph extraction → String formatting

## Code Patterns to Follow

- Use `pd.Series.unique()` instead of `np.unique()` for cluster IDs when working with clusters
- Use numpy boolean indexing: `np.where(clusters.values == cid)[0]` not enumerate loops
- Use `from __future__ import annotations` for forward references (already done)
- Test fixtures in conftest.py (small_array, array_with_nans, pseudo_laplacian_py)
- Validation tests separate from unit tests (tests/correctness/ vs tests/)
