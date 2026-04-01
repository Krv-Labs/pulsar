---
name: MCP module current state (2026-04-01)
description: Accurate summary of what exists in pulsar/mcp/ as of the sg_llm_demo branch review
type: project
---

As of 2026-04-01 on branch sg_llm_demo, pulsar/mcp/ contains:
- server.py (478 lines): 6 tools — run_topological_sweep, generate_cluster_dossier, compare_clusters_tool, export_labeled_data, characterize_dataset, diagnose_cosmic_graph. Session state implemented via _PulsarSession dataclass + _sessions dict.
- diagnostics.py (380 lines): SweepHistoryEntry, GraphMetrics, DiagnosisResult dataclasses; diagnose_model(), _classify(), _history_aware_epsilon(), _build_diagnosis(), _build_clustering_notes().
- interpreter.py (467 lines): resolve_clusters(), _cluster_by_components(), _cluster_by_spectral(), build_dossier(), dossier_to_markdown(), compare_clusters(), comparison_to_markdown().
- errors.py: .pyc exists but no .py source — effectively deleted. Error handling is now inline (return strings) in each tool function.

**Why:** The tools suggest_initial_config, explain_suggestion, and get_experiment_history mentioned in prior memory were removed. Session state global-bug was fixed with _PulsarSession dataclass.

**How to apply:** Do not reference suggest_initial_config, explain_suggestion, or get_experiment_history as existing tools. Do not suggest the session-state concurrency fix — it is already done. Focus review on issues that remain in current code.
