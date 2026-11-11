---
name: topological-analysis
description: Map the shape of a tabular dataset (CSV or Parquet) with Pulsar's topology tools - find groups, bridges and outliers, then explain what separates them. Use when the user wants to explore structure, segments, cohorts or subgroups in a table, asks what kinds of rows/records/patients/customers a file contains, or wants clustering they can defend.
---

# Topological analysis with Pulsar

Pulsar builds a calibrated graph of the rows in a table (a "cosmic graph" over many projections and
Ball Mapper scales) and reports measurements: graph state, threshold stability, cluster evidence and
provenance. It does not interpret them. You and the user supply the domain meaning.

Call `get_workflow_guide` once at the start of a session: it is the full procedure and tool map.
The outline below is the loop it describes.

## The loop

1. **Ingest** the file with `ingest_dataset(path)` (a path the server can read) and keep the
   `dataset_id`.
2. **Look** with `characterize_dataset` and, for specific columns, `probe_columns`.
3. **Plan preprocessing** with `draft_preprocessing_plan`. Read its open questions and
   `evidence.target_proxies` first. Answer what you can infer with `edit_preprocessing_plan`
   (`source="agent"`); ask the user only questions whose answer you cannot infer and whose stakes
   are material, in their own vocabulary, never in the tool's wording.
4. **Calibrate** with `create_config(dataset_id, plan_id=...)`. Epsilon and projection dimensions
   are measured from the data, not guessed. Leave `construction_threshold` on `"auto"` unless the
   user has a reason.
5. **Sweep** with `run_topological_sweep`. For large grids use `execution="async"` and poll with
   `wait_job`.
6. **Diagnose** with `diagnose_cosmic_graph` and `get_threshold_stability_curve` before any
   clustering. A giant hairball or all-singleton dust means the clusters will be too: adjust with
   `refine_config` and sweep again. Two or three deliberate sweeps beat one lucky one.
7. **Explain** with `generate_cluster_dossier(detail="summary")`, then `get_cluster_profile`,
   `get_feature_signal`, `get_cluster_signal_matrix` and `compare_clusters`. Describe each group by
   what separates it from its topological neighbours, not in isolation.
8. **Export** a labeled CSV, a Parquet bundle or a standalone HTML report when the user wants to
   keep the result.

Long-format panels (one row per entity and time step) take the longitudinal tools instead:
`build_longitudinal_graph`, then `diagnose_longitudinal_graph`, `get_trajectory_archetypes`,
`get_cross_time_neighbors` and `classify_trajectories`.

## Ground rules

- **Measurements, not conclusions.** Report what the tools measured and how stable it is; name
  groups only from the evidence and the user's domain knowledge.
- **One component can be the right answer.** Do not force a split the data does not support.
- **Keep outcomes out of the geometry.** Drop a label or outcome column from the features with
  `refine_config` (`preprocessing.drop_columns`); it still appears in the cluster evidence.
- **Errors say what to do.** Every error carries an `error_code` and an `agent_action`: act on
  the `agent_action` before retrying.
