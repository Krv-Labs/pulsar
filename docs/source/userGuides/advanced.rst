.. _advanced:

========
Advanced
========

Advanced topics include custom imputation strategies, reproducibility settings, and graph interpretation.

- Use ``seed`` values in each method to fix deterministic outputs.
- Combine ``run``, ``preprocessing``, and ``sweep`` sections for full experiment bookkeeping.
- Reuse the same ``ThemaRS`` instance once fit to compare graph statistics through ``stability_result``.

Troubleshooting
---------------

- Ensure numeric inputs are clean and free of mixed types.
- Prefer parquet for large datasets to speed input parsing.
