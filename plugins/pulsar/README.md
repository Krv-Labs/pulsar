# Pulsar plugin

Topological data analysis for tabular data, driven by your agent. Point it at a CSV or Parquet file
and it maps the shape of the data: groups, bridges between them and outliers, with calibrated
measurements, threshold stability, cluster evidence and provenance. Pulsar reports measurements; you
and the agent supply the domain meaning.

The plugin adds:

- **The Pulsar MCP server** (`@krv-labs/pulsar`, run with `npx`, so Node.js must be installed). One prebuilt
  binary per platform: macOS arm64/x64, Linux arm64/x64 (glibc and musl), Windows x64/arm64. No
  Rust or Python needed.
- **The `topological-analysis` skill**: the ingest, plan, calibrate, sweep, diagnose and explain
  loop, so the agent uses the tools in a sensible order.

Data stays on your machine: the server reads files from your disk and keeps its runs in a local
store.

## Install

```
/plugin marketplace add Krv-Labs/pulsar
/plugin install pulsar@krv-labs
```

Then ask something like "what kinds of rows are in `data/customers.csv`, and what separates them?".

## More

Documentation, the CLI and other agents: <https://github.com/Krv-Labs/pulsar>. License: BSD
3-Clause.
