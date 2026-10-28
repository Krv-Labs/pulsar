# Pulsar: MCP and agent-capability roadmap

Goal: the best topology workbench an agent can drive, on the current MCP
protocol (2026-07-28, tasks extension) and `rmcp` 3.5, designed around what
frontier models do well: long tool loops, structured data, judgment under
uncertainty. Items marked *(verify)* depend on spec or SDK details to confirm
against the live spec before building.

Principles carried over: tools expose **measurements, not conclusions**;
payloads are **summary-first and bounded**; errors carry an `agent_action`.

## Track A: the port (done)

Every tool of the Python server is ported and proven against frozen outputs
of the Python reference (see `tests/fixtures/golden/README.md`), except the
chunked-upload tools, which were dropped (ADR 0002).

## Track B: serving

- **Streamable HTTP** next to stdio, with per-session state keyed by session
  id (today: one global session). Bearer auth from an identity provider; `healthz` and a `get_runtime_context` that reports version,
  threads, cache occupancy and limits.
- **Job control:** `cancel_job` with a cooperative flag checked between grid
  cells; a bounded queue and concurrency limit so two sweeps cannot
  oversubscribe the rayon pool; sweep checkpoints so an interrupted job
  resumes instead of being marked `INTERRUPTED`.
- **Resources for large artifacts:** return `resource_link`s
  (`pulsar://runs/{id}/graph`, `.../dossier`) instead of inlining; paginated
  reads. *(verify resource-link and subscription behaviour in rmcp 3.5)*
- **Persisted projection cache:** Arrow IPC sidecar keyed by projection
  fingerprint, with size cap and eviction (in-memory cache already serves any
  live run by fingerprint).

## Track C: model-facing capability

- **Token budgets:** accept a `max_tokens`-style hint and size summaries to it,
  generalizing today's fixed 100 KB backstop; every truncated list reports
  returned/omitted counts.
- **Elicitation for plan questions:** the plan's open questions (blank meaning,
  code columns, outcome columns) are exactly "ask the human in domain words";
  surface them through MCP elicitation where the client supports it, fall
  back to the current tool-result form. *(verify client support matrix)*
- **Prompts:** serve the workflow guide as an MCP prompt as well as a tool, so
  hosts with prompt UIs can offer it; keep `instructions` a terse tool map.
- **Sampling, opt-in only:** let the server ask the client's model to *propose*
  cluster names from a dossier, always labeled as a proposal, never in default
  payloads (measurements, not conclusions). *(verify)*
- **Decision packets:** every analysis tool ends with compatible `next_tools`
  and the gate fields an agent needs to decide to stop (finalization gates,
  grid adequacy), as `get_threshold_stability_curve` does today.

## Track D: evaluation (what "best for models" means, measurably)

- An eval suite in the style of the `mcp-builder` skill: 10 to 20 read-only,
  multi-step questions over fixed datasets with verifiable answers, run against
  the built server through a real model. Track pass rate, tool calls per
  task, tokens per task and error-recovery rate per release.
- Contract tests keep the catalog under its character budget; raise the budget
  deliberately, never by accident.
- Criterion benchmarks as a gate for the hot paths (projection, ball mapper,
  pseudo-Laplacian, `neighborhood_overlap`); profile before optimizing.

## Track E: processing capabilities

- Chunked/streaming ingest for large Parquet and CSV; cost-estimate fields
  (ball maps, memory tier, runtime) before a sweep starts.
- `sample` execution mode: sweep on a subsample, refine on the full data.
- Incremental `fit_multi` and temporal appends without recomputing the graph.
- Sparse, streaming pseudo-Laplacian accumulation (no dense n x n).
