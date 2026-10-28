# ADR 0002: Pulsar 0.7 is Rust only and ships on npm

- **Status:** accepted (2026-10-28)
- **Supersedes:** ADR 0001 decision 7 (keep PyPI as thin bindings).

## Context

Through 0.6 Pulsar was a Python package (`thema-pulsar` on PyPI) with Rust
kernels behind PyO3. The MCP server is the product surface, and its users run
it from agent hosts, not notebooks; a Python runtime, `uv` and wheels were
installation cost without benefit for them.

## Decisions

1. **Rust only.** The workspace, CI and distribution contain no Python. The
   Python implementation served as the porting reference: every ported stage
   and tool was proven against frozen outputs of it (`tests/fixtures/golden/`,
   tolerances in its `manifest.json`), then the Python tree was removed. The
   fixtures are never regenerated; new behaviour gets Rust known-answer tests.
   Known divergences are documented in each module header.
2. **npm, platform-package pattern** (ADR 0001): `@krv-labs/pulsar` is a small
   launcher with one optional dependency per target
   (`@krv-labs/pulsar-<platform>`, eight targets), so npm installs exactly one
   prebuilt binary and nothing is downloaded at install or run time. Published
   publicly on npmjs.com with provenance from the tagged release workflow.
3. **PyPI is frozen at 0.6.** Existing `thema-pulsar` releases stay
   installable; there are no further Python releases. `pulsar install`
   recognizes and upgrades the agent entries the Python installer wrote.
4. **Version line continues** at 0.7.0, so 0.6 (Python) to 0.7 (Rust) reads as
   one project.
5. **Chunked uploads are dropped** (`begin_dataset_upload`,
   `append_dataset_chunk`, `finalize_dataset_upload`). Agents pass a
   host-visible path to `ingest_dataset`, which the workflow guide already
   required.

## Consequences

- One static binary per platform; `npx -y @krv-labs/pulsar mcp` is the whole
  setup for an MCP host.
- Users of the Python API (`ThemaRS`) stay on 0.6.
- Release = tag `vX.Y.Z` matching `[workspace.package].version`; CI builds the
  eight targets, publishes the platform packages then the launcher, and
  verifies the published packages install and run.
