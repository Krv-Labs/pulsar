# Pulsar

Topological data analysis for tabular data, built to be driven by an agent.
Pulsar implements the Thema pipeline (imputation, scaling, a grid of
projections, Ball Mapper per grid cell, a pseudo-Laplacian, and a cosmic graph
whose construction threshold is chosen by H0 persistent homology) and serves it
over MCP, with a CLI beside it. Pure Rust, shipped on npm as
[`@krv-labs/pulsar`](https://www.npmjs.com/package/@krv-labs/pulsar).

Tools return **measurements, not conclusions**: geometry, graph state,
threshold surfaces, cluster evidence and their provenance. The agent and the
user supply the domain meaning.

## Install

No Rust or Python needed; npm installs one prebuilt binary for your platform.

```bash
npx -y @krv-labs/pulsar install      # register the MCP server with every agent it detects
npx -y @krv-labs/pulsar status       # which agents have it, and whether each entry is healthy
```

`pulsar install` writes the entry each client expects for Claude Code, Claude
Desktop, Codex CLI, Gemini CLI, GitHub Copilot CLI, Cursor, VS Code and Google
Antigravity. Every config it edits is backed up first (`*.pulsar.backup`),
files with comments it would drop are refused, and only Pulsar's own entries
are ever changed. `pulsar uninstall` removes exactly what it added.

Or point any MCP host at it directly:

```json
{ "mcpServers": { "pulsar": { "command": "npx", "args": ["-y", "@krv-labs/pulsar", "mcp"] } } }
```

For a persistent command, `npm install -g @krv-labs/pulsar` (then
`pulsar --help`). Platforms: macOS arm64/x64, Linux x64/arm64 (glibc and
musl), Windows x64/arm64.

### Coming from the Python package (`thema-pulsar`)

Pulsar 0.7 replaces the Python library with a Rust implementation of the same
pipeline and MCP tools, proven against frozen reference outputs of the Python
version. `thema-pulsar` stays installable from PyPI at 0.6 but gets no further
releases. To move an agent setup over, run `npx -y @krv-labs/pulsar install`:
it recognizes the `uvx --from thema-pulsar …` entries the Python installer
wrote and rewrites them to the new server. Configs are unchanged (`params.yaml`
is read as before); the Python API (`ThemaRS`) is not part of 0.7.

## The agent loop

Topological data analysis used to mean a PhD in algebraic topology and a high
tolerance for parameter tuning ("why is my cosmic graph a hairball? what
should epsilon be?"). The MCP server lets the agent do that work, one
measured step at a time:

1. **Ingest** a host-visible CSV/Parquet path and get a stable `dataset_id`.
2. **Characterize** it (`characterize_dataset`, `probe_columns`) and, for messy
   data, draft a reviewable **preprocessing plan**.
3. **Create a calibrated config** with `create_config`: epsilon and projection
   dimensions are calibrated against the processed feature space, never guessed.
4. **Sweep** (`run_topological_sweep`, async jobs with progress for big grids).
5. **Diagnose** the graph (`diagnose_cosmic_graph`,
   `get_threshold_stability_curve`) and iterate with `refine_config`.
6. **Explain** the structure: `generate_cluster_dossier`, then
   `get_cluster_profile`, `get_feature_signal`, `get_cluster_signal_matrix` and
   `compare_clusters` (Welch t, KS, Cohen's d).
7. **Export**: labeled CSV, a Parquet bundle, or a standalone HTML report.

Longitudinal panels (one row per entity and time step) get their own tools:
`build_longitudinal_graph`, trajectory archetypes, cross-time neighbours and
trajectory classification. Call `get_workflow_guide` once at the start of a
session for the full procedure and tool map.

Things that save time:

- **Don't skip diagnosis.** If the graph is a giant hairball or all dust, the
  clusters will be too. The stability curve shows where the structure is.
- **Keep labels out of the geometry.** If your table has an outcome column,
  drop it with `refine_config` (`preprocessing.drop_columns`); it still shows
  up in the cluster evidence, which is where you want it.
- **Let `create_config` handle non-numeric columns**: low-cardinality strings
  are one-hot encoded, high-cardinality ones dropped.

## CLI

```bash
pulsar fit params.yaml            # run a sweep, print a JSON summary of the cosmic graph
pulsar runs list                  # persisted runs
pulsar config validate params.yaml
pulsar mcp                        # the MCP server on stdio
pulsar schema config              # JSON Schema of params.yaml
```

Runs, datasets and plans persist in `~/.local/share/pulsar` (or
`$PULSAR_STORE_DIR`), so handles survive restarts.

## Configuration

Thresholding is automatic by default; most configs only need data,
preprocessing and the sweep grid.

```yaml
run:
  name: my_experiment
  data: path/to/data.csv # CSV or Parquet

preprocessing:
  drop_columns: [id, timestamp]
  impute:
    age: { method: sample_normal, seed: 42 } # fill_mean | fill_median | fill_mode | sample_normal | sample_categorical | fill_value
    category: { method: sample_categorical, seed: 7 }

sweep:
  projection:
    method: jl # or pca
    dimensions: { values: [2, 3, 5] }
    seed: { values: [42, 7, 13] }
  ball_mapper:
    epsilon:
      range: { min: 0.1, max: 1.5, steps: 8 } # or: values: [0.3, 0.5, 0.8]

cosmic_graph:
  construction: minhash # or exact
  minhash_d: 256
  construction_threshold: auto
```

## Build from source

```bash
cargo build --release -p pulsar-cli      # target/release/pulsar
cargo test --workspace                   # unit, contract and golden-fixture parity tests
node npm/scripts/smoke.mjs               # pack + install + MCP handshake on this platform
scripts/install.sh                       # build, install to ~/.local/bin, register detected agents
```

| Crate | Role |
|---|---|
| `pulsar-core` | numeric engine: impute, scale, PCA/JL, Ball Mapper, pseudo-Laplacian, cosmic graph, H0, temporal |
| `pulsar-config` | `params.yaml` loader, JSON Schema, structured errors |
| `pulsar-data` | CSV/Parquet ingest (pandas-compatible typing), preprocessing executor, CSV export |
| `pulsar-pipeline` | `fit` / `fit_multi`, projection fingerprint, progress, `CosmicTrajectory` |
| `pulsar-store` | persistent store, typed handles, job model, model cache |
| `pulsar-analysis` | thresholds, clustering, diagnostics, evidence, dossier, preprocessing plan, longitudinal |
| `pulsar-report` | Parquet dataset bundle and the standalone HTML report |
| `pulsar-mcp` | the MCP server: typed tools, error envelope, async jobs |
| `pulsar-install` | `pulsar install / uninstall / status` |
| `pulsar-cli` | the `pulsar` binary |
| `pulsar-parity` | parity tests against frozen outputs of the Python reference ([provenance](tests/fixtures/golden/README.md)) |

Design notes: [`docs/adr/`](docs/adr) and [`docs/design/mcp-roadmap.md`](docs/design/mcp-roadmap.md).

## Citation

If you use Pulsar in research, please cite:

```bibtex
@article{Gathrid2025,
  author  = {Gathrid, Sidney and Wayland, Jeremy and Wayland, Stuart and Deshmukh, Ranjit and Wu, Grace C.},
  title   = {Strategies to accelerate US coal power phase-out using contextual retirement vulnerabilities},
  journal = {Nature Energy},
  year    = {2025},
  volume  = {10},
  number  = {10},
  pages   = {1274--1288},
  month   = {October},
  doi     = {10.1038/s41560-025-01871-0},
  url     = {https://doi.org/10.1038/s41560-025-01871-0},
  issn    = {2058-7546}
}
```

which introduced the original [Thema](https://github.com/Krv-Labs/Thema) algorithm.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
