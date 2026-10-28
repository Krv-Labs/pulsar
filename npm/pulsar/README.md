# @krv-labs/pulsar

Topological data analysis for tabular data, built to be driven by an agent:
an MCP server and CLI implementing the Thema pipeline (projection grid, Ball
Mapper, cosmic graph with an automatically chosen threshold). Pure Rust; npm
installs one prebuilt binary for your platform.

```bash
npx -y @krv-labs/pulsar install    # register the MCP server with every agent it detects
npx -y @krv-labs/pulsar mcp        # or run the MCP server on stdio yourself
npm install -g @krv-labs/pulsar    # persistent `pulsar` command
```

MCP host entry:

```json
{ "mcpServers": { "pulsar": { "command": "npx", "args": ["-y", "@krv-labs/pulsar", "mcp"] } } }
```

The binary ships in a per-platform package (`@krv-labs/pulsar-<platform>`:
macOS arm64/x64, Linux x64/arm64 on glibc and musl, Windows x64/arm64)
installed as an optional dependency; nothing is downloaded at install or run
time. Set `PULSAR_BINARY` to use a local build instead.

Documentation: https://github.com/Krv-Labs/pulsar
