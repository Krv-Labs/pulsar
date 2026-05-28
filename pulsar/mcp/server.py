"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import logging
from fastmcp import FastMCP
from fastmcp.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive patch: strip unknown kwargs from non-compliant MCP clients.
# Some clients (e.g. Gemini CLI) inject orchestration fields like
# ``wait_for_previous`` into tool calls.  FastMCP's Pydantic validation
# rejects these.  Patching FunctionTool.run at the class level filters
# known orchestration keys *before* validation — one patch protects every tool.
# Unknown keys outside the allowlist are logged at WARNING to surface caller bugs.
# ---------------------------------------------------------------------------
_original_function_tool_run = FunctionTool.run
_KNOWN_ORCHESTRATION_KEYS = frozenset({"wait_for_previous"})


async def _lenient_function_tool_run(self, arguments):
    if isinstance(arguments, dict) and arguments:
        valid_keys = set(self.parameters.get("properties", {}).keys())
        unknown_keys = set(arguments.keys()) - valid_keys
        if unknown_keys:
            unexpected = unknown_keys - _KNOWN_ORCHESTRATION_KEYS
            if unexpected:
                logger.warning(
                    "Stripped unexpected argument(s) %s from tool %s",
                    sorted(unexpected),
                    getattr(self, "name", "unknown"),
                )
            arguments = {k: v for k, v in arguments.items() if k in valid_keys}
    return await _original_function_tool_run(self, arguments)


FunctionTool.run = _lenient_function_tool_run

# ---------------------------------------------------------------------------
# Initialize FastMCP
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Pulsar",
    instructions=(
        "Topological data analysis for tabular datasets. Call "
        "`get_workflow_guide` for the recommended end-to-end phase-by-phase "
        "procedure.\n\n"
        "Tool map:\n"
        "- Primary loop: `ingest_dataset` -> `characterize_dataset` -> "
        "`create_config` -> `validate_config` -> `run_topological_sweep` -> "
        "`diagnose_cosmic_graph` -> `summarize_sweep_history` -> "
        "`generate_cluster_dossier`.\n"
        "- Targeted interpretation: `get_cluster_profile`, "
        "`get_feature_signal`, `compare_clusters_tool`, `export_html_report`.\n"
        "- Advanced/debug: chunked upload trio, `probe_columns`, "
        "`recommend_preprocessing`, `validate_preprocessing_config`, "
        "`repair_preprocessing_config`, `refine_config`, `refine_active_config`, "
        "`get_active_config`, `get_threshold_stability_curve`, "
        "`get_topological_skeleton`, `compare_sweeps`, `get_experiment_history`, "
        "`get_cluster_signal_matrix`, `export_labeled_data`, "
        "`get_runtime_context`."
    ),
)

# ---------------------------------------------------------------------------
# Dynamic Registration of All Tools
# ---------------------------------------------------------------------------
from pulsar.mcp.tools import ALL_TOOLS_LIST

for tool_fn in ALL_TOOLS_LIST:
    mcp.tool()(tool_fn)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
