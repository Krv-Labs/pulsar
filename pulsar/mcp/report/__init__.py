"""HTML report rendering for topological dossiers.

The public surface is ``dossier_to_html``. Submodules are implementation
detail: styles, script, formatting helpers, section renderers, and the
top-level composer.
"""

from __future__ import annotations

from pulsar.mcp.report.renderer import dossier_to_html

__all__ = ["dossier_to_html"]
