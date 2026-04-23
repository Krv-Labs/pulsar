"""External font links for the report."""

from __future__ import annotations


def _render_font_links() -> str:
    """Load Google-hosted body fonts with immediate local fallbacks when offline."""
    return "\n".join(
        [
            "<link rel='preconnect' href='https://fonts.googleapis.com'>",
            "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>",
            "<link href='https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500;700&display=swap' rel='stylesheet'>",
        ]
    )
