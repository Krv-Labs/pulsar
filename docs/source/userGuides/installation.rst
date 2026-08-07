.. _installation:

============
Installation
============

Agent / MCP (recommended)
-------------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then register Pulsar with your agent tools:

.. code-block:: bash

   uvx --from thema-pulsar pulsar install

No clone, venv, or Rust toolchain required. ``pulsar install`` detects Claude Code, Claude Desktop, Codex CLI, Gemini CLI, GitHub Copilot CLI, Cursor, VS Code, and Google Antigravity. See :ref:`mcp` for status/uninstall, pipx mode, headless ``--all``, and manual per-client setup.

Python API
----------

.. code-block:: bash

   pip install thema-pulsar

Or with the MCP extra for a persistent ``pulsar-mcp`` binary:

.. code-block:: bash

   pipx install "thema-pulsar[mcp]"
   # then: pulsar install --mode pipx

From source
-----------

For development from a source checkout (requires a Rust toolchain):

.. code-block:: bash

   uv sync
   uv run maturin develop --release

Supported Versions
------------------

- Python 3.10+
- Linux/macOS/Windows (with Rust toolchain for local builds)
