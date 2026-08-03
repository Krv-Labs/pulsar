.. _installation:

============
Installation
============

For the no-code MCP server, install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ and follow the :ref:`mcp` guide. ``uvx`` runs the published package without cloning this repository or creating a Python environment.

For Python users:

.. code-block:: bash

   pip install thema-pulsar

For development from a source checkout:

.. code-block:: bash

   uv sync
   uv run maturin develop --release

Supported Versions
------------------

- Python 3.10+
- Linux/macOS/Windows (with Rust toolchain for local builds)
