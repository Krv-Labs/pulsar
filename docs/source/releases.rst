.. _releases:

Releases
========

Pulsar follows `Semantic Versioning <https://semver.org/>`_ and documents changes in ``CHANGELOG.md`` using `Keep a Changelog <https://keepachangelog.com/>`_ format.

Version source of truth
-----------------------

``Cargo.toml`` ``[package].version`` is the canonical version. Maturin reads it when building wheels; ``pyproject.toml`` declares ``dynamic = ["version"]`` so the Python package metadata stays in sync automatically.

Editable checkouts resolve the same value through ``pulsar._version`` (which reads ``Cargo.toml``). Sphinx docs import ``pulsar.__version__`` in ``conf.py``.

CI enforces consistency with:

.. code-block:: bash

   python scripts/check_versions.py

Cutting a patch release
-----------------------

Example: shipping ``v0.2.5``.

1. **Branch**

   .. code-block:: bash

      git checkout main
      git pull
      git checkout -b release/v0.2.5

2. **Bump version** — edit ``Cargo.toml`` only:

   .. code-block:: toml

      [package]
      version = "0.2.5"

3. **Update changelog** — under ``## [Unreleased]``, finalize notes into a dated section:

   .. code-block:: markdown

      ## [Unreleased]

      ## [0.2.5] - 2026-07-11

      ### Added
      - ...

4. **Verify locally**

   .. code-block:: bash

      python scripts/check_versions.py
      uv run pytest tests/test_version.py -v
      uv run pytest -v

5. **Open a PR** to ``main`` from ``release/v0.2.5``. Wait for CI (build, tests, version check, lint).

6. **Tag after merge** — tags drive production publishing:

   .. code-block:: bash

      git checkout main
      git pull
      git tag v0.2.5
      git push origin v0.2.5

What the tag triggers
---------------------

``.github/workflows/release.yml`` runs on tag push:

- Builds wheels (Linux, Windows, macOS) and sdist
- Smoke-tests each wheel platform by installing the cp312 wheel and running ``scripts/smoke_mcp.py`` (import + MCP ping/list_tools/stdio launch):

  - Linux x86_64 on ``ubuntu-latest``
  - Linux aarch64 on ``ubuntu-24.04-arm``
  - Windows x86_64 on ``windows-latest``
  - macOS x86_64 on ``macos-15-intel``
  - macOS arm64 on ``macos-latest``

- Creates a GitHub Release with attached artifacts
- Publishes to PyPI via trusted publishing

Pull requests to ``main`` also exercise the release build matrix but do **not** publish. CI additionally runs ``scripts/smoke_mcp.py`` on ``macos-15-intel``.

Agent-friendly replication
--------------------------

Agents automating releases should read ``.agents/AGENTS.md`` (versioning section) and follow the same file touch list:

- ``Cargo.toml`` — version bump
- ``CHANGELOG.md`` — release notes + date
- ``scripts/check_versions.py`` — must pass (no manual doc version strings)

Do **not** hand-edit ``pyproject.toml`` version fields or hard-code versions in ``docs/source/conf.py``; both derive from ``Cargo.toml`` through the Python package.


