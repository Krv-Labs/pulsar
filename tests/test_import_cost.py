"""Guard the lazy import surface of `pulsar`.

`pulsar/__init__.py` resolves its public names lazily (PEP 562) so that the CLI
— which lives under this package but needs none of the scientific stack — does
not pay ~1.6s to print `--help`. That property is invisible at runtime and
silently regresses the moment someone adds a convenience import at module
scope, so it is asserted here rather than left to review.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import pulsar

# Imported by the pipeline, not by the CLI. If any of these lands in
# sys.modules after a bare `import pulsar`, the lazy surface has regressed.
HEAVY = ("sklearn", "scipy", "pandas", "networkx", "pyarrow")


def run(code: str) -> set[str]:
    """Report which HEAVY modules `code` leaves in sys.modules."""
    probe = f"{code}\nimport sys, json\nprint(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    import json

    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_importing_pulsar_does_not_load_the_scientific_stack():
    assert run("import pulsar") == set()


def test_importing_the_cli_does_not_load_the_scientific_stack():
    # This is the one that matters for `pulsar --help` / `pulsar install`:
    # pulsar.cli is a subpackage, so it executes pulsar/__init__.py.
    assert run("import pulsar.cli.main") == set()


def test_touching_a_pipeline_name_does_load_it():
    # The cost is deferred, not removed -- proving the lazy path is real and
    # not just a missing export. pandas rather than sklearn: the pipeline
    # genuinely needs pandas at import, whereas characterization now defers
    # scikit-learn into the two functions that use it.
    assert "pandas" in run("import pulsar; pulsar.ThemaRS")


def test_the_pipeline_no_longer_drags_in_scikit_learn():
    assert "sklearn" not in run("import pulsar; pulsar.ThemaRS")
    assert "sklearn" in run("import sklearn.impute"), "sanity: the probe can see it"


@pytest.mark.parametrize("name", sorted(pulsar.__all__))
def test_every_exported_name_resolves(name):
    assert getattr(pulsar, name) is not None


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        pulsar.nope


def test_star_import_matches_all():
    namespace: dict[str, object] = {}
    exec("from pulsar import *", namespace)
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == {n for n in pulsar.__all__ if not n.startswith("__")}


def test_dir_lists_the_public_surface():
    assert set(pulsar.__all__) <= set(dir(pulsar))
