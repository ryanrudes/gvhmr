"""The package version is declared twice — pin them together.

`pyproject.toml`'s `version` is what PyPI publishes; `gvhmr.__version__` is what users read at runtime.
Nothing else keeps them equal, and a release tag makes any drift permanent (PyPI version numbers cannot
be reused). Cheap test, unrecoverable bug.
"""

from __future__ import annotations

import tomllib

import gvhmr
from gvhmr import PROJ_ROOT


def _declared_version() -> str:
    with open(PROJ_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_runtime_version_matches_pyproject():
    assert gvhmr.__version__ == _declared_version()


def test_version_is_pep440_release():
    parts = _declared_version().split(".")
    assert len(parts) == 3, "expected MAJOR.MINOR.PATCH"
    assert all(p.isdigit() for p in parts), _declared_version()


def test_version_is_exported():
    assert "__version__" in gvhmr.__all__
