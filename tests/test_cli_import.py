"""The CLI must import on the **base install** — extras only gate running, never importing.

Regression: ``gvhmr/cli/demo.py`` imported ``SimpleVO`` (→ pycolmap, a ``preproc`` extra) at module
level, so on a base install every ``gvhmr demo`` invocation — even ``--static-cam``, which never uses
visual odometry — died with a bare ``ImportError`` at import time, before the fail-fast dependency
check could print the actual install command. CI runs without the preproc extra, so these imports
exercise the real thing.
"""

from __future__ import annotations

import importlib


def test_cli_package_imports_without_extras():
    mod = importlib.import_module("gvhmr.cli")
    assert mod.app is not None  # the Typer app; commands lazy-import their heavy bodies


def test_cli_demo_imports_without_extras():
    mod = importlib.import_module("gvhmr.cli.demo")
    # the demo entry point and the fail-fast dependency gate it runs before any download/compute
    assert callable(mod.run)
    assert callable(mod.ensure_deps)


def test_cli_info_and_download_import_without_extras():
    assert callable(importlib.import_module("gvhmr.cli.info").run)
    assert callable(importlib.import_module("gvhmr.cli.download").run)
