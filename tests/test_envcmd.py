"""``gvhmr env`` — the recorded-environment contract (record → sync command), no uv or GPU needed.

Pins the pieces that keep users away from raw uv: the pure ``sync_args`` construction (``--inexact`` so
nothing is ever pruned; the torch backend extra appended), the driver-version → extra mapping (including
its bounds), and that ``record_env`` merges into the config file without losing other fields.
"""

from __future__ import annotations

import tomllib

from gvhmr import PROJ_ROOT
from gvhmr.cli.envcmd import (
    EXTRA_COMPONENTS,
    SCRIPT_COMPONENTS,
    SYSTEM_COMPONENTS,
    TORCH_CHOICES,
    _extra_for_cuda,
    component_installed,
    record_env,
    sync_args,
    system_component_hint,
    system_component_installed,
)
from gvhmr.utils import localconfig


def test_sync_args_replays_extras_without_pruning():
    args = sync_args("cu124", ["preproc", "dev", "preproc"])  # duplicate collapses, order kept
    assert args == ["sync", "--inexact", "--extra=preproc", "--extra=dev", "--extra=cu124"]
    assert "--inexact" in args  # the anti-prune guarantee: DPVO / pip installs survive a sync


def test_sync_args_without_a_torch_extra():
    assert sync_args(None, ["preproc"]) == ["sync", "--inexact", "--extra=preproc"]
    assert sync_args("none", []) == ["sync", "--inexact"]  # macOS/MPS: the default PyPI wheel


def test_sync_args_dpvo_adds_the_locked_runtime_extra():
    # dpvo=true → the `dpvo` extra (numba/pypose) rides along, so uv itself enforces numba's numpy cap —
    # without it, a sync floats numpy past what numba supports and breaks a DPVO box.
    args = sync_args("cu124", ["preproc"], dpvo=True)
    assert args == ["sync", "--inexact", "--extra=preproc", "--extra=dpvo", "--extra=cu124"]
    assert sync_args(None, ["dpvo"], dpvo=True).count("--extra=dpvo") == 1  # no duplicate


def test_sync_args_scene_adds_the_locked_runtime_extra():
    # scene=true → the `scene` extra (roma) rides along. DUSt3R's global aligner imports roma at RUNTIME,
    # so without it `--camera dust3r` imports fine and then dies mid-reconstruction with
    # ModuleNotFoundError — which is exactly how it shipped broken: declared in DUSt3R's
    # requirements.txt, declared nowhere here, so it survived only until the next `uv sync` pruned it.
    args = sync_args("cu124", ["preproc"], scene=True)
    assert args == ["sync", "--inexact", "--extra=preproc", "--extra=scene", "--extra=cu124"]
    assert sync_args(None, ["scene"], scene=True).count("--extra=scene") == 1  # no duplicate
    # dpvo and scene compose
    assert sync_args(None, [], dpvo=True, scene=True) == ["sync", "--inexact", "--extra=dpvo", "--extra=scene"]


def test_cuda_version_maps_to_the_documented_extra():
    # mirrors the table in docs/INSTALL.md (and scripts/install.sh's case statement)
    assert _extra_for_cuda("12.0") == "cu124"
    assert _extra_for_cuda("12.5") == "cu124"
    assert _extra_for_cuda("12.6") == "cu126"
    assert _extra_for_cuda("12.7") == "cu126"
    assert _extra_for_cuda("12.8") == "cu128"
    assert _extra_for_cuda("13.0") == "cu128"
    assert set(TORCH_CHOICES) == {"none", "cpu", "cu124", "cu126", "cu128"}


def test_record_env_merges_without_losing_fields(tmp_path, monkeypatch):
    cfg = tmp_path / "gvhmr.toml"
    monkeypatch.setenv("GVHMR_CONFIG", str(cfg))
    record_env(torch="cu126", extras="preproc")
    assert localconfig.env_torch() == "cu126"
    assert localconfig.env_extras() == ["preproc"]
    record_env(dpvo=True)  # a later partial record must not clobber the earlier fields
    assert localconfig.env_dpvo() is True
    assert localconfig.env_torch() == "cu126"
    assert localconfig.env_extras() == ["preproc"]
    record_env(scene=True)  # the scene-camera setup script records this
    assert localconfig.env_scene() is True
    assert localconfig.env_dpvo() is True


def test_component_registry_matches_pyproject_and_disk():
    # The wizard's menu must stay truthful: every extra it offers exists in pyproject, the torch
    # backends are NOT offered as components (they're the separate torch prompt), and every setup
    # script it can run exists and is executable-ish (a file).
    with open(PROJ_ROOT / "pyproject.toml", "rb") as f:
        declared = set(tomllib.load(f)["project"]["optional-dependencies"])
    offered = set(EXTRA_COMPONENTS)
    assert offered <= declared, f"components not in pyproject: {offered - declared}"
    assert not offered & {"cpu", "cu124", "cu126", "cu128"}
    assert "preproc" in offered  # the demo's requirement leads the menu
    for key, (_desc, script, probe) in SCRIPT_COMPONENTS.items():
        assert (PROJ_ROOT / script).is_file(), f"{key}: missing {script}"
        kind = probe.split(":", 1)[0]
        assert kind in ("module", "dir"), f"{key}: bad probe {probe!r}"


def test_system_components_are_not_pip_extras():
    """System tools can't live in an extra — `uv sync` can never provide them, so they must stay out of
    EXTRA_COMPONENTS (which test_component_registry_matches_pyproject_and_disk requires to be in pyproject).
    """
    with open(PROJ_ROOT / "pyproject.toml", "rb") as f:
        declared = set(tomllib.load(f)["project"]["optional-dependencies"])
    assert not set(SYSTEM_COMPONENTS) & declared
    assert not set(SYSTEM_COMPONENTS) & set(EXTRA_COMPONENTS)


def test_system_component_registry_shape():
    for key, (label, desc, binary, cmds) in SYSTEM_COMPONENTS.items():
        assert label and desc and binary, key
        # macOS + Linux are the supported platforms; both need a real install command to advise.
        assert {"Darwin", "Linux"} <= set(cmds), f"{key}: missing an install command for a supported platform"
        assert all(cmds.values()), key
        assert isinstance(system_component_installed(key), bool)


def test_system_component_hint_is_platform_specific(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert system_component_hint("exiftool") == "brew install exiftool"
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert "apt" in system_component_hint("exiftool")
    monkeypatch.setattr("platform.system", lambda: "Plan9")  # unknown platform: list every option, don't crash
    assert "brew" in system_component_hint("exiftool") and "apt" in system_component_hint("exiftool")


def test_exiftool_is_registered():
    """The demo's focal detection depends on it, and its absence is silent — so it must be advertised."""
    assert "exiftool" in SYSTEM_COMPONENTS
    assert "depth" in SYSTEM_COMPONENTS["exiftool"][1].lower()  # the description must name the consequence


def test_component_installed_probes():
    assert component_installed("module:pytest") is True  # the suite itself proves it's importable
    assert component_installed("module:not-a-real-module") is False
    assert component_installed("dir:gvhmr") is True  # repo-relative dir
    assert component_installed("dir:not/a/real/dir") is False
