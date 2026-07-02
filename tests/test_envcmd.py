"""``gvhmr env`` — the recorded-environment contract (record → sync command), no uv or GPU needed.

Pins the pieces that keep users away from raw uv: the pure ``sync_args`` construction (``--inexact`` so
nothing is ever pruned; the torch backend extra appended), the driver-version → extra mapping (including
its bounds), and that ``record_env`` merges into the config file without losing other fields.
"""

from __future__ import annotations

from gvhmr.cli.envcmd import TORCH_CHOICES, _extra_for_cuda, record_env, sync_args
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
