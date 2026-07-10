"""Detector box-distribution adapter (docs/ROADMAP.md A4) — the transform + calibration, no weights/GPU.

Pins the inference-only lever: the default is the identity (released path byte-identical), the affine does
what it says, and ``fit_box_adapter`` recovers a known new→baseline mapping so it can be calibrated per
detector from paired boxes.
"""

from __future__ import annotations

import torch

from gvhmr.utils.preproc.box_adapter import BoxAdapter, fit_box_adapter


def test_default_is_identity():
    a = BoxAdapter()
    assert a.is_identity
    assert BoxAdapter.from_config(None).is_identity
    assert BoxAdapter.from_config({}).is_identity
    bbx = torch.tensor([[100.0, 120.0, 50.0], [200.0, 220.0, 60.0]])
    assert torch.equal(a.apply(bbx), bbx)  # identity leaves every box untouched


def test_from_config_and_affine_math():
    a = BoxAdapter.from_config({"scale": 1.2, "dx": 0.1, "dy": -0.2})
    assert (a.scale, a.dx, a.dy) == (1.2, 0.1, -0.2)
    assert not a.is_identity
    out = a.apply(torch.tensor([[100.0, 100.0, 50.0]]))
    # size*1.2=60; cx + 0.1*50 = 105; cy - 0.2*50 = 90
    assert torch.allclose(out, torch.tensor([[105.0, 90.0, 60.0]]), atol=1e-5)


def test_fit_recovers_a_known_mapping():
    new = torch.tensor([[100.0, 100.0, 50.0], [200.0, 150.0, 60.0], [300.0, 300.0, 40.0]])
    known = BoxAdapter(scale=1.1, dx=0.05, dy=-0.1)
    baseline = known.apply(new)  # pretend these are the frozen-baseline boxes
    fit = fit_box_adapter(new, baseline)
    assert abs(fit.scale - 1.1) < 1e-5 and abs(fit.dx - 0.05) < 1e-5 and abs(fit.dy + 0.1) < 1e-5
    assert torch.allclose(fit.apply(new), baseline, atol=1e-4)  # fit → apply maps new onto baseline
