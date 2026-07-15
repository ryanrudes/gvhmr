"""MotionResult.depth_reliability — the per-frame in-camera-depth (tz) trust proxy. CPU-only.

Downstream multi-view fusion (framelock) weights each view's tz by this, because tz = 2f/(s·b) biases
FAR on out-of-distribution crops (low-res/grayscale) via shape inflation. Pins the components, the
crop-size monotonicity of `weight`, and None-safety when the signals weren't produced.
"""

from __future__ import annotations

import torch

from gvhmr.inference.result import MotionResult


def _params(L: int) -> dict[str, torch.Tensor]:
    return {
        "transl": torch.zeros(L, 3),
        "betas": torch.zeros(L, 10),
        "body_pose": torch.zeros(L, 63),
        "global_orient": torch.zeros(L, 3),
    }


def _result(L: int, *, betas_per_frame=None, bbx_xys=None) -> MotionResult:
    K = torch.eye(3).repeat(L, 1, 1)
    return MotionResult(_params(L), _params(L), K, betas_per_frame=betas_per_frame, bbx_xys=bbx_xys)


def test_none_safe_without_signals():
    assert _result(5).depth_reliability() is None
    assert _result(5, betas_per_frame=torch.randn(5, 10)).depth_reliability() is None  # needs bbx too


def test_components_and_shapes():
    L = 5
    betas = torch.randn(L, 10)
    bbx = torch.zeros(L, 3)
    bbx[:, 2] = torch.tensor([100.0, 200.0, 200.0, 200.0, 50.0])
    dr = _result(L, betas_per_frame=betas, bbx_xys=bbx).depth_reliability()
    assert set(dr) == {"bbx_px", "betas_mag", "betas_std", "weight"}
    assert all(v.shape == (L,) for v in dr.values())
    assert torch.allclose(dr["betas_mag"], betas.norm(dim=-1))
    assert torch.allclose(dr["bbx_px"], bbx[:, 2])  # absolute pixel-height, not normalized


def test_weight_is_monotone_in_crop_size():
    L = 5
    betas = torch.ones(L, 10)  # constant -> betas_std == 0, weight is purely crop-size driven
    bbx = torch.zeros(L, 3)
    bbx[:, 2] = torch.tensor([100.0, 200.0, 200.0, 200.0, 50.0])
    dr = _result(L, betas_per_frame=betas, bbx_xys=bbx).depth_reliability()
    assert (dr["weight"] > 0).all() and (dr["weight"] <= 1.0 + 1e-6).all()
    assert dr["weight"].argmin().item() == 4  # the 50px (most OOD / smallest) crop is least reliable
    # larger crops than the median saturate at 1
    assert torch.allclose(dr["weight"][1:4], torch.ones(3))


def test_new_fields_serialized():
    L = 3
    r = _result(L, betas_per_frame=torch.randn(L, 10), bbx_xys=torch.ones(L, 3))
    r.kp2d = torch.zeros(L, 17, 3)
    d = r.to_dict()
    assert {"betas_per_frame", "bbx_xys", "kp2d"} <= set(d)
    raw = r._raw_from_fields()
    assert "bbx_xys" in raw and "kp2d" in raw
