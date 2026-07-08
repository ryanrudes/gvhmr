"""The intrinsics sidecar loader builds the per-frame ``(L,3,3)`` K the model consumes.

Covers the format contract in ``gvhmr/utils/intrinsics.py``: focal in pixels (fx, with fy defaulting to
fx), a true principal point (default = image centre), per-frame arrays, a full ``K``, resolution
rescaling, and the error paths. Hermetic — no GPU / checkpoints / datasets.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gvhmr.utils.geo.hmr_cam import convert_f_to_K
from gvhmr.utils.intrinsics import (
    _distortion_vector,
    load_intrinsics_file,
    load_intrinsics_for_undistort,
)

W, H, L = 1920, 1080, 8


def _write_json(tmp_path, payload) -> str:
    p = tmp_path / "cam.intrinsics.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_scalar_focal_defaults_principal_point_to_centre(tmp_path):
    K = load_intrinsics_file(_write_json(tmp_path, {"fx": 1450.0}), length=L, width=W, height=H)
    assert K.shape == (L, 3, 3)
    # fy defaults to fx; principal point defaults to the image centre; bottom-right stays 1.
    assert torch.allclose(K[:, 0, 0], torch.full((L,), 1450.0))
    assert torch.allclose(K[:, 1, 1], torch.full((L,), 1450.0))
    assert torch.allclose(K[:, 0, 2], torch.full((L,), W / 2))
    assert torch.allclose(K[:, 1, 2], torch.full((L,), H / 2))
    assert torch.allclose(K[:, 2, 2], torch.ones(L))
    # Identical to the historical scalar-focal construction, so the mm/heuristic paths stay comparable.
    assert torch.allclose(K, convert_f_to_K(1450.0, W, H).repeat(L, 1, 1))


def test_full_fx_fy_cx_cy_are_honoured(tmp_path):
    payload = {"fx": 1400.0, "fy": 1402.5, "cx": 964.2, "cy": 541.8}
    K = load_intrinsics_file(_write_json(tmp_path, payload), length=L, width=W, height=H)
    assert K[0, 0, 0].item() == pytest.approx(1400.0)
    assert K[0, 1, 1].item() == pytest.approx(1402.5)  # fy stored faithfully (model reads only fx)
    assert K[0, 0, 2].item() == pytest.approx(964.2)
    assert K[0, 1, 2].item() == pytest.approx(541.8)


def test_per_frame_focal_varies_across_frames(tmp_path):
    fx = [1400.0 + i for i in range(L)]
    K = load_intrinsics_file(_write_json(tmp_path, {"fx": fx, "cx": 960.0, "cy": 540.0}), length=L, width=W, height=H)
    assert torch.allclose(K[:, 0, 0], torch.tensor(fx))
    assert torch.allclose(K[:, 1, 1], torch.tensor(fx))  # fy mirrors per-frame fx when omitted


def test_per_frame_wrong_length_raises(tmp_path):
    with pytest.raises(ValueError, match="per-frame"):
        load_intrinsics_file(_write_json(tmp_path, {"fx": [1.0, 2.0, 3.0]}), length=L, width=W, height=H)


def test_full_K_3x3_is_broadcast(tmp_path):
    K_in = [[1500.0, 0.0, 900.0], [0.0, 1500.0, 500.0], [0.0, 0.0, 1.0]]
    K = load_intrinsics_file(_write_json(tmp_path, {"K": K_in}), length=L, width=W, height=H)
    assert K.shape == (L, 3, 3)
    assert torch.allclose(K[0], torch.tensor(K_in))
    assert torch.allclose(K[0], K[-1])


def test_full_K_per_frame_length_checked(tmp_path):
    good = load_intrinsics_file(_write_json(tmp_path, {"K": [torch.eye(3).tolist()] * L}), length=L, width=W, height=H)
    assert good.shape == (L, 3, 3)
    with pytest.raises(ValueError, match="per-frame 'K'"):
        load_intrinsics_file(_write_json(tmp_path, {"K": [torch.eye(3).tolist()] * 3}), length=L, width=W, height=H)


def test_resolution_rescale_from_calibration(tmp_path):
    # Calibrated at 4K, run at 1080p (half) → focal and principal point halve on each axis.
    payload = {"width": 3840, "height": 2160, "fx": 2900.0, "fy": 2900.0, "cx": 1920.0, "cy": 1080.0}
    K = load_intrinsics_file(_write_json(tmp_path, payload), length=L, width=W, height=H)
    assert K[0, 0, 0].item() == pytest.approx(1450.0)
    assert K[0, 1, 1].item() == pytest.approx(1450.0)
    assert K[0, 0, 2].item() == pytest.approx(960.0)
    assert K[0, 1, 2].item() == pytest.approx(540.0)


def test_npz_roundtrip(tmp_path):
    p = tmp_path / "cam.intrinsics.npz"
    np.savez(p, fx=1450.0, fy=1450.0, cx=960.0, cy=540.0)
    K = load_intrinsics_file(p, length=L, width=W, height=H)
    assert K.shape == (L, 3, 3)
    assert K[0, 0, 0].item() == pytest.approx(1450.0)


def test_npy_is_treated_as_K(tmp_path):
    p = tmp_path / "cam.npy"
    np.save(p, np.array([[1500.0, 0.0, 900.0], [0.0, 1500.0, 500.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    K = load_intrinsics_file(p, length=L, width=W, height=H)
    assert K.shape == (L, 3, 3)
    assert K[0, 0, 0].item() == pytest.approx(1500.0)


def test_missing_fx_and_K_raises(tmp_path):
    with pytest.raises(ValueError, match="fx"):
        load_intrinsics_file(_write_json(tmp_path, {"cx": 960.0}), length=L, width=W, height=H)


def test_unsupported_suffix_raises(tmp_path):
    p = tmp_path / "cam.txt"
    p.write_text("nope")
    with pytest.raises(ValueError, match="unsupported"):
        load_intrinsics_file(p, length=L, width=W, height=H)


# ------------------------------------------------------------------- distortion (undistort) -------------


def test_distortion_vector_from_list_and_dict():
    assert np.allclose(_distortion_vector({"distortion": [0.1, -0.2, 0.0, 0.0, 0.05]}), [0.1, -0.2, 0.0, 0.0, 0.05])
    # a dict fills missing coeffs with 0 in OpenCV order [k1, k2, p1, p2, k3]
    assert np.allclose(_distortion_vector({"distortion": {"k1": 0.1, "k2": -0.2}}), [0.1, -0.2, 0.0, 0.0, 0.0])
    assert _distortion_vector({"fx": 1000.0}) is None  # no 'distortion' key


def test_distortion_vector_bad_length_raises():
    with pytest.raises(ValueError, match="4, 5, 8"):
        _distortion_vector({"distortion": [0.1, 0.2, 0.3]})


def test_load_for_undistort_none_without_distortion(tmp_path):
    assert load_intrinsics_for_undistort(_write_json(tmp_path, {"fx": 1000.0}), width=W, height=H) is None


def test_load_for_undistort_returns_rescaled_K_and_coeffs(tmp_path):
    payload = {"width": 3840, "height": 2160, "fx": 1800.0, "cx": 1920.0, "cy": 1080.0,
               "distortion": [-0.28, 0.1, 0.0, 0.0], "undistort_alpha": 0.5}  # fmt: skip
    spec = load_intrinsics_for_undistort(_write_json(tmp_path, payload), width=W, height=H)
    assert spec is not None
    K, dist, meta = spec
    assert K.shape == (3, 3) and K.dtype == np.float64
    assert K[0, 0] == pytest.approx(900.0)  # 1800 rescaled 3840->1920 (half); dist coeffs are NOT rescaled
    assert K[0, 2] == pytest.approx(960.0)
    assert np.allclose(dist, [-0.28, 0.1, 0.0, 0.0])
    assert meta["alpha"] == pytest.approx(0.5)


def test_load_for_undistort_rejects_per_frame(tmp_path):
    payload = {"fx": [1000.0] * L, "distortion": [0.1, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="constant intrinsics"):
        load_intrinsics_for_undistort(_write_json(tmp_path, payload), width=W, height=H)
