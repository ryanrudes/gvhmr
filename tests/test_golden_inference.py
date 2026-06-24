"""Golden end-to-end inference test — the safety net for refactors and speedups.

Builds the *real* released GVHMR model from its checkpoint and runs the full demo
``predict`` path (NetworkEncoderRoPE → EnDecoder → postprocess → global rollout) on
seeded synthetic inputs, then asserts the output is byte-stable. This is what makes
the aggressive reorg and the performance work safe: any change that alters the model's
output trips this test.

Requires the released checkpoint + SMPL-X body models under ``inputs/checkpoints/``
(see docs/INSTALL.md). Skipped automatically when they're absent (e.g. in CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gvhmr import PROJ_ROOT

CKPT = PROJ_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
SMPLX = PROJ_ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"

pytestmark = pytest.mark.checkpoint

# Behavior fingerprint captured from the released checkpoint on CPU (float64 sums of
# each output param). A behavior-preserving change must reproduce these; a regression
# shifts them. Tolerance tolerates float re-association (e.g. vectorizing a loop) but
# catches real changes.
GOLDEN_SUMS = {
    "smpl_params_global.body_pose": -40.70616,
    "smpl_params_global.betas": -106.76443,
    "smpl_params_global.global_orient": -17.37176,
    "smpl_params_global.transl": 93.02473,
    "smpl_params_incam.body_pose": -40.70616,
    "smpl_params_incam.betas": -106.76443,
    "smpl_params_incam.global_orient": 31.25829,
    "smpl_params_incam.transl": 182.4404,
}
GOLDEN_SHAPES = {
    "body_pose": (32, 63),
    "betas": (32, 10),
    "global_orient": (32, 3),
    "transl": (32, 3),
}
ATOL = 0.05  # on float64 sums O(10–180); ~1e-3 relative


def _require_assets() -> None:
    if not CKPT.exists() or not SMPLX.exists():
        pytest.skip("released checkpoint / SMPL-X body models not available")


@pytest.fixture(scope="module")
def model():
    _require_assets()
    import hydra
    from hydra import compose, initialize_config_module

    import gvhmr.model.gvhmr.gvhmr_pl_demo  # noqa: F401  (registers the demo model)
    from gvhmr.configs import register_store_gvhmr

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=golden", "static_cam=True"])
    m = hydra.utils.instantiate(cfg.model, _recursive_=False)
    m.load_pretrained_model(cfg.ckpt_path)
    return m.eval()


def _synthetic_data(device="cpu") -> dict:
    torch.manual_seed(1234)
    L = 32
    data = {
        "length": torch.tensor(L),
        "kp2d": torch.randn(L, 17, 3),
        "bbx_xys": torch.rand(L, 3) * 200 + 100,
        "K_fullimg": torch.tensor([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]]).repeat(L, 1, 1),
        "cam_angvel": torch.zeros(L, 6),
        "f_imgseq": torch.randn(L, 1024),
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}


def _sums(pred, double=True) -> dict[str, float]:
    out = {}
    for grp in ("smpl_params_global", "smpl_params_incam"):
        for k, v in pred[grp].items():
            v = v.double() if double else v.float()
            out[f"{grp}.{k}"] = float(v.sum())
    return out


@torch.no_grad()
def test_golden_output_matches_fingerprint(model) -> None:
    pred = model.predict(_synthetic_data("cpu"), static_cam=True)
    # Shapes
    for k, shape in GOLDEN_SHAPES.items():
        assert tuple(pred["smpl_params_global"][k].shape) == shape
    # Output fingerprint
    sums = _sums(pred)
    for key, golden in GOLDEN_SUMS.items():
        assert sums[key] == pytest.approx(golden, abs=ATOL), f"{key}: {sums[key]} != golden {golden}"


@torch.no_grad()
def test_cpu_inference_is_deterministic(model) -> None:
    a = model.predict(_synthetic_data("cpu"), static_cam=True)
    b = model.predict(_synthetic_data("cpu"), static_cam=True)
    for grp in ("smpl_params_global", "smpl_params_incam"):
        for k in a[grp]:
            assert torch.equal(a[grp][k], b[grp][k]), f"non-deterministic: {grp}.{k}"


@torch.no_grad()
def test_mps_end_to_end_matches_cpu(model) -> None:
    # End-to-end MPS inference must produce finite output matching CPU within float32
    # tolerance. (This caught a real NaN: unclamped acos in qpow during the CCD IK.)
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    cpu = model.predict(_synthetic_data("cpu"), static_cam=True)
    model_mps = model.to("mps")
    try:
        mps = model_mps.predict(_synthetic_data("mps"), static_cam=True)
        for grp in ("smpl_params_global", "smpl_params_incam"):
            for k in cpu[grp]:
                a = cpu[grp][k].float()
                b = mps[grp][k].float().cpu()
                assert torch.isfinite(b).all(), f"non-finite MPS output at {grp}.{k}"
                assert torch.allclose(a, b, atol=1e-2), f"MPS/CPU mismatch at {grp}.{k}"
    finally:
        model.to("cpu")
