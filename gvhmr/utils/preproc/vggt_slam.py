"""Scene-aware SLAM backend: VGGT + Depth-Anything-V2 → a **metric** camera trajectory, on Mac.

An alternative to the DUSt3R backend (`dust3r_slam.py`) with the same output contract. VGGT (Visual
Geometry Grounded Transformer, CVPR 2025) is a single feed-forward pass that jointly predicts camera
poses *and* per-frame depth — replacing DUSt3R's pairwise-inference + global-alignment optimizer with one
forward, which is faster and often more robust. It is still **scale-ambiguous**, so — exactly as with
DUSt3R — Depth-Anything-V2 (metric) fixes the one global scale.

1. **VGGT** (pure-PyTorch, MPS/CPU/CUDA) on a handful of keyframes → per-frame extrinsics (OpenCV
   camera-from-world) + depth, up to one global scale.
2. **Depth-Anything V2** (metric) predicts monocular metric depth on the same keyframes; the median ratio
   of metric-to-VGGT depth fixes that global scale (the same recipe as the DUSt3R backend).
3. The metric keyframe poses are interpolated (slerp + lerp) to every frame and left as ``T_w2c`` — the
   same format ``load_data_dict`` consumes from SimpleVO/DPVO/DUSt3R, so the rest of the pipeline is
   unchanged, except the translation is now real metres.

VGGT weights auto-download from the HuggingFace hub (``facebook/VGGT-1B``, non-commercial) on first use.
The code is cloned into ``third-party/vggt`` by ``scripts/setup_scene_aware.sh`` and imported via sys.path
at call time. Depth-Anything weights default to ``~/Datasets/GVHMR/depth_anything/`` (see the DUSt3R backend).
"""

from __future__ import annotations

# VGGT / Depth-Anything-V2 are cloned under third-party/ and imported via sys.path at call time.
# pyright: reportMissingImports=false
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from gvhmr.utils.preproc.dust3r_slam import _DA_CFG, _DA_CKPT, _interp_poses

_THIRD_PARTY = Path(__file__).resolve().parents[3] / "third-party"
DEFAULT_VGGT_MODEL = "facebook/VGGT-1B"  # non-commercial; "facebook/VGGT-1B-Commercial" for commercial use


def _add_vendor_paths() -> None:
    for sub in ("vggt", "depth_anything_v2_repo/metric_depth"):
        p = str(_THIRD_PARTY / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def run_vggt_slam(
    video_path: str | Path,
    *,
    n_keyframes: int = 16,
    hf_model: str = DEFAULT_VGGT_MODEL,
    da_ckpt: Path = _DA_CKPT,
    max_depth: float = 80.0,
    device: str | None = None,
) -> dict:
    """Recover a metric camera trajectory from ``video_path`` on MPS/CPU/CUDA with VGGT.

    Returns ``{"T_w2c": (L,4,4) float32, "scale": float, "conf": float, "kf_idx": (K,)}`` — ``T_w2c``
    is metric (camera-from-world), one per video frame. ``conf`` is VGGT's mean depth confidence.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _add_vendor_paths()
    import cv2
    import imageio.v3 as iio
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    from gvhmr.utils.device import get_device

    dev = device or str(get_device())
    frames = iio.imread(str(video_path), plugin="pyav")
    length = len(frames)
    kf_idx = np.unique(np.linspace(0, length - 1, min(n_keyframes, length)).astype(int))

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i, n in enumerate(kf_idx):
            p = f"{tmp}/{i:04d}.jpg"
            iio.imwrite(p, frames[n])
            files.append(p)

        # 1) VGGT: one forward → per-keyframe extrinsics (w2c) + depth (arbitrary but self-consistent scale)
        model = VGGT.from_pretrained(hf_model).to(dev).eval()
        images = load_and_preprocess_images(files).to(dev)  # (K, 3, H, W), [0,1]
        # bf16/fp16 autocast only helps on CUDA; the MPS/CPU path stays fp32 (stable).
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()
        with torch.no_grad(), amp:
            images_b = images[None]  # (1, K, 3, H, W)
            tokens, ps_idx = model.aggregator(images_b)
            pose_enc = model.camera_head(tokens)[-1]  # (1, K, 9)
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images_b.shape[-2:])  # (1, K, 3, 4) w2c
            depth_map, depth_conf = model.depth_head(tokens, images_b, ps_idx)  # (1,K,H,W,1), (1,K,H,W)
        extr = extrinsic[0].float().cpu().numpy().astype("f4")  # (K, 3, 4) camera-from-world
        vggt_depth = depth_map[0, ..., 0].float().cpu().numpy()  # (K, H, W)
        conf = float(depth_conf.float().mean())
        # w2c (3,4) → c2w (4,4): scale is applied to camera centres, then inverted back at the end.
        w2c = np.tile(np.eye(4, dtype="f4"), (len(extr), 1, 1))
        w2c[:, :3, :4] = extr
        kf_c2w = np.linalg.inv(w2c).astype("f4")  # (K, 4, 4)
        del model, tokens, pose_enc, depth_map, depth_conf
        if dev == "mps":
            torch.mps.empty_cache()

        # 2) Depth-Anything V2 metric depth → global scale (median ratio over keyframes)
        da = DepthAnythingV2(**{**_DA_CFG, "max_depth": max_depth})
        da.load_state_dict(torch.load(str(da_ckpt), map_location="cpu", weights_only=False))
        da = da.to(dev).eval()
        ratios = []
        for i, n in enumerate(kf_idx):
            with torch.no_grad():
                md = da.infer_image(cv2.cvtColor(frames[n], cv2.COLOR_RGB2BGR), input_size=518)
            dd = vggt_depth[i]
            md_r = cv2.resize(md, (dd.shape[1], dd.shape[0]))
            valid = (dd > 1e-6) & (md_r > 1e-6)
            if valid.any():
                ratios.append(float(np.median(md_r[valid] / dd[valid])))
        del da
        if dev == "mps":
            torch.mps.empty_cache()

    scale = float(np.median(ratios)) if ratios else 1.0
    kf_c2w[:, :3, 3] *= scale  # metric camera centres

    # 3) interpolate to every frame, invert to T_w2c (camera-from-world)
    c2w = _interp_poses(kf_idx.astype(float), kf_c2w, length)
    T_w2c = np.linalg.inv(c2w).astype("f4")
    return {"T_w2c": T_w2c, "scale": scale, "conf": conf, "kf_idx": kf_idx}
