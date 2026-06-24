"""Scene-aware SLAM backend: DUSt3R + Depth Anything V2 → a **metric** camera trajectory, on Mac.

DPVO (the only camera backend that recovers translation) is CUDA-only. This is the device-agnostic
alternative — it runs on Apple-Silicon MPS (and CUDA) with no custom kernels:

1. **DUSt3R** (vendored, pure-PyTorch transformer + PyTorch global aligner) reconstructs the static
   scene from a handful of keyframes → camera-to-world poses + per-frame depth, robust where COLMAP
   fails, but only up to *one* global scale.
2. **Depth Anything V2** (metric) predicts monocular metric depth on the same keyframes; the median
   ratio of metric-to-DUSt3R depth fixes that global scale (the TRAM recipe).
3. The metric keyframe poses are interpolated (slerp + lerp) to every frame and inverted to ``T_w2c``,
   the same format ``load_data_dict`` consumes from SimpleVO/DPVO — so the rest of the pipeline is
   unchanged, except the translation is now real metres.

Heavy deps (torch, the two vendored models) are imported lazily so importing this module stays cheap.
Weights default to ``~/Datasets/GVHMR/{dust3r,depth_anything}/``. The indoor/outdoor metric model is
selected by ``max_depth`` (80 = outdoor/VKITTI, 20 = indoor/Hypersim).
"""

from __future__ import annotations

# DUSt3R / Depth-Anything-V2 are vendored under third-party/ and imported via sys.path at call time.
# pyright: reportMissingImports=false
import os
import sys
from pathlib import Path

import numpy as np

_THIRD_PARTY = Path(__file__).resolve().parents[3] / "third-party"
_DUST3R_CKPT = Path("~/Datasets/GVHMR/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth").expanduser()
_DA_CKPT = Path("~/Datasets/GVHMR/depth_anything/depth_anything_v2_metric_vkitti_vitb.pth").expanduser()
_DA_CFG = {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}


def _add_vendor_paths() -> None:
    for sub in ("dust3r", "dust3r/croco", "depth_anything_v2_repo/metric_depth"):
        p = str(_THIRD_PARTY / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def _interp_poses(kf_idx: np.ndarray, kf_c2w: np.ndarray, length: int) -> np.ndarray:
    """Interpolate keyframe camera-to-world poses to every frame: slerp rotation, lerp translation."""
    from scipy.spatial.transform import Rotation, Slerp

    slerp = Slerp(kf_idx, Rotation.from_matrix(kf_c2w[:, :3, :3]))
    all_idx = np.arange(length).clip(kf_idx[0], kf_idx[-1])
    R = slerp(all_idx).as_matrix()
    t = np.stack([np.interp(all_idx, kf_idx, kf_c2w[:, i, 3]) for i in range(3)], axis=1)
    c2w = np.broadcast_to(np.eye(4, dtype="f4"), (length, 4, 4)).copy()
    c2w[:, :3, :3], c2w[:, :3, 3] = R, t
    return c2w


def run_dust3r_slam(
    video_path: str | Path,
    *,
    n_keyframes: int = 16,
    scene_graph: str = "swin-3",
    dust3r_ckpt: Path = _DUST3R_CKPT,
    da_ckpt: Path = _DA_CKPT,
    max_depth: float = 80.0,
    device: str | None = None,
) -> dict:
    """Recover a metric camera trajectory from ``video_path`` on MPS/CPU/CUDA.

    Returns ``{"T_w2c": (L,4,4) float32, "scale": float, "conf": float, "kf_idx": (K,)}`` — ``T_w2c``
    is metric (camera-from-world), one per video frame. ``conf`` is DUSt3R's mean pairwise confidence
    (>~3 is a reliable reconstruction; low means the footage defeated the matcher).
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _add_vendor_paths()
    import cv2
    import imageio.v3 as iio
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images

    from gvhmr.utils.device import get_device

    dev = device or str(get_device())
    if dev == "mps":  # the global-alignment optimizer is unstable in fp16; keep fp32 (MPS default ok)
        pass

    frames = iio.imread(str(video_path), plugin="pyav")
    length = len(frames)
    kf_idx = np.unique(np.linspace(0, length - 1, min(n_keyframes, length)).astype(int))

    # write keyframes to disk (DUSt3R's loader wants files) under the system temp
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i, n in enumerate(kf_idx):
            p = f"{tmp}/{i:04d}.jpg"
            iio.imwrite(p, frames[n])
            files.append(p)

        # 1) DUSt3R reconstruction (arbitrary scale)
        d3r = AsymmetricCroCo3DStereo.from_pretrained(str(dust3r_ckpt)).to(dev).eval()
        imgs = load_images(files, size=512, verbose=False)
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        out = inference(pairs, d3r, dev, batch_size=4, verbose=False)
        conf = float(torch.cat([c.reshape(-1) for c in out["pred1"]["conf"]]).float().mean())
        scene = global_aligner(out, device=dev, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
        scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
        kf_c2w = scene.get_im_poses().detach().cpu().numpy().astype("f4")  # (K,4,4)
        d3r_depths = [dm.detach().cpu().numpy() for dm in scene.get_depthmaps()]
        del d3r, out, scene
        if dev == "mps":
            torch.mps.empty_cache()

        # 2) Depth Anything V2 metric depth → global scale (median ratio over keyframes)
        da = DepthAnythingV2(**{**_DA_CFG, "max_depth": max_depth})
        da.load_state_dict(torch.load(str(da_ckpt), map_location="cpu", weights_only=False))
        da = da.to(dev).eval()
        ratios = []
        for i, n in enumerate(kf_idx):
            with torch.no_grad():
                md = da.infer_image(cv2.cvtColor(frames[n], cv2.COLOR_RGB2BGR), input_size=518)
            dd = d3r_depths[i]
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
