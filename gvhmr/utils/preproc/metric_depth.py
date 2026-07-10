"""Swappable monocular **metric-depth** models for the scene-camera scale step (docs/ROADMAP.md, A2).

The DUSt3R / VGGT backends reconstruct a scene up to one global scale; a metric-depth model fixes that
scale via the median metric-to-recon depth ratio over the keyframes (the "TRAM recipe"). This module is
the swappable seam for that model:

- The default (`depth_anything_v2`) wraps the released Depth-Anything-V2 metric model and is **byte-identical**
  to the inline scale step it replaces — so the released dust3r/vggt camera stays unchanged.
- A registry + `MetricDepth` protocol let a modern metric-depth model (UniDepth, Metric3D-v2) drop in behind
  the same interface and be A/B'd end-to-end with ``tools/eval/eval_world.py`` (its ``gt-cam`` mode isolates
  the compose from SLAM error; ``dust3r`` measures the whole stack). See ``docs/ROADMAP.md`` Plan A2.

Heavy deps (torch, the vendored DA model) are imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

#: Implemented metric-depth backends (add a branch in ``make_metric_depth`` + list it here to register one).
METRIC_DEPTH_MODELS = ("depth_anything_v2",)


@runtime_checkable
class MetricDepth(Protocol):
    """Monocular metric-depth model → a metres-valued depth map for one RGB frame."""

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` metric depth (metres) for an ``(H, W, 3)`` uint8 **RGB** frame."""
        ...


class DepthAnythingV2Metric:
    """The released metric-depth model (Depth-Anything-V2). Byte-identical to the scale step it replaces."""

    def __init__(self, ckpt, max_depth: float = 80.0, device: str | None = None, input_size: int = 518):
        # Lazy: pull the DA config + vendor-path helper from the dust3r backend (its historical home) so this
        # module imports cheaply and there is no import cycle with dust3r_slam/vggt_slam.
        from gvhmr.utils.device import get_device
        from gvhmr.utils.preproc.dust3r_slam import _DA_CFG, _add_vendor_paths

        _add_vendor_paths()
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        self.input_size = input_size
        dev = device or str(get_device())
        model = DepthAnythingV2(**{**_DA_CFG, "max_depth": max_depth})
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=False))
        self.model = model.to(dev).eval()

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        with torch.no_grad():  # DA wants BGR; the backends hold RGB frames (imageio) — same as the old inline path
            return self.model.infer_image(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), input_size=self.input_size)


def make_metric_depth(
    name: str = "depth_anything_v2", *, ckpt: Path | None = None, max_depth: float = 80.0, device: str | None = None
) -> MetricDepth:
    """Construct a registered metric-depth model. Default = Depth-Anything-V2 (the released scale step)."""
    if name == "depth_anything_v2":
        if ckpt is None:
            from gvhmr.utils.preproc.dust3r_slam import _DA_CKPT

            ckpt = _DA_CKPT
        return DepthAnythingV2Metric(ckpt, max_depth=max_depth, device=device)
    if name in ("unidepth", "metric3d"):  # the A2 seam — implement mirroring DepthAnythingV2Metric
        raise NotImplementedError(
            f"metric-depth model {name!r} is a ROADMAP A2 stub — add a class emitting metres-valued depth "
            f"(the MetricDepth protocol) and a branch here, then A/B it with tools/eval/eval_world.py."
        )
    raise KeyError(f"unknown metric-depth model {name!r}; registered: {METRIC_DEPTH_MODELS}")


def metric_scale_from_depths(depth_model: MetricDepth, frames, kf_idx, recon_depths) -> float:
    """The one global scale: median metric/recon-depth ratio over keyframes (1.0 if none valid).

    ``recon_depths[i]`` is the reconstruction's (scale-ambiguous) depth map for keyframe ``kf_idx[i]``.
    Byte-identical to the loop the dust3r/vggt backends used inline before A2.
    """
    import cv2

    ratios = []
    for i, n in enumerate(kf_idx):
        md = depth_model.infer(frames[n])
        dd = recon_depths[i]
        md_r = cv2.resize(md, (dd.shape[1], dd.shape[0]))
        valid = (dd > 1e-6) & (md_r > 1e-6)
        if valid.any():
            ratios.append(float(np.median(md_r[valid] / dd[valid])))
    return float(np.median(ratios)) if ratios else 1.0
