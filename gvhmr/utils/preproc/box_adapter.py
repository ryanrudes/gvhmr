"""Detector box-distribution adapter (docs/ROADMAP.md, Plan A4) — an inference-only, no-retrain lever.

A "better" detector (higher COCO mAP / NMS-free) can *hurt* GVHMR: the whole pipeline — the ViT crops and
the box→depth cue — is calibrated to the released detector's (yolov8x) box **distribution**, not to
detection quality (measured: yolo26x −19% PA-MPJPE from box framing alone, boxes byte-identical elsewhere).
This renormalizes a new detector's boxes toward that distribution, so its robustness can be used without
paying the framing penalty.

``BoxAdapter`` applies a normalized affine to a square-box ``(cx, cy, size)`` track — default **identity**
(byte-identical, so the released path is untouched). Calibrate it per-detector with ``fit_box_adapter``
from paired boxes (a new detector vs the frozen baseline on the eval packs), then apply at inference and
validate the metric recovery with ``gvhmr eval``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BoxAdapter:
    """Normalized affine on a square-box track: ``size *= scale``; ``center += (dx, dy) * size``.

    ``dx``/``dy`` are in units of the (pre-adapt) box size, so the transform is scale-invariant across
    sequences. The default ``(1, 0, 0)`` is the identity — no change to any box.
    """

    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.dx == 0.0 and self.dy == 0.0

    def apply(self, bbx_xys: torch.Tensor) -> torch.Tensor:
        """``(F, 3)`` ``[cx, cy, size]`` → adapted ``(F, 3)``."""
        cx, cy, size = bbx_xys[:, 0], bbx_xys[:, 1], bbx_xys[:, 2]
        return torch.stack([cx + self.dx * size, cy + self.dy * size, size * self.scale], dim=-1)

    @classmethod
    def from_config(cls, cfg) -> BoxAdapter:
        """Build from a mapping / DictConfig / None (a config knob). Falsy → identity."""
        if not cfg:
            return cls()
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d: cfg[k] if k in cfg else d)
        return cls(scale=float(get("scale", 1.0)), dx=float(get("dx", 0.0)), dy=float(get("dy", 0.0)))


def fit_box_adapter(new_xys: torch.Tensor, baseline_xys: torch.Tensor) -> BoxAdapter:
    """Calibrate the affine that maps a detector's boxes toward the baseline's, from paired frames.

    ``scale`` = median(baseline_size / new_size); ``dx``/``dy`` = median per-frame center offset
    (baseline − new) normalized by the new box size. Median (not mean) so a few bad frames don't skew it.
    ``(F, 3), (F, 3)`` in the ``(cx, cy, size)`` convention. ``fit`` then ``apply`` maps new → baseline.
    """
    n = min(len(new_xys), len(baseline_xys))
    a, b = new_xys[:n].float(), baseline_xys[:n].float()
    size = a[:, 2].clamp(min=1e-6)
    return BoxAdapter(
        scale=float((b[:, 2] / size).median()),
        dx=float(((b[:, 0] - a[:, 0]) / size).median()),
        dy=float(((b[:, 1] - a[:, 1]) / size).median()),
    )
