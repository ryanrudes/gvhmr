"""Load real camera intrinsics for the demo/inference pipeline from a small sidecar file.

GVHMR's model consumes a per-frame pinhole ``K`` (full-image pixels). By default the demo builds a
single ``K`` from a diagonal-FOV heuristic (or ``--f_mm`` / ``--f_px``) and broadcasts it across the
clip. This module lets a user supply *measured* intrinsics — focal length **in pixels**, a true
principal point, and (optionally) **per-frame** values for a zoom / lens-switch — via a JSON or NPZ
sidecar.

Format (JSON ``<video>.intrinsics.json``, or an NPZ / NPY with the same keys)::

    {"width": 1920, "height": 1080,     # resolution the intrinsics were calibrated at (optional)
     "fx": 1450.0, "fy": 1450.0,         # focal length in pixels (fx required; fy defaults to fx)
     "cx": 964.2,  "cy": 541.8}          # principal point in pixels (optional; default = image center)

Any of ``fx / fy / cx / cy`` may instead be a length-``L`` list/array (one value per staged 30fps
frame) for per-frame intrinsics. A full matrix is also accepted as ``"K"`` of shape ``(3, 3)`` or
``(L, 3, 3)``.

Notes / landmines (see ``docs/ACCURACY.md``):
  * The model reads only ``K[0,0]`` (fx) as "the" focal, plus the principal point ``K[0,2]/K[1,2]``.
    ``fy`` is stored faithfully but never consumed (the model assumes square pixels) — so feeding
    ``fy`` separately gives no benefit, whereas a true off-center principal point *does*.
  * Intrinsics are resolution-bound. If ``width``/``height`` are given and differ from the frames the
    model actually processes (post-staging), the values are rescaled to match.
  * Per-frame arrays must have one value per *staged* frame — the clip is resampled to 30fps first, so
    a resampled source needs its per-frame intrinsics resampled the same way.
  * GVHMR is a pure pinhole model — lens-distortion coefficients, if any, are ignored (undistort the
    frames beforehand if the lens distorts noticeably).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _to_seq(value, length: int, name: str) -> torch.Tensor:
    """A scalar or length-``length`` sequence → a ``(length,)`` float tensor (scalars broadcast)."""
    arr = torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float32).reshape(-1)
    if arr.numel() == 1:
        return arr.expand(length).clone()
    if arr.numel() == length:
        return arr
    raise ValueError(
        f"intrinsics '{name}' has {arr.numel()} value(s); expected 1 (constant) or {length} "
        f"(per-frame — one per staged 30fps frame)"
    )


def _build_K_from_dict(data: dict, *, length: int, width: int, height: int) -> torch.Tensor:
    """Assemble a ``(length, 3, 3)`` pinhole ``K`` from a parsed intrinsics dict (see module docstring)."""
    cal_w = float(data.get("width") or width)
    cal_h = float(data.get("height") or height)

    if data.get("K") is not None:
        K = torch.as_tensor(np.asarray(data["K"], dtype=np.float64), dtype=torch.float32)
        if tuple(K.shape) == (3, 3):
            K = K.unsqueeze(0).repeat(length, 1, 1)
        elif K.ndim == 3 and tuple(K.shape[1:]) == (3, 3):
            if K.shape[0] != length:
                raise ValueError(
                    f"per-frame 'K' has {K.shape[0]} frame(s); expected {length} (one per staged 30fps frame)"
                )
            K = K.clone()
        else:
            raise ValueError(f"'K' must have shape (3, 3) or (L, 3, 3); got {tuple(K.shape)}")
    else:
        if data.get("fx") is None:
            raise ValueError("intrinsics must provide 'fx' (focal length in pixels) or a full 'K'")
        fx = _to_seq(data["fx"], length, "fx")
        fy = _to_seq(data["fy"] if data.get("fy") is not None else data["fx"], length, "fy")
        cx = _to_seq(data["cx"] if data.get("cx") is not None else cal_w / 2.0, length, "cx")
        cy = _to_seq(data["cy"] if data.get("cy") is not None else cal_h / 2.0, length, "cy")
        K = torch.zeros(length, 3, 3, dtype=torch.float32)
        K[:, 0, 0], K[:, 1, 1] = fx, fy
        K[:, 0, 2], K[:, 1, 2] = cx, cy
        K[:, 2, 2] = 1.0

    # Intrinsics are resolution-bound: rescale from the calibration resolution to the frames the model
    # actually sees (post-staging). fx/cx scale with width, fy/cy with height.
    if (round(cal_w), round(cal_h)) != (round(float(width)), round(float(height))):
        sx, sy = float(width) / cal_w, float(height) / cal_h
        K[:, 0, 0] *= sx
        K[:, 0, 2] *= sx
        K[:, 1, 1] *= sy
        K[:, 1, 2] *= sy
        from gvhmr.utils.pylogger import Log

        Log.info(f"Intrinsics rescaled from calibration {cal_w:.0f}x{cal_h:.0f} → {int(width)}x{int(height)}")
    return K


def load_intrinsics_file(path, *, length: int, width: int, height: int) -> torch.Tensor:
    """Load a JSON / NPZ / NPY intrinsics sidecar → a ``(length, 3, 3)`` full-image ``K``.

    See the module docstring for the accepted format. Raises ``ValueError`` on a malformed file or a
    per-frame array whose length doesn't match the staged clip.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} must contain a JSON object with fx/fy/cx/cy or K")
    elif suffix == ".npz":
        with np.load(path) as npz:
            data = {k: npz[k] for k in npz.files}
    elif suffix == ".npy":
        data = {"K": np.load(path)}
    else:
        raise ValueError(f"unsupported intrinsics file '{path.name}'; use .json, .npz, or .npy")
    return _build_K_from_dict(data, length=length, width=width, height=height)
