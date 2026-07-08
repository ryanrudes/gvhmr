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

Optional lens distortion (wide-angle / fisheye)::

    {"fx": 900, "fy": 900, "cx": 960, "cy": 540,
     "distortion": [k1, k2, p1, p2, k3],   # or {"k1":…, "k2":…, "p1":…, "p2":…, "k3":…}
     "undistort_alpha": 0.0}               # getOptimalNewCameraMatrix free-scaling (0 crop … 1 keep all)

GVHMR is a pinhole-only model, so distortion isn't fed to the network — instead the demo **undistorts
the staged frames** (``gvhmr.cli.demo._apply_undistortion``) and swaps in the corrected pinhole ``K``.
Distortion requires a single (constant) ``K``. Distortion parsing lives here
(:func:`load_intrinsics_for_undistort`); the actual pixel remap is in the demo (it needs the staged video).

Notes / landmines (see ``docs/ACCURACY.md``, ``docs/CAMERA_METADATA.md``):
  * The model reads only ``K[0,0]`` (fx) as "the" focal, plus the principal point ``K[0,2]/K[1,2]``.
    ``fy`` is stored faithfully but never consumed (the model assumes square pixels) — so feeding
    ``fy`` separately gives no benefit, whereas a true off-center principal point *does*.
  * Intrinsics are resolution-bound. If ``width``/``height`` are given and differ from the frames the
    model actually processes (post-staging), the values are rescaled to match.
  * Per-frame arrays must have one value per *staged* frame — the clip is resampled to 30fps first, so
    a resampled source needs its per-frame intrinsics resampled the same way.
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


def _read_sidecar(path) -> dict:
    """Parse a ``.json`` / ``.npz`` / ``.npy`` intrinsics sidecar into a plain dict."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} must contain a JSON object with fx/fy/cx/cy or K")
        return data
    if suffix == ".npz":
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}
    if suffix == ".npy":
        return {"K": np.load(path)}
    raise ValueError(f"unsupported intrinsics file '{path.name}'; use .json, .npz, or .npy")


def load_intrinsics_file(path, *, length: int, width: int, height: int) -> torch.Tensor:
    """Load a JSON / NPZ / NPY intrinsics sidecar → a ``(length, 3, 3)`` full-image ``K``.

    See the module docstring for the accepted format. Raises ``ValueError`` on a malformed file or a
    per-frame array whose length doesn't match the staged clip. Any ``distortion`` entry is ignored here
    (it is consumed earlier, at staging, by :func:`load_intrinsics_for_undistort`).
    """
    return _build_K_from_dict(_read_sidecar(path), length=length, width=width, height=height)


def _distortion_vector(data: dict) -> np.ndarray | None:
    """The OpenCV distortion coefficients from the sidecar's optional ``distortion`` entry, or ``None``.

    Accepts a list/array in OpenCV order ``[k1, k2, p1, p2, k3, …]`` (length 4/5/8/12/14) or a dict with
    any of ``{k1, k2, p1, p2, k3}`` (missing → 0). Distortion coefficients are dimensionless (defined in
    normalized image coordinates), so they need no resolution rescale."""
    d = data.get("distortion")
    if d is None:
        return None
    if isinstance(d, dict):
        return np.asarray([float(d.get(k) or 0.0) for k in ("k1", "k2", "p1", "p2", "k3")], dtype=np.float64)
    arr = np.asarray(d, dtype=np.float64).reshape(-1)
    if arr.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"'distortion' must have 4, 5, 8, 12, or 14 OpenCV coefficients; got {arr.size}")
    return arr


def _has_per_frame(data: dict) -> bool:
    """True if any focal/principal-point field (or ``K``) varies per frame."""
    for key in ("fx", "fy", "cx", "cy"):
        v = data.get(key)
        if isinstance(v, (list, tuple)) and len(v) > 1:
            return True
        if isinstance(v, np.ndarray) and v.size > 1:
            return True
    K = data.get("K")
    return K is not None and np.asarray(K).ndim == 3 and np.asarray(K).shape[0] > 1


def load_intrinsics_for_undistort(path, *, width: int, height: int):
    """If the sidecar declares lens ``distortion``, return ``(K_3x3, dist_vec, meta)`` for undistorting the
    frames; else ``None``.

    ``K_3x3`` is the (distorted) camera matrix as float64, rescaled to ``width``/``height``; ``dist_vec`` is
    the OpenCV coefficient array; ``meta`` carries ``alpha`` (the ``getOptimalNewCameraMatrix`` free-scaling,
    from the optional ``undistort_alpha`` key, default 0.0). Requires a single **constant** ``K`` — per-frame
    intrinsics + distortion is unsupported (a fixed lens has fixed distortion)."""
    data = _read_sidecar(path)
    dist = _distortion_vector(data)
    if dist is None:
        return None
    if _has_per_frame(data):
        raise ValueError("lens 'distortion' requires constant intrinsics (per-frame fx/cx/cy/K is unsupported)")
    K = _build_K_from_dict(data, length=1, width=width, height=height)[0].numpy().astype(np.float64)
    return K, dist, {"alpha": float(data.get("undistort_alpha") or 0.0)}
