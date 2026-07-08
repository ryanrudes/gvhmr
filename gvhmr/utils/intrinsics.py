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
``distortion`` may be **per-frame** too (a ``(L, N)`` array, or a dict whose coefficients are length-``L``
lists) — for a zoom, which changes focal *and* distortion together; pair it with per-frame ``fx/fy``.
Distortion parsing lives here (:func:`load_intrinsics_for_undistort`); the actual pixel remap is in the
demo (it needs the staged video).

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


_DIST_SIZES = (4, 5, 8, 12, 14)  # OpenCV distortion-coefficient counts


def _distortion_vector(data: dict, length: int) -> np.ndarray | None:
    """The OpenCV distortion coefficients from the sidecar's optional ``distortion`` entry, or ``None``.

    Returns a **constant** ``(N,)`` vector or, for a zoom / lens change, a **per-frame** ``(length, N)``
    array (one coefficient row per staged frame). Accepts:
      * a flat list/array ``[k1, k2, p1, p2, k3, …]`` (length 4/5/8/12/14) → constant;
      * a 2-D list/array of shape ``(length, N)`` → per-frame;
      * a dict ``{k1, k2, p1, p2, k3}`` where each value is a scalar **or** a length-``length`` list — any
        per-frame value makes the whole thing per-frame (missing coeff → 0).
    Distortion coefficients are dimensionless (defined in normalized image coordinates), so — unlike
    ``fx/fy/cx/cy`` — they need no resolution rescale."""
    d = data.get("distortion")
    if d is None:
        return None
    if isinstance(d, dict):
        cols, per_frame = [], False
        for k in ("k1", "k2", "p1", "p2", "k3"):
            arr = np.asarray(d[k] if d.get(k) is not None else 0.0, dtype=np.float64).reshape(-1)
            if arr.size not in (1, length):
                raise ValueError(f"distortion '{k}' has {arr.size} value(s); expected 1 (constant) or {length}")
            per_frame = per_frame or arr.size == length
            cols.append(arr)
        if not per_frame:
            return np.asarray([c.item() for c in cols], dtype=np.float64)  # (5,)
        return np.stack([np.broadcast_to(c, (length,)) for c in cols], axis=1)  # (length, 5)
    arr = np.asarray(d, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size not in _DIST_SIZES:
            raise ValueError(
                f"'distortion' must have {' / '.join(map(str, _DIST_SIZES))} OpenCV coeffs; got {arr.size}"
            )
        return arr  # (N,) constant
    if arr.ndim == 2:
        if arr.shape[0] != length:
            raise ValueError(
                f"per-frame 'distortion' has {arr.shape[0]} row(s); expected {length} (one per staged frame)"
            )
        if arr.shape[1] not in _DIST_SIZES:
            raise ValueError(f"per-frame 'distortion' rows must have {' / '.join(map(str, _DIST_SIZES))} coeffs")
        return arr  # (length, N) per-frame
    raise ValueError("'distortion' must be 1-D (constant) or 2-D (length, N) (per-frame)")


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


def load_intrinsics_for_undistort(path, *, length: int, width: int, height: int):
    """If the sidecar declares lens ``distortion``, return ``(K, dist, meta)`` for undistorting the frames;
    else ``None``.

    Handles both **constant** and **per-frame** lenses (a zoom changes focal *and* distortion). When either
    the focal/principal point (``fx/fy/cx/cy``/``K``) or the distortion is per-frame, both are returned as
    length-``length`` arrays (the constant one broadcast) and ``meta["per_frame"]`` is ``True``:
      * constant → ``K`` ``(3,3)`` float64, ``dist`` ``(N,)``;
      * per-frame → ``K`` ``(length,3,3)`` float64, ``dist`` ``(length,N)``.
    ``K`` is rescaled to ``width``/``height``; ``dist`` is not (coefficients are dimensionless). ``meta`` also
    carries ``alpha`` (the ``getOptimalNewCameraMatrix`` free-scaling; optional ``undistort_alpha`` key,
    default 0.0). Per-frame distortion needs a zoom-aware calibration — see docs/CAMERA_METADATA.md."""
    data = _read_sidecar(path)
    dist = _distortion_vector(data, length)
    if dist is None:
        return None
    per_frame = dist.ndim == 2 or _has_per_frame(data)
    K = _build_K_from_dict(data, length=length if per_frame else 1, width=width, height=height)
    K = K.numpy().astype(np.float64)  # (length,3,3) or (1,3,3)
    meta = {"alpha": float(data.get("undistort_alpha") or 0.0), "per_frame": per_frame}
    if not per_frame:
        return K[0], dist, meta  # (3,3), (N,)
    if dist.ndim == 1:  # per-frame K but constant distortion → broadcast the coefficients across frames
        dist = np.broadcast_to(dist, (length, dist.shape[0])).copy()
    return K, dist, meta  # (length,3,3), (length,N)
