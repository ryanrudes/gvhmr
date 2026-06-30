"""Core helpers for trainer-free **world-frame** evaluation (W/WA-MPJPE, RTE, jitter, foot-sliding).

The Lightning callback ``gvhmr/model/gvhmr/callbacks/metric_emdb.py`` computes these global metrics
inside the trainer; this module factors the same math (``compute_global_metrics``) into a small,
device-agnostic surface that a plain script can drive over any dataset with ground-truth *world-frame*
SMPL/SMPL-X. ``tools/eval/eval_world.py`` is the driver; the dataset adapters
(``gvhmr/dataset/{sloper4d,whac}/``) yield :class:`WorldSeqGT` records.

Both predictions and ground truth are reduced to a **common** representation — 6890-vertex SMPL meshes
plus the neutral 24-joint regressor — so SMPL and SMPL-X ground truth score against the same joints the
released SMPL-X model predicts. ``compute_global_metrics`` aligns per 100-frame chunk (Procrustes), so
the prediction's gravity-view world frame need not match the dataset's world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from einops import einsum
from torch import Tensor

from gvhmr.utils.device import to_device
from gvhmr.utils.eval.eval_utils import compute_global_metrics

# The five global metrics compute_global_metrics returns, lower-is-better, in report order.
GLOBAL_METRIC_KEYS = ("wa2_mpjpe", "waa_mpjpe", "rte", "jitter", "fs")

MODEL_FPS = 30.0  # GVHMR integrates per-frame velocities at 30 fps; inputs must be resampled to it


def resample_indices(length: int, src_fps: float, target_fps: float = MODEL_FPS) -> list[int]:
    """Frame indices that resample ``length`` frames from ``src_fps`` to ``target_fps``.

    Identical policy to the demo's video staging (``gvhmr/cli/demo.py``): a near-match passes through
    unchanged; otherwise nearest-frame pick. Adapters apply this to **both** the RGB frames and the
    per-frame ground truth (``smpl_params``, ``T_w2c``, ``mask``) so prediction and GT stay aligned.
    """
    if abs(src_fps - target_fps) <= 1.5:
        return list(range(length))
    n_out = max(1, round(length * target_fps / src_fps))
    return [min(length - 1, round(j * src_fps / target_fps)) for j in range(n_out)]


@dataclass
class WorldSeqGT:
    """One evaluation sequence: a staged RGB video + ground-truth world-frame body & camera.

    ``smpl_params`` are per-frame SMPL(-X) parameters in the **world** frame (axis-angle
    ``global_orient`` (L,3), ``body_pose`` (L,69 SMPL / L,63 SMPL-X), ``transl`` (L,3), ``betas``);
    ``T_w2c`` is the metric world→camera extrinsic the camera composition consumes. ``mask`` flags
    valid frames (``None`` ⇒ all valid).
    """

    vid: str
    frames_mp4: Path
    smpl_params: dict[str, Tensor]
    body_type: str  # "smpl" | "smplx"
    gender: str  # "neutral" | "male" | "female"
    T_w2c: Tensor  # (L, 4, 4)
    K_fullimg: Tensor  # (L, 3, 3) or (3, 3)
    fps: float
    length: int
    mask: Tensor | None = None


@dataclass
class BodyModels:
    """Eval body models / regressors, built once and reused across sequences (cf. eval_3dpw.py)."""

    smplx: object  # released SMPL-X ("supermotion") for predictions and SMPL-X GT
    smpl: dict[str, object]  # gendered SMPL for SMPL GT: {"male"|"female"|"neutral": model}
    smplx2smpl: Tensor  # (6890, 10475) sparse: SMPL-X verts → SMPL verts
    J_regressor: Tensor  # (24, 6890) neutral SMPL joint regressor
    device: torch.device


def _smplx_to_smpl_verts(smplx_model, smplx2smpl: Tensor, params: dict, device) -> Tensor:
    out = smplx_model(**to_device(params, device))
    return torch.stack([torch.matmul(smplx2smpl, v) for v in out.vertices])  # (L, 6890, 3)


def world_verts_j3d(bm: BodyModels, params: dict, *, body_type: str, gender: str) -> tuple[Tensor, Tensor]:
    """Run a body model → (verts (L,6890,3), j3d (L,24,3)) in its native (world) frame, on CPU."""
    if body_type == "smpl":
        out = bm.smpl[gender](**to_device(params, bm.device))
        verts = out.vertices  # (L, 6890, 3)
    elif body_type == "smplx":
        verts = _smplx_to_smpl_verts(bm.smplx, bm.smplx2smpl, params, bm.device)
    else:
        raise ValueError(f"body_type must be 'smpl' or 'smplx', got {body_type!r}")
    j3d = einsum(bm.J_regressor, verts, "j v, l v i -> l j i")
    return verts.cpu(), j3d.cpu()


def score_global(
    pred_verts: Tensor,
    pred_j3d: Tensor,
    gt_verts: Tensor,
    gt_j3d: Tensor,
    mask: Tensor | None = None,
) -> dict:
    """Thin wrapper over ``compute_global_metrics`` (the EMDB-2 protocol)."""
    batch = {
        "pred_j3d_glob": pred_j3d,
        "target_j3d_glob": gt_j3d,
        "pred_verts_glob": pred_verts,
        "target_verts_glob": gt_verts,
    }
    return compute_global_metrics(batch, mask=mask)
