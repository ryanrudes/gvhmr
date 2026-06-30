"""World-trajectory post-processing for a moving camera (no heavy deps).

Lifted out of ``gvhmr/cli/demo.py`` so it can be imported and unit-tested without pulling in the
preproc/render stack. ``compose_world_from_dust3r`` carries the in-cam human through a per-frame
*metric* camera (e.g. the DUSt3R backend, or a dataset's ground-truth ``T_w2c``) to recover the
scene traversal a following camera induces — which the velocity prior misses when the subject stays
centred in frame. Used by both the demo and the world-frame eval harness (``tools/eval/eval_world.py``).
"""

from __future__ import annotations

import numpy as np
import torch

from gvhmr.utils.geo.rotations import axis_angle_to_matrix


def lowpass(x, window: int):
    """Moving-average low-pass over the time axis, endpoints preserved."""
    xn = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if len(xn) < window or window < 3:
        return torch.as_tensor(xn).float()
    k = np.ones(window) / window
    sm = np.stack([np.convolve(xn[:, i], k, mode="same") for i in range(xn.shape[1])], axis=1)
    e = window // 2
    sm[:e], sm[-e:] = xn[:e], xn[-e:]
    return torch.from_numpy(sm).float()


def compose_world_from_dust3r(pred: dict, T_w2c) -> None:
    """Moving-camera world trajectory from a metric camera (modifies ``pred`` in place).

    Carries the in-cam human through the per-frame metric camera (``world = R_c2w·in-cam + cam_centre``),
    gravity-aligns that path to the prior's frame (via the frame-0 body orientation), then keeps its
    **gross** low-frequency path and grafts the prior's **local** high-frequency motion. This captures
    the scene traversal a moving/following camera induces — which the velocity prior misses when the
    subject stays centred — and reduces to the static-camera in-cam carry when the camera doesn't move.
    """
    ic, gp = pred["smpl_params_incam"], pred["smpl_params_global"]
    c2w = torch.linalg.inv(torch.as_tensor(T_w2c).float())
    R_c2w, cam_c = c2w[:, :3, :3], c2w[:, :3, 3]
    geom = torch.einsum("fij,fj->fi", R_c2w, ic["transl"]) + cam_c  # metric world, camera frame
    geom_o0 = R_c2w[0] @ axis_angle_to_matrix(ic["global_orient"][0])
    r_align = axis_angle_to_matrix(gp["global_orient"][0]) @ geom_o0.mT  # → prior gravity-view frame
    geom_gv = (r_align @ (geom - geom[0]).mT).mT + gp["transl"][0]
    window = max(15, (len(geom_gv) // 3) | 1)  # odd, ~a third of the clip
    gp["transl"] = lowpass(geom_gv, window) + (gp["transl"] - lowpass(gp["transl"], window))
