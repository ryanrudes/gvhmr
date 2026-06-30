"""Offline characterization of the world-eval plumbing — no data, checkpoints, or body models.

Two things are pinned here, both on synthetic tensors so they run in CI:

1. ``compose_world_from_dust3r`` recovers a *following-camera* traversal that the velocity prior
   misses, and does **not** fabricate motion when the camera is static.
2. ``score_global`` (the EMDB-2 metric driver) ranks a camera-composed trajectory above the static
   prior when the ground truth is moving — i.e. the metric actually rewards the composition.
"""

from __future__ import annotations

import torch

from gvhmr.utils.eval.world_eval import GLOBAL_METRIC_KEYS, score_global
from gvhmr.utils.postproc_world import compose_world_from_dust3r

L = 120  # frames (≥1 chunk for the 100-frame W/WA-MPJPE windows)


def _zeros_aa(n):
    return torch.zeros(n, 3)


def _identity_incam(transl):
    """In-cam SMPL params with a constant pelvis (subject centred in frame)."""
    return {
        "transl": transl,
        "global_orient": _zeros_aa(len(transl)),
        "body_pose": torch.zeros(len(transl), 69),
        "betas": torch.zeros(len(transl), 10),
    }


def _T_w2c_following(person_xyz, incam_d):
    """Camera that holds the subject at a constant in-cam offset ``incam_d`` (R = I)."""
    T = torch.eye(4).repeat(len(person_xyz), 1, 1)
    T[:, :3, 3] = incam_d - person_xyz  # t_w2c = d - p  ⇒  R_w2c·p + t = d
    return T


def test_compose_recovers_following_camera_traversal():
    t = torch.arange(L).float()
    person = torch.stack([0.02 * t, torch.zeros(L), torch.zeros(L)], dim=1)  # walks +x, ~2.4 m total
    incam_d = torch.tensor([0.0, 0.0, 4.0]).expand(L, 3)  # 4 m in front, centred

    pred = {
        "smpl_params_incam": _identity_incam(incam_d.clone()),
        # prior fails for a following camera: ~no world translation
        "smpl_params_global": _identity_incam(torch.zeros(L, 3)),
    }
    compose_world_from_dust3r(pred, _T_w2c_following(person, incam_d))

    composed = pred["smpl_params_global"]["transl"]
    # recovered traversal tracks the true walk (low-pass softens the linear-ramp endpoints)
    err = (composed - person).norm(dim=-1)
    assert err.median() < 0.05
    assert composed[:, 0].max() > 2.0  # actually traversed, not flattened to the prior's ~0


def test_compose_static_camera_does_not_fabricate_motion():
    incam_d = torch.tensor([0.0, 0.0, 4.0]).expand(L, 3)
    person = torch.zeros(L, 3)  # camera fixed at origin, subject stationary
    pred = {
        "smpl_params_incam": _identity_incam(incam_d.clone()),
        "smpl_params_global": _identity_incam(torch.zeros(L, 3)),
    }
    compose_world_from_dust3r(pred, _T_w2c_following(person, incam_d))
    span = pred["smpl_params_global"]["transl"].amax(0) - pred["smpl_params_global"]["transl"].amin(0)
    assert span.norm() < 0.05  # no traversal invented from a still camera


def _body_from_transl(transl, offsets):
    """Cheap (verts (L,6890,3), j3d (L,24,3)) whose pelvis tracks ``transl``."""
    j3d = transl[:, None, :] + offsets[None]
    verts = transl[:, None, :].expand(len(transl), 6890, 3).contiguous()
    return verts, j3d


def test_score_global_rewards_camera_composition():
    g = torch.Generator().manual_seed(0)
    offsets = torch.randn(24, 3, generator=g) * 0.1  # shared body shape, only trajectory differs
    t = torch.arange(L).float()
    gt_transl = torch.stack([0.03 * t, torch.zeros(L), torch.zeros(L)], dim=1)  # moving GT

    gt = _body_from_transl(gt_transl, offsets)
    prior = _body_from_transl(torch.zeros(L, 3), offsets)  # static prior
    gtcam = _body_from_transl(gt_transl + 0.01 * torch.randn(L, 3, generator=g), offsets)  # composed

    m_prior = score_global(*prior, *gt)
    m_gtcam = score_global(*gtcam, *gt)

    for k in GLOBAL_METRIC_KEYS:
        assert k in m_prior and k in m_gtcam
    # the composed trajectory is closer to GT than the static prior on both world metrics
    assert m_gtcam["waa_mpjpe"].mean() < m_prior["waa_mpjpe"].mean()
    assert m_gtcam["wa2_mpjpe"].mean() < m_prior["wa2_mpjpe"].mean()
