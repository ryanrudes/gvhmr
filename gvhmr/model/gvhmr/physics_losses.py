"""Physics / contact training losses (docs/ROADMAP.md, Plan A3) — opt-in, default-off.

Pure loss functions targeting world-motion realism (foot sliding, ground penetration, jitter). They are
weight-gated in the pipeline (a weight of 0 skips them), so the released training is byte-identical.

All three are WIRED into ``compute_extra_global_loss`` (weight-gated, default-off):
- ``velocity_smoothness_loss`` — ``weights.transl_w_accel`` — world-root jitter.
- ``foot_contact_loss`` — ``weights.foot_slide`` — no foot sliding while in (GT-derived) contact.
- ``ground_penetration_loss`` — ``weights.penetration`` — no joints below the (per-sequence GT-foot) floor.

The predicted **world** joints they need are FK'd fast + differentiably in training: roll out the predicted
local translation velocity with the GT world orientation (teacher-forced, exactly as the ``transl_w`` loss
does — a vectorized cumsum), then ``fk_v2``. No inference-time for-loop (``get_smpl_params_w_Rt_v2``) needed.
"""

from __future__ import annotations

import torch


def velocity_smoothness_loss(transl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize world-root **acceleration** (2nd time-difference) — a jitter/velocity prior.

    ``transl`` ``(B, L, 3)`` world-frame root translation; ``mask`` ``(B, L)`` valid-frame mask. Averaged
    over frame-triples where all three are valid; a constant-velocity path (zero acceleration) → 0.
    """
    accel = transl[:, 2:] - 2 * transl[:, 1:-1] + transl[:, :-2]  # (B, L-2, 3)
    valid = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2])[..., None].float()  # (B, L-2, 1)
    return (accel.pow(2) * valid).sum() / valid.sum().clamp(min=1.0)


def ground_penetration_loss(joints: torch.Tensor, mask: torch.Tensor, ground_y: float = 0.0) -> torch.Tensor:
    """Penalize joints below the ground plane (gravity-up = +y, so ``y < ground_y`` is penetration).

    ``joints`` ``(B, L, J, 3)`` world-frame; ``mask`` ``(B, L)``. Only the penetration depth is penalized
    (a hinge at the ground), so joints above ground contribute nothing.
    """
    below = (ground_y - joints[..., 1]).clamp(min=0.0)  # (B, L, J)
    valid = mask[..., None].float()  # (B, L, 1)
    return (below.pow(2) * valid).sum() / (valid.sum() * joints.shape[-2]).clamp(min=1.0)


def foot_contact_loss(joints: torch.Tensor, contact: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize foot-joint velocity while in contact (no foot sliding).

    ``joints`` ``(B, L, Jf, 3)`` world-frame foot joints; ``contact`` ``(B, L, Jf)`` in-contact indicator
    (0/1); ``mask`` ``(B, L)``. A foot flagged in contact across a frame-pair should not move.
    """
    vel = joints[:, 1:] - joints[:, :-1]  # (B, L-1, Jf, 3)
    both = contact[:, 1:] * contact[:, :-1]  # (B, L-1, Jf) — in contact across the pair
    valid = (mask[:, 1:] & mask[:, :-1]).float()[..., None]  # (B, L-1, 1)
    w = (both * valid).unsqueeze(-1).float()  # (B, L-1, Jf, 1)
    return (vel.pow(2) * w).sum() / w.sum().clamp(min=1.0)
