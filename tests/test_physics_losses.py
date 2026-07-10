"""Physics/contact training losses (docs/ROADMAP.md A3) — the loss math, pinned. No model/GPU.

These are opt-in (weight-gated) so the released training is byte-identical; the tests pin the math so the
opt-in physics retrain is trustworthy.
"""

from __future__ import annotations

import torch

from gvhmr.model.gvhmr.physics_losses import (
    foot_contact_loss,
    ground_penetration_loss,
    velocity_smoothness_loss,
)


def test_velocity_smoothness_zero_for_constant_velocity():
    mask = torch.ones(1, 5, dtype=torch.bool)
    v = torch.tensor([1.0, 2.0, 3.0])
    transl = torch.arange(5).float()[None, :, None] * v  # constant velocity → zero acceleration
    assert velocity_smoothness_loss(transl, mask).item() < 1e-6


def test_velocity_smoothness_known_acceleration():
    mask = torch.ones(1, 5, dtype=torch.bool)
    transl = torch.zeros(1, 5, 3)
    transl[0, :, 0] = torch.arange(5).float() ** 2  # x = t² → second difference = 2 everywhere
    assert abs(velocity_smoothness_loss(transl, mask).item() - 4.0) < 1e-5  # ‖(2,0,0)‖² = 4


def test_ground_penetration_only_penalizes_below():
    mask = torch.ones(1, 1, dtype=torch.bool)
    above = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]]])  # both above y=0
    assert ground_penetration_loss(above, mask).item() == 0.0
    mixed = torch.tensor([[[[0.0, -1.0, 0.0], [0.0, 3.0, 0.0]]]])  # one 1m below, one above
    assert abs(ground_penetration_loss(mixed, mask).item() - 0.5) < 1e-6  # (1² + 0²)/(1·2 joints)


def test_foot_contact_penalizes_sliding_only_in_contact():
    mask = torch.ones(1, 2, dtype=torch.bool)
    moving = torch.tensor([[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]])  # foot moves 1m between frames
    assert abs(foot_contact_loss(moving, torch.ones(1, 2, 1), mask).item() - 1.0) < 1e-6  # in contact → penalized
    assert foot_contact_loss(moving, torch.zeros(1, 2, 1), mask).item() == 0.0  # not in contact → ignored
    static = torch.tensor([[[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]])  # foot planted
    assert foot_contact_loss(static, torch.ones(1, 2, 1), mask).item() == 0.0  # in contact, no slide → 0
