"""Pin the training augmentation RNG stream — the load-bearing order landmine (``docs/BEHAVIOR.md``).

The dataset/step augmentors consume ``np.random``/``torch`` RNG in a specific order; reordering them (or
switching to a ``Generator``) silently shifts the whole training stream. These tests snapshot the CPU
augmentation output for a fixed seed, so such a reorder fails CI instead of silently changing training.

CI-safe: CPU only, no data/checkpoints. The pinned values are torch-RNG-dependent — if you intentionally
change an augmentor (or bump the locked torch and its RNG shifts), re-snapshot and update the constants.
"""

from __future__ import annotations

import numpy as np
import torch

from gvhmr.utils.geo.augment_noisy_pose import (
    get_invisible_legs_mask,
    get_visible_mask,
    get_wham_aug_kp3d,
)

SHAPE = (2, 120)  # (B, L) at the training motion length (get_invisible_legs_mask needs L > 90)


def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


def test_augmentors_are_deterministic_under_a_fixed_seed():
    # Version-robust: same seed ⇒ identical output (catches accidental nondeterminism / device leakage).
    _seed()
    a1, v1, l1 = get_wham_aug_kp3d(SHAPE, device="cpu"), get_visible_mask(SHAPE), get_invisible_legs_mask(SHAPE)
    _seed()
    a2, v2, l2 = get_wham_aug_kp3d(SHAPE, device="cpu"), get_visible_mask(SHAPE), get_invisible_legs_mask(SHAPE)
    assert torch.equal(a1, a2) and torch.equal(v1, v2) and torch.equal(l1, l2)


def test_wham_kp3d_stream_snapshot():
    # get_wham_aug_kp3d draws bias → lfhp → jitter in that order; reordering shifts this sum.
    _seed()
    aug = get_wham_aug_kp3d(SHAPE, device="cpu")
    assert aug.shape == (2, 120, 17, 3)
    assert abs(float(aug.double().sum()) - 4.29557) < 1e-3


def test_visible_mask_stream_snapshot():
    _seed()
    vis = get_visible_mask(SHAPE)
    assert vis.shape == (2, 120, 17)
    assert int(vis.sum()) == 3723
