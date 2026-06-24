"""Skeleton overlay geometry: subset resolution + mesh construction (no GL context needed)."""

from __future__ import annotations

import numpy as np
import pytest

from gvhmr.utils.vis.skeleton import (
    SMPL_BONES,
    SMPL_JOINT_NAMES,
    _CYL_V,
    _SPH_V,
    build_skeleton_mesh,
    resolve_joint_subset,
)


def test_resolve_all_and_none():
    assert resolve_joint_subset(None) == list(range(24))
    assert resolve_joint_subset("all") == list(range(24))
    assert resolve_joint_subset("  ") == list(range(24))


def test_resolve_groups_names_indices():
    # group expands to its joints; result is sorted + de-duplicated
    assert resolve_joint_subset("left_arm") == sorted(
        SMPL_JOINT_NAMES.index(n) for n in ("left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand")
    )
    # mixed name + index, with a duplicate collapsed
    assert resolve_joint_subset("left_knee, 4, right_knee") == [4, 5]


def test_resolve_bad_token():
    with pytest.raises(ValueError, match="Unknown skeleton joint/group"):
        resolve_joint_subset("nope")


def test_full_mesh_counts_and_validity():
    joints = np.arange(24 * 3, dtype="f4").reshape(24, 3)  # distinct positions → all bones non-degenerate
    v, f, c = build_skeleton_mesh(joints, resolve_joint_subset(None))
    # 24 spheres + 23 bones, each a unit primitive instance
    assert len(v) == 24 * len(_SPH_V) + len(SMPL_BONES) * len(_CYL_V)
    assert len(c) == len(v)  # per-vertex colors
    assert int(f.max()) < len(v) and int(f.min()) >= 0  # valid triangle indices


def test_subset_drops_unselected_bones():
    joints = np.arange(24 * 3, dtype="f4").reshape(24, 3)
    idx = resolve_joint_subset("left_arm")  # a 5-joint chain → 4 internal bones
    v, f, c = build_skeleton_mesh(joints, idx)
    n_bones = sum(1 for p, ch in SMPL_BONES if p in set(idx) and ch in set(idx))
    assert len(v) == len(idx) * len(_SPH_V) + n_bones * len(_CYL_V)
    assert len(v) < 24 * len(_SPH_V) + len(SMPL_BONES) * len(_CYL_V)  # strictly fewer than full


def test_empty_subset_is_renderer_safe():
    # no joints selected (e.g. a group with one joint and no internal bone still yields spheres);
    # the truly-empty path must not crash the renderer
    v, f, c = build_skeleton_mesh(np.zeros((24, 3), "f4"), [])
    assert v.shape == (1, 3) and f.shape == (1, 3) and c.shape == (1, 3)
