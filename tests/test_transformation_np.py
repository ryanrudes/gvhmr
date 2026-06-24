"""Characterization tests for :mod:`gvhmr.utils.preproc.relpose.transformation_np`.

This module is plain numpy (no cv2/pycolmap is exercised here): quaternions are
**wxyz** (real part first), and the camera-pose interpolation builds ``(F_all, 4, 4)``
``T_w2c`` stacks. These tests pin the rotation algebra (matrix<->quaternion
round-trips), the hand-rolled ``slerp`` (endpoints + antipodal/short-path handling),
and ``lerp_missing_frames`` so the reorg/rename/typing passes can't change numerics.

The package ``gvhmr.utils.preproc.__init__`` eagerly imports heavy optional deps
(ultralytics, etc.), so we load the target module directly from its file path to
keep the test importable on a bare CPU environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "gvhmr" / "utils" / "preproc" / "relpose" / "transformation_np.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("transformation_np_under_test", _SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load_module()

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])
ROT_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ROT_Z180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
ROT_X180 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _random_rotation() -> np.ndarray:
    """A proper rotation matrix (det +1) via QR of a random matrix."""
    a = np.random.randn(3, 3)
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def _make_pose(rot: np.ndarray, trans: np.ndarray | list[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(trans, dtype=float)
    return m


# --------------------------------------------------------------------------- #
# rotation_matrix_to_quaternion / quaternion_to_rotation_matrix
# --------------------------------------------------------------------------- #
def test_quaternion_to_matrix_of_identity_quat() -> None:
    assert np.allclose(T.quaternion_to_rotation_matrix(IDENTITY_Q), np.eye(3), atol=1e-12)


def test_identity_matrix_to_quaternion_is_identity() -> None:
    assert np.allclose(T.rotation_matrix_to_quaternion(np.eye(3)), IDENTITY_Q, atol=1e-12)


def test_rotation_matrix_to_quaternion_golden_z90() -> None:
    # 90-deg about +z lands in the trace>0 branch; golden captured from the code.
    q = T.rotation_matrix_to_quaternion(ROT_Z90)
    assert np.allclose(q, [0.7071067811865476, 0.0, 0.0, 0.7071067811865475], atol=1e-12)


def test_rotation_matrix_to_quaternion_golden_x180() -> None:
    # 180-deg about +x exercises the (m00 > m11 and m00 > m22) branch.
    q = T.rotation_matrix_to_quaternion(ROT_X180)
    assert np.allclose(q, [0.0, 1.0, 0.0, 0.0], atol=1e-12)


def test_rotation_matrix_to_quaternion_golden_z180() -> None:
    # 180-deg about +z exercises the final (qz) branch.
    q = T.rotation_matrix_to_quaternion(ROT_Z180)
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_quaternion_from_rotation_is_unit_norm() -> None:
    for _ in range(50):
        q = T.rotation_matrix_to_quaternion(_random_rotation())
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9)


def test_quaternion_to_matrix_is_special_orthogonal() -> None:
    for _ in range(50):
        q = T.rotation_matrix_to_quaternion(_random_rotation())
        m = T.quaternion_to_rotation_matrix(q)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(m), 1.0, atol=1e-9)


def test_matrix_quaternion_roundtrip() -> None:
    # R -> q -> R must recover the rotation (covers all four code branches).
    for _ in range(100):
        r = _random_rotation()
        q = T.rotation_matrix_to_quaternion(r)
        assert np.allclose(T.quaternion_to_rotation_matrix(q), r, atol=1e-9)


def test_quaternion_matrix_roundtrip_on_axis_rotations() -> None:
    for r in (np.eye(3), ROT_Z90, ROT_Z180, ROT_X180):
        q = T.rotation_matrix_to_quaternion(r)
        assert np.allclose(T.quaternion_to_rotation_matrix(q), r, atol=1e-12)


# --------------------------------------------------------------------------- #
# slerp
# --------------------------------------------------------------------------- #
def test_slerp_hits_endpoints() -> None:
    q0 = T.rotation_matrix_to_quaternion(_random_rotation())
    q1 = T.rotation_matrix_to_quaternion(_random_rotation())
    assert np.allclose(T.slerp(q0, q1, 0.0), q0, atol=1e-9)
    assert np.allclose(T.slerp(q0, q1, 1.0), q1, atol=1e-9)


def test_slerp_output_is_unit_norm() -> None:
    q0 = IDENTITY_Q
    q1 = T.rotation_matrix_to_quaternion(ROT_Z90)
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert np.isclose(np.linalg.norm(T.slerp(q0, q1, t)), 1.0, atol=1e-9)


def test_slerp_midpoint_golden() -> None:
    # Half-way between identity and a +z 90-deg quat is the +z 45-deg quat.
    q0 = IDENTITY_Q
    q1 = T.rotation_matrix_to_quaternion(ROT_Z90)
    mid = T.slerp(q0, q1, 0.5)
    assert np.allclose(mid, [0.9238795325112867, 0.0, 0.0, 0.3826834323650898], atol=1e-12)
    # ...and it corresponds to a proper 45-deg rotation about z.
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    expected_r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(T.quaternion_to_rotation_matrix(mid), expected_r, atol=1e-12)


def test_slerp_short_path_for_antipodal_inputs() -> None:
    # q1 == -q0 (dot == -1): the dot<0 flip makes q1 == q0, so the lerp branch
    # returns the (re-normalised) common rotation. Must stay finite.
    q0 = IDENTITY_Q
    out = T.slerp(q0, -q0, 0.5)
    assert np.all(np.isfinite(out))
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-9)
    # q and -q are the same rotation, so the result is that rotation (identity here).
    assert np.allclose(T.quaternion_to_rotation_matrix(out), np.eye(3), atol=1e-9)


def test_slerp_is_sign_invariant_on_second_quat() -> None:
    # Because slerp picks the short path, q1 and -q1 yield the same interpolated
    # *rotation* (the quaternion may differ by an overall sign).
    q0 = T.rotation_matrix_to_quaternion(_random_rotation())
    q1 = T.rotation_matrix_to_quaternion(_random_rotation())
    a = T.slerp(q0, q1, 0.5)
    b = T.slerp(q0, -q1, 0.5)
    assert np.allclose(T.quaternion_to_rotation_matrix(a), T.quaternion_to_rotation_matrix(b), atol=1e-12)


# --------------------------------------------------------------------------- #
# lerp_missing_frames
# --------------------------------------------------------------------------- #
def _known_poses() -> tuple[np.ndarray, list[int]]:
    poses = np.stack(
        [
            _make_pose(np.eye(3), [0.0, 0.0, 0.0]),
            _make_pose(ROT_Z90, [2.0, 0.0, 0.0]),
            _make_pose(ROT_Z180, [4.0, 0.0, 0.0]),
        ]
    )
    sample_idxs = [0, 2, 4]
    return poses, sample_idxs


def test_lerp_missing_frames_output_shape() -> None:
    poses, sample_idxs = _known_poses()
    out = T.lerp_missing_frames(poses, sample_idxs)
    # F_all == sample_idxs[-1] + 1.
    assert out.shape == (5, 4, 4)


def test_lerp_missing_frames_preserves_known_frames() -> None:
    poses, sample_idxs = _known_poses()
    out = T.lerp_missing_frames(poses, sample_idxs)
    for known_pos, idx in enumerate(sample_idxs):
        assert np.allclose(out[idx], poses[known_pos], atol=1e-12)


def test_lerp_missing_frames_first_frame_is_identity_pose() -> None:
    # sample_idxs[0] == 0 -> frame 0 is exactly the first known pose (identity here).
    poses, sample_idxs = _known_poses()
    out = T.lerp_missing_frames(poses, sample_idxs)
    assert np.allclose(out[0], np.eye(4), atol=1e-12)


def test_lerp_missing_frames_last_frame_matches_last_known() -> None:
    # sample_idxs[-1] == F_all - 1 -> the final row is exactly the last known pose.
    poses, sample_idxs = _known_poses()
    out = T.lerp_missing_frames(poses, sample_idxs)
    assert np.allclose(out[-1], poses[-1], atol=1e-12)


def test_lerp_missing_frames_translation_is_linear() -> None:
    poses, sample_idxs = _known_poses()
    out = T.lerp_missing_frames(poses, sample_idxs)
    # frame 1 sits halfway between known frames 0 (x=0) and 2 (x=2): x == 1.
    assert np.allclose(out[1][:3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    # frame 3 sits halfway between known frames 2 (x=2) and 4 (x=4): x == 3.
    assert np.allclose(out[3][:3, 3], [3.0, 0.0, 0.0], atol=1e-12)


def test_lerp_missing_frames_interpolated_rotation_is_valid() -> None:
    poses, sample_idxs = _known_poses()
    out = T.lerp_missing_frames(poses, sample_idxs)
    for idx in (1, 3):
        r = out[idx][:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-9)
        # bottom row of the homogeneous matrix is preserved.
        assert np.allclose(out[idx][3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_lerp_missing_frames_interpolated_rotation_golden() -> None:
    # Frame 1 = slerp(identity, Rz90, 0.5) => +z 45-deg rotation. Golden captured
    # from the current code.
    poses, sample_idxs = _known_poses()
    out = T.lerp_missing_frames(poses, sample_idxs)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    expected_r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(out[1][:3, :3], expected_r, atol=1e-12)


def test_lerp_missing_frames_all_known_is_identity_op() -> None:
    # When every frame is known, the output equals the input stack.
    poses, _ = _known_poses()
    out = T.lerp_missing_frames(poses, [0, 1, 2])
    assert out.shape == (3, 4, 4)
    assert np.allclose(out, poses, atol=1e-12)
