"""Characterization tests for :mod:`gvhmr.utils.geo.quaternion`.

Convention: quaternions are **wxyz** (real part first), matching pytorch3d.
These tests pin the rotation algebra (compose, invert, rotate, 6D, slerp) so the
reorg/rename/typing passes can't change numerics.
"""

from __future__ import annotations

import numpy as np
import torch

from gvhmr.utils.geo import quaternion as Q
from tests.conftest import random_unit_quaternions

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_qmul_identity_is_neutral() -> None:
    q = random_unit_quaternions(16)
    assert torch.allclose(Q.qmul(q, IDENTITY.expand_as(q)), q, atol=1e-6)
    assert torch.allclose(Q.qmul(IDENTITY.expand_as(q), q), q, atol=1e-6)


def test_qinv_is_left_and_right_inverse() -> None:
    q = random_unit_quaternions(32)
    prod = Q.qmul(q, Q.qinv(q))
    assert torch.allclose(prod, IDENTITY.expand_as(prod), atol=1e-6)


def test_qmul_matches_matrix_composition() -> None:
    q1 = random_unit_quaternions(8)
    q2 = random_unit_quaternions(8)
    m_from_q = Q.quaternion_to_matrix(Q.qmul(q1, q2))
    m_product = Q.quaternion_to_matrix(q1) @ Q.quaternion_to_matrix(q2)
    assert torch.allclose(m_from_q, m_product, atol=1e-5)


def test_qrot_matches_matrix_action() -> None:
    q = random_unit_quaternions(8)
    v = torch.randn(8, 3)
    rotated = Q.qrot(q, v)
    mat = Q.quaternion_to_matrix(q)
    expected = torch.einsum("bij,bj->bi", mat, v)
    assert torch.allclose(rotated, expected, atol=1e-5)


def test_qrot_identity_leaves_vectors_unchanged() -> None:
    v = torch.randn(10, 3)
    out = Q.qrot(IDENTITY.expand(10, 4), v)
    assert torch.allclose(out, v, atol=1e-6)


def test_quaternion_to_matrix_is_special_orthogonal() -> None:
    q = random_unit_quaternions(20)
    m = Q.quaternion_to_matrix(q)
    eye = torch.eye(3).expand_as(m)
    assert torch.allclose(m @ m.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(m), torch.ones(20), atol=1e-5)


def test_cont6d_to_matrix_is_special_orthogonal() -> None:
    # Build 6D from valid rotations so Gram-Schmidt has a well-defined answer.
    q = random_unit_quaternions(12)
    cont6d = Q.quaternion_to_cont6d(q)
    m = Q.cont6d_to_matrix(cont6d)
    eye = torch.eye(3).expand_as(m)
    assert torch.allclose(m @ m.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(m), torch.ones(12), atol=1e-5)


def test_quaternion_to_cont6d_roundtrips_through_matrix() -> None:
    q = random_unit_quaternions(12)
    mat = Q.quaternion_to_matrix(q)
    cont6d = Q.quaternion_to_cont6d(q)
    # cont6d packs columns 0 and 1 of the rotation matrix; cont6d_to_matrix must recover it.
    assert torch.allclose(Q.cont6d_to_matrix(cont6d), mat, atol=1e-5)


def test_qnormalize_unit_norm() -> None:
    q = torch.randn(7, 4) * 5.0
    qn = Q.qnormalize(q)
    assert torch.allclose(qn.norm(dim=-1), torch.ones(7), atol=1e-6)


def test_qbetween_rotates_v0_onto_v1() -> None:
    v0 = torch.randn(16, 3)
    v1 = torch.randn(16, 3)
    q = Q.qbetween(v0, v1)
    rotated = Q.qrot(q, v0 / v0.norm(dim=-1, keepdim=True))
    target = v1 / v1.norm(dim=-1, keepdim=True)
    assert torch.allclose(rotated, target, atol=1e-5)


def test_qbetween_handles_antipodal_vectors() -> None:
    # v0 == -v1 is the degenerate branch guarded in qbetween.
    v0 = torch.tensor([[0.0, 0.0, 1.0]])
    v1 = torch.tensor([[0.0, 0.0, -1.0]])
    q = Q.qbetween(v0, v1)
    assert torch.isfinite(q).all()
    rotated = Q.qrot(q, v0)
    assert torch.allclose(rotated, v1, atol=1e-5)


def test_qslerp_hits_endpoints() -> None:
    q0 = random_unit_quaternions(1, canonical=True)
    q1 = random_unit_quaternions(1, canonical=True)
    out0 = Q.qslerp(q0, q1, 0.0)
    out1 = Q.qslerp(q0, q1, 1.0)
    # slerp returns t.shape + q.shape; squeeze the leading point dim.
    assert torch.allclose(out0.reshape(1, 4), q0, atol=1e-5)
    assert torch.allclose(out1.reshape(1, 4), q1, atol=1e-5)


def test_qfix_enforces_temporal_continuity() -> None:
    q = random_unit_quaternions(10, 2).numpy()
    # Inject sign flips; qfix should remove the discontinuities.
    flipped = q.copy()
    flipped[3:] *= -1
    fixed = Q.qfix(flipped)
    dots = np.sum(fixed[1:] * fixed[:-1], axis=-1)
    assert (dots > -1e-4).all()


def test_numpy_and_torch_paths_agree() -> None:
    q = random_unit_quaternions(5).numpy()
    v = torch.randn(5, 3).numpy()
    assert np.allclose(Q.qrot_np(q, v), Q.qrot(torch.from_numpy(q), torch.from_numpy(v)).numpy(), atol=1e-6)
    r = random_unit_quaternions(5).numpy()
    assert np.allclose(Q.qmul_np(q, r), Q.qmul(torch.from_numpy(q), torch.from_numpy(r)).numpy(), atol=1e-6)


def test_expmap_to_quaternion_small_angle() -> None:
    # Zero rotation -> identity quaternion.
    e = np.zeros((4, 3), dtype=np.float64)
    quat = Q.expmap_to_quaternion(e)
    assert np.allclose(quat, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
