"""Characterization tests for the vendored ``gvhmr.network.hmr2.utils.geometry``.

This module holds pure-math rotation/projection helpers copied from HMR2.0
(``rot6d_to_rotmat``, ``aa_to_rotmat``, ``quat_to_rotmat``,
``perspective_projection``). It is **vendored / frozen** (see ``pyproject.toml``
``[tool.pyright].exclude``), and the package ``__init__`` chain pulls in heavy
deps (``yacs``, checkpoints), so we load the file directly by path rather than
``import gvhmr.network.hmr2.utils.geometry``. These tests pin the current
numerics — orthogonality, det +1, known axis-aligned rotations, and known pixel
coordinates — so the reorg/rename/typing passes can't change behaviour.

Quirk worth noting: ``aa_to_rotmat`` adds ``1e-8`` to the axis-angle vector
*before* normalizing, so even an exact 90-degree rotation comes back as
``1.0000001`` rather than ``1.0``. We characterise that with a loose tolerance
instead of pretending it is exact.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import torch

from tests.conftest import random_rotation_matrices

_GEOMETRY_PATH = Path(__file__).resolve().parents[1] / "gvhmr" / "network" / "hmr2" / "utils" / "geometry.py"


def _load_geometry() -> ModuleType:
    """Load the vendored geometry module straight from its file.

    Importing the parent package would execute ``hmr2/__init__.py``, which needs
    ``yacs`` and released checkpoints; the math we test has no such deps.
    """
    spec = importlib.util.spec_from_file_location("hmr2_geometry", _GEOMETRY_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


geom = _load_geometry()

EYE3 = torch.eye(3)


def _is_special_orthogonal(m: torch.Tensor, atol: float = 1e-5) -> None:
    """Assert ``m`` is a batch of proper rotations (R Rᵀ = I, det = +1)."""
    eye = torch.eye(3).expand_as(m)
    assert torch.allclose(m @ m.transpose(-1, -2), eye, atol=atol)
    assert torch.allclose(torch.linalg.det(m), torch.ones(m.shape[:-2]), atol=atol)


# --------------------------------------------------------------------------- #
# rot6d_to_rotmat
# --------------------------------------------------------------------------- #
def test_rot6d_to_rotmat_is_special_orthogonal() -> None:
    # Build 6D from valid rotations so Gram-Schmidt has a well-defined answer:
    # the first two columns of a rotation matrix, flattened.
    rots = random_rotation_matrices(24)
    x = rots[..., :2].permute(0, 2, 1).reshape(24, 6)
    m = geom.rot6d_to_rotmat(x)
    assert m.shape == (24, 3, 3)
    _is_special_orthogonal(m)


def test_rot6d_to_rotmat_recovers_source_rotation() -> None:
    # rot6d packs columns 0 and 1; the Gram-Schmidt result must rebuild the
    # original rotation when the input came from a genuine rotation matrix.
    rots = random_rotation_matrices(16)
    x = rots[..., :2].permute(0, 2, 1).reshape(16, 6)
    assert torch.allclose(geom.rot6d_to_rotmat(x), rots, atol=1e-5)


def test_rot6d_to_rotmat_identity_golden() -> None:
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    m = geom.rot6d_to_rotmat(x)
    assert torch.allclose(m[0], EYE3, atol=1e-6)


def test_rot6d_to_rotmat_is_orthonormalizing() -> None:
    # Non-orthonormal 6D input still yields a proper rotation (the point of the
    # Zhou et al. representation): scale + skew columns 0 and 1.
    x = torch.tensor([[2.0, 0.0, 0.0, 0.5, 3.0, 0.0]])
    m = geom.rot6d_to_rotmat(x)
    _is_special_orthogonal(m)
    # b1 is the normalized first column: (1, 0, 0) here.
    assert torch.allclose(m[0, :, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


# --------------------------------------------------------------------------- #
# quat_to_rotmat
# --------------------------------------------------------------------------- #
def test_quat_to_rotmat_identity_golden() -> None:
    m = geom.quat_to_rotmat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(m[0], EYE3, atol=1e-6)


def test_quat_to_rotmat_is_special_orthogonal() -> None:
    quats = torch.randn(20, 4)
    m = geom.quat_to_rotmat(quats)
    assert m.shape == (20, 3, 3)
    _is_special_orthogonal(m)


def test_quat_to_rotmat_normalizes_input() -> None:
    # A non-unit quaternion is normalized internally; scaling must not matter.
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    m_raw = geom.quat_to_rotmat(q)
    m_scaled = geom.quat_to_rotmat(q * 7.0)
    assert torch.allclose(m_raw, m_scaled, atol=1e-6)


def test_quat_to_rotmat_z_rotation_golden() -> None:
    # 90deg about +z: w = cos(45), z = sin(45)  ->  [[0,-1,0],[1,0,0],[0,0,1]].
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    m = geom.quat_to_rotmat(torch.tensor([[c, 0.0, 0.0, s]]))
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(m[0], expected, atol=1e-6)


def test_quat_to_rotmat_x_rotation_golden() -> None:
    # 90deg about +x: w = cos(45), x = sin(45)  ->  [[1,0,0],[0,0,-1],[0,1,0]].
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    m = geom.quat_to_rotmat(torch.tensor([[c, s, 0.0, 0.0]]))
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(m[0], expected, atol=1e-6)


def test_quat_to_rotmat_negation_is_same_rotation() -> None:
    # q and -q represent the same rotation.
    q = torch.randn(8, 4)
    assert torch.allclose(geom.quat_to_rotmat(q), geom.quat_to_rotmat(-q), atol=1e-6)


# --------------------------------------------------------------------------- #
# aa_to_rotmat
# --------------------------------------------------------------------------- #
def test_aa_to_rotmat_zero_is_identity() -> None:
    # The 1e-8 bias is negligible at the origin: zero axis-angle -> identity.
    m = geom.aa_to_rotmat(torch.zeros(1, 3))
    assert torch.allclose(m[0], EYE3, atol=1e-6)


def test_aa_to_rotmat_is_special_orthogonal() -> None:
    aa = torch.randn(20, 3)
    m = geom.aa_to_rotmat(aa)
    assert m.shape == (20, 3, 3)
    # Slightly looser tol: aa_to_rotmat's 1e-8 bias perturbs the norm.
    _is_special_orthogonal(m, atol=1e-4)


def test_aa_to_rotmat_z_rotation_golden() -> None:
    # 90deg about +z. The 1e-8 axis-angle bias makes entries ~1.0000001, so we
    # pin the *structure* with a loose tol rather than asserting exact 1.0.
    m = geom.aa_to_rotmat(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(m[0], expected, atol=1e-5)


def test_aa_to_rotmat_x_rotation_golden() -> None:
    # 90deg about +x  ->  [[1,0,0],[0,0,-1],[0,1,0]].
    m = geom.aa_to_rotmat(torch.tensor([[math.pi / 2, 0.0, 0.0]]))
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(m[0], expected, atol=1e-5)


def test_aa_to_rotmat_180_about_y_golden() -> None:
    # 180deg about +y  ->  diag(-1, 1, -1).
    m = geom.aa_to_rotmat(torch.tensor([[0.0, math.pi, 0.0]]))
    expected = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    assert torch.allclose(m[0], expected, atol=1e-5)


def test_aa_to_rotmat_matches_quat_to_rotmat() -> None:
    # aa_to_rotmat is implemented via quat_to_rotmat; cross-check the path for a
    # generic rotation (build the same half-angle quaternion by hand).
    axis = torch.tensor([0.0, 0.0, 1.0])
    angle = 1.2345
    aa = (axis * angle).unsqueeze(0)
    half = angle / 2.0
    quat = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
    assert torch.allclose(geom.aa_to_rotmat(aa), geom.quat_to_rotmat(quat), atol=1e-5)


# --------------------------------------------------------------------------- #
# perspective_projection
# --------------------------------------------------------------------------- #
def test_perspective_projection_centers_origin_ray() -> None:
    # A point on the optical axis projects to the camera center (here 0).
    pts = torch.tensor([[[0.0, 0.0, 10.0]]])
    out = geom.perspective_projection(pts, torch.zeros(1, 3), torch.tensor([[1000.0, 1000.0]]))
    assert out.shape == (1, 1, 2)
    assert torch.allclose(out[0, 0], torch.zeros(2), atol=1e-6)


def test_perspective_projection_known_pixels_golden() -> None:
    # x' = f * X / Z, y' = f * Y / Z with identity rotation, zero translation.
    pts = torch.tensor([[[0.0, 0.0, 10.0], [1.0, 2.0, 10.0]]])
    out = geom.perspective_projection(pts, torch.zeros(1, 3), torch.tensor([[1000.0, 1000.0]]))
    expected = torch.tensor([[[0.0, 0.0], [100.0, 200.0]]])
    assert torch.allclose(out, expected, atol=1e-5)


def test_perspective_projection_camera_center_offsets_pixels() -> None:
    pts = torch.tensor([[[0.0, 0.0, 10.0], [1.0, 2.0, 10.0]]])
    cc = torch.tensor([[320.0, 240.0]])
    out = geom.perspective_projection(pts, torch.zeros(1, 3), torch.tensor([[1000.0, 1000.0]]), camera_center=cc)
    expected = torch.tensor([[[320.0, 240.0], [420.0, 440.0]]])
    assert torch.allclose(out, expected, atol=1e-5)


def test_perspective_projection_translation_shifts_depth() -> None:
    # +z translation moves the point further away, shrinking the projection.
    pts = torch.tensor([[[1.0, 2.0, 10.0]]])
    out = geom.perspective_projection(
        pts,
        torch.tensor([[0.0, 0.0, 5.0]]),
        torch.tensor([[1000.0, 1000.0]]),
    )
    # X/Z = 1/15, Y/Z = 2/15, scaled by f = 1000.
    expected = torch.tensor([[[1000.0 / 15.0, 2000.0 / 15.0]]])
    assert torch.allclose(out, expected, atol=1e-4)


def test_perspective_projection_rotation_is_applied() -> None:
    # Rotating the world 90deg about +z maps (1,0,Z) -> (0,1,Z) before project.
    rot = geom.aa_to_rotmat(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    pts = torch.tensor([[[1.0, 0.0, 10.0]]])
    out = geom.perspective_projection(
        pts,
        torch.zeros(1, 3),
        torch.tensor([[1000.0, 1000.0]]),
        rotation=rot,
    )
    expected = torch.tensor([[[0.0, 100.0]]])
    assert torch.allclose(out, expected, atol=1e-4)


def test_perspective_projection_default_rotation_matches_explicit_identity() -> None:
    pts = torch.randn(2, 5, 3) + torch.tensor([0.0, 0.0, 8.0])
    trans = torch.zeros(2, 3)
    fl = torch.full((2, 2), 900.0)
    eye = EYE3.unsqueeze(0).expand(2, -1, -1)
    out_default = geom.perspective_projection(pts, trans, fl)
    out_explicit = geom.perspective_projection(pts, trans, fl, rotation=eye)
    assert torch.allclose(out_default, out_explicit, atol=1e-6)
