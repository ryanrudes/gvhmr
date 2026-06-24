"""Characterization tests for :mod:`gvhmr.utils.geo.transforms`.

The module exposes a single helper, :func:`axis_rotate_to_matrix`, which builds a
proper rotation matrix for a rotation about one principal axis (``"x"``/``"y"``/
``"z"``). It accepts either a Python ``float`` (wrapped to a length-1 batch) or an
``(N, 1)`` angle tensor and returns ``(N, 3, 3)`` rotation matrices.

These tests pin the current numerics (special-orthogonality, golden matrices at
known angles, composition along an axis, the float-scalar entry path) so the
upcoming reorg / rename / typing passes can't change behaviour. They also pin that
the sibling ``gvhmr.utils.geo.quaternion`` module — whose ``np.float`` fallback
(``try: np.finfo(np.float) except: np.finfo(float)``) is the kind of NumPy-2.x
landmine a refactor might trip over — still imports on the installed NumPy.

Pure torch / numpy; CPU-only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gvhmr.utils.geo.transforms import axis_rotate_to_matrix

AXES = ("x", "y", "z")

# Canonical basis vector that each axis-rotation must leave fixed.
_FIXED_AXIS = {"x": 0, "y": 1, "z": 2}


def _batch_angles(n: int) -> torch.Tensor:
    """A column of ``(n, 1)`` angles spanning a few sign/magnitude regimes."""
    return torch.linspace(-2.0, 2.0, n).reshape(n, 1)


@pytest.mark.parametrize("axis", AXES)
def test_matrices_are_special_orthogonal(axis: str) -> None:
    """Every output is a proper rotation: ``R Rᵀ = I`` and ``det R = +1``."""
    angles = _batch_angles(9)
    r = axis_rotate_to_matrix(angles, axis=axis)
    eye = torch.eye(3).expand_as(r)
    assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-6)
    assert torch.allclose(torch.linalg.det(r), torch.ones(9), atol=1e-6)


def test_output_shape_and_dtype() -> None:
    r = axis_rotate_to_matrix(_batch_angles(5), axis="x")
    assert r.shape == (5, 3, 3)
    assert r.dtype == torch.float32


def test_zero_angle_is_identity() -> None:
    for axis in AXES:
        r = axis_rotate_to_matrix(torch.zeros(4, 1), axis=axis)
        assert r.shape == (4, 3, 3)
        assert torch.allclose(r, torch.eye(3).expand_as(r), atol=1e-6)


@pytest.mark.parametrize("axis", AXES)
def test_rotation_fixes_its_own_axis(axis: str) -> None:
    """A rotation about ``axis`` leaves the corresponding basis vector unchanged."""
    e = torch.zeros(3)
    e[_FIXED_AXIS[axis]] = 1.0
    r = axis_rotate_to_matrix(_batch_angles(6), axis=axis)
    rotated = torch.einsum("nij,j->ni", r, e)
    assert torch.allclose(rotated, e.expand_as(rotated), atol=1e-6)


def test_float_scalar_path_wraps_to_unit_batch() -> None:
    """Passing a Python ``float`` takes the ``isinstance(angle, float)`` branch."""
    r = axis_rotate_to_matrix(math.pi / 2, axis="x")
    assert r.shape == (1, 3, 3)
    assert r.dtype == torch.float32
    tensor_path = axis_rotate_to_matrix(torch.tensor([[math.pi / 2]]), axis="x")
    assert torch.allclose(r, tensor_path, atol=1e-6)


def test_default_axis_is_x() -> None:
    angles = _batch_angles(5)
    assert torch.allclose(
        axis_rotate_to_matrix(angles),
        axis_rotate_to_matrix(angles, axis="x"),
        atol=1e-7,
    )


def test_composition_along_axis_adds_angles() -> None:
    """``R(a) @ R(b) == R(a + b)`` about a common axis (axis-rotations commute)."""
    a = torch.tensor([[0.4], [-1.3], [2.0]])
    b = torch.tensor([[0.9], [0.2], [-0.5]])
    for axis in AXES:
        ra = axis_rotate_to_matrix(a, axis=axis)
        rb = axis_rotate_to_matrix(b, axis=axis)
        rab = axis_rotate_to_matrix(a + b, axis=axis)
        assert torch.allclose(ra @ rb, rab, atol=1e-6)


def test_inverse_is_negative_angle_and_transpose() -> None:
    angles = _batch_angles(6)
    for axis in AXES:
        r = axis_rotate_to_matrix(angles, axis=axis)
        r_inv = axis_rotate_to_matrix(-angles, axis=axis)
        eye = torch.eye(3).expand_as(r)
        assert torch.allclose(r @ r_inv, eye, atol=1e-6)
        # Orthogonality means the inverse is also the transpose.
        assert torch.allclose(r_inv, r.transpose(-1, -2), atol=1e-6)


def test_golden_quarter_turn_matrices() -> None:
    """Golden values captured from the current code at a +90 degree turn."""
    q = math.pi / 2
    rx = axis_rotate_to_matrix(torch.tensor([[q]]), axis="x")[0]
    ry = axis_rotate_to_matrix(torch.tensor([[q]]), axis="y")[0]
    rz = axis_rotate_to_matrix(torch.tensor([[q]]), axis="z")[0]
    assert torch.allclose(rx, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(ry, torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(rz, torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), atol=1e-6)


def test_half_turn_about_y_flips_x_and_z() -> None:
    """A pi rotation about y maps e_x -> -e_x and e_z -> -e_z (golden action)."""
    r = axis_rotate_to_matrix(torch.full((1, 1), math.pi), axis="y")[0]
    assert torch.allclose(r @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(r @ torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)


def test_bad_axis_raises_assertion() -> None:
    with pytest.raises(AssertionError):
        axis_rotate_to_matrix(torch.tensor([[0.1]]), axis="w")


def test_quaternion_sibling_np_float_fallback_imports() -> None:
    """The ``np.float`` fallback in the sibling module must survive NumPy 2.x.

    ``np.float`` was removed in NumPy 2.x; the module guards it with a bare
    ``try/except`` that falls back to ``np.finfo(float)``. Pin that the import
    succeeds and yields the standard float64 machine epsilon so the refactor
    doesn't accidentally drop the guard.
    """
    from gvhmr.utils.geo import quaternion as Q

    assert np.finfo(float).eps == Q._FLOAT_EPS
