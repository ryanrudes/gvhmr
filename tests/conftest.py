"""Shared pytest fixtures and helpers for the GVHMR test suite.

The suite is a *characterization* / regression net: it pins the behaviour of the
pure-Python / torch / numpy math that the model depends on, so the modernization
refactors (rename, reorg, typing) cannot silently change numerics. Everything here
runs on CPU (and, where marked, Apple-Silicon MPS) with no checkpoints or datasets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed every test so random tensors are reproducible."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def mps_device() -> torch.device:
    """An Apple-Silicon MPS device, or skip the test if unavailable."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    return torch.device("mps")


def random_unit_quaternions(*shape: int, canonical: bool = False) -> torch.Tensor:
    """Random unit (wxyz, real-first) quaternions of shape ``(*shape, 4)``.

    With ``canonical=True`` the real part is forced non-negative (picks one of the
    two antipodal representatives) so comparisons aren't confused by ``q`` vs ``-q``.
    """
    q = torch.randn(*shape, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    if canonical:
        q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    return q


def random_rotation_matrices(*shape: int) -> torch.Tensor:
    """Random proper rotation matrices of shape ``(*shape, 3, 3)`` via QR."""
    a = torch.randn(*shape, 3, 3)
    q, r = torch.linalg.qr(a)
    # Fix sign so det = +1 (proper rotation).
    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * d.unsqueeze(-2)
    # Guarantee det +1 by flipping last column where det is -1.
    det = torch.linalg.det(q)
    q[..., -1] *= det.unsqueeze(-1).sign()
    return q
