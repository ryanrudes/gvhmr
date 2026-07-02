"""``renderer_gl._standalone_context`` — GL context creation with a headless EGL fallback.

Regression net for the HPC/cluster failure mode: moderngl's default backend uses GLX/X11 and
dies with ``XOpenDisplay: cannot open display`` on a displayless GPU node. The renderer must
fall back to EGL there, while still preferring the platform default on macOS / X11 desktops.
Stubs moderngl so the logic is exercised without a GPU, a display, or moderngl itself.
"""

from __future__ import annotations

import torch

from gvhmr.utils.vis.renderer import _look_at_rotation, get_global_cameras_static
from gvhmr.utils.vis.renderer_gl import _standalone_context


class _FakeMGL:
    """Minimal stand-in for the moderngl module, recording which backends were attempted."""

    def __init__(self, *, fail_default: bool) -> None:
        self.fail_default = fail_default
        self.calls: list[str | None] = []

    def create_standalone_context(self, backend: str | None = None):
        self.calls.append(backend)
        if backend is None and self.fail_default:
            raise Exception("(standalone) XOpenDisplay: cannot open display")
        return f"ctx(backend={backend})"


def test_prefers_platform_default_when_available() -> None:
    """A working default context (macOS CGL / X11 GLX) is used as-is; EGL is not attempted."""
    mgl = _FakeMGL(fail_default=False)
    ctx = _standalone_context(mgl)
    assert ctx == "ctx(backend=None)"
    assert mgl.calls == [None]  # never fell through to EGL


def test_falls_back_to_egl_when_headless() -> None:
    """When the default backend fails (no X display), retry with the EGL backend."""
    mgl = _FakeMGL(fail_default=True)
    ctx = _standalone_context(mgl)
    assert ctx == "ctx(backend=egl)"
    assert mgl.calls == [None, "egl"]  # tried default first, then EGL


def test_look_at_rotation_is_orthonormal() -> None:
    """The pure-torch look-at yields proper rotation matrices (RᵀR = I, det = +1)."""
    cam = torch.tensor([[3.0, 2.0, 5.0], [-4.0, 1.0, 0.0]])
    at = torch.zeros(2, 3)
    R = _look_at_rotation(cam, at)
    assert R.shape == (2, 3, 3)
    assert torch.allclose(R @ R.mT, torch.eye(3).expand(2, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(2), atol=1e-5)


def test_global_cameras_static_needs_no_pytorch3d() -> None:
    """Regression: this world-camera helper hard-required pytorch3d (look_at_rotation /
    PointLights) and raised NameError on the moderngl path. It must now run pure-torch."""
    torch.manual_seed(0)
    verts = torch.randn(10, 100, 3) * 0.3 + torch.tensor([0.0, 1.0, 0.0])
    R, T, _lights = get_global_cameras_static(verts)
    length = verts.shape[0]
    assert R.shape == (length, 3, 3)
    assert T.shape == (length, 3)
    assert torch.isfinite(R).all() and torch.isfinite(T).all()
    assert torch.allclose(R @ R.mT, torch.eye(3).expand(length, 3, 3), atol=1e-4)
