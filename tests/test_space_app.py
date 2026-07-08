"""Space app unit tests — no GPU, no live HF upload."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

# space/app.py does an unconditional `import gradio as gr`; gradio ships only in the `app` extra, which
# CI (dev + cpu) doesn't install. Skip the whole module rather than erroring at collection time there.
pytest.importorskip("gradio")

SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
_APP_MOD = None


def _load_space_app():
    """Import space/app.py once with a mocked ``spaces`` SDK (real torch/gradio)."""
    global _APP_MOD
    if _APP_MOD is not None:
        return _APP_MOD

    spaces_mod = ModuleType("spaces")

    def gpu_decorator(**_kwargs):
        def wrap(fn):
            return fn

        return wrap

    spaces_mod.GPU = gpu_decorator
    sys.modules["spaces"] = spaces_mod

    spec = importlib.util.spec_from_file_location("space_app_under_test", SPACE_DIR / "app.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _APP_MOD = mod
    return mod


def test_resolve_device_cpu_when_no_cuda(monkeypatch) -> None:
    monkeypatch.delenv("GVHMR_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    app = _load_space_app()
    assert app._resolve_device() == "cpu"


def test_resolve_device_cuda_when_available(monkeypatch) -> None:
    monkeypatch.delenv("GVHMR_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    app = _load_space_app()
    assert app._resolve_device() == "cuda"


def test_resolve_device_falls_back_when_env_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("GVHMR_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    app = _load_space_app()
    assert app._resolve_device() == "cpu"


def test_gradio_whitelist_includes_xethub() -> None:
    _load_space_app()
    from gradio import processing_utils as pu

    assert "cas-bridge-direct.xethub.hf.co" in pu.PUBLIC_HOSTNAME_WHITELIST


@pytest.mark.parametrize(
    "video,expected",
    [
        (None, None),
        ("/tmp/clip.mp4", "/tmp/clip.mp4"),
        ({"path": "/data/v.mp4"}, "/data/v.mp4"),
    ],
)
def test_video_path_normalization(video, expected) -> None:
    app = _load_space_app()
    assert app._video_path(video) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("/tmp/cam.intrinsics.json", "/tmp/cam.intrinsics.json"),
        ({"path": "/data/cam.npz"}, "/data/cam.npz"),
        ({"name": "/data/cam.json"}, "/data/cam.json"),
    ],
)
def test_intrinsics_path_normalization(value, expected) -> None:
    """The optional intrinsics upload normalizes to a path, or None (→ metadata/heuristic fallback)."""
    app = _load_space_app()
    assert app._intrinsics_path(value) == expected


def test_preflight_rejects_unavailable_camera() -> None:
    app = _load_space_app()
    import gradio as gr

    with pytest.raises(gr.Error, match="not set up"):
        app._preflight_stages(
            static_camera=False,
            camera="dust3r",
            detector="yolo",
            pose2d="vitpose",
            backbone="hmr2",
        )


def test_preflight_static_skips_camera_dep() -> None:
    # A clean static-camera run must not raise — but that requires the `preproc` stack (yolo→ultralytics)
    # to actually be importable, which CI's dev+cpu sync doesn't provide. Skip when it's absent.
    pytest.importorskip("ultralytics")
    app = _load_space_app()
    app._preflight_stages(
        static_camera=True,
        camera="simplevo",
        detector="yolo",
        pose2d="vitpose",
        backbone="hmr2",
    )


def test_pose2d_choices_include_rtmpose() -> None:
    app = _load_space_app()
    assert "rtmpose" in app.POSE2D_CHOICES
