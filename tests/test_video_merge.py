"""``merge_videos_horizontal`` / ``merge_videos_vertical`` — PyAV frame-stacking.

Regression net for the HPC/headless failure mode: the old merge shelled out to a *system*
``ffmpeg`` binary (ffmpeg-python ``hstack``/``vstack``), which crashed on nodes that only have
PyAV's bundled ffmpeg. The reimplementation stacks frames with numpy and writes via PyAV. These
tests stub the video reader/writer, so they verify the stacking axis + length alignment without
touching real video files or any ffmpeg.
"""

from __future__ import annotations

import numpy as np
import pytest

import gvhmr.utils.video_io_utils as vio


def _stub_io(monkeypatch: pytest.MonkeyPatch, clips: dict[str, np.ndarray]) -> dict:
    """Route get_video_reader at the given clips and capture what save_video would write."""
    monkeypatch.setattr(vio, "get_video_reader", lambda path: iter(clips[path]))
    captured: dict = {}
    monkeypatch.setattr(vio, "save_video", lambda arr, out: captured.update(shape=arr.shape, out=out, arr=arr))
    return captured


def test_horizontal_merge_concatenates_width(monkeypatch: pytest.MonkeyPatch) -> None:
    clips = {"a": np.zeros((5, 4, 6, 3), np.uint8), "b": np.ones((5, 4, 3, 3), np.uint8)}
    captured = _stub_io(monkeypatch, clips)
    vio.merge_videos_horizontal(["a", "b"], "out.mp4")
    assert captured["shape"] == (5, 4, 9, 3)  # same H=4, widths 6+3
    assert captured["out"] == "out.mp4"


def test_vertical_merge_concatenates_height(monkeypatch: pytest.MonkeyPatch) -> None:
    clips = {"a": np.zeros((5, 4, 6, 3), np.uint8), "b": np.ones((5, 2, 6, 3), np.uint8)}
    captured = _stub_io(monkeypatch, clips)
    vio.merge_videos_vertical(["a", "b"], "out.mp4")
    assert captured["shape"] == (5, 6, 6, 3)  # same W=6, heights 4+2


def test_merge_aligns_to_shortest_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mismatched lengths align to the shortest, mirroring ffmpeg's hstack behavior."""
    clips = {"a": np.zeros((5, 4, 6, 3), np.uint8), "b": np.ones((3, 4, 6, 3), np.uint8)}
    captured = _stub_io(monkeypatch, clips)
    vio.merge_videos_horizontal(["a", "b"], "out.mp4")
    assert captured["shape"] == (3, 4, 12, 3)  # min(5, 3) == 3 frames


def test_merge_requires_two_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_io(monkeypatch, {"a": np.zeros((5, 4, 6, 3), np.uint8)})
    with pytest.raises(ValueError):
        vio.merge_videos_horizontal(["a"], "out.mp4")
