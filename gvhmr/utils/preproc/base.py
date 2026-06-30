"""Pluggable preprocessing backends — detector and 2D-pose — selectable by name.

Mirrors the camera ``--slam`` selector: each stage has a small **Protocol** (the output contract a
replacement must satisfy) and a tiny **registry** so a newer/arbitrary model drops in by name without
editing call sites, as long as it emits the same format. See ``docs/EXTENSIBILITY.md`` (Tier A).

The factories **lazy-import** their implementations, so importing this module is cheap and does not pull
in ultralytics / the vendored ViTPose (which live behind the ``preproc`` extra) — keeping it import-safe
on the base/CI install.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Detector(Protocol):
    """Person detector/tracker → one smoothed bbox track for the primary subject."""

    def get_one_track(self, video_path) -> torch.Tensor:
        """Return ``bbx_xyxy (F, 4)`` ``[x1, y1, x2, y2]`` per frame (gaps interpolated)."""
        ...


@runtime_checkable
class Pose2D(Protocol):
    """2D keypoint estimator → **COCO-17** keypoints (the skeleton the trained net asserts)."""

    def extract(self, video_path, bbx_xys) -> torch.Tensor:
        """Return ``(F, 17, 3)`` ``[x, y, conf]`` in full-image pixels."""
        ...


# Registered backend names (kept here so callers/tests can enumerate without importing the heavy impls).
DETECTORS = ("yolo",)
POSE2D = ("vitpose",)


def make_detector(name: str = "yolo", **kwargs) -> Detector:
    """Construct a registered detector. ``kwargs`` (e.g. ``ckpt``, ``conf``) pass to its ctor."""
    if name == "yolo":
        from gvhmr.utils.preproc.tracker import Tracker

        return Tracker(**kwargs)
    raise KeyError(f"unknown detector {name!r}; registered: {DETECTORS}")


def make_pose2d(name: str = "vitpose", **kwargs) -> Pose2D:
    """Construct a registered 2D-pose estimator. Must emit COCO-17 ``(F,17,3)``."""
    if name == "vitpose":
        from gvhmr.utils.preproc.vitpose import VitPoseExtractor

        return VitPoseExtractor(**kwargs)
    raise KeyError(f"unknown pose2d {name!r}; registered: {POSE2D}")
