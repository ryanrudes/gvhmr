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


@runtime_checkable
class FeatureBackbone(Protocol):
    """Per-frame image-feature extractor → ``f_imgseq`` (the network's *learned conditioning*).

    Unlike the detector/pose stages, this is NOT freely swappable at inference: the trained GVHMR
    checkpoint's ``imgseq_embedder`` is fit to a specific feature space (default HMR2, 1024-d). A new
    backbone requires **retraining** the core (re-extract features with declared ``feat_dim``, set
    ``network.imgseq_dim``, retrain). See ``docs/EXTENSIBILITY.md`` (Tier B) and ``docs/TRAINING.md``.
    """

    feat_dim: int  # declared output width D; must match the trained network's imgseq_dim

    def extract_video_features(self, video_path, bbx_xys) -> torch.Tensor:
        """Return ``(F, feat_dim)`` per-frame features."""
        ...


# Registered backend names (kept here so callers/tests can enumerate without importing the heavy impls).
DETECTORS = ("yolo",)
POSE2D = ("vitpose", "rtmpose")
BACKBONES = ("hmr2", "dinov2")


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
    if name == "rtmpose":
        from gvhmr.utils.preproc.rtmpose import RTMPoseExtractor

        return RTMPoseExtractor(**kwargs)
    raise KeyError(f"unknown pose2d {name!r}; registered: {POSE2D}")


def make_backbone(name: str = "hmr2", **kwargs) -> FeatureBackbone:
    """Construct a registered image-feature backbone. Emits ``(F, feat_dim)``; swapping it needs a retrain."""
    if name == "hmr2":
        from gvhmr.utils.preproc.vitfeat_extractor import Extractor

        return Extractor(**kwargs)
    if name == "dinov2":
        from gvhmr.utils.preproc.dinov2_backbone import DINOv2Backbone

        return DINOv2Backbone(**kwargs)
    raise KeyError(f"unknown backbone {name!r}; registered: {BACKBONES}")
