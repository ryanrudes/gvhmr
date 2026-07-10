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
BACKBONES = ("hmr2", "dinov2", "sapiens")

#: What each selectable backend needs beyond the base install: ``stage:backend`` → (probe module, how to
#: install it). Lets the demo fail fast with the exact command instead of a mid-run ImportError traceback.
#: Every YOLO detector preset (yolov8x, yolo26n, …) shares the ultralytics implementation.
REQUIREMENTS: dict[str, tuple[str, str]] = {
    "detector:yolo": ("ultralytics", "uv sync --extra preproc"),
    "pose2d:rtmpose": ("rtmlib", "uv sync --extra rtmpose"),
    "camera:simplevo": ("pycolmap", "uv sync --extra preproc"),
    "camera:dpvo": ("dpvo", "scripts/setup_dpvo.sh  (CUDA box only)"),
}


def missing_requirements(selections: dict[str, str], have=None) -> list[tuple[str, str, str]]:
    """Unmet runtime requirements for the chosen backends: ``(stage 'name', module, install hint)`` rows.

    ``selections`` maps stage → selected backend name (e.g. ``{"detector": "yolo26x", "camera": "simplevo"}``);
    stages resolved to a backend with no extra requirement contribute nothing. ``have`` is an injectable
    module-presence predicate (for tests); the default probes ``importlib.util.find_spec``.
    """
    import importlib.util

    if have is None:

        def have(module: str) -> bool:
            return importlib.util.find_spec(module) is not None

    out = []
    for stage, name in selections.items():
        backend = "yolo" if stage == "detector" and name.startswith("yolo") else name
        req = REQUIREMENTS.get(f"{stage}:{backend}")
        if req is not None and not have(req[0]):
            out.append((f"{stage} '{name}'", req[0], req[1]))
    return out


# --- Resident-model cache -------------------------------------------------------------------------
# Loading the preproc models dominates short clips (ViTPose 2.5GB + HMR2 2.7GB ≈ 13s/clip): keep the
# last-built instance of each stage resident so folder/library/Space loops reload nothing. One slot per
# stage (a different name/kwargs evicts the previous instance). All three resident together is ~5-6GB of
# GPU weights — disable on low-memory boxes with GVHMR_NO_MODEL_CACHE=1 to restore build-use-free.
_MODEL_CACHE: dict[str, tuple] = {}


def _resident(stage: str, name: str, kwargs: dict, build):
    import os

    if os.environ.get("GVHMR_NO_MODEL_CACHE", "").strip().lower() in ("1", "true", "yes"):
        return build()
    try:
        key = (name, tuple(sorted(kwargs.items())))
    except TypeError:  # unhashable kwarg → just build uncached
        return build()
    entry = _MODEL_CACHE.get(stage)
    if entry is not None and entry[0] == key:
        return entry[1]
    model = build()
    _MODEL_CACHE[stage] = (key, model)
    return model


def clear_model_cache() -> None:
    """Drop all resident preproc models (frees their GPU/CPU memory on the next GC)."""
    _MODEL_CACHE.clear()


def make_detector(name: str = "yolo", **kwargs) -> Detector:
    """Construct (or reuse a resident) registered detector. ``kwargs`` (e.g. ``ckpt``, ``conf``) pass to its ctor."""
    if name == "yolo":

        def build():
            from gvhmr.utils.preproc.tracker import Tracker

            return Tracker(**kwargs)

        return _resident("detector", name, kwargs, build)
    raise KeyError(f"unknown detector {name!r}; registered: {DETECTORS}")


def make_pose2d(name: str = "vitpose", **kwargs) -> Pose2D:
    """Construct (or reuse a resident) registered 2D-pose estimator. Must emit COCO-17 ``(F,17,3)``."""
    if name == "vitpose":

        def build():
            from gvhmr.utils.preproc.vitpose import VitPoseExtractor

            return VitPoseExtractor(**kwargs)

        return _resident("pose2d", name, kwargs, build)
    if name == "rtmpose":

        def build():
            from gvhmr.utils.preproc.rtmpose import RTMPoseExtractor

            return RTMPoseExtractor(**kwargs)

        return _resident("pose2d", name, kwargs, build)
    raise KeyError(f"unknown pose2d {name!r}; registered: {POSE2D}")


def make_backbone(name: str = "hmr2", **kwargs) -> FeatureBackbone:
    """Construct (or reuse a resident) registered feature backbone. Emits ``(F, feat_dim)``; swapping needs a retrain."""
    if name == "hmr2":

        def build():
            from gvhmr.utils.preproc.vitfeat_extractor import Extractor

            return Extractor(**kwargs)

        return _resident("backbone", name, kwargs, build)
    if name == "dinov2":

        def build():
            from gvhmr.utils.preproc.dinov2_backbone import DINOv2Backbone

            return DINOv2Backbone(**kwargs)

        return _resident("backbone", name, kwargs, build)
    if name == "sapiens":

        def build():
            from gvhmr.utils.preproc.sapiens_backbone import SapiensBackbone

            return SapiensBackbone(**kwargs)

        return _resident("backbone", name, kwargs, build)
    raise KeyError(f"unknown backbone {name!r}; registered: {BACKBONES}")
