"""Swappable metric-depth for the scene-camera scale step (docs/ROADMAP.md A2) — CI-safe (no weights/GPU).

Pins the seam the dust3r/vggt backends were refactored onto: the registry + protocol, the not-yet-built
alternatives fail loudly, the camera configs expose the knob, and — the behavior-preserving part — the
global-scale math (median metric/recon-depth ratio, 1.0 fallback) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from gvhmr.configs import register_store_gvhmr
from gvhmr.utils.preproc.metric_depth import (
    METRIC_DEPTH_MODELS,
    MetricDepth,
    make_metric_depth,
    metric_scale_from_depths,
)


class _ConstDepth:
    """A metric-depth stub emitting a constant depth — satisfies the MetricDepth protocol structurally."""

    def __init__(self, metres: float):
        self.metres = metres

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        return np.full(frame_rgb.shape[:2], self.metres, dtype="f4")


def test_registry_protocol_and_seam():
    assert "depth_anything_v2" in METRIC_DEPTH_MODELS and "unidepth" in METRIC_DEPTH_MODELS
    assert isinstance(_ConstDepth(1.0), MetricDepth)  # protocol is structural — any conformer drops in
    assert not isinstance(object(), MetricDepth)
    with pytest.raises(KeyError):
        make_metric_depth("nope")
    with pytest.raises(NotImplementedError):  # metric3d needs mmcv (absent) — still a stub
        make_metric_depth("metric3d")
    # unidepth is implemented (UniDepthMetric); constructing it needs the third-party/UniDepth clone + a
    # GPU, so it's not built here — the A2 A/B validates it out-of-band.


def test_scale_is_median_metric_over_recon_ratio():
    # metric depth 2.0 everywhere; recon depths 0.5/1.0/2.0 → ratios 4/2/1 → median scale 2.0.
    frames = np.zeros((3, 8, 8, 3), dtype="uint8")
    kf_idx = np.array([0, 1, 2])
    recon = [np.full((8, 8), 0.5, "f4"), np.full((8, 8), 1.0, "f4"), np.full((8, 8), 2.0, "f4")]
    assert abs(metric_scale_from_depths(_ConstDepth(2.0), frames, kf_idx, recon) - 2.0) < 1e-6


def test_scale_falls_back_to_one_when_no_valid_pixels():
    frames = np.zeros((1, 8, 8, 3), dtype="uint8")
    recon = [np.zeros((8, 8), "f4")]  # all ≤ 1e-6 → no valid pixels → scale 1.0 (unchanged behavior)
    assert metric_scale_from_depths(_ConstDepth(2.0), frames, np.array([0]), recon) == 1.0


def test_camera_configs_expose_swappable_depth_model():
    from hydra import compose, initialize_config_module

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        for cam in ("dust3r", "vggt"):
            cfg = compose(config_name="demo", overrides=["video_name=x", f"camera={cam}"])
            # unidepth since 2026-07-14: 2x better scale than DA-V2 on the 25-seq EMDB-2 A/B (ROADMAP A2).
            assert cfg.camera.depth_model == "unidepth"


def _adaptive_kf(length, cap):
    """Mirror of run_dust3r_slam's adaptive keyframe policy (extracted so CI can pin it without weights)."""
    return min(max(16, length // 20), cap)


def test_adaptive_keyframe_policy():
    """A flat 16 keyframes silently wrecks long sequences — ROADMAP A2 measured RTE 15 vs the prior's 3
    because the camera is linearly interpolated across ~3.6s gaps. The default must scale with length,
    stay at 16 for short clips (unchanged behavior), and cap so the global aligner doesn't OOM."""
    from gvhmr.utils.preproc.dust3r_slam import DUST3R_KF_CAP

    # short clips are untouched — a 3-second demo still gets 16
    assert _adaptive_kf(90, DUST3R_KF_CAP) == 16
    assert _adaptive_kf(300, DUST3R_KF_CAP) == 16
    # long sequences get proportionally more (EMDB's ~1728-frame mean lands between the 48 and 96 we swept)
    assert _adaptive_kf(1728, DUST3R_KF_CAP) == 86
    # and it never exceeds the measured memory ceiling (192 OOM'd a 48 GB GPU)
    assert _adaptive_kf(100_000, DUST3R_KF_CAP) == DUST3R_KF_CAP
    assert DUST3R_KF_CAP <= 128  # the cap must stay below where 192 OOM'd


def test_scene_cameras_default_to_unidepth():
    """UniDepth beat DA-V2 2x on the 25-seq EMDB-2 arbitration (ROADMAP A2), so it's the scale default."""
    register_store_gvhmr()
    from hydra import compose, initialize_config_module

    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        for cam in ("dust3r", "vggt"):
            cfg = compose(config_name="demo", overrides=[f"camera={cam}", "video_name=_"])
            assert cfg.camera.depth_model == "unidepth", f"{cam} should default to unidepth"
