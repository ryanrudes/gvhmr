"""``gvhmr sweep`` — the W&B sweep-config builder and trial-combo mapping (no wandb needed).

Pins the sweep contract: the config is a valid W&B sweep spec (method/metric/parameters), 'all'
expands to the real config-group presets with the canonical baseline included, unsupported dataset ×
variant combinations are rejected at create time, and a trial's wandb.config maps to exactly the same
(detector, pose2d, slug) semantics `gvhmr eval --detector/--pose2d` uses.
"""

from __future__ import annotations

import pytest

from gvhmr.cli.sweepcmd import build_sweep_config, resolve_combo


def test_default_sweep_config_shape():
    cfg = build_sweep_config("3dpw")
    assert cfg["method"] == "grid"
    assert cfg["metric"] == {"name": "3DPW/pa_mpjpe", "goal": "minimize"}
    assert cfg["parameters"]["datasets"] == {"value": "3dpw"}
    assert cfg["parameters"]["detector"]["values"] == ["canonical"]  # baseline always present by default
    assert cfg["parameters"]["pose2d"]["values"] == ["canonical", "rtmpose"]


def test_all_expands_to_real_presets():
    from gvhmr.cli.config import _group_options

    cfg = build_sweep_config("3dpw,emdb", detectors="all", pose2ds="all")
    det = cfg["parameters"]["detector"]["values"]
    assert det[0] == "canonical" and set(det[1:]) == set(_group_options("detector"))
    assert "yolo26x" in det
    assert set(cfg["parameters"]["pose2d"]["values"]) == {"canonical", *_group_options("pose2d")}
    assert cfg["metric"]["name"] == "3DPW/pa_mpjpe"  # first dataset anchors the sweep metric


def test_create_rejections():
    with pytest.raises(KeyError):
        build_sweep_config("3dpw", detectors="yolo9000x")  # unknown preset
    with pytest.raises(KeyError):
        build_sweep_config("rich", detectors="yolo26x")  # variants unsupported on RICH
    # …but an all-canonical sweep of RICH is fine (it's just the paper protocol)
    cfg = build_sweep_config("rich", pose2ds="canonical")
    assert cfg["parameters"]["detector"]["values"] == ["canonical"]


def test_resolve_combo_matches_eval_semantics():
    assert resolve_combo({"detector": "canonical", "pose2d": "canonical"}) == (None, None, None)
    det, pose, slug = resolve_combo({"detector": "yolo26x", "pose2d": "canonical"})
    assert (det, pose, slug) == ("yolo26x", None, "yolo26x-vitpose")
    det, pose, slug = resolve_combo({"detector": "canonical", "pose2d": "rtmpose"})
    assert (det, pose, slug) == (None, "rtmpose", "yolo-rtmpose")
    assert resolve_combo({})[2] is None  # absent keys → canonical
