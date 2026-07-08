"""``gvhmr sweep`` — the W&B sweep-config builder and trial-combo mapping (no wandb needed).

Pins the sweep contract: the config is a valid W&B sweep spec (method/metric/parameters), 'all'
expands to the real config-group presets (with the yolov8x/vitpose baseline among them), unsupported
dataset × variant combinations are rejected at create time, and a trial's wandb.config maps to exactly
the same (detector, pose2d, slug) semantics `gvhmr eval --detector/--pose2d` uses. The baseline is named
by its models (yolov8x/vitpose); the legacy `canonical` is still accepted as an alias.
"""

from __future__ import annotations

import pytest

from gvhmr.cli.sweepcmd import build_sweep_config, resolve_combo


def test_default_sweep_config_shape():
    cfg = build_sweep_config("3dpw")
    assert cfg["method"] == "grid"
    assert cfg["metric"] == {"name": "3DPW/pa_mpjpe", "goal": "minimize"}
    assert cfg["parameters"]["datasets"] == {"value": "3dpw"}
    assert cfg["parameters"]["detector"]["values"] == ["yolov8x"]  # baseline always present by default
    assert cfg["parameters"]["pose2d"]["values"] == ["vitpose", "rtmpose"]


def test_all_expands_to_real_presets():
    from gvhmr.cli.config import _group_options

    cfg = build_sweep_config("3dpw,emdb", detectors="all", pose2ds="all")
    det = cfg["parameters"]["detector"]["values"]
    assert set(det) == set(_group_options("detector"))  # every preset; yolov8x (baseline) among them
    assert "yolov8x" in det and "yolo26x" in det
    assert set(cfg["parameters"]["pose2d"]["values"]) == set(_group_options("pose2d"))
    assert cfg["metric"]["name"] == "3DPW/pa_mpjpe"  # first dataset anchors the sweep metric


def test_create_rejections():
    with pytest.raises(KeyError):
        build_sweep_config("3dpw", detectors="yolo9000x")  # unknown preset
    with pytest.raises(KeyError):
        build_sweep_config("rich", detectors="yolo26x")  # variants unsupported on RICH
    # …but a baseline-only sweep of RICH is fine (it's just the paper protocol)
    cfg = build_sweep_config("rich", pose2ds="vitpose")
    assert cfg["parameters"]["detector"]["values"] == ["yolov8x"]


def test_resolve_combo_matches_eval_semantics():
    assert resolve_combo({"detector": "yolov8x", "pose2d": "vitpose"}) == (None, None, None)  # baseline → pack
    det, pose, slug = resolve_combo({"detector": "yolo26x", "pose2d": "vitpose"})
    assert (det, pose, slug) == ("yolo26x", None, "yolo26x-vitpose")
    det, pose, slug = resolve_combo({"detector": "yolov8x", "pose2d": "rtmpose"})
    assert (det, pose, slug) == (None, "rtmpose", "yolov8x-rtmpose")
    assert resolve_combo({})[2] is None  # absent keys → baseline
    # the legacy 'canonical' token still resolves to the baseline (back-compat alias)
    assert resolve_combo({"detector": "canonical", "pose2d": "canonical"}) == (None, None, None)


def test_sweep_metric_names_cover_every_reference_metric():
    from gvhmr.cli.sweepcmd import sweep_metric_names

    assert sweep_metric_names(["3dpw"]) == ["3DPW/pa_mpjpe", "3DPW/mpjpe", "3DPW/pve", "3DPW/accel"]
    assert sweep_metric_names(["3dpw"], vs_paper=True)[0] == "3DPW/pa_mpjpe_vs_paper"
    emdb = sweep_metric_names(["emdb"])  # both splits, world metrics included
    assert "EMDB_1/pa_mpjpe" in emdb and "EMDB_2/wa2_mpjpe" in emdb and "EMDB_2/fs" in emdb


def test_print_dimensions_smokes():
    from gvhmr.cli.sweepcmd import print_dimensions

    print_dimensions(build_sweep_config("3dpw", detectors="yolov8x,yolo26x"))  # must not raise
