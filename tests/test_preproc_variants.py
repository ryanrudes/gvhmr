"""Preproc-variant plumbing (`gvhmr eval --detector/--pose2d`) — the pure parts, no weights needed.

The generator itself (real detector/pose/backbone over videos) is exercised out-of-band; these tests pin
what must never drift: the slug naming, the multi-person identity guard's geometry, the raw-dataset
layout mapping, and — critically — that the `preproc_variant` config key actually reaches every test
dataset node while defaulting to None (the canonical paper protocol) everywhere else.
"""

from __future__ import annotations

import torch

from gvhmr.utils.eval.preproc_variants import (
    IDENTITY_IOU_THR,
    _raw_image_dir,
    same_identity,
    variant_slug,
    xys_iou,
)


def test_variant_slug_spells_out_defaults():
    assert variant_slug("yolo26x", None) == "yolo26x-vitpose"
    assert variant_slug(None, "rtmpose") == "yolo-rtmpose"
    assert variant_slug(None, None) == "yolo-vitpose"
    assert variant_slug("yolo26x", "rtmpose", "hmr2") == "yolo26x-rtmpose"  # default backbone omitted
    assert variant_slug("yolo26x", None, "dinov2") == "yolo26x-vitpose-dinov2"


def test_xys_iou_geometry():
    a = torch.tensor([[100.0, 100.0, 50.0]])
    assert xys_iou(a, a.clone()).item() == 1.0  # identical squares
    b = torch.tensor([[500.0, 500.0, 50.0]])
    assert xys_iou(a, b).item() == 0.0  # disjoint
    c = torch.tensor([[125.0, 100.0, 50.0]])  # half-overlap in x
    assert abs(xys_iou(a, c).item() - 1 / 3) < 1e-6  # inter=1250, union=3750


def test_identity_guard_separates_people():
    track = torch.tensor([[100.0, 100.0, 50.0]]).repeat(30, 1)
    jittered = track.clone()
    jittered[:, :2] += 5.0  # same person, slightly different boxes (a better detector)
    assert same_identity(jittered, track)
    other = track.clone()
    other[:, 0] += 200.0  # a different person across the frame
    assert not same_identity(other, track)
    assert not same_identity(track[:0], track)  # empty track is never the same person
    assert 0.0 < IDENTITY_IOU_THR < 1.0


def test_raw_autofetch_registry():
    # 3DPW's official host serves the files directly → auto-fetchable; EMDB is credential-gated → NOT.
    from gvhmr.utils.eval.preproc_variants import RAW_AUTOFETCH

    assert "3dpw" in RAW_AUTOFETCH and "emdb" not in RAW_AUTOFETCH
    url, size, license_url = RAW_AUTOFETCH["3dpw"]
    assert url.startswith("https://") and license_url.startswith("https://")
    assert size > 1e9  # exact-size verification guards truncated downloads


def test_raw_image_dir_layouts(tmp_path):
    assert _raw_image_dir("3dpw", tmp_path, "downtown_bar_00") == tmp_path / "imageFiles/downtown_bar_00"
    assert (
        _raw_image_dir("emdb", tmp_path, "P3_28_outdoor_walk_lunges") == tmp_path / "P3/28_outdoor_walk_lunges/images"
    )
    assert _raw_image_dir("emdb", tmp_path, "P0_09_outdoor_walk") == tmp_path / "P0/09_outdoor_walk/images"


def test_preproc_variant_key_reaches_every_test_dataset_node():
    # The hydra override grammar can't address `test_datasets.3dpw.*` (identifiers can't start with a
    # digit), so the loaders interpolate the root `preproc_variant` key — one override, all datasets.
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from gvhmr.configs import register_store_gvhmr

    register_store_gvhmr()
    base = ["exp=gvhmr/mixed/mixed", "ckpt_path=x"]
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="train", overrides=["global/task=gvhmr/test_3dpw_emdb_rich", *base])
        nodes = cfg.data.dataset_opts.test
        for key in ("3dpw", "emdb1", "emdb2"):
            assert OmegaConf.to_container(nodes[key], resolve=True)["preproc_variant"] is None  # canonical default
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(
            config_name="train", overrides=["global/task=gvhmr/test_emdb", *base, "preproc_variant=yolo26x-vitpose"]
        )
        for key in ("emdb1", "emdb2"):
            node = OmegaConf.to_container(cfg.data.dataset_opts.test[key], resolve=True)
            assert node["preproc_variant"] == "yolo26x-vitpose"


def test_eval_rejects_variant_mode_for_rich():
    import pytest

    from gvhmr.cli.evalcmd import DATASETS, PACK_DIRS, VARIANT_GROUPS, run

    assert set(VARIANT_GROUPS) == {"3dpw", "emdb"}  # RICH: no videos in the pack, raw is gated
    assert set(PACK_DIRS) == set(DATASETS)
    with pytest.raises(SystemExit):
        run("rich", detector="yolo26x")
    with pytest.raises(SystemExit):
        run("all", detector="yolo26x")  # 'all' includes rich — must ask for 3dpw,emdb explicitly
