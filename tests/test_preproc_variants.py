"""Preproc-variant plumbing (`gvhmr eval --detector/--pose2d`) — the pure parts, no weights needed.

The generator itself (real detector/pose/backbone over videos) is exercised out-of-band; these tests pin
what must never drift: the slug naming, the multi-person identity guard's geometry, the raw-dataset
layout mapping, and — critically — that the `preproc_variant` config key actually reaches every test
dataset node while defaulting to None (the released baseline / paper protocol) everywhere else.
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
    assert variant_slug(None, "rtmpose") == "yolov8x-rtmpose"
    assert variant_slug(None, None) == "yolov8x-vitpose"
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
            assert OmegaConf.to_container(nodes[key], resolve=True)["preproc_variant"] is None  # baseline default
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


def test_normalize_stage_maps_baseline_to_none():
    # A baseline spelling means the FROZEN pack (not a regeneration) and must collapse to None so `gvhmr
    # eval --detector yolov8x` shares one slug / stage-cache key with the None form — matching the sweep.
    from gvhmr.utils.eval.preproc_variants import normalize_stage

    assert normalize_stage("detector", "yolov8x") is None
    assert normalize_stage("pose2d", "vitpose") is None
    assert normalize_stage("backbone", "hmr2") is None
    assert normalize_stage("detector", "canonical") is None  # legacy alias
    assert normalize_stage("detector", None) is None
    assert normalize_stage("detector", "yolo26x") == "yolo26x"  # a real swap is preserved
    assert normalize_stage("pose2d", "rtmpose") == "rtmpose"


class _RaisingStages:
    """A _LazyStages stand-in that fails if the (expensive) ViT/backbone pass is touched."""

    def get(self, stage):
        raise AssertionError(f"stages.get({stage!r}) must not run when frozen features are reused")

    def override_tag(self, *groups):
        return ""


def test_stage_feats_reuses_frozen_baseline_features(tmp_path):
    # A pose2d-only swap keeps baseline boxes AND baseline backbone, so the features must be REUSED from
    # the frozen pack, never recomputed from the re-encoded video (that confounds the swept-stage delta).
    from gvhmr.utils.eval.preproc_variants import _stage_feats

    sentinel = {
        "features": torch.zeros(4, 1024),
        "flip_features": torch.zeros(4, 1024),
        "flip_bbx_xys": torch.zeros(4, 3),
    }
    calls: list[str] = []

    def loader(vid):
        calls.append(vid)
        return sentinel

    out = _stage_feats(
        tmp_path,
        tmp_path / "v.mp4",
        "seqA",
        None,
        None,
        torch.zeros(4, 3),
        _RaisingStages(),
        False,
        baseline_feats=loader,
    )
    assert out is sentinel  # frozen features returned verbatim
    assert calls == ["seqA"]  # loader consulted once; _RaisingStages.get (the ViT pass) never ran


def test_stage_feats_recomputes_when_a_feature_stage_is_swapped(tmp_path):
    # When the detector (→ different boxes) OR the backbone is swapped, features genuinely differ, so the
    # frozen-reuse shortcut must NOT fire — the code falls through to recompute (and here promptly fails
    # on the missing video, proving it took the recompute path without consulting the baseline loader).
    import pytest

    from gvhmr.utils.eval.preproc_variants import _stage_feats

    calls: list[str] = []

    def loader(vid):
        calls.append(vid)
        return {"features": None, "flip_features": None, "flip_bbx_xys": None}

    for detector, backbone in (("yolo26x", None), (None, "dinov2")):
        with pytest.raises(Exception):  # noqa: B017 — recompute path hits the absent video/model
            _stage_feats(
                tmp_path, tmp_path / "missing.mp4", "seqA", detector, backbone,
                torch.zeros(4, 3), _RaisingStages(), False, baseline_feats=loader,
            )  # fmt: skip
    assert calls == []  # the reuse guard correctly skipped the frozen-feature shortcut


def test_variant_complete_3dpw_requires_kp2d(tmp_path):
    # Regression: a resume interrupted between the bbx and kp2d saves leaves bbx+feats present but a vid
    # missing from kp2d. variant_complete must report that as INCOMPLETE (the old check ignored kp2d and
    # returned True, so eval later KeyError'd on vid2kp2d[vid]).
    from gvhmr.utils.eval.preproc_variants import variant_complete

    torch.save({"vidA": {"vname": "s"}}, tmp_path / "test_3dpw_gt_labels.pt")
    out = tmp_path / "preproc_variants" / "yolov8x-rtmpose"
    (out / "imgfeats/3dpw_test").mkdir(parents=True)
    torch.save({"features": torch.zeros(2, 1024)}, out / "imgfeats/3dpw_test/vidA.pt")
    torch.save({"vidA": {"bbx_xys": torch.zeros(2, 3)}}, out / "preproc_test_bbx.pt")

    assert variant_complete("3dpw", tmp_path, "yolov8x-rtmpose") is False  # bbx+feats but no kp2d file
    torch.save({}, out / "preproc_test_kp2d.pt")
    assert variant_complete("3dpw", tmp_path, "yolov8x-rtmpose") is False  # kp2d file present but missing vidA
    torch.save({"vidA": torch.zeros(2, 17, 3)}, out / "preproc_test_kp2d.pt")
    assert variant_complete("3dpw", tmp_path, "yolov8x-rtmpose") is True  # all three present


def test_atomic_save_leaves_no_partial_file(tmp_path):
    # The combo-dir checkpoints go through _atomic_save (tmp + rename) so a crash mid-write can't leave a
    # truncated pickle that wedges every later resume/variant_complete load.
    from gvhmr.utils.eval.preproc_variants import _atomic_save

    p = tmp_path / "preproc_test_bbx.pt"
    _atomic_save({"vidA": torch.ones(3)}, p)
    assert p.exists() and not (tmp_path / "preproc_test_bbx.pt.tmp").exists()
    assert torch.equal(torch.load(p, weights_only=False)["vidA"], torch.ones(3))


def test_generate_3dpw_variant_reuses_features_and_resumes(tmp_path, monkeypatch):
    # End-to-end generator loop for a pose2d-only swap (models faked): the features must be reused from the
    # frozen pack (Bug 1), bbx+kp2d both land and variant_complete agrees (Bug 3), and a second call finds
    # everything cached — nothing regenerated (resume).
    from types import SimpleNamespace

    import gvhmr.utils.video_io_utils as vio
    from gvhmr.utils.eval import preproc_variants as pv

    n_frames = 5
    torch.save({"vidA": {"vname": "seqA", "img_wh": torch.tensor([1920, 1080])}}, tmp_path / "test_3dpw_gt_labels.pt")
    baseline_xys = torch.arange(n_frames * 3).reshape(n_frames, 3).float()
    torch.save({"vidA": {"bbx_xys": baseline_xys}}, tmp_path / "preproc_test_bbx.pt")
    (tmp_path / "imgfeats/3dpw_test").mkdir(parents=True)
    (tmp_path / "imgfeats/3dpw_test_flip").mkdir(parents=True)
    frozen_feat, frozen_flip, frozen_flip_xys = torch.randn(n_frames, 1024), torch.randn(n_frames, 1024), torch.randn(n_frames, 3)  # fmt: skip
    torch.save(
        {"features": frozen_feat, "bbx_xys": baseline_xys, "img_wh": torch.tensor([1920, 1080])},
        tmp_path / "imgfeats/3dpw_test/vidA.pt",
    )
    torch.save({"features": frozen_flip, "bbx_xys": frozen_flip_xys}, tmp_path / "imgfeats/3dpw_test_flip/vidA.pt")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/seqA.mp4").write_bytes(b"x")  # existence + frame count are all the generator needs
    monkeypatch.setattr(vio, "get_video_lwh", lambda p: (n_frames, 1920, 1080))

    fake_kp2d = torch.randn(n_frames, 17, 3)

    class _FakeStages:
        def get(self, stage):
            assert stage == "pose2d"  # boxes are baseline (no detector); features are reused (no backbone)
            return SimpleNamespace(extract=lambda vp, bbx: fake_kp2d)

        def override_tag(self, *groups):
            return ""

    monkeypatch.setattr(pv, "_LazyStages", lambda *a, **k: _FakeStages())

    report = pv.generate_3dpw_variant(tmp_path, "yolov8x-rtmpose", None, "rtmpose")
    assert report.generated == ["vidA"]
    out = tmp_path / "preproc_variants/yolov8x-rtmpose"
    assert "vidA" in torch.load(out / "preproc_test_kp2d.pt", weights_only=False)
    assert "vidA" in torch.load(out / "preproc_test_bbx.pt", weights_only=False)
    # features written into the variant cache are the FROZEN pack features, not a recompute
    assert torch.equal(torch.load(out / "imgfeats/3dpw_test/vidA.pt", weights_only=False)["features"], frozen_feat)
    assert pv.variant_complete("3dpw", tmp_path, "yolov8x-rtmpose")

    report2 = pv.generate_3dpw_variant(tmp_path, "yolov8x-rtmpose", None, "rtmpose")
    assert report2.cached == ["vidA"] and report2.generated == []
