"""Smoke tests for Hydra config registration and composition.

These assert the config *store* registers and the demo/train configs *compose*
to a resolvable ``DictConfig`` on a CPU-only box — no GPU, checkpoints, datasets,
or pytorch3d. They are the regression guard for the lazy-import work that keeps
the package importable on macOS / Apple-Silicon, and for the rename to come.
"""

from __future__ import annotations

import warnings

import pytest
from hydra import compose, initialize_config_module

from gvhmr.configs import MainStore, register_store_gvhmr


@pytest.fixture(scope="module", autouse=True)
def _registered() -> None:
    register_store_gvhmr()


def test_registration_populates_expected_groups() -> None:
    # MainStore.repo is nested: {group: {name.yaml: node}} (and groups can nest further,
    # e.g. model -> gvhmr -> gvhmr_pl.yaml). Assert the key entries the pipelines need.
    repo = MainStore.repo
    assert "gvhmr_pl.yaml" in repo["model"]["gvhmr"]
    assert "gvhmr_pl_demo.yaml" in repo["model"]["gvhmr"]
    assert "gvhmr" in repo["network"]
    assert any(name.startswith("adamw") for name in repo["optimizer"])
    assert "metric_3dpw.yaml" in repo["callbacks"]


def test_demo_config_composes() -> None:
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(
            config_name="demo",
            overrides=["video_name=unit_test", "static_cam=True", "verbose=False", "use_dpvo=False"],
        )
    # The live demo path uses NetworkEncoderRoPE (V1) — guard against the V2 drift.
    assert cfg.pipeline.args_denoiser3d._target_.endswith("NetworkEncoderRoPE")
    assert cfg.ckpt_path.endswith("gvhmr_siga24_release.ckpt")
    assert cfg.output_dir == "outputs/demo/unit_test"
    assert cfg.paths.bbx.endswith("bbx.pt")


def test_train_config_composes() -> None:
    # `train` requires an experiment selection (exp=<...>); compose the released mix.
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="train", overrides=["exp=gvhmr/mixed/mixed"])
    assert "model" in cfg
    assert "pl_trainer" in cfg
    assert cfg.exp_name
    # real training defaults to the Weights & Biases logger (configs/logger group)
    assert cfg.logger is not None and cfg.logger._target_.endswith("WandbLogger")
    assert cfg.logger.project == "gvhmr"


def test_logger_group_swaps_by_name() -> None:
    # `logger` is a config group (none / tensorboard / wandb): `mixed` defaults to wandb, the smoke
    # test to none (no backend needed), and any run can swap by name.
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        smoke = compose(config_name="train", overrides=["exp=gvhmr/mixed/smoke_3dpw"])
        assert smoke.logger is None  # inherits the `none` default
        tb = compose(config_name="train", overrides=["exp=gvhmr/mixed/smoke_3dpw", "logger=tensorboard"])
        assert tb.logger._target_.endswith("TensorBoardLogger") and tb.logger.version == "tb"
        off = compose(config_name="train", overrides=["exp=gvhmr/mixed/mixed", "logger=none"])
        assert off.logger is None  # wandb default can be turned off by name


def test_preproc_groups_compose_and_swap_by_name() -> None:
    # The pluggable stages are first-class Hydra config groups. Defaults are the released models
    # (so composition stays byte-identical); a name selector swaps the group; a recipe bundles
    # field values; and a raw override still wins over a recipe.
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=x"])
        assert cfg.detector.name == "yolo"
        assert cfg.pose2d.name == "vitpose"
        assert cfg.backbone.name == "hmr2"
        assert cfg.camera.name == "simplevo"
        # group defaults reproduce the old hard-coded construction (behaviour-preserving: a null weight
        # ⇒ the ctor default, and the knobs match what run_preprocess used to pass literally)
        assert cfg.detector.conf == 0.5 and cfg.detector.ckpt is None
        assert cfg.pose2d.model_name == "ViTPose_huge_coco_256x192" and cfg.pose2d.ckpt_path is None
        assert cfg.camera.scale == 0.5 and cfg.camera.step == 8 and cfg.camera.method == "sift"
        # name selection swaps the whole group (here: a newer YOLO weight)
        swapped = compose(config_name="demo", overrides=["video_name=x", "detector=yolo11x"])
        assert swapped.detector.ckpt.endswith("yolo11x.pt")
        # the camera backend (formerly the --slam selector) is a group too, carrying its knobs
        cam = compose(config_name="demo", overrides=["video_name=x", "camera=dust3r"])
        assert cam.camera.name == "dust3r" and cam.camera.max_depth == 80.0
        vggt = compose(config_name="demo", overrides=["video_name=x", "camera=vggt"])
        assert vggt.camera.name == "vggt" and vggt.camera.max_depth == 80.0
        # a recipe applies a bundle of choices; an explicit override still takes precedence
        rec = compose(config_name="demo", overrides=["video_name=x", "+recipe=hq", "render_scale=0.25"])
        assert rec.render_scale == 0.25


def test_recipes_bundle_fields_and_group_overrides() -> None:
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        # a field-setting recipe (benchmark quality)
        acc = compose(config_name="demo", overrides=["video_name=x", "+recipe=accurate"])
        assert acc.flip_test is True and acc.render_scale == 1.0
        # a recipe can also select a config *group*; an explicit --camera still wins over it
        scene = compose(config_name="demo", overrides=["video_name=x", "+recipe=scene"])
        assert scene.camera.name == "dust3r"
        scene_cli = compose(config_name="demo", overrides=["video_name=x", "+recipe=scene", "camera=simplevo"])
        assert scene_cli.camera.name == "simplevo"


@pytest.mark.filterwarnings("ignore")
def test_demo_focal_length_override_propagates() -> None:
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=x", "f_mm=24"])
    assert cfg.f_mm == 24


def test_list_options_renders_every_stage() -> None:
    """`gvhmr list` (print_options) enumerates each swappable stage from the config groups + recipes."""
    import typer

    from gvhmr.cli.config import LISTABLE, _group_options, print_options

    assert set(LISTABLE) == {"detector", "pose2d", "backbone", "camera", "recipe"}
    for stage in LISTABLE:  # config-group presets exist for every listable stage
        assert _group_options(stage), f"no presets discovered for {stage}"
    print_options()  # all stages — must not raise
    print_options("camera")  # a single stage
    with pytest.raises(typer.Exit):
        print_options("bogus")


# Silence the pervasive autocast FutureWarning during this module's imports.
warnings.filterwarnings("ignore", category=FutureWarning)
