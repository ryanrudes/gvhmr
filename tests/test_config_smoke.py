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


@pytest.mark.filterwarnings("ignore")
def test_demo_focal_length_override_propagates() -> None:
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=x", "f_mm=24"])
    assert cfg.f_mm == 24


# Silence the pervasive autocast FutureWarning during this module's imports.
warnings.filterwarnings("ignore", category=FutureWarning)
