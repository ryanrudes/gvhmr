"""Sapiens backbone scaffold (docs/ROADMAP.md Plan A1) — the retrain plumbing, CI-safe (no weights/GPU).

Pins what a backbone bake-off needs to compose: the Sapiens backend is registered and selectable as a
config group, it fails cleanly without weights, the smoke experiment composes at the new feature width, and
— the subtle one — the motion-only AMASS placeholder's ``f_imgseq`` width follows ``network.imgseq_dim`` so
batches still collate under a non-1024 backbone.
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from gvhmr.configs import register_store_gvhmr


def test_sapiens_registered_and_fails_without_weights():
    from gvhmr.utils.preproc.base import BACKBONES, make_backbone

    assert "sapiens" in BACKBONES
    # No checkpoint → a clear, actionable error before any torch.jit/GPU work (CI-safe).
    with pytest.raises(FileNotFoundError):
        make_backbone("sapiens")


def test_sapiens_backbone_config_group_composes():
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=x", "backbone=sapiens"])
    assert cfg.backbone.name == "sapiens"
    assert cfg.backbone.model_name == "sapiens_0.6b"
    assert cfg.backbone.checkpoint is None  # user must point this at a downloaded sapiens-lite encoder
    assert list(cfg.backbone.input_hw) == [1024, 1024]  # verified: the pretrain encoders are traced at 1024²


def _amass_node(train):
    """The AMASS dataset entry in a composed (resolved) train-dataset mix (keyed by group name)."""
    for d in train.values():
        if isinstance(d, dict) and "AmassDataset" in d.get("_target_", ""):
            return d
    raise AssertionError("AMASS dataset not found in the train mix")


def test_amass_placeholder_width_follows_network_imgseq_dim():
    # The motion-only AMASS f_imgseq is a masked-off zeros placeholder, but its WIDTH must equal the
    # network's imgseq_dim or batches won't stack with the feature datasets. It interpolates network.imgseq_dim.
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        base = compose(config_name="train", overrides=["exp=gvhmr/mixed/mixed", "ckpt_path=x"])
        train = OmegaConf.to_container(base.data.dataset_opts.train, resolve=True)
        assert _amass_node(train)["imgseq_dim"] == 1024  # released HMR2 default, unchanged

        swapped = compose(
            config_name="train", overrides=["exp=gvhmr/mixed/mixed", "ckpt_path=x", "network.imgseq_dim=1280"]
        )
        train2 = OmegaConf.to_container(swapped.data.dataset_opts.train, resolve=True)
        assert _amass_node(train2)["imgseq_dim"] == 1280  # follows the backbone width on a retrain


def test_smoke_sapiens_experiment_composes():
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="train", overrides=["exp=gvhmr/mixed/smoke_3dpw_sapiens", "ckpt_path=x"])
    assert cfg.network.imgseq_dim == 1280  # the swapped feature width
    train = OmegaConf.to_container(cfg.data.dataset_opts.train, resolve=True)
    assert any(
        isinstance(d, dict) and "imgfeats/3dpw_train_sapiens" in str(d.get("imgfeat_subdir", ""))
        for d in train.values()
    )
