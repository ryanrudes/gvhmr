"""``gvhmr train`` — train / test the model (Hydra-driven).

Training is GPU-only and Hydra-managed (output dirs, logging, sweeps), so this keeps the
``@hydra.main`` entry point intact and the Typer command just forwards key=value
overrides into it (e.g. ``gvhmr train exp=gvhmr/mixed/mixed``).
"""

from __future__ import annotations

import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks.checkpoint import Checkpoint

from gvhmr.configs import register_store_gvhmr
from gvhmr.utils.net_utils import get_resume_ckpt_path, load_pretrained_model
from gvhmr.utils.pylogger import Log
from gvhmr.utils.vis.rich_logger import print_cfg


def get_callbacks(cfg: DictConfig) -> list | None:
    """Parse and instantiate all the callbacks in the config."""
    if not hasattr(cfg, "callbacks") or cfg.callbacks is None:
        return None
    enable_checkpointing = cfg.pl_trainer.get("enable_checkpointing", True)
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            cb = hydra.utils.instantiate(callback, _recursive_=False)
            if not enable_checkpointing and isinstance(cb, Checkpoint):
                continue
            callbacks.append(cb)
    return callbacks


def train(cfg: DictConfig) -> None:
    """Train/Test."""
    Log.info(f"Experiment: [gvhmr]{cfg.exp_name}[/]")
    if cfg.task == "fit":
        Log.info(f"[GPU x Batch] = {cfg.pl_trainer.devices} x {cfg.data.loader_opts.train.batch_size}")
    pl.seed_everything(cfg.seed)

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if cfg.ckpt_path is not None:
        load_pretrained_model(model, cfg.ckpt_path)

    callbacks = get_callbacks(cfg)
    has_ckpt_cb = any(isinstance(cb, Checkpoint) for cb in callbacks)
    if not has_ckpt_cb and cfg.pl_trainer.get("enable_checkpointing", True):
        Log.warning("No checkpoint-callback found. Disabling PL auto checkpointing.")
        cfg.pl_trainer = {**cfg.pl_trainer, "enable_checkpointing": False}
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)

    if cfg.task == "test":
        Log.info("Test mode forces full-precision.")
        cfg.pl_trainer = {**cfg.pl_trainer, "precision": 32}
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    if cfg.task == "fit":
        resume_path = None
        if cfg.resume_mode is not None:
            resume_path = get_resume_ckpt_path(cfg.resume_mode, ckpt_dir=cfg.callbacks.model_checkpoint.dirpath)
            Log.info(f"Resume training from {resume_path}")
        Log.info("Start fitting...")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=resume_path)
    elif cfg.task == "test":
        Log.info("Start testing...")
        trainer.test(model, datamodule.test_dataloader())
    else:
        raise ValueError(f"Unknown task: {cfg.task}")
    Log.info("[ok]End of script.[/]")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def _hydra_main(cfg: DictConfig) -> None:
    print_cfg(cfg, use_rich=True)
    train(cfg)


def run(overrides: list[str]) -> None:
    """Entry point for ``gvhmr train`` — forwards overrides into the Hydra main."""
    register_store_gvhmr()
    sys.argv = ["gvhmr train", *overrides]
    _hydra_main()
