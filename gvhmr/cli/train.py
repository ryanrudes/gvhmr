"""``gvhmr train`` — train / test the model (Hydra-driven).

Real training runs on CUDA (multi-GPU, fp16-mixed), but the accelerator is **device-aware**
(``get_device()`` → cuda/mps/cpu, honouring ``$GVHMR_DEVICE``) so a small ``fit`` smoke test can run
on CPU/MPS. Off-CUDA it forces full precision + a single device (fp16 autocast is CUDA-only). Hydra
manages output dirs/logging/sweeps; the Typer command forwards ``key=value`` overrides into the
``@hydra.main`` entry point (e.g. ``gvhmr train exp=gvhmr/mixed/mixed``).
"""

from __future__ import annotations

import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks.checkpoint import Checkpoint

from gvhmr.configs import register_store_gvhmr
from gvhmr.utils.device import device_name, get_device
from gvhmr.utils.net_utils import get_resume_ckpt_path, load_pretrained_model
from gvhmr.utils.pylogger import Log
from gvhmr.utils.vis.rich_logger import print_cfg

# torch device type → PyTorch-Lightning accelerator string.
_PL_ACCELERATOR = {"cuda": "gpu", "mps": "mps", "cpu": "cpu"}

#: After a `task=test` run: {dataset_id: {metric: value}} from the metric callbacks (see `gvhmr eval`).
LAST_TEST_METRICS: dict[str, dict[str, float]] = {}


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


def _ckpt_dir_for_resume(cfg: DictConfig) -> str | None:
    """Directory the checkpoint callback writes to, for ``resume_mode=last``.

    The released ``mixed`` config checkpoints via ``simple_ckpt_saver`` (dir
    ``${output_dir}/checkpoints/``), not a ``model_checkpoint`` callback — so accept either and
    fall back to that default. Uses ``.get`` so a missing key doesn't crash OmegaConf's struct mode
    (the old code hard-referenced ``callbacks.model_checkpoint.dirpath`` and broke resume for ``mixed``).
    """
    cbs = cfg.get("callbacks") or {}
    for key in ("model_checkpoint", "simple_ckpt_saver"):
        cb = cbs.get(key) if hasattr(cbs, "get") else None
        if cb is not None:
            d = cb.get("dirpath") or cb.get("output_dir")
            if d is not None:
                return d
    return f"{cfg.output_dir}/checkpoints"


def train(cfg: DictConfig) -> None:
    """Train/Test."""
    Log.info(f"Experiment: [gvhmr]{cfg.exp_name}[/]")
    device = get_device()
    accelerator = _PL_ACCELERATOR[device.type]
    # fp16-mixed autocast + multi-device are CUDA-only; off-CUDA force 32-bit on a single device.
    if device.type != "cuda":
        cfg.pl_trainer = {**cfg.pl_trainer, "precision": 32, "devices": 1}
    if cfg.task == "fit":
        Log.info(f"[{device_name(device)} x Batch] = {cfg.pl_trainer.get('devices', 1)} x "
                 f"{cfg.data.loader_opts.train.batch_size}")  # fmt: skip
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
    # The logger group defaults to wandb for real training; degrade gracefully if its backend
    # isn't installed (base installs won't have wandb/tensorboard) rather than crashing a run.
    try:
        logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    except (ImportError, ModuleNotFoundError) as e:
        Log.warning(
            f"Logger backend unavailable ({type(e).__name__}: {e}); training without a logger. "
            "Install it with [gvhmr]uv sync --extra train[/] (wandb + tensorboard)."
        )
        logger = None

    if cfg.task == "test":
        Log.info("Test mode forces full-precision.")
        cfg.pl_trainer = {**cfg.pl_trainer, "precision": 32}
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    if cfg.task == "fit":
        resume_path = None
        if cfg.resume_mode is not None:
            resume_path = get_resume_ckpt_path(cfg.resume_mode, ckpt_dir=_ckpt_dir_for_resume(cfg))
            Log.info(f"Resume training from {resume_path}")
        Log.info("Start fitting...")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=resume_path)
    elif cfg.task == "test":
        Log.info("Start testing...")
        trainer.test(model, datamodule.test_dataloader())
        # Expose the metric callbacks' epoch averages to in-process callers (`gvhmr eval`'s table).
        global LAST_TEST_METRICS
        LAST_TEST_METRICS = dict(getattr(model, "metrics_summary", {}))
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
