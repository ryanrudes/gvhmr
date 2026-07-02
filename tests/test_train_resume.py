"""`gvhmr.cli.train._ckpt_dir_for_resume` — locate the checkpoint dir for `resume_mode=last`.

Regression net: the old code hard-referenced `cfg.callbacks.model_checkpoint.dirpath`, which crashes
OmegaConf's struct mode for the released `mixed` config (it checkpoints via `simple_ckpt_saver`), so
`resume_mode=last` was unusable on the one config real training uses. The helper must accept either
callback and fall back to the `simple_ckpt_saver` default under `output_dir`.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from gvhmr.cli.train import _ckpt_dir_for_resume


def test_resolves_simple_ckpt_saver_dir() -> None:
    """The `mixed` shape: a simple_ckpt_saver with output_dir and no model_checkpoint."""
    cfg = OmegaConf.create(
        {"output_dir": "/runs/mixed", "callbacks": {"simple_ckpt_saver": {"output_dir": "/runs/mixed/checkpoints/"}}}
    )
    assert _ckpt_dir_for_resume(cfg) == "/runs/mixed/checkpoints/"


def test_prefers_model_checkpoint_dirpath() -> None:
    """A model_checkpoint callback's dirpath wins (backward compatible)."""
    cfg = OmegaConf.create({"output_dir": "/runs/x", "callbacks": {"model_checkpoint": {"dirpath": "/custom/ckpts"}}})
    assert _ckpt_dir_for_resume(cfg) == "/custom/ckpts"


def test_falls_back_to_output_dir_when_no_ckpt_callback() -> None:
    """No checkpoint callback at all (e.g. a smoke config) → the conventional default; no crash."""
    cfg = OmegaConf.create({"output_dir": "/runs/y", "callbacks": {"prog_bar": {"_target_": "x"}}})
    assert _ckpt_dir_for_resume(cfg) == "/runs/y/checkpoints"


def test_no_callbacks_key_does_not_crash() -> None:
    """Struct-mode safety: a config without a callbacks key must not raise."""
    cfg = OmegaConf.create({"output_dir": "/runs/z"})
    assert _ckpt_dir_for_resume(cfg) == "/runs/z/checkpoints"
