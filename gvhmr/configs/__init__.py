import argparse
import os
from dataclasses import dataclass

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra_zen import builds

os.environ["HYDRA_FULL_ERROR"] = "1"

MainStore = ConfigStore.instance()


def register_store_gvhmr():
    """Register group options to MainStore (+ the asset-root resolver used by the yaml configs)."""
    from omegaconf import OmegaConf

    # ${gvhmr_checkpoints:} → the fully-resolved checkpoint root (env var > config file > default).
    # A plain ${oc.env:GVHMR_CHECKPOINTS,...} only honored the env var, so relocating checkpoints via
    # the config file broke every yaml-declared checkpoint path (e.g. the demo's ckpt_path).
    def _checkpoint_root() -> str:
        from gvhmr.utils.assets import CHECKPOINT_ROOT

        return str(CHECKPOINT_ROOT)

    OmegaConf.register_new_resolver("gvhmr_checkpoints", _checkpoint_root, replace=True)

    from . import store_gvhmr


def parse_args_to_cfg():
    """
    Use minimal Hydra API to parse args and return cfg.
    This function don't do _run_hydra which create log file hierarchy.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", "-cn", default="train")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    # Cfg
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    return cfg
