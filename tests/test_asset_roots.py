"""The configurable asset roots honour their env vars — so weights AND datasets relocate with one setting.

``CHECKPOINT_ROOT`` / ``BODY_MODEL_ROOT`` / ``DATA_ROOT`` are read from the environment at *import* time
(``gvhmr/utils/assets.py``), so each override is checked in a fresh subprocess (setting os.environ in-process
wouldn't re-trigger the import). This pins the "download it where I want, and have the code read it there"
contract that the checkpoint reorg + the dataset-loader routing rely on.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _resolve(var: str, attr: str, value: str) -> str:
    env = {**os.environ, var: value}
    out = subprocess.run(
        [sys.executable, "-c", f"from gvhmr.utils.assets import {attr}; print({attr})"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.mark.parametrize(
    ("var", "attr"),
    [
        ("GVHMR_CHECKPOINTS", "CHECKPOINT_ROOT"),
        ("GVHMR_BODY_MODELS", "BODY_MODEL_ROOT"),
        ("GVHMR_DATA_ROOT", "DATA_ROOT"),
    ],
)
def test_root_honours_env(tmp_path, var, attr):
    assert _resolve(var, attr, str(tmp_path)) == str(tmp_path)


def test_scene_weights_honour_gvhmr_data(tmp_path):
    # The DUSt3R / VGGT scene-camera weights resolve under $GVHMR_DATA — the same root
    # scripts/setup_scene_aware.sh downloads them to (dust3r_slam has light module-level imports only).
    env = {**os.environ, "GVHMR_DATA": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "-c", "from gvhmr.utils.preproc.dust3r_slam import _DA_CKPT; print(_DA_CKPT)"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0, out.stderr
    assert str(tmp_path) in out.stdout.strip()
