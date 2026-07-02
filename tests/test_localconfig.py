"""The local config file (asset paths + default model versions) — resolution, precedence, round-trip.

Roots/models are read at *import* time, so precedence cases run in a subprocess with a fresh env. Confirms
the config file drives both `[paths]` and `[models]`, that an env var still overrides the file, and that the
writer round-trips through `tomllib`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib

from gvhmr.utils import localconfig


def _child(env: dict, code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_config_file_drives_paths_and_models(tmp_path):
    cfg = tmp_path / "gvhmr.toml"
    cfg.write_text("[paths]\ndata = '/vol/data'\n[models]\ndetector = 'yolo11'\ncamera = 'vggt'\n")
    # GVHMR_CONFIG selects the file; clear any inherited data-root env so the file is what's tested.
    env = {"GVHMR_CONFIG": str(cfg), "GVHMR_DATA_ROOT": ""}
    out = _child(
        env,
        "from gvhmr.utils import assets, localconfig;"
        "print(assets.DATA_ROOT); print(localconfig.model_default('detector')); print(localconfig.model_default('camera'))",
    )
    assert "/vol/data" in out
    assert "yolo11" in out
    assert "vggt" in out


def test_env_overrides_config_file(tmp_path):
    cfg = tmp_path / "gvhmr.toml"
    cfg.write_text("[paths]\ndata = '/from/file'\n")
    out = _child(
        {"GVHMR_CONFIG": str(cfg), "GVHMR_DATA_ROOT": "/from/env"},
        "from gvhmr.utils import assets; print(assets.DATA_ROOT)",
    )
    assert out == "/from/env"


def test_no_config_no_env_uses_default(tmp_path):
    # An absent config (point $GVHMR_CONFIG at a non-file) + no env ⇒ the built-in default under the repo.
    out = _child(
        {"GVHMR_CONFIG": str(tmp_path / "nope.toml"), "GVHMR_DATA_ROOT": ""},
        "from gvhmr.utils import assets; print(assets.ROOTS['data'][2])",  # the 'source' field
    )
    assert out == "default"


def test_write_config_round_trips(tmp_path):
    target = localconfig.write_config(
        tmp_path / "c.toml",
        {"data": "/d", "checkpoints": "/c"},
        {"detector": "yolo11"},
        model_comments={"detector": "options: yolo, yolo11"},
    )
    with open(target, "rb") as f:
        data = tomllib.load(f)
    assert data["paths"]["data"] == "/d"
    assert data["models"]["detector"] == "yolo11"
    assert "options: yolo, yolo11" in target.read_text()  # the option-menu comment is written
