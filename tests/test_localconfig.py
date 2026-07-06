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
    cfg.write_text("[paths]\ndata = '/vol/data'\n[models]\ndetector = 'yolo11x'\ncamera = 'vggt'\n")
    # GVHMR_CONFIG selects the file; clear any inherited data-root env so the file is what's tested.
    env = {"GVHMR_CONFIG": str(cfg), "GVHMR_DATA_ROOT": ""}
    out = _child(
        env,
        "from gvhmr.utils import assets, localconfig;"
        "print(assets.DATA_ROOT); print(localconfig.model_default('detector')); print(localconfig.model_default('camera'))",
    )
    assert "/vol/data" in out
    assert "yolo11x" in out
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


def test_gvhmr_config_env_is_authoritative(tmp_path, monkeypatch):
    # When $GVHMR_CONFIG is set it is the ONLY candidate: a missing file means "no config", with no
    # silent fallback to a repo-local / legacy file (this is also what keeps the test suite hermetic).
    repo_file = tmp_path / "gvhmr.toml"
    repo_file.write_text("[paths]\ndata = '/from/repo/file'\n")
    monkeypatch.setattr(localconfig, "DEFAULT_CONFIG_PATH", repo_file)
    monkeypatch.setenv("GVHMR_CONFIG", str(tmp_path / "missing.toml"))
    assert localconfig.config_file() is None


def test_lookup_falls_back_to_repo_then_legacy(tmp_path, monkeypatch):
    # Without $GVHMR_CONFIG: ./gvhmr.toml → <repo>/gvhmr.toml → legacy XDG path.
    monkeypatch.delenv("GVHMR_CONFIG", raising=False)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")  # so no ./gvhmr.toml candidate exists
    repo_file = tmp_path / "repo" / "gvhmr.toml"
    legacy_file = tmp_path / "xdg" / "config.toml"
    monkeypatch.setattr(localconfig, "DEFAULT_CONFIG_PATH", repo_file)
    monkeypatch.setattr(localconfig, "LEGACY_CONFIG_PATH", legacy_file)
    assert localconfig.config_file() is None  # nothing exists yet
    assert localconfig.target_config_path() == repo_file  # writes default into the repo
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text("[paths]\n")
    assert localconfig.config_file() == legacy_file  # legacy still honored…
    repo_file.parent.mkdir(parents=True)
    repo_file.write_text("[paths]\n")
    assert localconfig.config_file() == repo_file  # …but the repo-local file wins


def test_env_table_accessors(tmp_path, monkeypatch):
    cfg = tmp_path / "gvhmr.toml"
    cfg.write_text("[env]\ntorch = 'cu126'\nextras = 'preproc, dev'\ndpvo = 'true'\nscene = 'true'\n")
    monkeypatch.setenv("GVHMR_CONFIG", str(cfg))
    assert localconfig.env_torch() == "cu126"
    assert localconfig.env_extras() == ["preproc", "dev"]
    assert localconfig.env_dpvo() is True
    assert localconfig.env_scene() is True
    # 'none' means the default PyPI wheel — reported as no torch extra
    cfg.write_text("[env]\ntorch = 'none'\n")
    assert localconfig.env_torch() is None
    assert localconfig.env_extras() == []
    assert localconfig.env_dpvo() is False


def test_body_models_mirror_resolution(tmp_path, monkeypatch):
    cfg = tmp_path / "gvhmr.toml"
    cfg.write_text("[body_models]\nmirror = 'me/gvhmr-body-models'\n")
    monkeypatch.setenv("GVHMR_CONFIG", str(cfg))
    monkeypatch.delenv("GVHMR_BODY_MODELS_MIRROR", raising=False)
    assert localconfig.body_models_mirror() == "me/gvhmr-body-models"
    # env var wins over the config file
    monkeypatch.setenv("GVHMR_BODY_MODELS_MIRROR", "other/repo")
    assert localconfig.body_models_mirror() == "other/repo"


def test_body_models_mirror_none_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("GVHMR_CONFIG", str(tmp_path / "nope.toml"))
    monkeypatch.delenv("GVHMR_BODY_MODELS_MIRROR", raising=False)
    assert localconfig.body_models_mirror() is None


def test_config_set_preserves_body_models_mirror(tmp_path, monkeypatch):
    """Setting the mirror, then changing an unrelated key, must not drop the mirror (whole-file rewrite)."""
    from gvhmr.cli import config as cfgcmd

    target = tmp_path / "gvhmr.toml"
    monkeypatch.setenv("GVHMR_CONFIG", str(target))
    monkeypatch.delenv("GVHMR_BODY_MODELS_MIRROR", raising=False)

    cfgcmd.write_settings(body_models={"mirror": "you/bm"}, target=target)
    assert localconfig.body_models_mirror() == "you/bm"

    # A later, unrelated write (a model change) carries the mirror over.
    cfgcmd.write_settings(models={**cfgcmd._current_models(), "detector": "yolo11x"}, target=target)
    assert localconfig.body_models_mirror() == "you/bm"
    assert localconfig.model_default("detector") == "yolo11x"


def test_write_config_round_trips(tmp_path):
    # sections: [(section, [(key, value, comment_lines)])] — comments render ABOVE each key.
    target = localconfig.write_config(
        tmp_path / "c.toml",
        [
            ("paths", [("data", "/d", ["data — training/eval packs"])]),
            ("models", [("detector", "yolo11x", ["detector — options:", "  yolo11x  YOLO11-x"])]),
        ],
    )
    with open(target, "rb") as f:
        data = tomllib.load(f)
    assert data["paths"]["data"] == "/d"
    assert data["models"]["detector"] == "yolo11x"
    text = target.read_text()
    assert "# detector — options:" in text  # multiline comment block above the key
    assert "YOLO11-x" in text
