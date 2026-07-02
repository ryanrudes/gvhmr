"""Human-friendly local configuration — a readable **TOML file** for asset locations, instead of env vars.

All the big-asset roots (checkpoints, data packs, body models, scene-camera weights) can be pointed at a
high-storage volume from one readable place. Manage it with ``gvhmr config`` (``show`` / ``init`` / ``set``).

Resolution precedence for every root — **env var > config file > built-in default** — so the config file
is the nice everyday mechanism, while a ``$GVHMR_*`` env var still wins for CI / one-off overrides.

The config file is the first that exists of: ``$GVHMR_CONFIG``, ``./gvhmr.toml``,
``~/.config/gvhmr/config.toml``. Read with the stdlib ``tomllib``; written from a commented template
(no extra deps).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))


#: Where ``gvhmr config init`` writes by default (unless a project-local ./gvhmr.toml is chosen).
DEFAULT_CONFIG_PATH = _xdg_config_home() / "gvhmr" / "config.toml"


def config_file() -> Path | None:
    """The active config file (first that exists of ``$GVHMR_CONFIG`` / ``./gvhmr.toml`` / the XDG path)."""
    for cand in (os.environ.get("GVHMR_CONFIG"), "gvhmr.toml", str(DEFAULT_CONFIG_PATH)):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file():
                return p
    return None


def _load_table(name: str) -> dict:
    """A top-level table (e.g. ``paths`` / ``models``) of the active config file (empty if none/unreadable)."""
    f = config_file()
    if f is None:
        return {}
    try:
        with open(f, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    t = data.get(name, {})
    return t if isinstance(t, dict) else {}


def resolve(key: str, env_var: str, default) -> tuple[Path, str]:
    """Resolve an asset root to ``(path, human-readable source)``. Precedence: env var > config file > default."""
    if (v := os.environ.get(env_var)) is not None and v != "":
        return Path(v).expanduser(), f"env ${env_var}"
    if (fv := _load_table("paths").get(key)) is not None and str(fv) != "":
        return Path(str(fv)).expanduser(), "config file"
    return Path(default), "default"


def resolve_path(key: str, env_var: str, default) -> Path:
    """Just the resolved :class:`Path` (see :func:`resolve`)."""
    return resolve(key, env_var, default)[0]


def models() -> dict:
    """The ``[models]`` table — default stage selections (detector/pose2d/backbone/camera) for the demo."""
    return _load_table("models")


def model_default(key: str) -> str | None:
    """The configured default for a swappable stage (``[models][key]``), or ``None`` if unset."""
    v = models().get(key)
    return str(v) if v not in (None, "") else None


def _render_section(name: str, entries: dict[str, str], comments: dict[str, str]) -> list[str]:
    out = [f"[{name}]"]
    width = max((len(k) for k in entries), default=0)
    for k, v in entries.items():
        line = f"{k.ljust(width)} = '{v}'"  # TOML literal string: path-safe, no escaping needed
        if c := comments.get(k):
            line += f"  # {c}"
        out.append(line)
    return out


def write_config(
    target: Path,
    paths: dict[str, str],
    models_: dict[str, str] | None = None,
    *,
    path_comments: dict[str, str] | None = None,
    model_comments: dict[str, str] | None = None,
) -> Path:
    """Write a readable ``[paths]`` (+ optional ``[models]``) config file, creating parent dirs. Returns it."""
    out = [
        "# GVHMR configuration — all your local settings in one readable place.",
        "# A friendly alternative to the $GVHMR_* env vars + CLI flags; edit freely, or use `gvhmr config`.",
        "# Precedence: env var / CLI flag > this file > built-in default.  Inspect with `gvhmr config show`.",
        "",
        *_render_section("paths", paths, path_comments or {}),
    ]
    if models_:
        out += ["", *_render_section("models", models_, model_comments or {})]
    target = Path(target).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out) + "\n")
    return target
