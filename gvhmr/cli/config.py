"""``gvhmr config`` — a friendly view + editor for the local config file (asset paths + model choices).

The config file (``~/.config/gvhmr/config.toml`` by default) is the one readable place for machine-local
settings: where large assets live, and which model version each swappable stage uses. Precedence is always
**env var / CLI flag > config file > built-in default**, so it's a convenience, never a trap.

- ``gvhmr config`` / ``gvhmr config show`` — table of every setting, its value, and where it came from.
- ``gvhmr config init``                    — interactive wizard that writes the file (with option menus).
- ``gvhmr config set <key> <value>``       — set one path or model choice non-interactively.
- ``gvhmr config path``                    — print the active (or default) config file path.
"""

from __future__ import annotations

import typer

from gvhmr.utils.console import console, rule

app = typer.Typer(
    help="View & edit the local config file (asset paths + default model versions).", no_args_is_help=False
)

# The swappable stages the config file can pin a default for — the same Hydra groups the CLI flags select.
MODEL_KEYS = ("detector", "pose2d", "backbone", "camera")
DEMO_DEFAULTS = {"detector": "yolo", "pose2d": "vitpose", "backbone": "hmr2", "camera": "simplevo"}
MODEL_NOTES = {
    "pose2d": "rtmpose needs `uv sync --extra rtmpose`",
    "backbone": "non-hmr2 requires a retrain (Tier B)",
    "camera": "dpvo=CUDA-only; dust3r/vggt=scene-aware metric (setup_scene_aware.sh)",
}


def _group_options(group: str) -> list[str]:
    """The available choices for a stage = the config-group yaml files (source of truth, stays in sync)."""
    from gvhmr import PROJ_ROOT

    return sorted(p.stem for p in (PROJ_ROOT / "gvhmr" / "configs" / group).glob("*.yaml"))


def _model_comment(key: str) -> str:
    opts = ", ".join(_group_options(key)) or "?"
    note = MODEL_NOTES.get(key)
    return f"options: {opts}" + (f" — {note}" if note else "")


def _current_paths() -> dict[str, str]:
    from gvhmr.utils import assets

    return {k: str(assets.ROOTS[k][1]) for k in assets.ROOTS}


def _current_models() -> dict[str, str]:
    from gvhmr.utils import localconfig

    return {k: (localconfig.model_default(k) or DEMO_DEFAULTS[k]) for k in MODEL_KEYS}


def _write(paths: dict[str, str], models: dict[str, str], target) -> None:
    from gvhmr.utils import assets, localconfig

    path_comments = {k: assets.ROOTS[k][3] for k in assets.ROOTS}
    model_comments = {k: _model_comment(k) for k in MODEL_KEYS}
    written = localconfig.write_config(
        target, paths, models, path_comments=path_comments, model_comments=model_comments
    )
    console.print(f"[ok]✓ wrote[/] [muted]{written}[/]")


@app.command()
def show() -> None:
    """Show every setting, its resolved value, and where it came from."""
    from rich.table import Table

    from gvhmr.utils import assets, localconfig

    active = localconfig.config_file()
    rule("[gvhmr]gvhmr config[/]")
    console.print(
        f"config file: [ok]{active}[/]" if active else "config file: [muted]none[/] — using defaults / env vars"
    )
    console.print(f"[dim](edit it, or `gvhmr config init`; default location {localconfig.DEFAULT_CONFIG_PATH})[/]\n")

    paths = Table(title="paths  ·  where large assets live", expand=False)
    paths.add_column("key")
    paths.add_column("path")
    paths.add_column("source", style="muted")
    paths.add_column("exists", justify="center")
    for key, (_env, path, source, _desc) in assets.ROOTS.items():
        exists = "[ok]✓[/]" if path.exists() else "[muted]—[/]"
        paths.add_row(key, str(path), source, exists)
    console.print(paths)

    models = Table(title="models  ·  default version per stage (CLI flags override)", expand=False)
    models.add_column("stage")
    models.add_column("using")
    models.add_column("options", style="muted")
    for key in MODEL_KEYS:
        chosen = localconfig.model_default(key)
        using = f"[gvhmr]{chosen}[/] [dim](config)[/]" if chosen else f"{DEMO_DEFAULTS[key]} [dim](default)[/]"
        models.add_row(key, using, ", ".join(_group_options(key)))
    console.print(models)


@app.command()
def path() -> None:
    """Print the active config file path (or the default location if none exists yet)."""
    from gvhmr.utils import localconfig

    console.print(str(localconfig.config_file() or localconfig.DEFAULT_CONFIG_PATH))


@app.command("set")
def set_(
    key: str = typer.Argument(..., help="A path key (checkpoints/data/body_models/scene) or model stage (detector/…)."),
    value: str = typer.Argument(..., help="The path, or the model version to select."),
) -> None:
    """Set one path or model choice, non-interactively (writes the config file)."""
    from gvhmr.utils import localconfig

    paths, models = _current_paths(), _current_models()
    if key in paths:
        paths[key] = value
    elif key in models:
        opts = _group_options(key)
        if value not in opts:
            console.print(f"[err]'{value}' is not a valid {key}[/] — choose from: {', '.join(opts)}")
            raise typer.Exit(1)
        models[key] = value
    else:
        console.print(f"[err]unknown key '{key}'[/] — paths: {', '.join(paths)} · models: {', '.join(models)}")
        raise typer.Exit(1)
    _write(paths, models, localconfig.config_file() or localconfig.DEFAULT_CONFIG_PATH)


@app.command()
def init() -> None:
    """Interactive wizard: choose asset locations + default model versions, then write the config file."""
    from pathlib import Path

    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    from gvhmr.utils import localconfig

    console.print(
        Panel.fit(
            "Configure [gvhmr]GVHMR[/] — where large assets live, and which model version each stage uses.\n"
            "[dim]Everything lands in one readable file you can re-edit anytime.[/]",
            border_style="gvhmr",
        )
    )

    # 1) Asset locations: one base folder → derive, with optional per-path override.
    default_base = str(Path.home() / "gvhmr")
    base = Prompt.ask("Base folder for all GVHMR assets [dim](e.g. a high-storage volume)[/]", default=default_base)
    paths = {k: str(Path(base).expanduser() / k) for k in ("checkpoints", "data", "body_models", "scene")}
    if Confirm.ask("Customize individual asset paths?", default=False):
        for k in paths:
            paths[k] = Prompt.ask(f"  {k}", default=paths[k])

    # 2) Model versions per stage (option menus). Defaults = the released models.
    models = dict(DEMO_DEFAULTS)
    if Confirm.ask("Choose default model versions now? [dim](else keep the released defaults)[/]", default=False):
        for k in MODEL_KEYS:
            opts = _group_options(k)
            note = MODEL_NOTES.get(k)
            if note:
                console.print(f"  [dim]{k}: {note}[/]")
            models[k] = Prompt.ask(f"  {k}", choices=opts, default=DEMO_DEFAULTS[k])

    # 3) Where to write.
    target = Path(Prompt.ask("Write config to", default=str(localconfig.DEFAULT_CONFIG_PATH))).expanduser()
    _write(paths, models, target)
    show()

    # 4) Offer to fetch now.
    if Confirm.ask("\nFetch the model checkpoints now? [dim](gvhmr download)[/]", default=False):
        from gvhmr.cli.download import run as download_run

        download_run("demo")
