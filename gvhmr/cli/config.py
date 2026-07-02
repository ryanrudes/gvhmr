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
STAGE_DESC = {
    "detector": "person detector/tracker (ultralytics YOLO)",
    "pose2d": "2D-pose estimator (must emit COCO-17)",
    "backbone": "image-feature backbone",
    "camera": "moving-camera backend",
}
# Per-option one-liners for the stages with a handful of choices (detector is grouped by family instead).
OPTION_DOCS = {
    "pose2d": {
        "vitpose": "ViTPose-Huge — released default (heatmap top-down)",
        "rtmpose": "RTMPose-m — SimCC via rtmlib/ONNX; needs `uv sync --extra rtmpose`",
    },
    "backbone": {
        "hmr2": "HMR2.0a ViT, 1024-d — the released, trained conditioning",
        "dinov2": "DINOv2 (ungated) — inference needs a retrain (Tier B)",
    },
    "camera": {
        "simplevo": "SIFT visual odometry — rotation only (default; any device)",
        "dpvo": "deep patch VO — full 6-DoF incl. translation (CUDA-only; setup_dpvo.sh)",
        "dust3r": "DUSt3R + Depth-Anything — scene-aware metric camera (MPS/CPU/CUDA)",
        "vggt": "VGGT + Depth-Anything — scene-aware metric, single forward pass",
    },
}


def _group_options(group: str) -> list[str]:
    """The available choices for a stage = the config-group yaml files (source of truth, stays in sync)."""
    from gvhmr import PROJ_ROOT

    return sorted(p.stem for p in (PROJ_ROOT / "gvhmr" / "configs" / group).glob("*.yaml"))


def _fam_num(family: str) -> int:
    return int("".join(c for c in family if c.isdigit()) or 0)


_SIZE_RANK = {c: i for i, c in enumerate("tnsmbclxe")}  # ultralytics size letters, small → large


def _detector_block() -> list[str]:
    """A grouped-by-family comment for the (many) YOLO detector presets — readable at any count."""
    fams: dict[str, list[str]] = {}
    for o in _group_options("detector"):
        if o == "yolo":
            continue  # the default alias (= yolov8x), noted separately
        fams.setdefault(o[:-1], []).append(o)
    lines = [f"detector — {STAGE_DESC['detector']}. Pick a preset (default `yolo` = yolov8x):"]
    width = max((len(f) for f in fams), default=0)
    for fam in sorted(fams, key=_fam_num):
        sizes = sorted(fams[fam], key=lambda o: _SIZE_RANK.get(o[-1], 99))  # n<s<m<l<x
        latest = "   (latest, NMS-free)" if fam == "yolo26" else ""
        lines.append(f"  {fam.ljust(width)}  {' '.join(sizes)}{latest}")
    lines.append("sizes n<s<m<l<x: accuracy up / speed down. Any other or newer weight: --detector-ckpt <name>.pt")
    return lines


def _model_block(key: str) -> list[str]:
    """The multiline comment block written above a `[models]` entry (option menu, per stage)."""
    if key == "detector":
        return _detector_block()
    lines = [f"{key} — {STAGE_DESC[key]}. Options:"]
    docs = OPTION_DOCS.get(key, {})
    opts = _group_options(key)
    width = max((len(o) for o in opts), default=0)
    lines += [f"  {o.ljust(width)}  {docs.get(o, '')}".rstrip() for o in opts]
    return lines


def _options_summary(key: str) -> str:
    """A compact one-liner for `config show` (families for the big detector list, else the names)."""
    opts = _group_options(key)
    if len(opts) <= 6:
        return ", ".join(opts)
    fams = sorted({o[:-1] for o in opts if o != "yolo"}, key=_fam_num)
    return f"{len(opts)} presets — families {', '.join(fams)} (sizes n/s/m/l/x)"


def _current_paths() -> dict[str, str]:
    from gvhmr.utils import assets

    return {k: str(assets.ROOTS[k][1]) for k in assets.ROOTS}


def _current_models() -> dict[str, str]:
    from gvhmr.utils import localconfig

    return {k: (localconfig.model_default(k) or DEMO_DEFAULTS[k]) for k in MODEL_KEYS}


def _write(paths: dict[str, str], models: dict[str, str], target) -> None:
    from gvhmr.utils import assets, localconfig

    path_entries = [(k, paths[k], [f"{k} — {assets.ROOTS[k][3]}"]) for k in paths]
    model_entries = [(k, models[k], _model_block(k)) for k in MODEL_KEYS]
    written = localconfig.write_config(target, [("paths", path_entries), ("models", model_entries)])
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
        models.add_row(key, using, _options_summary(key))
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
    _write(paths, models, localconfig.target_config_path())


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
            for line in _model_block(k):
                console.print(f"  [dim]{line}[/]")
            if k == "detector":  # too many YOLO presets to list as choices — free text (default yolo)
                models[k] = Prompt.ask(f"  {k}", default=DEMO_DEFAULTS[k])
            else:
                models[k] = Prompt.ask(f"  {k}", choices=_group_options(k), default=DEMO_DEFAULTS[k])

    # 3) Where to write (honors $GVHMR_CONFIG / an existing file as the default).
    target = Path(Prompt.ask("Write config to", default=str(localconfig.target_config_path()))).expanduser()
    _write(paths, models, target)
    show()

    # 4) Offer to fetch now.
    if Confirm.ask("\nFetch the model checkpoints now? [dim](gvhmr download)[/]", default=False):
        from gvhmr.cli.download import run as download_run

        download_run("demo")
