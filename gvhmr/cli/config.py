"""``gvhmr config`` — a friendly view + editor for the local config file (paths, model choices, env).

The config file (``<repo>/gvhmr.toml`` by default — machine-local, gitignored) is the one readable place
for local settings: where large assets live, which model version each swappable stage uses, and the
recorded Python-environment choices (``[env]``, replayed by ``gvhmr env sync``). Precedence is always
**env var / CLI flag > config file > built-in default**, so it's a convenience, never a trap.

- ``gvhmr config`` / ``gvhmr config show`` — table of every setting, its value, and where it came from.
- ``gvhmr config init``                    — interactive wizard that writes the file (with option menus).
- ``gvhmr config set <key> <value>``       — set one path / model choice / env field non-interactively.
- ``gvhmr config path``                    — print the active (or default) config file path.
"""

from __future__ import annotations

import typer

from gvhmr.utils.console import console, rule

app = typer.Typer(
    help="View & edit the local config file (asset paths + default model versions + env).", no_args_is_help=False
)

# The swappable stages the config file can pin a default for — the same Hydra groups the CLI flags select.
MODEL_KEYS = ("detector", "pose2d", "backbone", "camera")
DEMO_DEFAULTS = {"detector": "yolo", "pose2d": "vitpose", "backbone": "hmr2", "camera": "simplevo"}
# [env] fields (see gvhmr/cli/envcmd.py — recorded by the installer/wizard, replayed by `gvhmr env sync`).
ENV_KEYS = ("torch", "extras", "dpvo")
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
        "rtmpose": "RTMPose-m — SimCC via rtmlib/ONNX; needs the rtmpose extra",
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
ENV_DOCS = {
    "torch": "torch — the torch backend for this box: none (PyPI wheel: macOS/MPS), cpu, or cu124/cu126/cu128",
    "extras": "extras — comma-separated install extras `gvhmr env sync` applies (preproc = the demo's models)",
    "dpvo": "dpvo — 'true' when DPVO (CUDA SLAM) is installed out-of-band by scripts/setup_dpvo.sh",
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


def _current_env() -> dict[str, str]:
    from gvhmr.utils import localconfig

    return {str(k): str(v) for k, v in localconfig.env_table().items()}


def write_settings(
    paths: dict[str, str] | None = None,
    models: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    target=None,
) -> None:
    """Write the config file: the given sections, with any omitted one carried over from the current
    state — so updating one thing never loses another. Warns when the target is outside the lookup."""
    from gvhmr.utils import assets, localconfig

    paths = paths if paths is not None else _current_paths()
    models = models if models is not None else _current_models()
    env = env if env is not None else _current_env()
    target = target if target is not None else localconfig.target_config_path()

    path_entries = [(k, paths[k], [f"{k} — {assets.ROOTS[k][3]}"]) for k in paths]
    model_entries = [(k, models[k], _model_block(k)) for k in MODEL_KEYS]
    sections: list[localconfig.Section] = [("paths", path_entries), ("models", model_entries)]
    if env:
        sections.append(("env", [(k, env[k], [ENV_DOCS.get(k, k)]) for k in ENV_KEYS if k in env]))
    written = localconfig.write_config(target, sections)
    console.print(f"[ok]✓ wrote[/] [muted]{written}[/]")
    # Catch the silent-misconfig trap: a file outside the lookup chain would simply never be read.
    active = localconfig.config_file()
    if active is None or active.resolve() != written.resolve():
        console.print(
            f"[warn]note:[/] GVHMR won't read this location — the lookup is $GVHMR_CONFIG → ./gvhmr.toml → "
            f"[muted]{localconfig.DEFAULT_CONFIG_PATH}[/] → [muted]{localconfig.LEGACY_CONFIG_PATH}[/] (legacy)"
            + (f", and it currently finds [muted]{active}[/]" if active else "")
            + f". To use this file: [gvhmr]export GVHMR_CONFIG={written}[/]"
        )


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

    if localconfig.env_table():
        console.print(
            f"\nenv: torch=[gvhmr]{localconfig.env_torch() or 'none'}[/] "
            f"extras=[gvhmr]{','.join(localconfig.env_extras()) or '(none)'}[/] "
            f"dpvo=[gvhmr]{str(localconfig.env_dpvo()).lower()}[/] — re-apply anytime with [gvhmr]gvhmr env sync[/]"
        )


@app.command()
def path() -> None:
    """Print the active config file path (or the default location if none exists yet)."""
    from gvhmr.utils import localconfig

    console.print(str(localconfig.config_file() or localconfig.DEFAULT_CONFIG_PATH))


@app.command("set")
def set_(
    key: str = typer.Argument(
        ...,
        help="A path key (checkpoints/data/body_models/scene), model stage (detector/…), or env field (torch/extras/dpvo).",
    ),
    value: str = typer.Argument(..., help="The path, model version, or env value to set."),
) -> None:
    """Set one path, model choice, or env field, non-interactively (writes the config file)."""
    from gvhmr.cli.envcmd import TORCH_CHOICES
    from gvhmr.utils import localconfig

    paths, models = _current_paths(), _current_models()
    if key in paths:
        paths[key] = value
        write_settings(paths=paths, target=localconfig.target_config_path())
    elif key in models:
        opts = _group_options(key)
        if value not in opts:
            console.print(f"[err]'{value}' is not a valid {key}[/] — choose from: {', '.join(opts)}")
            raise typer.Exit(1)
        models[key] = value
        write_settings(models=models, target=localconfig.target_config_path())
    elif key in ENV_KEYS:
        if key == "torch" and value not in TORCH_CHOICES:
            console.print(f"[err]'{value}' is not a torch backend[/] — choose from: {', '.join(TORCH_CHOICES)}")
            raise typer.Exit(1)
        if key == "dpvo" and value not in ("true", "false"):
            console.print("[err]dpvo must be 'true' or 'false'[/]")
            raise typer.Exit(1)
        env = _current_env()
        env[key] = value
        write_settings(env=env, target=localconfig.target_config_path())
    else:
        console.print(
            f"[err]unknown key '{key}'[/] — paths: {', '.join(paths)} · models: {', '.join(models)} · "
            f"env: {', '.join(ENV_KEYS)}"
        )
        raise typer.Exit(1)


@app.command()
def init() -> None:
    """Interactive wizard: asset locations, default models, and the managed environment — then write."""
    from pathlib import Path

    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    from gvhmr.cli.envcmd import TORCH_CHOICES, detect_torch_extra
    from gvhmr.utils import localconfig

    console.print(
        Panel.fit(
            "Configure [gvhmr]GVHMR[/] — where assets live, which models to use, and the Python environment.\n"
            "[dim]Everything lands in one readable file you can re-edit anytime.[/]",
            border_style="gvhmr",
        )
    )

    # 1) Asset locations: one base folder → derive, with optional per-path override. Default matches
    # the scene-weights default (~/Datasets/GVHMR) and stays clear of a repo clone at ~/gvhmr.
    default_base = str(Path.home() / "Datasets" / "GVHMR")
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

    # 3) The managed environment: record the torch build + extras so `gvhmr env sync` can always restore
    # the env — no uv flags to remember, nothing silently pruned.
    env = _current_env()
    if Confirm.ask(
        "Let gvhmr manage the Python environment? [dim](records the torch build + extras; "
        "`gvhmr env sync` re-applies them)[/]",
        default=True,
    ):
        detected = detect_torch_extra()
        console.print(f"  detected torch backend for this box: [gvhmr]{detected or 'none (PyPI wheel — macOS/MPS)'}[/]")
        env["torch"] = Prompt.ask("  torch backend", choices=list(TORCH_CHOICES), default=detected or "none")
        env["extras"] = Prompt.ask("  extras [dim](comma-separated)[/]", default=env.get("extras") or "preproc")
        env.setdefault("dpvo", "false")

    # 4) Where to write (honors $GVHMR_CONFIG / an existing file; default is <repo>/gvhmr.toml).
    target = Path(Prompt.ask("Write config to", default=str(localconfig.target_config_path()))).expanduser()
    write_settings(paths=paths, models=models, env=env, target=target)
    show()

    # 5) Offer to apply everything now — environment first, then the checkpoints.
    if env and Confirm.ask("\nSync the environment now? [dim](gvhmr env sync)[/]", default=False):
        from gvhmr.cli.envcmd import sync as env_sync

        env_sync(dry_run=False)
    if Confirm.ask("Fetch the model checkpoints now? [dim](gvhmr download)[/]", default=False):
        from gvhmr.cli.download import run as download_run

        download_run("demo")
