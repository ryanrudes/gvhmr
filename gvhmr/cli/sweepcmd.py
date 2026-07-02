"""``gvhmr sweep`` — compare benchmark evals across preprocessing combinations with a real W&B sweep.

Each trial is one (detector, pose2d) combination: the agent regenerates that combo's preprocessing
variant if needed (cached under ``preproc_variants/<slug>/`` — see ``gvhmr eval --detector``), runs the
paper-protocol benchmark, and logs every metric as ``<DATASET>/<metric>`` (plus ``…_vs_paper`` deltas),
so the W&B sweep UI's table / parallel-coordinates views compare combinations directly. ``canonical``
is a first-class value — the unmodified pack preprocessing — giving every sweep its baseline point.

This is the standard W&B sweep workflow (https://docs.wandb.ai/guides/sweeps), not a homegrown loop:

    gvhmr sweep run 3dpw --detectors canonical,yolov8x,yolo26x        # create + run locally, one command
    gvhmr sweep create 3dpw --detectors all --pose2ds all             # → sweep id (grid over everything)
    gvhmr sweep agent <sweep_id> --raw-dir ~/ds/3DPW                  # run trials (any machine, many agents)

Requirements: ``wandb login`` once (sweeps are scheduled by the W&B service — offline mode can't run
them), the ``train`` extra (wandb), and — for any non-canonical combo — the raw videos (``--raw-dir``,
see docs/EVAL.md). Costs are dominated by first-time variant generation (~minutes/sequence, cached
forever after); a full grid over every detector preset is a *large* budget — trim with ``--detectors``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from gvhmr.utils.console import console
from gvhmr.utils.pylogger import Log

app = typer.Typer(help="W&B sweeps comparing benchmark evals across preprocessing combinations.")

DEFAULT_PROJECT = "gvhmr-eval"


def _group_values(group: str, spec: str | None, default: list[str]) -> list[str]:
    """Expand a --detectors/--pose2ds spec: None → default, 'all' → canonical + every config-group
    preset, else a validated CSV. 'canonical' (the pack's frozen preprocessing) is always allowed."""
    from gvhmr.cli.config import _group_options

    options = _group_options(group)
    if spec is None:
        values = default
    elif spec.strip().lower() == "all":
        values = ["canonical", *options]
    else:
        values = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [v for v in values if v != "canonical" and v not in options]
    if unknown:
        raise KeyError(f"unknown {group}(s) {unknown}; choose from ['canonical', *{options}]")
    return list(dict.fromkeys(values))


def build_sweep_config(
    datasets: str,
    detectors: str | None = None,
    pose2ds: str | None = None,
    method: str = "grid",
    metric: str | None = None,
    name: str | None = None,
) -> dict:
    """The W&B sweep config (pure dict — pass to ``wandb.sweep`` or dump as YAML).

    Defaults: every detector preset + canonical × {canonical, rtmpose}; grid search; the sweep metric
    is the first dataset's PA-MPJPE (used by bayes/random; informational for grid)."""
    from gvhmr.cli.evalcmd import DATASETS, VARIANT_GROUPS, parse_datasets

    names = parse_datasets(datasets)
    detector_values = _group_values("detector", detectors, ["canonical"])
    pose2d_values = _group_values("pose2d", pose2ds, ["canonical", "rtmpose"])
    non_canonical = [v for v in detector_values + pose2d_values if v != "canonical"]
    unsupported = [n for n in names if n not in VARIANT_GROUPS]
    if non_canonical and unsupported:
        raise KeyError(
            f"preproc variants aren't supported for {unsupported} (see docs/EVAL.md) — "
            f"sweep 3dpw/emdb, or restrict to canonical values"
        )
    first_id = DATASETS[names[0]][2][0]
    return {
        "name": name or f"gvhmr-eval-{'-'.join(names)}",
        "method": method,
        "metric": {"name": metric or f"{first_id}/pa_mpjpe", "goal": "minimize"},
        "parameters": {
            "datasets": {"value": ",".join(names)},
            "detector": {"values": detector_values},
            "pose2d": {"values": pose2d_values},
        },
    }


def resolve_combo(config: dict) -> tuple[str | None, str | None, str | None]:
    """A trial's wandb.config → (detector, pose2d, variant slug). canonical/absent → None; a fully
    canonical combo has slug None (the unmodified paper protocol)."""
    from gvhmr.utils.eval.preproc_variants import variant_slug

    detector = config.get("detector")
    pose2d = config.get("pose2d")
    detector = None if detector in (None, "canonical") else str(detector)
    pose2d = None if pose2d in (None, "canonical") else str(pose2d)
    slug = variant_slug(detector, pose2d) if (detector or pose2d) else None
    return detector, pose2d, slug


def _require_wandb():
    try:
        import wandb  # noqa: F401

        return wandb
    except ImportError:
        console.print(
            "[err]wandb is not installed[/] — it lives in the [gvhmr]train[/] extra:\n"
            "  gvhmr config set extras <current>,train && gvhmr env sync   [dim](or: uv sync --extra train)[/]\n"
            "then authenticate once with [gvhmr]wandb login[/] (sweeps are scheduled by the W&B service)."
        )
        raise typer.Exit(1) from None


def _trial(raw_dir: Path | None, ckpt: str | None) -> None:
    """One sweep trial: regenerate the combo's variant if needed, benchmark, log to W&B."""
    import wandb

    from gvhmr.cli.evalcmd import (
        PAPER_REFERENCE,
        ensure_inputs,
        ensure_variant,
        parse_datasets,
        print_summary,
        run_benchmarks,
    )
    from gvhmr.utils import assets

    with wandb.init() as run:
        detector, pose2d, slug = resolve_combo(dict(run.config))
        names = parse_datasets(run.config["datasets"])
        run.name = f"{slug or 'canonical'} · {','.join(names)}"
        Log.info(f"[gvhmr]sweep trial[/] detector={detector or 'canonical'} pose2d={pose2d or 'canonical'}")

        ensure_inputs(names)
        if slug is not None:
            ensure_variant(names, slug, detector, pose2d, None, raw_dir, regen=False)
        results = run_benchmarks(names, str(ckpt or assets.GVHMR_CKPT), slug=slug)
        if not results:
            raise RuntimeError("no metrics produced — see the run log")

        flat: dict[str, float] = {}
        for ds, metrics in results.items():
            ref = PAPER_REFERENCE.get(ds, {})
            for k, v in metrics.items():
                flat[f"{ds}/{k}"] = v
                if k in ref:
                    flat[f"{ds}/{k}_vs_paper"] = v - ref[k]
        run.log(flat)
        run.summary["preproc_variant"] = slug or "canonical"
        print_summary(results, variant=slug)


@app.command()
def create(
    datasets: str = typer.Argument("3dpw", help="Benchmarks each trial runs (3dpw/emdb/rich, CSV, or all)."),
    detectors: str | None = typer.Option(
        None,
        "--detectors",
        help="CSV of detector presets, or 'all'; 'canonical' = pack preproc. [dim](default: canonical)[/]",
    ),
    pose2ds: str | None = typer.Option(
        None, "--pose2ds", help="CSV of 2D-pose backends, or 'all'. [dim](default: canonical,rtmpose)[/]"
    ),
    method: str = typer.Option("grid", "--method", help="W&B sweep method (grid/random/bayes)."),
    metric: str | None = typer.Option(
        None, "--metric", help="Sweep metric [dim](default: <first dataset>/pa_mpjpe)[/]."
    ),
    project: str = typer.Option(DEFAULT_PROJECT, "--project", help="W&B project."),
    entity: str | None = typer.Option(None, "--entity", help="W&B entity (team/user)."),
    config_yaml: Path | None = typer.Option(
        None, "--config", help="Hand-written W&B sweep YAML (overrides the flags)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the sweep config without creating it."),
) -> None:
    """Create the sweep on W&B and print the agent command [dim](grid over the chosen combos)[/]."""
    import json

    if config_yaml is not None:
        import yaml

        cfg = yaml.safe_load(Path(config_yaml).read_text())
    else:
        cfg = build_sweep_config(datasets, detectors, pose2ds, method=method, metric=metric)
    n_combos = len(cfg["parameters"]["detector"]["values"]) * len(cfg["parameters"]["pose2d"]["values"])
    console.print(f"sweep config ({n_combos} grid combination(s)):\n[muted]{json.dumps(cfg, indent=2)}[/]")
    if dry_run:
        return
    wandb = _require_wandb()
    sweep_id = wandb.sweep(cfg, project=project, entity=entity)
    ent = f" --entity {entity}" if entity else ""
    console.print(
        f"\n[ok]sweep created[/]: [gvhmr]{sweep_id}[/]\nrun trials with:\n"
        f"  [gvhmr]gvhmr sweep agent {sweep_id} --project {project}{ent}[/]  "
        f"[dim](add --raw-dir for non-canonical combos; run on any/multiple GPU boxes)[/]"
    )


@app.command()
def agent(
    sweep_id: str = typer.Argument(..., help="The sweep id printed by [gvhmr]gvhmr sweep create[/]."),
    count: int | None = typer.Option(
        None, "--count", help="Max trials for this agent [dim](default: until the sweep is done)[/]."
    ),
    raw_dir: Path | None = typer.Option(
        None, "--raw-dir", help="Official raw-dataset download (composes missing videos once)."
    ),
    ckpt: str | None = typer.Option(
        None, "--ckpt", help="Checkpoint to evaluate [dim](default: the released ckpt)[/]."
    ),
    project: str = typer.Option(DEFAULT_PROJECT, "--project", help="W&B project."),
    entity: str | None = typer.Option(None, "--entity", help="W&B entity (team/user)."),
) -> None:
    """Run sweep trials on this machine [dim](launch on several GPU boxes to parallelize)[/]."""
    wandb = _require_wandb()
    wandb.agent(sweep_id, function=lambda: _trial(raw_dir, ckpt), count=count, project=project, entity=entity)


@app.command(name="run")
def run_(
    datasets: str = typer.Argument("3dpw", help="Benchmarks each trial runs (3dpw/emdb/rich, CSV, or all)."),
    detectors: str | None = typer.Option(None, "--detectors", help="CSV of detector presets, or 'all'."),
    pose2ds: str | None = typer.Option(None, "--pose2ds", help="CSV of 2D-pose backends, or 'all'."),
    method: str = typer.Option("grid", "--method", help="W&B sweep method (grid/random/bayes)."),
    metric: str | None = typer.Option(None, "--metric", help="Sweep metric."),
    raw_dir: Path | None = typer.Option(None, "--raw-dir", help="Official raw-dataset download."),
    ckpt: str | None = typer.Option(None, "--ckpt", help="Checkpoint to evaluate."),
    count: int | None = typer.Option(None, "--count", help="Max trials for this agent."),
    project: str = typer.Option(DEFAULT_PROJECT, "--project", help="W&B project."),
    entity: str | None = typer.Option(None, "--entity", help="W&B entity (team/user)."),
) -> None:
    """Create the sweep and run every trial locally — the one-command comparison."""
    wandb = _require_wandb()
    cfg = build_sweep_config(datasets, detectors, pose2ds, method=method, metric=metric)
    n = len(cfg["parameters"]["detector"]["values"]) * len(cfg["parameters"]["pose2d"]["values"])
    Log.info(f"creating sweep with {n} grid combination(s) in project [gvhmr]{project}[/]")
    sweep_id = wandb.sweep(cfg, project=project, entity=entity)
    wandb.agent(sweep_id, function=lambda: _trial(raw_dir, ckpt), count=count, project=project, entity=entity)
