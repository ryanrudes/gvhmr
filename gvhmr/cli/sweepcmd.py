"""``gvhmr sweep`` — compare benchmark evals across preprocessing combinations with a real W&B sweep.

**What is swept, exactly:** the grid is ``detector × pose2d`` (the two preprocessing stages that swap
freely at inference), with ``canonical`` — the pack's frozen paper preprocessing — as a first-class
value so every sweep contains its baseline. The other two stages are deliberately **not** swept:

- **backbone** — the features are *learned conditioning*; swapping them under the released checkpoint
  produces meaningless numbers (a swap requires a retrain — docs/EXTENSIBILITY.md Tier B). Sweep it
  only with retrained checkpoints via ``--ckpt`` per sweep.
- **camera** — the benchmark protocol feeds the model the dataset's **ground-truth** camera rotation
  (no visual odometry runs at all), so simplevo/dpvo/dust3r/vggt cannot change these numbers by
  construction. To A/B camera backends, use the world-eval harness (``tools/eval/eval_world.py``).

Each trial regenerates its combo's preprocessing if needed (cached under ``preproc_variants/<slug>/``
— see ``gvhmr eval --detector``), runs the paper-protocol benchmark, and logs every metric as
``<DATASET>/<metric>`` plus ``<DATASET>/<metric>_vs_paper`` deltas. A **report** with parallel-
coordinates across all metrics, per-metric bar charts, and accuracy scatter plots is generated
automatically (wandb-workspaces); the sweep page's own table/parallel-coordinates work too.

This is the standard W&B sweep workflow (https://docs.wandb.ai/guides/sweeps), not a homegrown loop:

    gvhmr sweep run 3dpw --detectors canonical,yolov8x,yolo26x --raw-dir ~/ds/3DPW
    gvhmr sweep create 3dpw --detectors all                            # → sweep id (grid over presets)
    gvhmr sweep agent <sweep_id> --raw-dir ~/ds/3DPW                   # any machine, many agents
    gvhmr sweep report <sweep_id>                                      # (re)build the comparison report

Requirements: ``wandb login`` once (sweeps are scheduled by the W&B service — offline mode can't run
them), the ``train`` extra (wandb + wandb-workspaces), and — for any non-canonical combo — the raw
videos (3DPW auto-downloads; EMDB needs ``--raw-dir``; checked ONCE up front, not per trial).
Generation caches per STAGE, not per combo (boxes and the heavy feature pass are per-detector; only
keypoints are per-combo — see docs/EVAL.md "Grid economics"), so a full ``--detectors all`` grid is
O(detectors) heavy passes, one-time; every later sweep is ~1 min/trial.
"""

from __future__ import annotations

import os
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
    diagnostics: bool = False,
) -> dict:
    """The W&B sweep config (pure dict — pass to ``wandb.sweep`` or dump as YAML).

    Defaults: {canonical} detectors × {canonical, rtmpose} pose2d; grid search; the sweep metric is
    the first dataset's PA-MPJPE (drives bayes/random; informational for grid)."""
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
            "diagnostics": {"value": bool(diagnostics)},  # trials read this to capture full-distribution stats
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


def sweep_metric_names(names: list[str], vs_paper: bool = False) -> list[str]:
    """Every metric the swept datasets produce, in reference order (e.g. ``3DPW/pa_mpjpe``)."""
    from gvhmr.cli.evalcmd import DATASETS, PAPER_REFERENCE

    suffix = "_vs_paper" if vs_paper else ""
    return [f"{ds_id}/{k}{suffix}" for n in names for ds_id in DATASETS[n][2] for k in PAPER_REFERENCE.get(ds_id, {})]


def print_dimensions(cfg: dict) -> None:
    """Say exactly what the sweep tests — the swept grid AND what's fixed (and why)."""
    from rich.table import Table

    p = cfg["parameters"]
    det, pose = p["detector"]["values"], p["pose2d"]["values"]
    table = Table(title=f"sweep dimensions  ·  {len(det) * len(pose)} grid combination(s)", expand=False)
    table.add_column("dimension")
    table.add_column("values / setting")
    table.add_column("", style="muted")
    table.add_row("detector", ", ".join(det), "swept (person boxes; canonical = the pack's YOLOv8x)")
    table.add_row("pose2d", ", ".join(pose), "swept (2D keypoints; canonical = the pack's ViTPose)")
    table.add_row("datasets", p["datasets"]["value"], "fixed — every trial runs the same benchmarks")
    table.add_row("backbone", "hmr2 (fixed)", "learned conditioning — a swap needs a retrained --ckpt")
    table.add_row("camera", "GT rotation (fixed)", "benchmark protocol; no VO runs — see docs/EVAL.md")
    table.add_row("protocol", "flip-test + postproc", "the paper protocol, identical for every trial")
    console.print(table)


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


def _preflight_variants(sweep_cfg: dict, raw_dir: Path | None) -> None:
    """Fail ONCE, up front, when non-canonical combos can't be generated (missing videos / stage
    deps) — instead of crashing trial after trial. Composes the videos here when --raw-dir is given.
    Fully-cached combos need no videos, so re-sweeping a generated grid works on any box."""
    from itertools import product

    from gvhmr.cli.evalcmd import PACK_DIRS, ensure_stage_deps, ensure_variant_inputs, parse_datasets
    from gvhmr.utils import assets
    from gvhmr.utils.eval import preproc_variants as pv

    p = sweep_cfg["parameters"]
    det_values, pose_values = list(p["detector"]["values"]), list(p["pose2d"]["values"])
    names = parse_datasets(p["datasets"]["value"])
    ensure_stage_deps(
        [d for d in det_values if d != "canonical"],
        [v for v in pose_values if v != "canonical"] or ["vitpose"],
        ["hmr2"],
    )
    needs_videos = []
    for n in names:
        support = assets.DATA_ROOT / PACK_DIRS[n] / "hmr4d_support"
        for d, q in product(det_values, pose_values):
            _, _, slug = resolve_combo({"detector": d, "pose2d": q})
            if slug is not None and not pv.variant_complete(n, support, slug):
                needs_videos.append(n)
                break
    if needs_videos:
        ensure_variant_inputs(needs_videos, raw_dir)


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

        # Opt-in rich diagnostics: `gvhmr sweep run --diagnostics` (or $GVHMR_EVAL_DIAGNOSTICS / a sweep-config
        # `diagnostics: true`). Captures the full distribution and logs it alongside the unchanged means.
        from gvhmr.cli.evalcmd import _provenance
        from gvhmr.model.gvhmr.callbacks._diagnostics import diagnostics_enabled

        diagnostics = diagnostics_enabled(bool(run.config.get("diagnostics", False)))
        detailed: dict = {}
        raw: dict = {}
        ckpt_str = str(ckpt or assets.GVHMR_CKPT)
        if diagnostics:
            os.environ["GVHMR_EVAL_DIAGNOSTICS"] = "1"
            results, detailed, raw = run_benchmarks(names, ckpt_str, slug=slug, collect_detailed=True)
        else:
            results = run_benchmarks(names, ckpt_str, slug=slug)
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
        if diagnostics:
            _log_diagnostics_to_wandb(run, wandb, detailed, raw, _provenance(ckpt_str, slug, names, detailed))
        print_summary(results, variant=slug)


def _log_diagnostics_to_wandb(run, wandb, detailed: dict, raw: dict, provenance: dict) -> None:
    """Log the opt-in diagnostics to a W&B trial: extended distribution scalars, per-metric histograms,
    a per-sequence table, per-joint bars, a raw-arrays artifact, and the provenance record."""
    import tempfile

    import numpy as np

    try:
        run.summary["provenance"] = provenance
        scalars: dict[str, float] = {}
        for ds, detail in detailed.items():
            for metric, dist in detail.get("distribution", {}).items():
                for stat in ("std", "median", "p05", "p25", "p75", "p95", "p99", "min", "max"):
                    if stat in dist:
                        scalars[f"{ds}/{metric}_{stat}"] = dist[stat]
        if scalars:
            run.log(scalars)
        # per-metric histograms + per-sequence tables from the raw arrays
        for ds, metrics in raw.items():
            for metric, vid2arr in metrics.items():
                if metric.startswith("perjoint_"):
                    continue
                pooled = np.concatenate([np.asarray(a).ravel() for a in vid2arr.values()]) if vid2arr else np.array([])
                if pooled.size:
                    run.log({f"hist_{ds}/{metric}": wandb.Histogram(pooled)})
            table = wandb.Table(columns=["sequence", "metric", "mean", "std", "median", "count"])
            for metric, per_seq in detailed.get(ds, {}).get("per_sequence", {}).items():
                for vid, s in per_seq.items():
                    table.add_data(vid, metric, s.get("mean"), s.get("std"), s.get("median"), s.get("count"))
            run.log({f"per_sequence/{ds}": table})
            # per-joint bars
            for metric, pj in detailed.get(ds, {}).get("per_joint", {}).items():
                if pj.get("labels"):
                    bar = wandb.Table(data=list(zip(pj["labels"], pj["mean"], strict=False)), columns=["joint", "mean"])
                    run.log(
                        {
                            f"per_joint/{ds}_{metric}": wandb.plot.bar(
                                bar, "joint", "mean", title=f"{ds} {metric} per joint"
                            )
                        }
                    )
        # raw arrays as an artifact
        with tempfile.TemporaryDirectory() as td:
            art = wandb.Artifact(f"eval-raw-{run.id}", type="eval-raw")
            for ds, metrics in raw.items():
                arrays = {f"{m}__{vid}": np.asarray(a) for m, vid2arr in metrics.items() for vid, a in vid2arr.items()}
                if arrays:
                    p = Path(td) / f"{ds}_raw.npz"
                    np.savez_compressed(p, **arrays)
                    art.add_file(str(p))
            run.log_artifact(art)
    except Exception as exc:  # noqa: BLE001 — diagnostics logging must never fail a sweep trial
        Log.warning(f"[warn]diagnostics logging skipped[/]: {type(exc).__name__}: {exc}")


def build_report(entity: str, project: str, sweep_id: str, names: list[str]) -> str:
    """Create/refresh the W&B comparison report for a sweep: parallel coordinates across every metric
    (absolute + vs-paper), a bar chart per metric, and accuracy scatter plots. Returns the URL."""
    import wandb_workspaces.reports.v2 as wr

    from gvhmr.cli.evalcmd import DATASETS

    metrics = sweep_metric_names(names)
    deltas = sweep_metric_names(names, vs_paper=True)
    runset = wr.Runset(entity=entity, project=project, name="sweep runs", filters=f"Sweep == '{sweep_id}'")

    def pcols(names_: list[str]):
        cols = [wr.ParallelCoordinatesPlotColumn(metric=wr.Config("detector"), display_name="detector")]
        cols.append(wr.ParallelCoordinatesPlotColumn(metric=wr.Config("pose2d"), display_name="pose2d"))
        cols += [wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric(m)) for m in names_]
        return cols

    full = wr.Layout(x=0, y=0, w=24, h=10)
    half = wr.Layout(w=12, h=8)
    panels = [
        wr.ParallelCoordinatesPlot(columns=pcols(metrics), title="all metrics (lower is better)", layout=full),
        wr.ParallelCoordinatesPlot(columns=pcols(deltas), title="Δ vs paper (0 = published number)", layout=full),
    ]
    panels += [wr.BarPlot(title=m, metrics=[m], layout=half) for m in metrics]
    for n in names:
        for ds_id in DATASETS[n][2]:
            if f"{ds_id}/pa_mpjpe" in metrics and f"{ds_id}/mpjpe" in metrics:
                panels.append(
                    wr.ScatterPlot(
                        title=f"{ds_id}: PA-MPJPE vs MPJPE",
                        x=wr.SummaryMetric(f"{ds_id}/pa_mpjpe"),
                        y=wr.SummaryMetric(f"{ds_id}/mpjpe"),
                        layout=half,
                    )
                )

    report = wr.Report(
        entity=entity,
        project=project,
        title=f"GVHMR preprocessing sweep · {sweep_id}",
        description="Benchmark evals across preprocessing combinations (gvhmr sweep).",
        blocks=[
            wr.H1(text=f"GVHMR preprocessing sweep · {', '.join(names)}"),
            wr.MarkdownBlock(
                text=(
                    "**Swept:** detector × pose2d ('canonical' = the pack's frozen paper preprocessing — "
                    "the baseline every combo is compared against). **Fixed:** backbone (hmr2 — a swap needs "
                    "a retrain), camera (the protocol feeds GT rotation; no VO runs), flip-test + postproc.\n\n"
                    "Every metric is logged as `DATASET/metric` with a `_vs_paper` delta against the "
                    "published numbers (arXiv 2409.06662). All metrics: lower is better; a canonical run's "
                    "deltas sit at ≈0 by construction."
                )
            ),
            wr.PanelGrid(runsets=[runset], panels=panels),
        ],
    ).save()
    return getattr(report, "url", f"https://wandb.ai/{entity}/{project}/reports")


def _entity_or_default(entity: str | None) -> str:
    import wandb

    return entity or wandb.Api().default_entity


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
    diagnostics: bool = typer.Option(
        False,
        "--diagnostics",
        help="Log the full distribution per trial (std/percentiles/per-seq/per-joint "
        "histograms + raw-arrays artifact + provenance), not just the means.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the sweep config without creating it."),
) -> None:
    """Create the sweep on W&B (+ its comparison report) and print the agent command."""
    if config_yaml is not None:
        import yaml

        cfg = yaml.safe_load(Path(config_yaml).read_text())
    else:
        cfg = build_sweep_config(datasets, detectors, pose2ds, method=method, metric=metric, diagnostics=diagnostics)
    print_dimensions(cfg)
    if dry_run:
        import json

        console.print(f"[muted]{json.dumps(cfg, indent=2)}[/]")
        return
    wandb = _require_wandb()
    sweep_id = wandb.sweep(cfg, project=project, entity=entity)
    from gvhmr.cli.evalcmd import parse_datasets

    url = build_report(
        _entity_or_default(entity), project, sweep_id, parse_datasets(cfg["parameters"]["datasets"]["value"])
    )
    ent = f" --entity {entity}" if entity else ""
    console.print(
        f"\n[ok]sweep created[/]: [gvhmr]{sweep_id}[/]   [ok]report[/]: [muted]{url}[/]\nrun trials with:\n"
        f"  [gvhmr]gvhmr sweep agent {sweep_id} --project {project}{ent}[/]  "
        f"[dim](add --raw-dir for non-canonical combos; run on any/multiple GPU boxes)[/]"
    )


@app.command()
def agent(
    sweep_id: str = typer.Argument(..., help="The sweep id printed by [gvhmr]gvhmr sweep create[/]."),
    count: int | None = typer.Option(None, "--count", help="Max trials for this agent [dim](default: until done)[/]."),
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
    ent = _entity_or_default(entity)
    try:  # preflight ONCE (videos/deps) instead of crashing every non-canonical trial
        sweep_cfg = dict(wandb.Api().sweep(f"{ent}/{project}/{sweep_id}").config)
        _preflight_variants(sweep_cfg, raw_dir)
    except SystemExit:
        raise
    except Exception as e:  # API hiccup → let the trials surface any real problem
        Log.warning(f"could not preflight the sweep config ({type(e).__name__}: {e}); proceeding")
    wandb.agent(sweep_id, function=lambda: _trial(raw_dir, ckpt), count=count, project=project, entity=entity)


@app.command()
def report(
    sweep_id: str = typer.Argument(..., help="The sweep id to (re)build the comparison report for."),
    datasets: str = typer.Option("3dpw", "--datasets", help="Datasets the sweep ran (sets the metric panels)."),
    project: str = typer.Option(DEFAULT_PROJECT, "--project", help="W&B project."),
    entity: str | None = typer.Option(None, "--entity", help="W&B entity (team/user)."),
) -> None:
    """(Re)generate the comparison report: parallel coordinates over every metric + per-metric charts."""
    from gvhmr.cli.evalcmd import parse_datasets

    _require_wandb()
    url = build_report(_entity_or_default(entity), project, sweep_id, parse_datasets(datasets))
    console.print(f"[ok]report[/]: {url}")


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
    diagnostics: bool = typer.Option(
        False, "--diagnostics", help="Log the full distribution per trial (not just the means)."
    ),
) -> None:
    """Create the sweep and run every trial locally — the one-command comparison."""
    wandb = _require_wandb()
    cfg = build_sweep_config(datasets, detectors, pose2ds, method=method, metric=metric, diagnostics=diagnostics)
    print_dimensions(cfg)
    _preflight_variants(cfg, raw_dir)  # fail once, before creating the sweep — not on all N trials
    sweep_id = wandb.sweep(cfg, project=project, entity=entity)
    from gvhmr.cli.evalcmd import parse_datasets

    names = parse_datasets(cfg["parameters"]["datasets"]["value"])
    url = build_report(_entity_or_default(entity), project, sweep_id, names)
    console.print(f"[ok]report[/] (fills in live as trials finish): [muted]{url}[/]")
    wandb.agent(sweep_id, function=lambda: _trial(raw_dir, ckpt), count=count, project=project, entity=entity)
