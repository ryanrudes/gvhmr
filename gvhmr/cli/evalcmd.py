"""``gvhmr eval`` — the paper benchmarks (3DPW / EMDB / RICH) as one friendly command.

Wraps the canonical Lightning test tasks (``global/task=gvhmr/test_*`` — flip-test TTA + test-time
postprocessing, exactly the paper's protocol) with everything around them handled: the preprocessed
data packs and the released checkpoint auto-fetch, the registration-gated body models are checked up
front with the exact files/sign-up needed, and the run ends in one consolidated table with the paper's
published numbers next to yours (plus an optional ``--json`` dump for tracking).

The numbers are produced by the same callbacks/tasks as the raw
``gvhmr train global/task=gvhmr/test_3dpw_emdb_rich …`` invocation — this command only removes the
ceremony around them.
"""

from __future__ import annotations

from pathlib import Path

from gvhmr.utils.console import console, rule
from gvhmr.utils.pylogger import Log

#: dataset key → (hydra test task, data pack name, dataset_ids its callbacks report).
DATASETS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "3dpw": ("gvhmr/test_3dpw", "3dpw", ("3DPW",)),
    "emdb": ("gvhmr/test_emdb", "emdb", ("EMDB_1", "EMDB_2")),
    "rich": ("gvhmr/test_rich", "rich", ("RICH",)),
}
COMBINED_TASK = "gvhmr/test_3dpw_emdb_rich"  # all three in one run (the README's reproduce command)

#: Body-model files each dataset's GT/prediction pipeline needs (all registration-gated).
BODY_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "3dpw": ("smplx/SMPLX_NEUTRAL.npz", "smpl/SMPL_MALE.pkl", "smpl/SMPL_FEMALE.pkl"),
    "emdb": ("smplx/SMPLX_NEUTRAL.npz", "smpl/SMPL_MALE.pkl", "smpl/SMPL_FEMALE.pkl"),
    "rich": (
        "smplx/SMPLX_NEUTRAL.npz",
        "smplx/SMPLX_MALE.npz",
        "smplx/SMPLX_FEMALE.npz",
        "smpl/SMPL_NEUTRAL.pkl",
    ),
}

#: GVHMR's published numbers (arXiv 2409.06662, Tables 1–2; FlipEval + postprocessing — the same
#: protocol these test tasks run). ``wa2_mpjpe`` is the first-2-frame-aligned W-MPJPE₁₀₀ and
#: ``waa_mpjpe`` the fully-aligned WA-MPJPE₁₀₀ (per 100-frame chunk), matching eval_utils.
PAPER_REFERENCE: dict[str, dict[str, float]] = {
    "3DPW": {"pa_mpjpe": 36.2, "mpjpe": 55.6, "pve": 67.2, "accel": 5.0},
    "EMDB_1": {"pa_mpjpe": 42.7, "mpjpe": 72.6, "pve": 84.2, "accel": 3.6},
    "EMDB_2": {"wa2_mpjpe": 274.9, "waa_mpjpe": 109.1, "rte": 1.9, "jitter": 16.5, "fs": 3.5},
    "RICH": {
        "pa_mpjpe": 39.5,
        "mpjpe": 66.0,
        "pve": 74.4,
        "accel": 4.1,
        "wa2_mpjpe": 126.3,
        "waa_mpjpe": 78.8,
        "rte": 2.4,
        "jitter": 12.8,
        "fs": 3.0,
    },
}

#: metric key → display label with units (mm for joint/vertex errors; RTE is % of GT path length;
#: accel is m/s² against GT; jitter is the WHAM-convention 10·m/s³ of the prediction alone).
METRIC_LABELS: dict[str, str] = {
    "pa_mpjpe": "PA-MPJPE (mm) ↓",
    "mpjpe": "MPJPE (mm) ↓",
    "pve": "PVE (mm) ↓",
    "accel": "Accel (m/s²) ↓",
    "wa2_mpjpe": "W-MPJPE₁₀₀ (mm) ↓",
    "waa_mpjpe": "WA-MPJPE₁₀₀ (mm) ↓",
    "rte": "RTE (%) ↓",
    "jitter": "Jitter (10·m/s³) ↓",
    "fs": "Foot sliding (mm) ↓",
}


def parse_datasets(datasets: str) -> list[str]:
    """Normalize the CLI argument to dataset keys (raises KeyError listing valid names)."""
    names = [s.strip().lower() for s in datasets.replace(",", " ").split() if s.strip()]
    if not names or "all" in names:
        return list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise KeyError(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)} or 'all'")
    return list(dict.fromkeys(names))


def missing_body_models(names: list[str], root: Path) -> list[str]:
    """The gated body-model files (relative to ``root``) these datasets need but that are absent."""
    need = dict.fromkeys(f for n in names for f in BODY_MODEL_FILES[n])
    return [f for f in need if not (root / f).exists()]


def ensure_inputs(names: list[str]) -> None:
    """Auto-fetch the data packs + checkpoint; fail fast (with the sign-up) on gated body models."""
    from gvhmr.utils import assets

    for n in names:
        _, pack, _ = DATASETS[n]
        _, ds_dir, size = assets.DATA_PACKS[pack]
        if (assets.DATA_ROOT / ds_dir / "hmr4d_support").exists():
            continue
        with console.status(f"Fetching the {pack} eval pack ({size / 1e9:.1f}GB) + extracting…"):
            assets.fetch_data_pack(pack)
        Log.info(f"[ok]{pack} pack ready[/]")
    if not assets.is_present(assets.ASSETS["gvhmr"]):
        with console.status("Fetching the released GVHMR checkpoint…"):
            assets.fetch({"gvhmr": assets.ASSETS["gvhmr"]})
    gap = missing_body_models(names, assets.BODY_MODEL_ROOT)
    if gap:
        console.print(
            f"[err]Missing registration-gated body models[/] under [muted]{assets.BODY_MODEL_ROOT}[/]:\n"
            + "\n".join(f"  • {f}" for f in gap)
            + "\nSign up at https://smpl.is.tue.mpg.de/ + https://smpl-x.is.tue.mpg.de/ and place them "
            "there (or set $GVHMR_BODY_MODELS / `gvhmr config set body_models …`)."
        )
        raise SystemExit(1)


def print_summary(metrics: dict[str, dict[str, float]]) -> None:
    """One consolidated table: your numbers next to the paper's, with the delta."""
    from rich.table import Table

    table = Table(title="GVHMR benchmark  ·  flip-test + postproc (paper protocol)", expand=False)
    table.add_column("dataset")
    table.add_column("metric")
    table.add_column("this run", justify="right")
    table.add_column("paper", justify="right", style="muted")
    table.add_column("Δ", justify="right")
    for ds in sorted(metrics):
        ref = PAPER_REFERENCE.get(ds, {})
        for k, v in metrics[ds].items():
            r = ref.get(k)
            delta = "" if r is None else f"{v - r:+.1f}"
            style = "" if r is None else ("ok" if v - r <= 0.05 * max(abs(r), 1.0) else "warn")
            table.add_row(
                ds,
                METRIC_LABELS.get(k, k),
                f"{v:.1f}",
                "" if r is None else f"{r:.1f}",
                f"[{style}]{delta}[/]" if delta else "",
            )
        table.add_section()
    console.print(table)
    console.print(
        "[dim]paper = arXiv 2409.06662 Tables 1–2 (same protocol). Small deviations (<~1mm) are expected "
        "across GPUs/library versions; a large Δ means the pipeline changed.[/]"
    )


def run(
    datasets: str = "all", ckpt: str | None = None, json_out: Path | None = None, set_overrides: list[str] | None = None
) -> None:
    """Run the benchmark eval end-to-end and print the consolidated table."""
    import importlib
    import json

    import torch

    from gvhmr.utils import assets

    # The Typer command *function* `train` shadows the submodule on the package, so import explicitly.
    train_cli = importlib.import_module("gvhmr.cli.train")

    names = parse_datasets(datasets)
    rule(f"[gvhmr]gvhmr eval[/] · {', '.join(names)}")
    if not torch.cuda.is_available():
        Log.warning(
            "[warn]No CUDA GPU[/] — the eval runs but will be slow (the paper protocol runs the model "
            "with flip-test on every sequence)."
        )
    ensure_inputs(names)
    ckpt = str(ckpt or assets.GVHMR_CKPT)

    # One combined run when evaluating all three (the README's reproduce command); else one task per
    # dataset, sequentially in-process.
    tasks = [COMBINED_TASK] if set(names) == set(DATASETS) else [DATASETS[n][0] for n in names]
    results: dict[str, dict[str, float]] = {}
    for task in tasks:
        overrides = [f"global/task={task}", "exp=gvhmr/mixed/mixed", f"ckpt_path={ckpt}"]
        overrides += list(set_overrides or [])
        Log.info(f"Running [gvhmr]gvhmr train {' '.join(overrides)}[/]")
        train_cli.run(overrides)
        results.update(train_cli.LAST_TEST_METRICS)

    if not results:
        Log.warning("No metrics were produced — check the run output above.")
        raise SystemExit(1)
    print_summary(results)

    if json_out is not None:
        from datetime import datetime, timezone

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ckpt": ckpt,
            "datasets": names,
            "protocol": "flip-test + postproc (paper protocol)",
            "metrics": results,
            "paper_reference": {ds: PAPER_REFERENCE.get(ds, {}) for ds in results},
        }
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n")
        Log.info(f"[ok]metrics written[/] → [muted]{json_out}[/]")
