"""``gvhmr eval`` — the paper benchmarks (3DPW / EMDB / RICH) as one friendly command.

Wraps the standard Lightning test tasks (``global/task=gvhmr/test_*`` — flip-test TTA + test-time
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

#: dataset key → the `test_datasets.<key>` config nodes a preproc variant must be applied to.
VARIANT_GROUPS: dict[str, tuple[str, ...]] = {"3dpw": ("3dpw",), "emdb": ("emdb1", "emdb2")}
#: dataset key → its pack dir under DATA_ROOT (where hmr4d_support lives).
PACK_DIRS: dict[str, str] = {"3dpw": "3DPW", "emdb": "EMDB", "rich": "RICH"}
RAW_DATASET_URLS = {"3dpw": "https://virtualhumans.mpi-inf.mpg.de/3DPW/", "emdb": "https://eth-ait.github.io/emdb/"}

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


def print_summary(metrics: dict[str, dict[str, float]], variant: str | None = None) -> None:
    """One consolidated table: your numbers next to the paper's, with the delta."""
    from rich.table import Table

    title = "GVHMR benchmark  ·  flip-test + postproc (paper protocol)"
    if variant is not None:
        title = f"GVHMR benchmark  ·  preproc variant [gvhmr]{variant}[/]  ·  flip-test + postproc"
    table = Table(title=title, expand=False)
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
    if variant is None:
        console.print(
            "[dim]paper = arXiv 2409.06662 Tables 1–2 (same protocol). Small deviations (<~1mm) are expected "
            "across GPUs/library versions; a large Δ means the pipeline changed.[/]"
        )
    else:
        console.print(
            "[dim]paper = arXiv 2409.06662 with the released [gvhmr]yolov8x + vitpose[/] preprocessing — the Δ "
            "column shows what the swapped stage(s) change relative to it. Run without stage flags for that "
            "baseline.[/]"
        )


def ensure_stage_deps(detectors: list[str], pose2ds: list[str], backbones: list[str]) -> None:
    """Fail fast (with the exact fix) when any chosen stage's dependency is missing."""
    from gvhmr.utils.preproc.base import missing_requirements

    rows: list[tuple[str, str, str]] = []
    for stage, values in (("detector", detectors), ("pose2d", pose2ds), ("backbone", backbones)):
        for v in values:
            rows += missing_requirements({stage: v})
    rows = list(dict.fromkeys(rows))
    if rows:
        fixes = "\n".join(f"  {fix}" for fix in dict.fromkeys(fix for _, _, fix in rows))
        console.print(
            "[err]Missing dependencies to regenerate preprocessing:[/]\n"
            + "\n".join(f"  • {module} — needed by {why}" for why, module, _ in rows)
            + f"\nInstall with:\n{fixes}"
        )
        raise SystemExit(1)


def ensure_variant_inputs(names: list[str], raw_dir: Path | None) -> None:
    """Ensure each dataset's pack has its per-sequence videos — the pixels every preprocessing
    regeneration needs. The raw dataset auto-downloads when its official host serves it directly
    (3DPW); ``--raw-dir`` points at an existing copy instead. Credential-gated raws (EMDB) fail ONCE
    with the download pointer (callers preflight this so a sweep doesn't crash trial after trial)."""
    from gvhmr.utils import assets
    from gvhmr.utils.eval import preproc_variants as pv

    for n in names:
        support = assets.DATA_ROOT / PACK_DIRS[n] / "hmr4d_support"
        video_names = pv.dataset_video_names(n, support)
        missing_videos = [v for v in video_names if not (support / f"videos/{v}.mp4").exists()]
        if not missing_videos:
            continue
        n_raw_dir = raw_dir
        if n_raw_dir is None and n in pv.RAW_AUTOFETCH:
            n_raw_dir = pv.fetch_raw_dataset(n, assets.DATA_ROOT / "raw" / PACK_DIRS[n])
        if n_raw_dir is not None:
            Log.info(f"[{n}] composing {len(missing_videos)} missing video(s) from the raw dataset…")
            pv.build_videos_from_raw(n, Path(n_raw_dir), video_names, support / "videos")
        else:
            console.print(
                f"[err]{n}: {len(missing_videos)} video(s) missing[/] under [muted]{support}/videos[/] — the "
                f"packs ship no videos (not redistributable) and this dataset's raw download is "
                f"credential-gated. Download it ({RAW_DATASET_URLS[n]}) and pass "
                f"[gvhmr]--raw-dir <download>[/] to compose them once."
            )
            raise SystemExit(1)


def ensure_variant(
    names: list[str],
    slug: str,
    detector: str | None,
    pose2d: str | None,
    backbone: str | None,
    raw_dir: Path | None,
    regen: bool,
    stage_overrides: list[str] | None = None,
) -> None:
    """Generate (or reuse) the regenerated-preprocessing cache for each dataset."""
    from gvhmr.utils import assets
    from gvhmr.utils.eval import preproc_variants as pv

    ensure_stage_deps([detector] if detector else [], [pose2d or "vitpose"], [backbone or "hmr2"])
    for n in names:
        support = assets.DATA_ROOT / PACK_DIRS[n] / "hmr4d_support"
        if not regen and pv.variant_complete(n, support, slug):
            Log.info(f"[ok]{n}[/]: variant '{slug}' already generated [muted]({support}/preproc_variants/{slug})[/]")
            continue
        ensure_variant_inputs([n], raw_dir)
        Log.info(f"[{n}] regenerating preprocessing as variant '{slug}' (resumable; ~minutes/sequence)…")
        gen = pv.generate_3dpw_variant if n == "3dpw" else pv.generate_emdb_variant
        report = gen(support, slug, detector, pose2d, backbone, overwrite=regen, stage_overrides=stage_overrides)
        report.log()


def run_benchmarks(
    names: list[str],
    ckpt: str,
    slug: str | None = None,
    set_overrides: list[str] | None = None,
    collect_detailed: bool = False,
):
    """Run the Lightning test task(s) for the given datasets and return {dataset_id: {metric: value}}.

    The shared engine under both ``gvhmr eval`` and ``gvhmr sweep``: one combined run for all three
    standard datasets, else one task per dataset sequentially in-process; a ``slug`` applies a
    regenerated preprocessing variant via the root ``preproc_variant`` config key.

    With ``collect_detailed`` (used by ``--diagnostics``), returns ``(results, detailed, raw)`` where
    ``detailed``/``raw`` are the opt-in diagnostics stashed by the metric callbacks; otherwise returns
    just ``results`` (unchanged, so ``gvhmr sweep``'s existing call is untouched)."""
    import importlib

    # The Typer command *function* `train` shadows the submodule on the package, so import explicitly.
    train_cli = importlib.import_module("gvhmr.cli.train")

    combined = set(names) == set(DATASETS) and slug is None
    tasks = [COMBINED_TASK] if combined else [DATASETS[n][0] for n in names]
    results: dict[str, dict[str, float]] = {}
    detailed: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    for task in tasks:
        overrides = [f"global/task={task}", "exp=gvhmr/mixed/mixed", f"ckpt_path={ckpt}"]
        if slug is not None:
            overrides.append(f"preproc_variant={slug}")  # the test dataset nodes interpolate this key
        overrides += list(set_overrides or [])
        Log.info(f"Running [gvhmr]gvhmr train {' '.join(overrides)}[/]")
        train_cli.run(overrides)
        results.update(train_cli.LAST_TEST_METRICS)
        if collect_detailed:
            detailed.update(train_cli.LAST_TEST_DETAILED)
            raw.update(train_cli.LAST_TEST_RAW)
    if collect_detailed:
        return results, detailed, raw
    return results


def _provenance(ckpt: str, slug: str | None, names: list[str], detailed: dict) -> dict:
    """A reproducibility record for a diagnostics dump: git sha, ckpt hash, env, dataset sizes."""
    import hashlib
    import subprocess
    from datetime import datetime, timezone

    import torch

    def _git_sha() -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
        except Exception:  # noqa: BLE001
            return None

    def _sha256(path: str) -> str | None:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:  # noqa: BLE001
            return None

    sizes = {ds: {"n_sequences": d.get("n_sequences"), "n_frames": d.get("n_frames")} for ds, d in detailed.items()}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "ckpt": ckpt,
        "ckpt_sha256": _sha256(ckpt),
        "preproc_variant": slug,
        "datasets": names,
        "dataset_sizes": sizes,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run(
    datasets: str = "all",
    ckpt: str | None = None,
    json_out: Path | None = None,
    set_overrides: list[str] | None = None,
    detector: str | None = None,
    pose2d: str | None = None,
    backbone: str | None = None,
    variant: str | None = None,
    raw_dir: Path | None = None,
    regen: bool = False,
    diagnostics: bool = False,
    dump_raw: bool = False,
    diagnostics_out: Path | None = None,
) -> None:
    """Run the benchmark eval end-to-end and print the consolidated table."""
    import json
    import os

    import torch

    from gvhmr.utils import assets
    from gvhmr.utils.eval.preproc_variants import normalize_stage

    names = parse_datasets(datasets)
    # Normalize a baseline spelling (`--detector yolov8x` / `--pose2d vitpose` / `--backbone hmr2`, or the
    # legacy `canonical`) to None: it means the FROZEN pack, not a regeneration from the re-encoded video,
    # and must share ONE slug / stage-cache key with the None form (matching sweep's resolve_combo) so the
    # two can't diverge or collide. Without this, `--detector yolov8x` silently regenerates a fresh track.
    detector = normalize_stage("detector", detector)
    pose2d = normalize_stage("pose2d", pose2d)
    backbone = normalize_stage("backbone", backbone)
    # Split --set overrides: stage-scoped ones (detector./pose2d./backbone.) drive the preproc REGEN
    # (e.g. `--set pose2d.flip_test=false` to certify --fast); the rest tune the eval/model phase.
    _STAGE_GROUPS = ("detector", "pose2d", "backbone")
    stage_overrides = [o for o in (set_overrides or []) if o.split("=", 1)[0].split(".", 1)[0] in _STAGE_GROUPS]
    eval_overrides = [o for o in (set_overrides or []) if o not in stage_overrides]
    variant_mode = any(v is not None for v in (detector, pose2d, backbone, variant)) or bool(stage_overrides)
    slug = None
    if variant_mode:
        from gvhmr.utils.eval.preproc_variants import variant_slug

        slug = variant or variant_slug(detector, pose2d, backbone, stage_overrides)
        unsupported = [n for n in names if n not in VARIANT_GROUPS]
        if unsupported:
            console.print(
                f"[err]Preproc variants aren't supported for: {', '.join(unsupported)}[/] — the RICH pack has "
                f"no per-sequence videos and the raw dataset is registration-gated. "
                f"Run e.g. [gvhmr]gvhmr eval 3dpw,emdb --detector …[/] instead."
            )
            raise SystemExit(1)
        if backbone not in (None, "hmr2") and ckpt is None:
            Log.warning(
                "[warn]A swapped feature backbone with the RELEASED checkpoint produces meaningless numbers[/] "
                "— the features are learned conditioning (docs/EXTENSIBILITY.md Tier B). Pass --ckpt with a "
                "checkpoint retrained on those features."
            )
    rule(f"[gvhmr]gvhmr eval[/] · {', '.join(names)}" + (f" · preproc={slug}" if slug else ""))
    if not torch.cuda.is_available():
        Log.warning(
            "[warn]No CUDA GPU[/] — the eval runs but will be slow (the paper protocol runs the model "
            "with flip-test on every sequence)."
        )
    ensure_inputs(names)
    if variant_mode:
        ensure_variant(names, slug, detector, pose2d, backbone, raw_dir, regen, stage_overrides=stage_overrides)
    ckpt = str(ckpt or assets.GVHMR_CKPT)

    diagnostics = diagnostics or dump_raw or diagnostics_out is not None
    detailed: dict = {}
    raw: dict = {}
    if diagnostics:
        os.environ["GVHMR_EVAL_DIAGNOSTICS"] = "1"  # in-process toggle read by the metric callbacks
        results, detailed, raw = run_benchmarks(
            names, ckpt, slug=slug, set_overrides=eval_overrides, collect_detailed=True
        )
    else:
        results = run_benchmarks(names, ckpt, slug=slug, set_overrides=eval_overrides)
    if not results:
        Log.warning("No metrics were produced — check the run output above.")
        raise SystemExit(1)
    print_summary(results, variant=slug)

    provenance = _provenance(ckpt, slug, names, detailed) if diagnostics else None

    if json_out is not None:
        from datetime import datetime, timezone

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ckpt": ckpt,
            "datasets": names,
            "protocol": "flip-test + postproc (paper protocol)",
            "preproc_variant": slug,
            "metrics": results,
            "paper_reference": {ds: PAPER_REFERENCE.get(ds, {}) for ds in results},
        }
        if provenance is not None:
            payload["provenance"] = provenance
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n")
        Log.info(f"[ok]metrics written[/] → [muted]{json_out}[/]")

    if diagnostics:
        out = (
            Path(diagnostics_out)
            if diagnostics_out is not None
            else (Path(json_out).parent if json_out else Path("outputs/eval")) / "diagnostics.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"provenance": provenance, "detailed": detailed}, indent=2) + "\n")
        Log.info(f"[ok]diagnostics written[/] → [muted]{out}[/] [dim](std/percentiles/per-seq/per-joint/timing)[/]")
        if dump_raw:
            import numpy as np

            raw_dir_out = out.parent / "results_diagnostics"
            raw_dir_out.mkdir(parents=True, exist_ok=True)
            for ds, metrics in raw.items():
                arrays = {}
                for metric, vid2arr in metrics.items():
                    for vid, arr in vid2arr.items():
                        arrays[f"{metric}__{vid}"] = np.asarray(arr)
                npz_path = raw_dir_out / f"{ds}_raw.npz"
                np.savez_compressed(npz_path, **arrays)
                Log.info(f"[ok]raw arrays[/] → [muted]{npz_path}[/] ({len(arrays)} arrays)")
