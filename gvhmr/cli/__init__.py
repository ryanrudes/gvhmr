"""The ``gvhmr`` command-line interface (Typer + Rich).

Command *bodies* lazy-import their implementations so ``gvhmr --help`` / ``gvhmr info``
stay instant (no torch import until a command actually runs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gvhmr.cli.config import app as config_app
from gvhmr.cli.envcmd import app as env_app
from gvhmr.cli.hubcmd import auth_app, body_models_app, publish_hub, publish_space
from gvhmr.cli.sweepcmd import app as sweep_app

app = typer.Typer(
    name="gvhmr",
    help=(
        ":runner: [gvhmr]GVHMR[/] — world-grounded human motion recovery from video "
        "(SIGGRAPH Asia 2024).\n\n"
        "Recover SMPL/SMPL-X motion in camera & world frames. Start with [gvhmr]gvhmr info[/]."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
# `gvhmr config …` — view/edit the local config file (asset paths + default model versions + env).
app.add_typer(config_app, name="config")
# `gvhmr env …` — record & re-sync this box's Python environment (so users never run uv by hand).
app.add_typer(env_app, name="env")
# `gvhmr sweep …` — W&B sweeps comparing benchmark evals across preproc combinations (wandb loads lazily).
app.add_typer(sweep_app, name="sweep")
# `gvhmr auth …` — store credentials for auto-fetching gated assets (the SMPL/SMPL-X body models).
app.add_typer(auth_app, name="auth")
# `gvhmr body-models …` — push/pull your own private HF mirror of the gated SMPL/SMPL-X models.
app.add_typer(body_models_app, name="body-models")
# `gvhmr publish-hub` / `publish-space` — publish weights + the gradio demo to the HuggingFace Hub.
app.command(name="publish-hub")(publish_hub)
app.command(name="publish-space")(publish_space)


@app.callback()
def _setup() -> None:
    """Runs before every command — quiet benign warnings, repair broken TLS cert config."""
    import warnings

    # torch emits this whenever it loads our committed sparse-tensor assets (regressors,
    # smplx2smpl); the tensors are valid, so the check being disabled is fine.
    warnings.filterwarnings("ignore", message=".*Sparse invariant checks are implicitly disabled.*")
    # flip-test's rotation averaging uses SVD, which MPS runs on CPU — correct, just a perf note.
    warnings.filterwarnings("ignore", message=".*aten::linalg_svd' is not currently supported on the MPS.*")

    # Some clusters (notably HPC login shells) export a misconfigured SSL_CERT_DIR/FILE that
    # breaks OpenSSL issuer lookup → downloads fail with CERTIFICATE_VERIFY_FAILED. Repair it
    # process-wide (no-op on sane environments) so download/demo just work. See gvhmr.utils.net.
    from gvhmr.utils.net import ensure_ca_bundle

    ensure_ca_bundle()


@app.command()
def demo(
    video: Annotated[Path, typer.Argument(help="Input video file (.mp4).", exists=True, dir_okay=False)],
    static_cam: Annotated[
        bool, typer.Option("--static-cam", "-s", help="Camera is static — skip visual odometry.")
    ] = False,
    output_root: Annotated[
        str | None, typer.Option("--output-root", "-o", help="Output root [dim](default outputs/demo)[/].")
    ] = None,
    camera: Annotated[
        str | None,
        typer.Option(
            "--camera",
            help="Camera backend for a moving camera: [gvhmr]simplevo[/] (default, rotation only), "
            "[gvhmr]dpvo[/] (CUDA-only), or the scene-aware [bold]metric[/] cameras [gvhmr]dust3r[/] / "
            "[gvhmr]vggt[/] — run on Apple-Silicon/CPU and recover world translation (the following-camera fix).",
        ),
    ] = None,
    use_dpvo: Annotated[bool, typer.Option(help="[dim]Deprecated alias for [/][gvhmr]--camera dpvo[/].")] = False,
    slam: Annotated[str | None, typer.Option("--slam", help="[dim]Deprecated alias for [/][gvhmr]--camera[/].")] = None,
    f_mm: Annotated[
        int | None, typer.Option(help="Full-frame focal length in mm [dim](iPhone 1x≈24, 2x≈48)[/].")
    ] = None,
    render_scale: Annotated[
        float | None, typer.Option(help="Overlay resolution fraction [dim](0.5 default; 1.0 = full)[/].")
    ] = None,
    no_render: Annotated[bool, typer.Option("--no-render", help="Skip overlay videos — recover motion only.")] = False,
    flip_test: Annotated[
        bool,
        typer.Option(
            "--flip-test",
            help="Flip-test TTA: average the prediction with its mirror for accuracy "
            "[dim](the benchmark setting; +1 feature pass)[/].",
        ),
    ] = False,
    incam_world_traj: Annotated[
        bool,
        typer.Option(
            "--incam-world-traj/--no-incam-world-traj",
            help="On a [bold]static camera[/], take the world trajectory from the in-cam motion "
            "(captures gliding/skateboarding the velocity prior misses). Default on; no-op for moving cameras.",
        ),
    ] = True,
    skeleton: Annotated[
        bool, typer.Option("--skeleton", help="Also export a [bold]world-frame skeleton[/] video (joints + bones).")
    ] = False,
    skeleton_overlay: Annotated[
        bool,
        typer.Option("--skeleton-overlay", help="Also export the mesh videos with the [bold]skeleton overlaid[/]."),
    ] = False,
    skeleton_joints: Annotated[
        str | None,
        typer.Option(
            "--skeleton-joints",
            help="Subset to draw [dim](default all)[/]: groups (legs, left_arm, arms, torso, spine, head, "
            "hands, feet) and/or joint names/indices, comma-separated — e.g. [gvhmr]legs,left_arm[/].",
        ),
    ] = None,
    smplx: Annotated[
        bool,
        typer.Option(
            "--smplx",
            help="Render the [bold]native ~10475-vertex SMPL-X surface[/] (SMPL-X faces) instead of the "
            "default SMPL-topology mesh [dim](→ _smplx-suffixed videos; needs only the SMPL-X body model)[/].",
        ),
    ] = False,
    detector: Annotated[
        str | None,
        typer.Option(
            "--detector",
            help="Detector preset [dim](yolo=default; any family×size: [gvhmr]yolo11x[/], [gvhmr]yolo26x[/], … — `gvhmr config` lists them)[/].",
        ),
    ] = None,
    pose2d: Annotated[
        str | None,
        typer.Option("--pose2d", help="2D-pose backend by name [dim](e.g. [gvhmr]vitpose[/], [gvhmr]rtmpose[/])[/]."),
    ] = None,
    backbone: Annotated[
        str | None,
        typer.Option("--backbone", help="Feature backbone by name [dim](hmr2; others need a retrain)[/]."),
    ] = None,
    detector_ckpt: Annotated[
        str | None,
        typer.Option("--detector-ckpt", help="Detector weight override [dim](e.g. a yolov11x.pt)[/]."),
    ] = None,
    pose2d_ckpt: Annotated[
        str | None, typer.Option("--pose2d-ckpt", help="2D-pose weight override [dim](must emit COCO-17)[/].")
    ] = None,
    recipe: Annotated[
        str | None,
        typer.Option(
            "--recipe", help="Apply a bundled config recipe [dim](e.g. [gvhmr]hq[/]; see configs/recipe/)[/]."
        ),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Raw Hydra override(s), repeatable [dim](e.g. --set detector.conf=0.4)[/]."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Also save intermediate-result overlays.")] = False,
) -> None:
    """Recover motion from a [bold]single video[/] and render in-cam + world overlays."""
    from gvhmr.cli.demo import run

    run(
        video,
        output_root=output_root,
        static_cam=static_cam,
        use_dpvo=use_dpvo,
        slam=slam,
        camera=camera,
        f_mm=f_mm,
        verbose=verbose,
        render_scale=render_scale,
        no_render=no_render,
        flip_test=flip_test,
        incam_world_traj=incam_world_traj,
        skeleton=skeleton,
        skeleton_overlay=skeleton_overlay,
        skeleton_joints=skeleton_joints,
        smplx=smplx,
        detector=detector,
        pose2d=pose2d,
        backbone=backbone,
        detector_ckpt=detector_ckpt,
        pose2d_ckpt=pose2d_ckpt,
        recipe=recipe,
        set_overrides=set_,
    )


@app.command(name="demo-folder")
def demo_folder(
    folder: Annotated[Path, typer.Argument(help="Folder of .mp4 videos.", exists=True, file_okay=False)],
    static_cam: Annotated[bool, typer.Option("--static-cam", "-s", help="Cameras are static.")] = False,
    output_root: Annotated[str | None, typer.Option("--output-root", "-o", help="Output root.")] = None,
    camera: Annotated[
        str | None, typer.Option("--camera", help="Camera backend [dim](simplevo/dpvo/dust3r/vggt)[/].")
    ] = None,
    use_dpvo: Annotated[bool, typer.Option(help="[dim]Deprecated alias for [/][gvhmr]--camera dpvo[/].")] = False,
    f_mm: Annotated[int | None, typer.Option(help="Focal length in mm.")] = None,
    render_scale: Annotated[float | None, typer.Option(help="Overlay resolution fraction.")] = None,
    no_render: Annotated[bool, typer.Option("--no-render", help="Skip overlays.")] = False,
    smplx: Annotated[
        bool, typer.Option("--smplx", help="Render the native ~10475-vertex SMPL-X surface (→ _smplx videos).")
    ] = False,
) -> None:
    """Run the demo over [bold]every video in a folder[/]."""
    from gvhmr.cli.demo_folder import run

    run(
        folder,
        output_root=output_root,
        static_cam=static_cam,
        camera=camera,
        use_dpvo=use_dpvo,
        f_mm=f_mm,
        render_scale=render_scale,
        no_render=no_render,
        smplx=smplx,
    )


@app.command()
def bench(
    lengths: Annotated[list[int] | None, typer.Option(help="Sequence lengths (frames) to time.")] = None,
    repeats: Annotated[int, typer.Option(help="Timed repeats per length.")] = 5,
) -> None:
    """Benchmark core inference latency (model forward + decode + postprocess)."""
    from gvhmr.cli.bench import run

    run(lengths=lengths or [64, 128, 256, 512], repeats=repeats)


@app.command()
def train(
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Hydra overrides, e.g. [gvhmr]exp=gvhmr/mixed/mixed[/]."),
    ] = None,
) -> None:
    """Train or test the model [dim](Hydra-driven, GPU-only)[/]."""
    from gvhmr.cli.train import run

    run(overrides or [])


@app.command(name="eval")
def eval_(
    datasets: Annotated[
        str,
        typer.Argument(
            help="Benchmarks to run: [gvhmr]3dpw[/], [gvhmr]emdb[/], [gvhmr]rich[/] (comma/space list) or [gvhmr]all[/]."
        ),
    ] = "all",
    ckpt: Annotated[
        str | None, typer.Option("--ckpt", help="Checkpoint to evaluate [dim](default: the released ckpt)[/].")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the metrics (+ paper reference) to this JSON file.")
    ] = None,
    detector: Annotated[
        str | None,
        typer.Option(
            "--detector",
            help="[bold]Preproc swap:[/] regenerate boxes with this detector (e.g. [gvhmr]yolo26x[/]) and "
            "benchmark with them [dim](3dpw/emdb only; needs the raw videos once — see --raw-dir)[/].",
        ),
    ] = None,
    pose2d: Annotated[
        str | None,
        typer.Option("--pose2d", help="[bold]Preproc swap:[/] regenerate 2D keypoints with this backend."),
    ] = None,
    backbone: Annotated[
        str | None,
        typer.Option(
            "--backbone",
            help="[bold]Preproc swap:[/] regenerate features [dim](needs a retrained --ckpt to be meaningful)[/].",
        ),
    ] = None,
    variant: Annotated[
        str | None, typer.Option("--variant", help="Custom name for the regenerated-preproc cache.")
    ] = None,
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            "--raw-dir",
            help="Official raw-dataset download; composes the pack's missing videos/ once "
            "[dim](3DPW imageFiles/ · EMDB P*/)[/].",
        ),
    ] = None,
    regen: Annotated[bool, typer.Option("--regen", help="Regenerate the variant cache even if present.")] = False,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Raw Hydra override(s), repeatable [dim](passed to the test task)[/]."),
    ] = None,
    diagnostics: Annotated[
        bool,
        typer.Option(
            "--diagnostics",
            help="Also capture the full distribution [dim](std/percentiles/per-sequence/per-joint/timing "
            "+ provenance) → diagnostics.json[/].",
        ),
    ] = False,
    dump_raw: Annotated[
        bool,
        typer.Option("--dump-raw", help="With --diagnostics, also dump the raw per-sequence arrays as <DS>_raw.npz."),
    ] = False,
    diagnostics_out: Annotated[
        Path | None,
        typer.Option("--diagnostics-out", help="Where to write diagnostics.json [dim](default: next to --json)[/]."),
    ] = None,
) -> None:
    """Run the paper benchmarks (auto-fetches data; prints your numbers next to the paper's)."""
    from gvhmr.cli.evalcmd import run

    run(
        datasets,
        ckpt=ckpt,
        json_out=json_out,
        set_overrides=set_,
        detector=detector,
        pose2d=pose2d,
        backbone=backbone,
        variant=variant,
        raw_dir=raw_dir,
        regen=regen,
        diagnostics=diagnostics,
        dump_raw=dump_raw,
        diagnostics_out=diagnostics_out,
    )


@app.command(name="extract-features")
def extract_features(
    videos: Annotated[Path, typer.Argument(help="A video file or a folder of them.", exists=True)],
    out: Annotated[Path, typer.Argument(help="Output feature dir [dim](e.g. imgfeats/<ds>_<backbone>)[/].")],
    backbone: Annotated[str, typer.Option("--backbone", help="Feature backbone [dim](hmr2, dinov2)[/].")] = "hmr2",
    detector: Annotated[
        str, typer.Option("--detector", help="Detector for the person box [dim](when not --bbx-from)[/].")
    ] = "yolo",
    bbx_from: Annotated[
        Path | None,
        typer.Option("--bbx-from", help="Reuse bbx_xys/img_wh from an existing feature-cache dir (exact re-extract)."),
    ] = None,
    pattern: Annotated[
        str, typer.Option("--pattern", help="Glob for videos when [gvhmr]videos[/] is a folder.")
    ] = "*.mp4",
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Recompute even if the .pt already exists.")] = False,
    set_: Annotated[
        list[str] | None,
        typer.Option(
            "--set", help="Raw Hydra override(s) on the backbone [dim](e.g. backbone.model_name=dinov2_vitl14)[/]."
        ),
    ] = None,
) -> None:
    """Extract image features to the training-cache format [dim](Tier B: retrain on a new backbone)[/]."""
    from gvhmr.cli.extract_features import run

    run(
        videos,
        out,
        backbone=backbone,
        detector=detector,
        bbx_from=bbx_from,
        pattern=pattern,
        overwrite=overwrite,
        set_overrides=set_,
    )


@app.command()
def info() -> None:
    """Show device, installed features, and checkpoint status."""
    from gvhmr.cli.info import run

    run()


@app.command()
def download(
    what: Annotated[
        str,
        typer.Argument(
            help="Group ([gvhmr]demo[/] = gvhmr+hmr2+vitpose+yolo, [gvhmr]slam[/] = dpvo, [gvhmr]all[/]) "
            "or comma-separated asset names.",
        ),
    ] = "demo",
    force: Annotated[bool, typer.Option("--force", help="Re-download even if already present.")] = False,
    data: Annotated[
        str | None,
        typer.Option("--data", help="Also fetch training/eval data packs [dim](3dpw,amass,h36m,emdb,rich,bedlam)[/]."),
    ] = None,
) -> None:
    """Fetch model checkpoints into the right place [dim](no manual download/placement)[/]."""
    from gvhmr.cli.download import run

    run(what, force=force, data=data)
