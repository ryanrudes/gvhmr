""":class:`GVHMRPipeline` — end-to-end video → :class:`MotionResult`, HuggingFace ``pipeline`` style.

Bundles the four preprocessing stages (detector → 2D-pose → feature backbone → camera), the trained
:class:`~gvhmr.inference.model.GVHMR` model, and post-processing behind a single callable — the same shape
as ``transformers.pipeline(...)``. It reuses the *exact* code paths behind ``gvhmr demo``
(:mod:`gvhmr.cli.demo`), so results are byte-identical; the win is a clean Python object, weights loaded
once and reused across videos, and a rich return value.

    >>> import gvhmr
    >>> pipe = gvhmr.pipeline("human-motion-recovery", device="cuda")
    >>> result = pipe("dance.mp4")
    >>> result.render("overlay.mp4")
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

from gvhmr.inference import hub
from gvhmr.inference.model import GVHMR
from gvhmr.inference.result import MotionResult

#: Task aliases accepted by :func:`pipeline`.
TASKS = ("human-motion-recovery", "hmr", "motion-recovery", "smpl", "video-to-smpl")


@contextlib.contextmanager
def _maybe_quiet(quiet: bool):
    """Temporarily silence the shared Rich console (progress=False)."""
    from gvhmr.utils.console import console

    if not quiet:
        yield
        return
    prev = console.quiet
    console.quiet = True
    try:
        yield
    finally:
        console.quiet = prev


class GVHMRPipeline:
    """Callable video → :class:`MotionResult` pipeline. Construct with :meth:`from_pretrained`."""

    task = "human-motion-recovery"

    def __init__(
        self,
        model: GVHMR,
        *,
        detector: str | None = None,
        pose2d: str | None = None,
        backbone: str | None = None,
        camera: str | None = None,
        device=None,
    ):
        self.model = model
        self.detector = detector
        self.pose2d = pose2d
        self.backbone = backbone
        self.camera = camera
        self.device = device if device is not None else model.device

    @classmethod
    def from_pretrained(
        cls,
        model: str = hub.DEFAULT_REPO,
        *,
        device=None,
        detector: str | None = None,
        pose2d: str | None = None,
        backbone: str | None = None,
        camera: str | None = None,
        cache_dir=None,
        revision: str | None = None,
        token: str | None = None,
        ckpt_path=None,
    ) -> GVHMRPipeline:
        """Load weights from a Hub repo id (or ``ckpt_path``) and set the default preprocessing stages."""
        gv = GVHMR.from_pretrained(
            model, device=device, cache_dir=cache_dir, revision=revision, token=token, ckpt_path=ckpt_path
        )
        return cls(gv, detector=detector, pose2d=pose2d, backbone=backbone, camera=camera, device=gv.device)

    # -------------------------------------------------------------------------------------------------
    def __call__(
        self,
        video,
        *,
        static_camera: bool = False,
        camera: str | None = None,
        detector: str | None = None,
        pose2d: str | None = None,
        backbone: str | None = None,
        f_mm: int | None = None,
        f_px: float | None = None,
        intrinsics=None,
        flip_test: bool = False,
        world_from_incam: bool = True,
        render: bool = False,
        mesh: str = "smpl",
        output_dir=None,
        output_root: str | None = None,
        render_scale: float | None = None,
        recipe: str | None = None,
        set_overrides: list[str] | None = None,
        progress: bool = True,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> MotionResult:
        """Recover SMPL motion from ``video`` and return a :class:`MotionResult`.

        Args:
            static_camera: the camera doesn't move — skip visual odometry (faster, cleaner world frame).
            camera: moving-camera backend (``simplevo`` / ``dpvo`` / ``dust3r`` / ``vggt``); overrides the default.
            detector / pose2d / backbone: per-call stage overrides (else the pipeline / config defaults).
            f_mm: full-frame focal length in mm (else read from metadata, else a FOV heuristic).
            f_px: focal length in **pixels** — the faithful route (bypasses the mm→px conversion; overrides f_mm).
            intrinsics: measured camera intrinsics — a sidecar path (JSON/NPZ), a dict with fx/fy/cx/cy
                (each a scalar or per-frame list) or K, or a 3x3 / (L,3,3) K array/tensor. Highest
                precedence; carries a true principal point and per-frame values. See docs/ACCURACY.md.
            flip_test: mirror-averaging test-time augmentation (the benchmark setting; +1 feature pass).
            world_from_incam: for a static camera, take the world trajectory from the in-cam motion.
            render: also render the overlay videos now (else call ``result.render()`` later).
            mesh: ``'smpl'`` (default SMPL-topology mesh) or ``'smplx'`` (native ~10475-vertex SMPL-X surface).
            output_dir / output_root: where to stage artifacts (default: ``outputs/demo/<video>``, reusable cache).
            render_scale: overlay resolution fraction (0.5 default).
            recipe / set_overrides: a bundled config recipe and/or raw Hydra overrides (advanced).
            progress: show the Rich progress display (set False for quiet library use).
            progress_callback: optional ``(fraction, description)`` hook for UIs (e.g. Gradio); called
                as ``(0..1, stage_label)`` — each stage (:func:`progress_phase`) drives its own 0→1 bar.
        """
        from gvhmr.cli.demo import (
            build_demo_cfg,
            ensure_assets,
            ensure_deps,
            load_data_dict,
            recover_motion,
            run_preprocess,
        )
        from gvhmr.utils import localconfig
        from gvhmr.utils.console import progress_hook, progress_phase

        video = Path(video)
        # Stage selection — precedence: call arg > pipeline default > config-file default > released default.
        detector = detector or self.detector or localconfig.model_default("detector")
        pose2d = pose2d or self.pose2d or localconfig.model_default("pose2d")
        backbone = backbone or self.backbone or localconfig.model_default("backbone")
        cam = camera or self.camera or localconfig.model_default("camera")

        config_overrides: list[str] = []
        if recipe is not None:
            config_overrides.append(f"+recipe={recipe}")
        if detector is not None:
            config_overrides.append(f"detector={detector}")
        if pose2d is not None:
            config_overrides.append(f"pose2d={pose2d}")
        if backbone is not None:
            config_overrides.append(f"backbone={backbone}")
        if cam is not None:
            config_overrides.append(f"camera={cam}")
        if flip_test:
            config_overrides.append("flip_test=true")
        if output_dir is not None:
            config_overrides.append(f"output_dir={output_dir}")
        config_overrides += list(set_overrides or [])

        with _maybe_quiet(not progress), progress_hook(progress_callback):
            with progress_phase("Preparing video"):
                cfg = build_demo_cfg(
                    video,
                    output_root=output_root,
                    static_cam=static_camera,
                    use_dpvo=False,
                    f_mm=f_mm,
                    f_px=f_px,
                    intrinsics=intrinsics,
                    verbose=False,
                    render_scale=render_scale,
                    config_overrides=config_overrides,
                )
            cam_name = cfg.camera.name
            resolved_flip = cfg.flip_test

            with progress_phase("Fetching assets"):
                ensure_deps(cfg)
                ensure_assets(cam_name, want_render=render, repo=self.model.repo_id)

            run_preprocess(cfg, flip_test=resolved_flip)

            with progress_phase("Loading motion data"):
                data = load_data_dict(cfg, flip_test=resolved_flip)

            from gvhmr.utils.device import predict_device

            self.model.to(predict_device()).eval()  # network is ~6.7x faster on CPU (launch-bound)
            with progress_phase("Recovering motion"):
                pred = recover_motion(
                    self.model.pl,
                    data,
                    static_cam=static_camera,
                    flip_test_data=data.get("flip_test"),
                    world_from_incam=(static_camera and world_from_incam),
                    cam_name=cam_name,
                    slam_path=cfg.paths.slam,
                )
            import torch

            torch.save(pred, cfg.paths.hmr4d_results)  # persist so result.render()/caching work

            result = MotionResult(
                smpl_params_world=pred["smpl_params_global"],
                smpl_params_camera=pred["smpl_params_incam"],
                intrinsics=pred["K_fullimg"],
                betas_per_frame=pred.get("betas_per_frame"),
                bbx_xys=pred.get("bbx_xys"),
                kp2d=pred.get("kp2d"),
                fps=30.0,
                camera="static" if static_camera else cam_name,
                video_path=Path(cfg.video_path),
                output_dir=Path(cfg.output_dir),
                raw=pred,
                _cfg=cfg,
            )
            if render:
                with progress_phase("Rendering overlay"):
                    result.render(render_scale=render_scale, mesh=mesh)
        return result

    # ``recover`` reads a touch nicer than calling the object; identical behaviour.
    recover = __call__

    def __repr__(self) -> str:
        return f"GVHMRPipeline(task={self.task!r}, model={self.model!r})"


# -----------------------------------------------------------------------------------------------------
# Factories
# -----------------------------------------------------------------------------------------------------
def pipeline(task: str = "human-motion-recovery", *, model: str = hub.DEFAULT_REPO, device=None, **kwargs):
    """Build a :class:`GVHMRPipeline` — the HuggingFace ``pipeline()`` entrypoint.

    >>> pipe = gvhmr.pipeline("human-motion-recovery", device="cuda")
    >>> result = pipe("dance.mp4")
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; GVHMR provides: {TASKS}")
    return GVHMRPipeline.from_pretrained(model, device=device, **kwargs)


_DEFAULT_PIPELINE: dict[tuple, GVHMRPipeline] = {}


def recover(video, *, model: str = hub.DEFAULT_REPO, device=None, **kwargs) -> MotionResult:
    """One-liner: recover motion from a video with a cached default pipeline.

    >>> result = gvhmr.recover("dance.mp4")

    Reuses a per-(model, device) pipeline across calls, so a loop of videos loads the weights once.
    ``**kwargs`` are forwarded to :meth:`GVHMRPipeline.__call__`.
    """
    call_keys = {
        "static_camera",
        "camera",
        "detector",
        "pose2d",
        "backbone",
        "f_mm",
        "f_px",
        "intrinsics",
        "flip_test",
        "world_from_incam",
        "render",
        "mesh",
        "output_dir",
        "output_root",
        "render_scale",
        "recipe",
        "set_overrides",
        "progress",
        "progress_callback",
    }
    call_kwargs = {k: v for k, v in kwargs.items() if k in call_keys}
    build_kwargs = {k: v for k, v in kwargs.items() if k not in call_keys}
    key = (model, str(device))
    pipe = _DEFAULT_PIPELINE.get(key)
    if pipe is None:
        pipe = _DEFAULT_PIPELINE[key] = pipeline(model=model, device=device, **build_kwargs)
    return pipe(video, **call_kwargs)
