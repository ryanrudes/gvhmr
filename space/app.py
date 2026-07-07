"""HuggingFace Space (Gradio + ZeroGPU) for GVHMR — World-Grounded Human Motion Recovery.

Upload a video → recover SMPL human motion → get a side-by-side (in-camera + world)
mesh overlay video plus an .npz of the SMPL parameters.

Uses Hugging Face **ZeroGPU** (free shared GPU for visitors; the Space owner selects the
ZeroGPU hardware flavor in Settings — requires a PRO/Team/Enterprise plan to host). Only the GPU
work (preproc models + the network) runs inside ``@spaces.GPU`` (``_recover_gpu``); dependency
bootstrap, preflight, the moderngl render and the npz save run OUTSIDE it, so the short ZeroGPU
budget isn't spent on non-CUDA work.

Body models: configure a private mirror (`GVHMR_BODY_MODELS_MIRROR` + `HF_TOKEN`) or MPI
Secrets — see space/README.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

# ZeroGPU: `spaces` must be imported before any torch/CUDA import (HF requirement). gradio above is fine —
# it doesn't import torch — and torch is only imported later, well after this line.
import spaces

# The moderngl overlay renderer needs a GL context; on a GPU-less Space (cpu-basic) there's no hardware
# GL ("libEGL.so not loaded"), so force Mesa's software rasterizer (llvmpipe) — with the EGL libs from
# packages.txt this gives a headless context so rendering works instead of failing to pytorch3d (absent).
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def _patch_gradio_file_whitelist() -> None:
    """SSR mode re-fetches uploads over HTTP; HF xethub bridge hosts must be allow-listed."""
    try:
        from gradio import processing_utils as pu

        for host in (
            "cas-bridge-direct.xethub.hf.co",
            "cas-bridge.xethub.hf.co",
            "xethub.hf.co",
        ):
            if host not in pu.PUBLIC_HOSTNAME_WHITELIST:
                pu.PUBLIC_HOSTNAME_WHITELIST.append(host)
    except Exception:  # noqa: BLE001 — gradio internals differ across versions
        pass


_patch_gradio_file_whitelist()

REPO_URL = "https://github.com/ryanrudes/gvhmr"

DEFAULT_HUB_REPO = os.getenv("GVHMR_HUB_REPO", "ryanrudes/gvhmr")

# All presets from gvhmr/configs/{detector,pose2d,backbone,camera}/ — keep in sync with the repo.
DETECTOR_CHOICES = [
    "yolo",
    "yolo11l",
    "yolo11m",
    "yolo11n",
    "yolo11s",
    "yolo11x",
    "yolo12l",
    "yolo12m",
    "yolo12n",
    "yolo12s",
    "yolo12x",
    "yolo26l",
    "yolo26m",
    "yolo26n",
    "yolo26s",
    "yolo26x",
    "yolov10b",
    "yolov10l",
    "yolov10m",
    "yolov10n",
    "yolov10s",
    "yolov10x",
    "yolov8l",
    "yolov8m",
    "yolov8n",
    "yolov8s",
    "yolov8x",
    "yolov9c",
    "yolov9e",
    "yolov9m",
    "yolov9s",
    "yolov9t",
]
POSE2D_CHOICES = ["vitpose", "rtmpose"]  # rtmpose: lazy-pip rtmlib+onnxruntime on first use
BACKBONE_CHOICES = ["hmr2"]  # dinov2 needs a retrained checkpoint
CAMERA_CHOICES = ["simplevo"]  # dpvo/dust3r/vggt are not set up on this Space

DESCRIPTION = f"""
# GVHMR — World-Grounded Human Motion Recovery

Upload a video of a person and GVHMR recovers their **SMPL** body motion in both the
**camera** frame and a gravity-aligned **world** frame. You get a side-by-side mesh
overlay video and an `.npz` of the SMPL parameters.

*World-Grounded Human Motion Recovery via Gravity-View Coordinates* (SIGGRAPH Asia 2024).
Source & docs: [{REPO_URL}]({REPO_URL}).

> **ZeroGPU** (Space owner selects that hardware) is much faster; **CPU basic** also works but
> is slow — keep clips short (~10–15 s), use *Static camera*, and skip flip-test when possible.
> On ZeroGPU, each inference call gets **60 seconds of GPU time** on the free tier (HF default for
> `@spaces.GPU`); PRO owners can raise the cap with a `ZERO_GPU_DURATION` Secret (seconds).
"""


def _hub_repo_choices() -> list[str]:
    choices = [DEFAULT_HUB_REPO, "camenduru/GVHMR"]
    if extra := os.getenv("GVHMR_HUB_REPO_OPTIONS", ""):
        choices.extend(item.strip() for item in extra.split(",") if item.strip())
    return list(dict.fromkeys(choices))


def _apply_bootstrap_smpl_compat() -> None:
    """Legacy shims before ``gvhmr`` is installed (mirrors ``gvhmr.utils._smpl_compat``)."""
    import inspect

    import numpy as np

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, target in {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, target)


def _pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def _bootstrap_deps() -> None:
    """Install chumpy (no-build-isolation) then gvhmr — cannot run at Space build time."""
    _apply_bootstrap_smpl_compat()

    try:
        import chumpy  # noqa: F401
    except ImportError:
        _pip_install("--no-build-isolation", "numpy>=1.26", "chumpy==0.70")

    try:
        import gvhmr  # noqa: F401
    except ImportError:
        # ZeroGPU preinstalls torch 2.11 — chumpy must be present first (see requirements.txt).
        _pip_install("gvhmr[preproc]>=1.0.6")

    _verify_preproc_imports()

    from gvhmr.utils._smpl_compat import apply

    apply()


def _verify_preproc_imports() -> None:
    """Fail fast if the runtime preproc extra did not land (common on first pip bootstrap)."""
    import importlib

    missing: list[str] = []
    for mod in ("ultralytics", "pycolmap", "cv2"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return
    _pip_install("gvhmr[preproc]>=1.0.6", "pycolmap>=0.6", "opencv-python-headless>=4.8")
    still: list[str] = []
    for mod in missing:
        try:
            importlib.import_module(mod)
        except ImportError:
            still.append(mod)
    if still:
        raise RuntimeError(f"preproc dependencies missing after bootstrap: {', '.join(still)}")


def _ensure_pose2d_deps(pose2d: str) -> None:
    """Lazy-install RTMPose extras (rtmlib + onnxruntime) when selected — keeps vitpose-only runs lighter."""
    if pose2d != "rtmpose":
        return
    import importlib

    try:
        importlib.import_module("rtmlib")
    except ImportError:
        _pip_install("rtmlib", "onnxruntime>=1.16")
        try:
            importlib.import_module("rtmlib")
        except ImportError as exc:
            raise RuntimeError("RTMPose (rtmlib) failed to install — try again or use vitpose.") from exc


_BOOTSTRAPPED = False


def _ensure_bootstrapped() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _bootstrap_deps()
    _BOOTSTRAPPED = True


def _startup_bootstrap() -> None:
    """Best-effort dependency install at app *startup* (module import), not inside ``@spaces.GPU``.

    ZeroGPU reservations are short (60 s on the free tier); running the multi-minute chumpy/gvhmr[preproc]
    install here — once, before any request is GPU-scheduled — keeps that budget for the actual inference.
    ``spaces`` is already imported (top of file), so importing torch during the install is allowed. On
    failure we leave ``_BOOTSTRAPPED`` False so ``run`` retries in-request as a fallback.
    """
    try:
        _ensure_bootstrapped()
    except Exception:  # noqa: BLE001 — don't crash module import; run() will retry and surface a gr.Error
        traceback.print_exc()


# --------------------------------------------------------------------------------------
# Pipeline cache: lazy load inside @spaces.GPU (keyed by hub repo + stage presets).
# --------------------------------------------------------------------------------------
_PIPES: dict[tuple, Any] = {}


def _resolve_device() -> str:
    """Pick cuda → mps → cpu; honour ``GVHMR_DEVICE`` but fall back when unavailable.

    Uses only ``torch`` (HF preinstall) so this works before the runtime ``gvhmr`` pip bootstrap.
    """
    import torch

    def _auto() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return "mps"
        return "cpu"

    prefer = os.getenv("GVHMR_DEVICE", "").strip()
    if prefer == "cuda" and not torch.cuda.is_available():
        return _auto()
    if prefer == "mps":
        mps = getattr(torch.backends, "mps", None)
        if not (mps and mps.is_available()):
            return _auto()
    if prefer:
        return prefer
    return _auto()


def _cpu_preproc_overrides() -> list[str]:
    """On a CPU Space, shrink the ViT batch so activations don't OOM the 16 GB cpu-basic box: batch 32 fp32
    ViT-Huge (ViTPose + HMR2) is fine on a GPU but tips CPU RAM over. No-op on a GPU Space.

    Guarded: only emit keys the *installed* gvhmr's config actually defines — the Space pins gvhmr from
    PyPI, and a release predating the batch_size knob would ``ConfigCompositionException`` on the override.
    """
    if _resolve_device() != "cpu":
        return []
    try:
        from hydra import compose, initialize_config_module
        from omegaconf import OmegaConf

        from gvhmr.configs import register_store_gvhmr

        register_store_gvhmr()
        with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
            cfg = compose(config_name="demo", overrides=["video_name=_"])
        ov = []
        if OmegaConf.select(cfg, "pose2d.batch_size") is not None:
            ov.append("pose2d.batch_size=4")
        if OmegaConf.select(cfg, "backbone.batch_size") is not None:
            ov.append("backbone.batch_size=4")
        return ov
    except Exception:  # noqa: BLE001 — never let a memory-tuning override break inference
        return []


def _pipeline_key(
    model_repo: str,
    revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
) -> tuple:
    rev = (revision or "").strip()
    return (model_repo.strip(), rev, detector, pose2d, backbone, _resolve_device())


def _get_pipeline(
    *,
    model_repo: str,
    revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
):
    """Return a cached pipeline for the selected weights + preprocessing presets."""
    key = _pipeline_key(model_repo, revision, detector, pose2d, backbone)
    if key in _PIPES:
        return _PIPES[key]
    try:
        _ensure_bootstrapped()
        import gvhmr

        rev = (revision or "").strip() or None
        pipe = gvhmr.pipeline(
            "human-motion-recovery",
            model=model_repo.strip(),
            revision=rev,
            detector=detector,
            pose2d=pose2d,
            backbone=backbone,
            device=key[-1],
        )
        _PIPES[key] = pipe
        return pipe
    except Exception as exc:  # noqa: BLE001 — surface to the UI but do NOT cache the failure:
        # a transient first-call error (cold-bootstrap timeout, network blip, creds not yet set) must not
        # permanently disable this combo — the next request retries from scratch.
        traceback.print_exc()
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def _preflight_stages(*, static_camera: bool, camera: str, detector: str, pose2d: str, backbone: str) -> None:
    """Mirror ``ensure_deps`` checks but raise ``gr.Error`` (never ``SystemExit``)."""
    from gvhmr.utils.preproc.base import missing_requirements

    if pose2d not in POSE2D_CHOICES:
        raise gr.Error(f"2D pose '{pose2d}' is not available on this Space. Use vitpose or rtmpose.")
    _ensure_pose2d_deps(pose2d)
    if backbone not in BACKBONE_CHOICES:
        raise gr.Error(f"Backbone '{backbone}' is not available on this Space. Use hmr2.")
    if not static_camera and camera not in CAMERA_CHOICES:
        raise gr.Error(f"Camera '{camera}' is not set up on this Space. Check Static camera, or use simplevo.")

    selections: dict[str, str] = {"detector": detector, "pose2d": pose2d, "backbone": backbone}
    if not static_camera:
        selections["camera"] = camera
    missing = missing_requirements(selections)
    if missing:
        detail = "; ".join(f"{why} requires {mod}" for why, mod, _ in missing)
        raise gr.Error(f"Missing dependencies for this run: {detail}")


def _is_credentials_error(exc: Exception) -> bool:
    """Heuristic: does this exception look like missing gated-body-model credentials?"""
    text = f"{type(exc).__name__} {exc}".lower()
    # Specific to the gated body-model flow. Deliberately NOT bare "401"/"403"/"mirror"/"permissionerror":
    # a generic Hub 403 (rate limit, private checkpoint repo) is not a body-model creds problem, and
    # mislabelling it "SMPL/SMPL-X are gated…" sends the user down the wrong fix.
    needles = (
        "smplx_user",
        "smplx_pw",
        "smpl_user",
        "smpl_pw",
        "body model",
        "body_model",
        "registration-gated",
        "smpl-x body model",
        "is.tue.mpg.de",
    )
    return any(n in text for n in needles)


def _seconds_of_motion(num_frames: int, fps: float) -> float:
    return round(num_frames / fps, 2) if fps else 0.0


def _video_path(video) -> str | None:
    """Normalize Gradio Video value (path str or FileData dict) to a filesystem path."""
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("path") or video.get("video")
    path = getattr(video, "path", None)
    return str(path) if path else str(video)


def _zero_gpu_duration_cap() -> int:
    """Max seconds requested per ``@spaces.GPU`` call (HF free tier default: 60)."""
    return int(os.getenv("ZERO_GPU_DURATION", "60"))


def _estimate_gpu_duration(
    video_path,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    *args,
    **kwargs,
) -> int:
    """ZeroGPU queue budget (seconds) for one ``@spaces.GPU`` call — clamped to the per-call cap.

    HF's free tier caps every call at 60 s; PRO owners can raise it via ``$ZERO_GPU_DURATION``. Deps are
    bootstrapped at app startup (see ``_startup_bootstrap``), so the GPU reservation only pays for the
    checkpoint load + inference, not pip.
    """
    cap = _zero_gpu_duration_cap()

    video_path = _video_path(video_path)
    secs = 30.0
    if video_path:
        try:
            import cv2

            cap_vid = cv2.VideoCapture(video_path)
            fps = cap_vid.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap_vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap_vid.release()
            if fps > 0 and frames > 0:
                secs = float(frames / fps)
        except Exception:  # noqa: BLE001
            pass
    per_sec = 5.0 if flip_test else 3.5
    load = 20  # first-call checkpoint load / warm-up
    est = load + secs * per_sec
    return int(max(15, min(cap, est)))


# --------------------------------------------------------------------------------------
# Inference callback (GPU allocated for the duration of this function on ZeroGPU Spaces)
# --------------------------------------------------------------------------------------
@spaces.GPU(duration=_estimate_gpu_duration)
def _recover_gpu(
    video_path: str,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    hub_repo: str,
    hub_revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
    out_dir: str,
    progress: gr.Progress,
):
    """The GPU-bound core (ZeroGPU allocation held ONLY here): load the pipeline + run preproc models +
    the network. Render (moderngl/EGL — not CUDA), npz save, bootstrap and preflight all run OUTSIDE this
    window (in ``run``), so the 60s free-tier budget isn't spent on non-GPU work. Returns the MotionResult
    with ``render=False`` (the pred is torch.saved to ``out_dir`` on the shared FS for ``run`` to render)."""
    from gvhmr.utils.console import progress_hook, progress_phase

    def _hook(frac: float, desc: str) -> None:
        progress(max(0.0, min(1.0, frac)), desc=desc)

    with progress_hook(_hook):
        with progress_phase("Loading model"):
            pipe = _get_pipeline(
                model_repo=hub_repo, revision=hub_revision, detector=detector, pose2d=pose2d, backbone=backbone
            )
        camera = None if static_camera else camera_backend
        return pipe(
            video_path,
            static_camera=static_camera,
            camera=camera,
            flip_test=flip_test,
            render=False,
            progress=False,
            progress_callback=_hook,
            output_dir=str(out_dir),
            set_overrides=_cpu_preproc_overrides(),  # small CPU batch → don't OOM cpu-basic
        )


def run(
    video_path: str | None,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    hub_repo: str,
    hub_revision: str,
    detector: str,
    pose2d: str,
    backbone: str,
    native_smplx: bool = False,
    render_overlay: bool = False,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 — gvhmr uses Rich track(), not tqdm; we drive it via progress(...)
) -> tuple[str, str, dict]:
    """Gradio handler: validate + bootstrap + preflight (CPU), run inference under @spaces.GPU, then render
    + save (CPU / moderngl — no CUDA) OUTSIDE the GPU window so ZeroGPU time is spent only on the GPU work."""
    video_path = _video_path(video_path)
    if not video_path:
        raise gr.Error("Please upload a video first.")
    if video_path.startswith(("http://", "https://")):
        raise gr.Error(
            "The uploaded video could not be read as a local file. Hard-refresh the page, "
            "upload the clip again, and click Recover motion immediately — do not reuse a stale tab."
        )

    def _hook(frac: float, desc: str) -> None:
        progress(max(0.0, min(1.0, frac)), desc=desc)

    try:
        progress(0.0, desc="Preparing environment…")
        _ensure_bootstrapped()  # CPU / pip — outside the GPU window
        from gvhmr.utils.console import progress_hook, progress_phase

        progress(0.01, desc="Starting…")
        _preflight_stages(  # CPU dependency checks — outside the GPU window
            static_camera=static_camera,
            camera=camera_backend if not static_camera else "simplevo",
            detector=detector,
            pose2d=pose2d,
            backbone=backbone,
        )
        out_dir = Path(tempfile.mkdtemp(prefix="gvhmr_"))

        try:  # GPU window: preproc models + network only
            result = _recover_gpu(
                video_path, static_camera, camera_backend, flip_test, hub_repo, hub_revision,
                detector, pose2d, backbone, str(out_dir), progress,
            )  # fmt: skip
        except RuntimeError as exc:
            raise gr.Error(f"The GVHMR pipeline failed, so inference is unavailable. Details: {exc}") from exc

        # Render (moderngl/EGL — not CUDA) + save: run OUTSIDE the @spaces.GPU allocation. Native SMPL-X
        # (default here) needs only the SMPL-X body model; the SMPL-topology mesh needs the gated SMPL one.
        mesh = "smplx" if native_smplx else "smpl"
        with progress_hook(_hook):
            with progress_phase("Saving SMPL params"):  # always produced — the reliable output
                npz_path = result.save_npz(out_dir / "motion.npz")
            overlay_path = None
            # Render is OFF by default on this CPU Space (opt-in checkbox): software (llvmpipe) rasterization of
            # the mesh per frame is slow and memory-heavy on cpu-basic, and the .npz is the reliable output.
            if not render_overlay:
                render_status = "not requested (tick 'Render overlay video' to also produce the mesh video)"
            else:
                render_status = "ok"
                with progress_phase("Rendering overlay"):
                    try:  # best-effort: never fail the request over a missing SMPL model / GL context
                        overlay_path = result.render(out_dir / "overlay_both.mp4", view="both", mesh=mesh)
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        hint = (
                            " — the SMPL-topology mesh needs the gated SMPL body model; tick 'Native SMPL-X mesh' "
                            "(needs only SMPL-X)"
                            if mesh == "smpl"
                            else " — the renderer needs a working GL context (moderngl/EGL) on this Space"
                        )
                        render_status = f"skipped ({type(exc).__name__}: {exc}){hint}"

        summary = {
            "frames": result.num_frames,
            "fps": result.fps,
            "camera": result.camera,
            "seconds_of_motion": _seconds_of_motion(result.num_frames, result.fps),
            "mesh": "SMPL-X (native)" if native_smplx else "SMPL topology",
            "overlay": render_status,  # 'ok', or why the overlay was skipped (the .npz is always returned)
            "weights_repo": hub_repo.strip(),
            "weights_revision": (hub_revision or "").strip() or "default",
            "detector": detector,
            "pose2d": pose2d,
            "backbone": backbone,
        }
        progress(1.0, desc="Done")
        return (str(overlay_path) if overlay_path else None), str(npz_path), summary

    except gr.Error:
        raise
    except SystemExit as exc:
        raise gr.Error(
            "GVHMR dependency check failed (see Space Logs for the package list). "
            "On this Space: keep Static camera checked unless you need simplevo, "
            "and use the default vitpose / hmr2 / yolo presets."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — convert everything to a friendly UI error
        if _is_credentials_error(exc):
            raise gr.Error(
                "SMPL/SMPL-X body models are gated and could not be fetched. Configure a private mirror "
                "(GVHMR_BODY_MODELS_MIRROR + HF_TOKEN) or MPI Secrets (SMPLX_USER/SMPLX_PW and, for "
                "rendering, SMPL_USER/SMPL_PW). Details: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise gr.Error(f"Inference failed: {type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def _toggle_camera(static_camera: bool) -> gr.Dropdown:
    """The camera backend is meaningless (and ignored) for a static camera."""
    return gr.Dropdown(interactive=not static_camera)


# Install the runtime deps now, at import — before the platform GPU-schedules the first `run` call.
_startup_bootstrap()


def _probe_gl() -> None:
    """Log whether the moderngl/EGL overlay renderer can init here (diagnostic; the real render is
    best-effort in ``run``). glcontext's standalone EGL needs an *enumerable* device — on a device-less
    cpu-basic box software rendering may be unavailable no matter which Mesa libs are installed, so this
    line tells us (from the Space Logs) which case we're in without running a full job."""
    try:
        import moderngl

        from gvhmr.utils.vis.renderer_gl import _standalone_context

        ctx = _standalone_context(moderngl)
        info = ctx.info
        print(f"[gvhmr] GL overlay context OK — renderer={info.get('GL_RENDERER')!r} ({info.get('GL_VERSION')!r})")
        ctx.release()
    except Exception as exc:  # noqa: BLE001 — purely diagnostic; the app runs (npz-only) either way
        print(f"[gvhmr] GL overlay context UNAVAILABLE ({type(exc).__name__}: {exc}) — overlays fall back to npz-only")


_probe_gl()


with gr.Blocks(title="GVHMR — Human Motion Recovery") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video", sources=["upload"], format="mp4")
            static_in = gr.Checkbox(
                value=True,
                label="Static camera",
                info="Recommended on this Space (faster; skips visual odometry). Uncheck only for simplevo.",
            )
            camera_in = gr.Dropdown(
                choices=CAMERA_CHOICES,
                value="simplevo",
                label="Camera backend",
                info="Ignored when 'Static camera' is checked. Scene-aware backends (dpvo/dust3r/vggt) "
                "aren't set up on this Space.",
            )
            flip_in = gr.Checkbox(value=False, label="Flip-test (slower, more accurate)")
            smplx_in = gr.Checkbox(
                value=True,
                label="Native SMPL-X mesh",
                info="Render the full ~10475-vertex SMPL-X surface (the model predicts SMPL-X natively). "
                "Default ON: it needs only the SMPL-X body model. Uncheck for the SMPL-topology mesh — "
                "which additionally requires the gated SMPL body model.",
            )
            render_in = gr.Checkbox(
                value=False,
                label="Render overlay video (slow on CPU)",
                info="OFF by default on this free CPU Space: the .npz (SMPL params) is always returned fast. "
                "Tick to also rasterize the mesh overlay in software (llvmpipe) — adds minutes for a short clip.",
            )

            with gr.Accordion("Model settings", open=False):
                hub_repo_in = gr.Dropdown(
                    choices=_hub_repo_choices(),
                    value=DEFAULT_HUB_REPO,
                    allow_custom_value=True,
                    label="Weights repo (Hub model id)",
                    info="GVHMR checkpoint repo — default is the released weights.",
                )
                hub_revision_in = gr.Textbox(
                    label="Hub revision (optional)",
                    placeholder="main, a tag, or commit hash — blank = default branch",
                )
                detector_in = gr.Dropdown(
                    choices=DETECTOR_CHOICES,
                    value="yolo",
                    allow_custom_value=True,
                    label="Detector",
                    info="All YOLO presets from gvhmr config (`yolo` = yolov8x). Custom weights via name or .pt stem.",
                )
                pose2d_in = gr.Dropdown(
                    choices=POSE2D_CHOICES,
                    value="vitpose",
                    label="2D pose",
                    info="`rtmpose` (RTMPose-m via rtmlib/ONNX) auto-installs on first use; ONNX weights download then too.",
                )
                backbone_in = gr.Dropdown(
                    choices=BACKBONE_CHOICES,
                    value="hmr2",
                    label="Feature backbone",
                    info="`dinov2` only works with a GVHMR checkpoint trained on DINOv2 features.",
                )

            run_btn = gr.Button("Recover motion", variant="primary")

        with gr.Column():
            overlay_out = gr.Video(label="Overlay (in-camera | world)")
            npz_out = gr.File(label="SMPL parameters (.npz)")
            summary_out = gr.JSON(label="Summary")

    static_in.change(_toggle_camera, inputs=static_in, outputs=camera_in)
    run_btn.click(
        run,
        inputs=[
            video_in,
            static_in,
            camera_in,
            flip_in,
            hub_repo_in,
            hub_revision_in,
            detector_in,
            pose2d_in,
            backbone_in,
            smplx_in,
            render_in,
        ],
        outputs=[overlay_out, npz_out, summary_out],
        show_progress="full",
    )


if __name__ == "__main__":
    # ssr_mode=False is load-bearing on HF Spaces: with SSR on, the browser uploads inputs to the HF Xet
    # CDN and passes a signed URL; gradio then re-fetches it server-side via an *unauthenticated* SSRF-
    # protected GET that 403s (`Failed to download file. Status code: 403`), so uploads never reach the
    # pipeline. Forcing it off here makes uploads go to the Space's own /gradio_api/upload (a local path),
    # deterministically — not reliant on the GRADIO_SSR_MODE env var applying before the container starts.
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        ssr_mode=False,
    )
