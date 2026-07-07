"""HuggingFace Space (Gradio + ZeroGPU) for GVHMR — World-Grounded Human Motion Recovery.

Upload a video → recover SMPL human motion → get a side-by-side (in-camera + world)
mesh overlay video plus an .npz of the SMPL parameters.

Uses Hugging Face **ZeroGPU** (free shared GPU for visitors; the Space owner selects the
ZeroGPU hardware flavor in Settings — requires a PRO/Team/Enterprise plan to host). Inference
runs inside ``@spaces.GPU``; the pipeline loads lazily on first request (inside the GPU decorator).

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

# ZeroGPU: import `spaces` before any torch/CUDA import (HF requirement).
import spaces
import gradio as gr

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
POSE2D_CHOICES = ["vitpose", "rtmpose"]
BACKBONE_CHOICES = ["hmr2", "dinov2"]
CAMERA_CHOICES = ["simplevo", "dpvo", "dust3r", "vggt"]

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
        _pip_install("gvhmr[preproc]>=1.0.3")

    from gvhmr.utils._smpl_compat import apply

    apply()


_BOOTSTRAPPED = False


def _ensure_bootstrapped() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _bootstrap_deps()
    _BOOTSTRAPPED = True


# --------------------------------------------------------------------------------------
# Pipeline cache: lazy load inside @spaces.GPU (keyed by hub repo + stage presets).
# --------------------------------------------------------------------------------------
_PIPES: dict[tuple, Any] = {}
_PIPE_ERRORS: dict[tuple, str] = {}


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
    if key in _PIPE_ERRORS:
        raise RuntimeError(_PIPE_ERRORS[key])
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
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI, keep it up
        msg = f"{type(exc).__name__}: {exc}"
        _PIPE_ERRORS[key] = msg
        traceback.print_exc()
        raise RuntimeError(msg) from exc


def _is_credentials_error(exc: Exception) -> bool:
    """Heuristic: does this exception look like missing gated-body-model credentials?"""
    text = f"{type(exc).__name__} {exc}".lower()
    needles = (
        "smplx_user",
        "smplx_pw",
        "body model",
        "body_model",
        "credential",
        "registration",
        "mpi",
        "registration-gated",
        "smpl-x body model",
        "mirror",
        "401",
        "403",
        "permissionerror",
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
    """ZeroGPU queue budget — HF free tier defaults to 60 s per ``@spaces.GPU`` call."""
    cap = _zero_gpu_duration_cap()
    if cap <= 60:
        return 60

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
    if not static_camera and camera_backend in ("dust3r", "vggt"):
        per_sec *= 1.6
    bootstrap = 35  # cold-start pip/bootstrap + checkpoint load (first call)
    return int(min(cap, max(60, bootstrap + secs * per_sec)))


# --------------------------------------------------------------------------------------
# Inference callback (GPU allocated for the duration of this function on ZeroGPU Spaces)
# --------------------------------------------------------------------------------------
@spaces.GPU(duration=_estimate_gpu_duration)
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
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008
) -> tuple[str, str, dict]:
    """Run GVHMR on the uploaded video and return (overlay_mp4, npz_path, summary)."""
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

    def _inference_hook(frac: float, desc: str) -> None:
        # Reserve 0–5% for model load; pipeline reports 0–1 over the remaining 88%.
        _hook(0.05 + max(0.0, min(1.0, frac)) * 0.88, desc)

    try:
        progress(0.0, desc="Preparing environment…")
        _ensure_bootstrapped()
        from gvhmr.utils.console import progress_hook, progress_phase

        progress(0.01, desc="Starting…")
        with progress_hook(_hook):
            with progress_phase(0.0, 0.05, "Loading model"):
                try:
                    pipe = _get_pipeline(
                        model_repo=hub_repo,
                        revision=hub_revision,
                        detector=detector,
                        pose2d=pose2d,
                        backbone=backbone,
                    )
                except RuntimeError as exc:
                    raise gr.Error(
                        f"The GVHMR pipeline failed to load, so inference is unavailable. Details: {exc}"
                    ) from exc

            camera = None if static_camera else camera_backend
            out_dir = Path(tempfile.mkdtemp(prefix="gvhmr_"))

            result = pipe(
                video_path,
                static_camera=static_camera,
                camera=camera,
                flip_test=flip_test,
                render=False,
                progress=False,
                progress_callback=_inference_hook,
                output_dir=str(out_dir),
            )

            with progress_phase(0.93, 0.98, "Rendering overlay"):
                overlay_path = result.render(out_dir / "overlay_both.mp4", view="both")

            with progress_phase(0.98, 0.995, "Saving SMPL params"):
                npz_path = result.save_npz(out_dir / "motion.npz")

            summary = {
                "frames": result.num_frames,
                "fps": result.fps,
                "camera": result.camera,
                "seconds_of_motion": _seconds_of_motion(result.num_frames, result.fps),
                "weights_repo": hub_repo.strip(),
                "weights_revision": (hub_revision or "").strip() or "default",
                "detector": detector,
                "pose2d": pose2d,
                "backbone": backbone,
            }
        progress(1.0, desc="Done")
        return str(overlay_path), str(npz_path), summary

    except gr.Error:
        raise
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


with gr.Blocks(title="GVHMR — Human Motion Recovery") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video", sources=["upload"], format="mp4")
            static_in = gr.Checkbox(value=False, label="Static camera")
            camera_in = gr.Dropdown(
                choices=CAMERA_CHOICES,
                value="simplevo",
                label="Camera backend",
                info="Ignored when 'Static camera' is checked. dpvo needs compiled DPVO (not on this Space).",
            )
            flip_in = gr.Checkbox(value=False, label="Flip-test (slower, more accurate)")

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
                    allow_custom_value=True,
                    label="2D pose",
                    info="`rtmpose` needs the optional rtmlib extra (not installed here).",
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
        ],
        outputs=[overlay_out, npz_out, summary_out],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
