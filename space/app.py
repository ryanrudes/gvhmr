"""HuggingFace Space (Gradio + ZeroGPU) for GVHMR — World-Grounded Human Motion Recovery.

Upload a video → recover SMPL human motion → get a side-by-side (in-camera + world)
mesh overlay video plus an .npz of the SMPL parameters.

Uses Hugging Face **ZeroGPU** (free shared GPU for visitors; the Space owner selects the
ZeroGPU hardware flavor in Settings — requires a PRO/Team/Enterprise plan to host). Inference
runs inside ``@spaces.GPU``; the pipeline is loaded once at startup on ``cuda``.

Body models: configure a private mirror (`GVHMR_BODY_MODELS_MIRROR` + `HF_TOKEN`) or MPI
Secrets — see space/README.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ZeroGPU: import `spaces` before any torch/CUDA import (HF requirement).
import spaces
import gradio as gr

REPO_URL = "https://github.com/ryanrudes/gvhmr"

# Camera backends shown in the UI. DPVO is CUDA-only and intentionally omitted.
CAMERA_BACKENDS = ["simplevo", "dust3r", "vggt"]

DESCRIPTION = f"""
# GVHMR — World-Grounded Human Motion Recovery

Upload a video of a person and GVHMR recovers their **SMPL** body motion in both the
**camera** frame and a gravity-aligned **world** frame. You get a side-by-side mesh
overlay video and an `.npz` of the SMPL parameters.

*World-Grounded Human Motion Recovery via Gravity-View Coordinates* (SIGGRAPH Asia 2024).
Source & docs: [{REPO_URL}]({REPO_URL}).

> Runs on **ZeroGPU** when the Space owner has selected that hardware — much faster than CPU.
> Free HF accounts get a few minutes of GPU time per day; [HF PRO](https://huggingface.co/pricing)
> gets a larger daily quota.
"""


def _bootstrap_deps() -> None:
    """Install chumpy (no-build-isolation) and gvhmr before the heavy imports."""
    try:
        import chumpy  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-build-isolation",
                "numpy>=1.26",
                "chumpy==0.70",
            ]
        )
    try:
        import gvhmr  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gvhmr[preproc]>=1.0.1"])


_bootstrap_deps()

# --------------------------------------------------------------------------------------
# Pipeline: load once on cuda (ZeroGPU emulation outside @spaces.GPU, real GPU inside).
# --------------------------------------------------------------------------------------
_PIPE = None
_LOAD_ERROR: str | None = None


def _load_pipeline() -> None:
    """Populate the module-global pipeline, capturing any load error as a string."""
    global _PIPE, _LOAD_ERROR
    try:
        import gvhmr

        # ZeroGPU expects weights on cuda at module scope; override with GVHMR_DEVICE=cpu locally.
        device = os.getenv("GVHMR_DEVICE", "cuda")
        _PIPE = gvhmr.pipeline("human-motion-recovery", model="ryanrudes/gvhmr", device=device)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI, keep it up
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"


_load_pipeline()


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
    )
    return any(n in text for n in needles)


def _seconds_of_motion(num_frames: int, fps: float) -> float:
    return round(num_frames / fps, 2) if fps else 0.0


def _estimate_gpu_duration(
    video_path: str | None,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
) -> int:
    """ZeroGPU queue budget — scale with clip length and the heavier camera backends."""
    secs = 30.0
    if video_path:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and frames > 0:
                secs = float(frames / fps)
        except Exception:  # noqa: BLE001
            pass
    per_sec = 5.0 if flip_test else 3.5
    if not static_camera and camera_backend in ("dust3r", "vggt"):
        per_sec *= 1.6
    return int(min(600, max(120, 60 + secs * per_sec)))


# --------------------------------------------------------------------------------------
# Inference callback (GPU allocated for the duration of this function on ZeroGPU Spaces)
# --------------------------------------------------------------------------------------
@spaces.GPU(duration=_estimate_gpu_duration)
def run(
    video_path: str | None,
    static_camera: bool,
    camera_backend: str,
    flip_test: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 — gradio's documented DI pattern
) -> tuple[str, str, dict]:
    """Run GVHMR on the uploaded video and return (overlay_mp4, npz_path, summary)."""
    if _PIPE is None:
        raise gr.Error(
            f"The GVHMR pipeline failed to load, so inference is unavailable. Details: {_LOAD_ERROR or 'unknown error'}"
        )
    if not video_path:
        raise gr.Error("Please upload a video first.")

    camera = None if static_camera else camera_backend
    out_dir = Path(tempfile.mkdtemp(prefix="gvhmr_"))

    def _hook(frac: float, desc: str) -> None:
        progress(max(0.0, min(1.0, frac)), desc=desc)

    try:
        progress(0.0, desc="Starting…")
        result = _PIPE(
            video_path,
            static_camera=static_camera,
            camera=camera,
            flip_test=flip_test,
            render=False,
            progress=False,
            progress_callback=_hook,
            output_dir=str(out_dir),
        )

        progress(0.93, desc="Rendering overlay…")
        overlay_path = result.render(out_dir / "overlay_both.mp4", view="both")

        progress(0.98, desc="Saving SMPL params…")
        npz_path = result.save_npz(out_dir / "motion.npz")

        summary = {
            "frames": result.num_frames,
            "fps": result.fps,
            "camera": result.camera,
            "seconds_of_motion": _seconds_of_motion(result.num_frames, result.fps),
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

    if _PIPE is None:
        gr.Markdown(
            f"> ⚠️ **The pipeline failed to load — inference is disabled.** `{_LOAD_ERROR}`",
        )

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video", sources=["upload"])
            static_in = gr.Checkbox(value=False, label="Static camera")
            camera_in = gr.Dropdown(
                choices=CAMERA_BACKENDS,
                value="simplevo",
                label="Camera backend",
                info="Ignored when 'Static camera' is checked. dpvo is CUDA-only and not offered here.",
            )
            flip_in = gr.Checkbox(value=False, label="Flip-test (slower, more accurate)")
            run_btn = gr.Button("Recover motion", variant="primary")

        with gr.Column():
            overlay_out = gr.Video(label="Overlay (in-camera | world)")
            npz_out = gr.File(label="SMPL parameters (.npz)")
            summary_out = gr.JSON(label="Summary")

    static_in.change(_toggle_camera, inputs=static_in, outputs=camera_in)
    run_btn.click(
        run,
        inputs=[video_in, static_in, camera_in, flip_in],
        outputs=[overlay_out, npz_out, summary_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
