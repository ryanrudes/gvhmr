"""HuggingFace Space (Gradio) for GVHMR — World-Grounded Human Motion Recovery.

Upload a video → recover SMPL human motion → get a side-by-side (in-camera + world)
mesh overlay video plus an .npz of the SMPL parameters.

The pipeline is loaded once at module import and cached in a global. Body models
(SMPL/SMPL-X) are registration-gated and auto-fetched from MPI using the
`SMPLX_USER` / `SMPLX_PW` env vars — on a Space these are set as repo Secrets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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

> **A GPU is strongly recommended** — on CPU a short clip can take many minutes.
"""

# --------------------------------------------------------------------------------------
# Pipeline: load once, guarded. A failure here (missing extras, etc.) must not take the
# UI down — we surface it as a banner and let each request report a friendly error.
# --------------------------------------------------------------------------------------
_PIPE = None
_LOAD_ERROR: str | None = None


def _load_pipeline() -> None:
    """Populate the module-global pipeline, capturing any load error as a string."""
    global _PIPE, _LOAD_ERROR
    try:
        import gvhmr

        _PIPE = gvhmr.pipeline("human-motion-recovery", model="ryanrudes/gvhmr", device=None)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI, keep it up
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"


_load_pipeline()


def _is_credentials_error(exc: Exception) -> bool:
    """Heuristic: does this exception look like missing gated-body-model credentials?"""
    text = f"{type(exc).__name__} {exc}".lower()
    needles = ("smplx_user", "smplx_pw", "body model", "body_model", "credential", "registration", "mpi")
    return any(n in text for n in needles)


def _seconds_of_motion(num_frames: int, fps: float) -> float:
    return round(num_frames / fps, 2) if fps else 0.0


# --------------------------------------------------------------------------------------
# Inference callback
# --------------------------------------------------------------------------------------
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

    # A static camera means we skip the scene-aware camera backend entirely.
    camera = None if static_camera else camera_backend

    # Keep outputs for the whole session; a per-call temp dir avoids collisions.
    out_dir = Path(tempfile.mkdtemp(prefix="gvhmr_"))

    try:
        progress(0.0, desc="Recovering motion…")
        result = _PIPE(
            video_path,
            static_camera=static_camera,
            camera=camera,
            flip_test=flip_test,
            render=True,
            progress=True,
            output_dir=str(out_dir),
        )

        progress(0.7, desc="Rendering overlay…")
        overlay_path = result.render(out_dir / "overlay_both.mp4", view="both")

        progress(0.9, desc="Saving SMPL params…")
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
                "SMPL/SMPL-X body models are gated and could not be fetched. "
                "The Space owner must set the SMPLX_USER and SMPLX_PW secrets to their own "
                "MPI (smpl-x.is.tue.mpg.de) login so the body models can be downloaded. "
                f"Details: {type(exc).__name__}: {exc}"
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
    demo.launch()
