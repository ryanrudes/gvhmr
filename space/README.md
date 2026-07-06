---
title: GVHMR — Human Motion Recovery
emoji: 🏃
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 30m
pinned: false
license: other
short_description: World-grounded SMPL motion from a single video
---

# GVHMR — World-Grounded Human Motion Recovery

GVHMR recovers **SMPL** human motion from a single video, in both the camera frame and a
gravity-aligned **world** frame — *World-Grounded Human Motion Recovery via Gravity-View
Coordinates* (SIGGRAPH Asia 2024).

Upload a video of a person and the Space returns a side-by-side mesh overlay (in-camera on
the left, world on the right) plus an `.npz` file containing the recovered SMPL parameters
(`global_orient`, `body_pose`, `betas`, `transl`) for every frame. You can pick a scene-aware
camera backend (`simplevo`, `dust3r`, `vggt`) for moving cameras, or check *Static camera* for
a fixed viewpoint.

> **Space owners:** this Space targets **ZeroGPU** (free shared GPU for visitors — you must select
> the **ZeroGPU** hardware flavor under Space Settings; hosting ZeroGPU requires
> [HF PRO / Team / Enterprise](https://huggingface.co/pricing)). Docker Spaces cannot use ZeroGPU.
> Each `@spaces.GPU` call gets **60 seconds of GPU time by default** on the free tier; optional Secret
> `ZERO_GPU_DURATION=600` (seconds) raises the per-call cap for longer clips on PRO.
>
> **Body models** are registration-gated and never bundled. **Recommended — private mirror:** on your
> laptop, register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) +
> [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de), run `gvhmr auth smpl`, then
> `gvhmr body-models push you/gvhmr-body-models`. Set Space **Secrets**:
> `GVHMR_BODY_MODELS_MIRROR=you/gvhmr-body-models` and `HF_TOKEN` (read access).
>
> **Alternative — MPI at runtime:** `SMPLX_USER` / `SMPLX_PW` (required) and, for mesh overlays,
> `SMPL_USER` / `SMPL_PW` from separate logins.
>
> A legacy **CPU Docker** build (`space/Dockerfile`) remains for local/offline testing without ZeroGPU.

Source, documentation, and the full CLI: <https://github.com/ryanrudes/gvhmr>.
