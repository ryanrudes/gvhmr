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
(`global_orient`, `body_pose`, `betas`, `transl`) for every frame. Check *Static camera* for a
fixed viewpoint, or uncheck it to use the `simplevo` camera backend for a moving camera. (The
scene-aware backends `dust3r`/`vggt` aren't set up on this Space — use the full CLI for those.)

> **Space owners:** this Space targets **ZeroGPU** (free shared GPU for visitors — you must select
> the **ZeroGPU** hardware flavor under Space Settings; hosting ZeroGPU requires
> [HF PRO / Team / Enterprise](https://huggingface.co/pricing)). Docker Spaces cannot use ZeroGPU.
> Each `@spaces.GPU` call gets **60 seconds of GPU time by default** on the free tier; optional Secret
> `ZERO_GPU_DURATION=600` (seconds) raises the per-call cap for longer clips on PRO.
>
> **Torch:** ZeroGPU preinstalls `torch==2.11.0` (builder adds `torch<=2.11.0`). Do **not** pin
> torch/torchvision in `requirements.txt`.
>
> **gvhmr / chumpy:** also **not** in `requirements.txt` — `chumpy` cannot build during the Space
> image build (PEP 517 isolation). `app.py` installs `chumpy` with `--no-build-isolation`, then
> `gvhmr[preproc]`, on the first inference request.
>
> **SSR / uploads:** Gradio 6 SSR re-fetches uploads over HTTP and often returns **403** on Spaces.
> `gvhmr publish-space` sets Space variable **`GRADIO_SSR_MODE=false`** (HF ignores `launch(ssr_mode=…)`).
> After changing it, republish or restart the Space so the runtime picks it up.
>
> **Body models (recommended — private mirror):** the gated SMPL/SMPL-X models aren't shipped. From your
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

The Space UI exposes **Model settings** with every swappable stage preset from `gvhmr config`
(detector / 2D-pose / backbone / camera + Hub weights repo/revision). Some choices need extras
this Space does not install (`dpvo`, `dust3r`/`vggt`, `dinov2` without a matching checkpoint). `rtmpose`
lazy-installs `rtmlib` + `onnxruntime` on first use. Space owners
can set `GVHMR_HUB_REPO` / `GVHMR_HUB_REPO_OPTIONS` Secrets to pre-seed extra repo choices.
