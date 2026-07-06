---
title: GVHMR — Human Motion Recovery
emoji: 🏃
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: other
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

> **Space owners:** the body models are registration-gated and **never bundled**. GVHMR auto-fetches
> them on first run (mirror first, then MPI). **SMPL and SMPL-X are separate MPI registrations.**
>
> **Recommended — private mirror** (no MPI login in the Space runtime): on your laptop, register at
> [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) + [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de),
> run `gvhmr auth smpl`, then `gvhmr body-models push you/gvhmr-body-models` to a **private** HF repo.
> Set Space **Secrets**: `GVHMR_BODY_MODELS_MIRROR=you/gvhmr-body-models` and `HF_TOKEN` (read access).
>
> **Alternative — MPI at runtime:** set `SMPLX_USER` / `SMPLX_PW` (required) and, for mesh overlays,
> `SMPL_USER` / `SMPL_PW` from separate [SMPL-X](https://smpl-x.is.tue.mpg.de) /
> [SMPL](https://smpl.is.tue.mpg.de) logins.
>
> A **GPU** is strongly recommended — CPU inference on a short clip can take many minutes.

Source, documentation, and the full CLI: <https://github.com/ryanrudes/gvhmr>.
