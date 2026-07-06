---
title: GVHMR — Human Motion Recovery
emoji: 🏃
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
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

> **Space owners:** the SMPL/SMPL-X body models are registration-gated and are auto-fetched
> from MPI at run time. You must set the `SMPLX_USER` and `SMPLX_PW` repo **Secrets** to your
> own login at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) so the body models can be
> downloaded. A **GPU** is strongly recommended — CPU inference on a short clip can take many
> minutes.

Source, documentation, and the full CLI: <https://github.com/ryanrudes/gvhmr>.
