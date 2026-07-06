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

> **Space owners:** the body models are registration-gated and auto-fetched from MPI at run time.
> **SMPL and SMPL-X are separate registrations**, each with its own login. Set the `SMPLX_USER` /
> `SMPLX_PW` repo **Secrets** from your [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) login
> (required for motion recovery) and, to enable the mesh-overlay video, the `SMPL_USER` / `SMPL_PW`
> Secrets from a *separate* [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) login. A **GPU** is
> strongly recommended — CPU inference on a short clip can take many minutes.

Source, documentation, and the full CLI: <https://github.com/ryanrudes/gvhmr>.
