# The GVHMR Python library

A small, HuggingFace-style Python API over the `gvhmr demo` pipeline: feed it a video, get back
SMPL/SMPL-X human motion in the **camera** and **world** frames as a rich `MotionResult` object.
It lives in `gvhmr/inference/` and reuses the *exact* code paths behind the CLI, so its numbers are
byte-identical to `gvhmr demo` (the golden test guards this). Prefer the shell? See the
[CLI usage](../README.md#usage) and [docs/CONFIGURATION.md](CONFIGURATION.md).

```python
import gvhmr

result = gvhmr.recover("dance.mp4")            # video → MotionResult
result.render("overlay.mp4")                   # in-cam ∥ world overlay video
result.save_npz("dance.npz")                   # portable SMPL params + intrinsics
```

## Install

```bash
pip install gvhmr                    # base: import + model + mesh rendering (CPU / Apple-Silicon MPS)
pip install "gvhmr[preproc]"         # + YOLO / ViTPose / pycolmap — REQUIRED to run on a video
pip install "gvhmr[preproc,cu128]"   # Linux + CUDA: also pick a torch build (cu124 / cu126 / cu128)
pip install "gvhmr[app]"             # + the gradio demo app
```

- **`preproc` is required to run the pipeline on a video** (it provides the detector, 2D-pose, and
  camera stages). The base install can still load the model, hold a `MotionResult`, and render meshes.
- **Linux + CUDA:** pick the torch extra nearest to (but not above) your driver's CUDA version —
  `cu124` / `cu126` / `cu128` (cu128 covers 12.8–13.x; **V100/P100 need cu126**). **macOS** uses the
  default MPS wheel — do *not* add a `cuXXX` extra.
- **Device** is auto-selected (cuda → mps → cpu). Override per call with `device=` or globally with
  `$GVHMR_DEVICE`.

## Body models & `gvhmr auth smpl`

Motion recovery renders meshes and joints from the **SMPL/SMPL-X body models**, which are
registration-gated by the Max Planck Institute (MPI) and **licensed for non-commercial research with no
redistribution**. They are therefore **never bundled with the package or re-hosted on the Hub**. Only
the GVHMR/HMR2/ViTPose/YOLO *checkpoints* come from the Hub repo (with fallback to the community mirror
`camenduru/GVHMR`).

Instead, GVHMR fetches the body models from the **official MPI source using your own account** — exactly
what the official `smplx`/SMPLify-X scripts do. Register once at
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/),
then either:

```bash
gvhmr auth smpl                      # prompts for your MPI email + password; stored 0600, fetches now
```

or set environment variables (handy for CI / containers):

```bash
export SMPLX_USER="you@example.com"
export SMPLX_PW="..."
```

The models are downloaded and placed on first use (or verified immediately by `gvhmr auth smpl`).
Without credentials, the library falls back to printing the manual sign-up + target-path instructions.
`gvhmr info` shows what's present and where.

## Quickstart

### One-liner

`gvhmr.recover(...)` recovers motion from a single video. It caches a default pipeline per
`(model, device)`, so a loop of videos loads the weights only once.

```python
import gvhmr

result = gvhmr.recover("dance.mp4")                 # moving camera (SimpleVO), world_from_incam
result = gvhmr.recover("clip.mp4", static_camera=True)   # static camera — faster, cleaner world frame

for path in ["a.mp4", "b.mp4", "c.mp4"]:            # weights load once, reused across the loop
    r = gvhmr.recover(path)
    print(r)                                        # MotionResult(frames=..., fps=30, camera='simplevo')
```

`import gvhmr` stays light — torch/hydra are only imported the first time you touch the API.

### Reusable pipeline

For explicit control over when weights load (and to share them across many calls), build a
`GVHMRPipeline` once and call it:

```python
import gvhmr

pipe = gvhmr.pipeline("human-motion-recovery", device="cuda")   # loads weights once
result = pipe("dance.mp4", flip_test=True, render=True)

# task aliases: "human-motion-recovery" | "hmr" | "motion-recovery" | "smpl" | "video-to-smpl"
# pipe.recover(...) is an alias for pipe(...)
```

`pipeline(...)` forwards `model=`, `device=`, and stage defaults (`detector=`, `pose2d=`, `backbone=`,
`camera=`) to `GVHMRPipeline.from_pretrained`.

## The `MotionResult`

Every entrypoint returns a `MotionResult` — the recovered motion for one video in both frames, with lazy
accessors for meshes, joints, files, and overlay videos. **`world`** = gravity-aligned world/global
frame (the paper's global frame); **`camera`** = in-camera frame. `L` is the number of frames.

| Member | Type | What |
|---|---|---|
| `.smpl_params_world` | `dict` | world-frame SMPL params: `global_orient (L,3)`, `body_pose (L,63)`, `betas (L,10)`, `transl (L,3)` |
| `.smpl_params_camera` | `dict` | same keys, in the camera frame |
| `.intrinsics` | `(L,3,3)` | per-frame pinhole `K` (full-image pixels) |
| `.fps` | `float` | frames per second (GVHMR runs at `30.0`) |
| `.camera` | `str` | backend used (`simplevo` / `dpvo` / `dust3r` / `vggt` / `static`) |
| `.video_path` | `Path \| None` | the staged 30fps input video |
| `.output_dir` | `Path \| None` | where the pipeline staged artifacts |
| `.num_frames` / `len(result)` | `int` | number of frames |
| `.vertices_world` / `.vertices_camera` | `(L,6890,3)` | SMPL mesh vertices *(lazy — needs body models)* |
| `.joints_world` / `.joints_camera` | `(L,24,3)` | 24 SMPL joints *(lazy — needs body models)* |
| `.faces` | `(13776,3)` | SMPL triangle faces (shared across frames) |
| `.to_dict()` | `dict` | plain dict of the friendly tensor fields |
| `.save(path)` | `Path` | full raw prediction as a `.pt` — the exact `hmr4d_results` format `gvhmr demo` writes |
| `.save_npz(path)` | `Path` | portable `.npz` (numpy; no torch needed to read) |
| `.render(path=None, *, view, skeleton, skeleton_overlay, skeleton_joints, render_scale)` | `Path` | overlay video *(needs staged artifacts + SMPL body model)* |

Meshes, joints, and rendering resolve the gated body models on first access; if they're missing you'll
get an error pointing at `gvhmr auth smpl` — the motion itself is always available via
`.save()` / `.save_npz()` / the `smpl_params_*` dicts.

```python
result = gvhmr.recover("dance.mp4")

joints = result.joints_world               # (L, 24, 3) world-frame joints
verts  = result.vertices_camera            # (L, 6890, 3) in-camera mesh

result.save_npz("dance.npz")               # world_/camera_ params + intrinsics, fps, camera
result.save("dance.pt")                    # round-trips with the rest of the CLI toolchain

result.render("overlay.mp4")               # side-by-side in-cam ∥ world (view="both", default)
result.render("world.mp4", view="world")   # world-frame mesh only
result.render("skel.mp4", skeleton_overlay=True, skeleton_joints="legs,left_arm")
```

`.render()` needs the on-disk artifacts the pipeline stages, so it only works on a `MotionResult`
produced by `pipeline(...)(video)` / `gvhmr.recover(...)` (not one you build from raw tensors).
`view` is `"both"` | `"world"` | `"camera"`; pass no `path` to get back the path of the staged video.

## Cameras, static vs moving, and accuracy levers

The pipeline `__call__` (and `gvhmr.recover`, which forwards to it) takes:

```python
result = pipe(
    "clip.mp4",
    static_camera=False,     # True → skip visual odometry (faster, cleaner world frame)
    camera="vggt",           # moving-camera backend: simplevo | dpvo | dust3r | vggt
    detector=None,           # per-call stage overrides (else pipeline/config defaults)
    pose2d=None,
    backbone=None,
    f_mm=35,                 # true full-frame focal in mm (else metadata, else a FOV heuristic)
    flip_test=True,          # mirror-averaging TTA — the benchmark setting (+1 feature pass)
    world_from_incam=True,   # static camera: take the world trajectory from in-cam motion
    render=False,            # render overlays now, else call result.render() later
    output_dir=None,         # where to stage artifacts (reusable cache)
    render_scale=0.5,        # overlay resolution fraction
    recipe=None,             # a committable config recipe (advanced)
    set_overrides=None,      # raw Hydra overrides, e.g. ["detector.conf=0.4"] (advanced)
    progress=True,           # Rich progress display — set False for quiet library use
)
```

**Static vs moving.** `static_camera=True` skips visual odometry — use it when the camera doesn't move.
For a moving camera, the default backend is **SimpleVO** (recovers rotation only). To also recover the
camera's **translation** (e.g. a following/tracking shot):

- `camera="dust3r"` or `camera="vggt"` — scene-aware **metric** cameras that run on any device
  (Apple-Silicon MPS / CPU / CUDA). Set them up once with `scripts/setup_scene_aware.sh`.
- `camera="dpvo"` — classic DPVO SLAM, **CUDA-only** (`scripts/setup_dpvo.sh`).

**Accuracy levers** (byte-identical default path stays golden; these are opt-in):
`flip_test=True` (mirror-averaging TTA, the benchmark-time setting) and `f_mm=<mm>` (pass the true focal
length if you know it; phone metadata is read automatically). Evidence: [docs/ACCURACY.md](ACCURACY.md).

Stage/backbone swaps mirror the CLI's config groups — see [docs/CONFIGURATION.md](CONFIGURATION.md).

## The model "power path"

If you've already run (and cached) your own preprocessing, skip the pipeline and drive the trained model
directly. `GVHMR` wraps the `DemoPL` module (NetworkEncoderRoPE denoiser + EnDecoder), built exactly the
way the golden test / `gvhmr demo` build it.

```python
from gvhmr import GVHMR

model = GVHMR.from_pretrained("ryanrudes/gvhmr", device="cuda")

pred = model.predict(
    data,                    # tensor dict (bring your own preproc), keys below
    static_cam=False,
    flip_test_data=None,     # optional mirrored data dict for flip-test TTA
    world_from_incam=False,
)
```

`predict` is the tensor-level API. Its `data` dict carries the per-frame preprocessing:

| key | shape | what |
|---|---|---|
| `length` | scalar | number of frames `F` |
| `kp2d` | `(F, 17, 3)` | 2D keypoints (COCO-17) + confidence |
| `bbx_xys` | `(F, 3)` | person bbox center-x, center-y, scale |
| `K_fullimg` | `(F, 3, 3)` | per-frame camera intrinsics |
| `cam_angvel` | `(F, 6)` | camera angular velocity |
| `f_imgseq` | `(F, 1024)` | HMR2 ViT image features |

`predict` returns the raw `pred` dict (`smpl_params_global` / `smpl_params_incam` / `K_fullimg` /
`net_outputs`). For the full video→motion path with this model, `model.recover(video, **kwargs)` is a
convenience that runs a `GVHMRPipeline` under the hood. `model.to(device)` / `model.eval()` behave as
expected.

## Sharing models

Publish a fine-tune or re-host the released checkpoints (body models are **never** uploaded):

```python
model.save_pretrained("my-gvhmr/")                 # self-contained folder: ckpt + config.json + card
url = model.push_to_hub("me/gvhmr-ft", private=True)   # push ckpt + generated model card to the Hub
```

From the CLI:

```bash
gvhmr publish-hub                    # upload the released checkpoints + model card to ryanrudes/gvhmr
gvhmr publish-hub me/gvhmr --private --names gvhmr,hmr2 --dry-run
gvhmr publish-space me/gvhmr-demo    # push the bundled gradio Space under space/
```

`from_pretrained` resolves weights from the Hub repo (default `ryanrudes/gvhmr`), transparently falling
back to the community mirror `camenduru/GVHMR` until your own repo is published. Pass `ckpt_path=` to
load a local checkpoint and skip the download.

## Device & performance

Device is auto (cuda → mps → cpu); set `device=` per call/pipeline or `$GVHMR_DEVICE` globally. Note the
GVHMR model's `predict` is actually **faster on CPU** than MPS (it's latency-bound on the IK's small
ops), while MPS accelerates the batched preprocessing models — so the fastest end-to-end setup often
mixes the two. Full profiling notes: [docs/PERFORMANCE.md](PERFORMANCE.md).

## See also

- [CLI usage](../README.md#usage) and [docs/CONFIGURATION.md](CONFIGURATION.md) — the `gvhmr` command,
  config file, and swapping every pipeline stage.
- [docs/EVAL.md](EVAL.md) — reproduce the paper benchmarks (`gvhmr eval`).
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — the code map, data flow, and the 151-dim latent.
- [docs/ACCURACY.md](ACCURACY.md) / [docs/PERFORMANCE.md](PERFORMANCE.md) — accuracy levers and latency.
