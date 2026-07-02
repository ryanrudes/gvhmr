# Configuring & swapping models

GVHMR's preprocessing pipeline is **modular**: the detector, 2D-pose estimator, feature backbone, and
camera backend are each a **Hydra config group** you can swap by name, bundle into a shareable *recipe*,
or tweak with a raw override — one config system, shared with `gvhmr train`. No forked YAML, no editing
call sites.

```
video ─► detector ─► pose2d ─► backbone ─► camera ─► motion recovery ─► render
         (YOLO)     (ViTPose)  (HMR2 ViT)  (SimpleVO)
          ▲          ▲          ▲           ▲
          └──────────┴──────────┴───────────┴─  each a config group under gvhmr/configs/
```

## Your config file — one readable place (`gvhmr config`)

Machine-local settings — **where large assets live**, **which model version each stage uses by default**,
and **the recorded Python environment** — go in one readable TOML file instead of scattered env vars. It
lives **inside the repository** by default (`<repo>/gvhmr.toml`, gitignored). Set it up interactively, or
edit it by hand:

```bash
gvhmr config init                       # wizard: asset folders + default models + managed env → writes the file
gvhmr config show                       # table of every setting, its value, and where it came from
gvhmr config set camera vggt            # change one thing non-interactively (validated against the options)
gvhmr config set checkpoints /vol/gvhmr/ckpts
gvhmr config set torch cu126            # env fields too — apply with `gvhmr env sync`
```

Lookup order: **`$GVHMR_CONFIG`** (when set it is authoritative — that exact file or nothing, no
fallback) → `./gvhmr.toml` (current directory) → `<repo>/gvhmr.toml` → `~/.config/gvhmr/config.toml`
(legacy). `gvhmr config` warns if it ever writes to a location outside this chain.

The file is **self-documenting** — every model line lists all its options:

```toml
[paths]
checkpoints = '/vol/gvhmr/checkpoints'   # model checkpoints (gvhmr, hmr2, vitpose, yolo, dpvo)
data        = '/vol/gvhmr/data'          # training/eval data packs (<DS>/hmr4d_support)
body_models = '/vol/gvhmr/body_models'   # SMPL / SMPL-X body models (registration-gated)
scene       = '/vol/gvhmr/scene'         # DUSt3R / Depth-Anything scene-camera weights

[models]
# detector — ultralytics YOLO. Pick a preset (default `yolo` = yolov8x):
#   yolov8   yolov8n yolov8s yolov8m yolov8l yolov8x
#   …        (families v8 / v9 / v10 / 11 / 12 / 26, each n<s<m<l<x — accuracy up, speed down)
#   yolo26   yolo26n yolo26s yolo26m yolo26l yolo26x   (latest, NMS-free)
# any other or newer weight: --detector-ckpt <name>.pt
detector = 'yolo'

# pose2d — must emit COCO-17. Options:
#   rtmpose  RTMPose-m — needs `uv sync --extra rtmpose`
#   vitpose  ViTPose-Huge — released default
pose2d = 'vitpose'

# camera — options: simplevo (default), dpvo (CUDA), dust3r, vggt (scene-aware metric)
camera = 'simplevo'

[env]
# torch — the torch backend for this box: none (PyPI wheel: macOS/MPS), cpu, or cu124/cu126/cu128
torch = 'cu128'
# extras — comma-separated install extras `gvhmr env sync` applies (preproc = the demo's models)
extras = 'preproc'
# dpvo — 'true' when DPVO (CUDA SLAM) is installed out-of-band by scripts/setup_dpvo.sh
dpvo = 'false'
```

(`gvhmr config init` generates the full file — every stage's options listed above its value.)

`gvhmr download` fetches into `[paths]`; `gvhmr demo` reads `[models]` as its defaults; **`gvhmr env
sync`** replays `[env]` through uv (`--inexact`, so nothing is ever pruned) — after the installer or
wizard records it once, you never run uv by hand. **Precedence for everything is: env var / CLI flag >
this file > built-in default** — the file is your baseline, and a one-off flag or `$GVHMR_*` env var
still wins (`gvhmr config show` reveals the source of each value, so a stray env var is never a mystery).

## Overriding per run (CLI)

These override the config file for a single invocation.

**1. Name flags — pick an implementation (the 90% case):**
```bash
gvhmr demo clip.mp4 --detector yolo26x --pose2d rtmpose --camera dust3r
```

**2. Recipes — a committable bundle of choices (shareable, reproducible):**
```bash
gvhmr demo clip.mp4 --recipe accurate       # flip-test TTA + full-res overlays
gvhmr demo clip.mp4 --recipe scene          # scene-aware metric camera (DUSt3R)
```
A recipe is one file in [`gvhmr/configs/recipe/`](../gvhmr/configs/recipe). It can set fields *and* select
groups — e.g. `scene.yaml` selects `camera: dust3r`. Write your own by dropping a file there:
```yaml
# gvhmr/configs/recipe/my_setup.yaml
# @package _global_
defaults:
  - override /pose2d: rtmpose
  - override /camera: dust3r
flip_test: true
```
`gvhmr demo clip.mp4 --recipe my_setup`.

**3. Raw overrides — tweak any knob (power users):**
```bash
gvhmr demo clip.mp4 --set detector.conf=0.4 --set backbone.model_name=dinov2_vitl14
```

Precedence (later wins): **recipe → name flag → `--set`**. So `--pose2d vitpose` overrides a recipe's
pose choice, and a `--set` overrides everything.

## What you can swap

| Stage | Flag | Options today | Swap needs |
|---|---|---|---|
| **Detector** | `--detector` | `yolo` (=yolov8x) + every family×size preset (`yolo26x`, …) | nothing — any weight drops in |
| **2D pose** | `--pose2d` | `vitpose`, `rtmpose` | nothing — **must emit COCO-17** |
| **Camera** | `--camera` | `simplevo`, `dpvo` (CUDA), `dust3r` / `vggt` (scene-aware, metric) | nothing |
| **Feature backbone** | `--backbone` | `hmr2` (released), `dinov2` | **a retrain** (see below) |
| **Body model** | config `[paths].body_models` | SMPL / SMPL-X file | file-swap within topology only |

- **Detector:** every YOLO family × size (v8/v9/v10/11/12/26 × n/s/m/l/x) is a preset — `--detector yolo26x`
  auto-downloads `yolo26x.pt` (ultralytics). Any other or newer weight: `--detector-ckpt yolo27x.pt`. The
  presets are generated by `scripts/gen_detector_presets.py` (add a family + re-run when a new one ships).
- **2D pose:** `rtmpose` needs the extra — `uv sync --extra rtmpose` (rtmlib + ONNXRuntime, ungated). Any
  estimator emitting **COCO-17** `(F,17,3)` fits the slot; the network asserts `J==17`.
- **Camera:** `dpvo` is CUDA-only; on Apple-Silicon/CPU use a scene-aware **metric** camera — `dust3r`
  or `vggt` (both recover world translation; VGGT is a single feed-forward pass, often faster/more
  robust). Set them up with `scripts/setup_scene_aware.sh` (VGGT weights auto-download). `--slam` /
  `--use-dpvo` are deprecated aliases for `--camera`.
- **Backbone:** the released checkpoint's `imgseq_embedder` is fit to HMR2's 1024-d features, so a different
  backbone is **not** a drop-in at inference — it needs a retrain (next section).

## Swapping the feature backbone (needs a retrain)

The backbone feeds the trained core, so a swap = re-extract features + retrain. The tooling makes this a
short loop (see [`docs/EXTENSIBILITY.md`](EXTENSIBILITY.md) Tier B, [`docs/TRAINING.md`](TRAINING.md)):

```bash
# 1) extract features with the new backbone into the training-cache format
gvhmr extract-features VIDEOS_DIR inputs/3DPW/hmr4d_support/imgfeats/3dpw_train_dinov2 \
    --backbone dinov2 --bbx-from inputs/3DPW/hmr4d_support/imgfeats/3dpw_train_smplx_refit

# 2) train the small core on them (imgseq_dim must match the backbone's feature width)
gvhmr train exp=gvhmr/mixed/smoke_3dpw_dinov2      # sets network.imgseq_dim=384 for dinov2 vits14
```
`extract-features` resolves the backbone through the **same `configs/backbone` group** as the demo, so the
feature width stays consistent across extraction, the demo, and the training config.

## Where the configs live

```
gvhmr/configs/
├── detector/   yolo.yaml + yolov8/9/10/11/12/26 × n/s/m/l/x  (gen_detector_presets.py)
├── pose2d/     vitpose.yaml  rtmpose.yaml
├── backbone/   hmr2.yaml  dinov2.yaml
├── camera/     simplevo.yaml  dpvo.yaml  dust3r.yaml  vggt.yaml
└── recipe/     hq.yaml  accurate.yaml  scene.yaml
```

Each stage file is a small node: a `name` (which implementation) plus that implementation's ctor knobs.
Defaults are the released models, so the default `gvhmr demo` path is **byte-identical** to before this
config surface existed (golden-guarded). `gvhmr demo --help` lists every flag; `gvhmr info` shows what's
installed.
