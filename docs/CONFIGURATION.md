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

## Three ways to configure

**1. Name flags — pick an implementation (the 90% case):**
```bash
gvhmr demo clip.mp4 --detector yolo11 --pose2d rtmpose --camera dust3r
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
| **Detector** | `--detector` | `yolo` (yolov8x), `yolo11` | nothing — any ultralytics weight drops in |
| **2D pose** | `--pose2d` | `vitpose`, `rtmpose` | nothing — **must emit COCO-17** |
| **Camera** | `--camera` | `simplevo`, `dpvo` (CUDA), `dust3r` / `vggt` (scene-aware, metric) | nothing |
| **Feature backbone** | `--backbone` | `hmr2` (released), `dinov2` | **a retrain** (see below) |
| **Body model** | `$GVHMR_BODY_MODELS` | SMPL / SMPL-X file | file-swap within topology only |

- **Detector:** `--detector yolo11` pulls `yolo11x.pt` automatically (ultralytics). Point at any weight with
  `--detector-ckpt path/to.pt`.
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
├── detector/   yolo.yaml  yolo11.yaml
├── pose2d/     vitpose.yaml  rtmpose.yaml
├── backbone/   hmr2.yaml  dinov2.yaml
├── camera/     simplevo.yaml  dpvo.yaml  dust3r.yaml  vggt.yaml
└── recipe/     hq.yaml  accurate.yaml  scene.yaml
```

Each stage file is a small node: a `name` (which implementation) plus that implementation's ctor knobs.
Defaults are the released models, so the default `gvhmr demo` path is **byte-identical** to before this
config surface existed (golden-guarded). `gvhmr demo --help` lists every flag; `gvhmr info` shows what's
installed.
