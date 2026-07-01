# Extensibility & Retraining Plan

How to turn GVHMR from a **frozen released checkpoint** into a **re-trainable, swappable system** — swap
any preprocessing model for a newer/arbitrary version, and retrain the (small) core on a new feature
backbone or new data, so it's usable "for whatever we want."

This is a living roadmap. It's grounded in a full read of the pipeline (anchors throughout); update it as
phases land. Prime directive is unchanged: the **released-checkpoint inference path stays byte-identical**
(golden-guarded, `tests/test_golden_inference.py`) until/unless we deliberately retrain.

---

## 1. The architecture insight (why this is tractable)

A deep map of training + inference (see `docs/ARCHITECTURE.md`, and the per-model contracts below) surfaced
three facts that make broad extensibility far cheaper than "retrain a SIGGRAPH model" sounds:

1. **The trainable core is small and feed-forward — not diffusion.** Despite the `denoiser3d` naming, the
   network (`NetworkEncoderRoPE`, `gvhmr/network/gvhmr/relative_transformer.py:14`) is a **12-layer RoPE
   transformer, latent 512**, run **once** per forward (`gvhmr/model/gvhmr/pipeline/gvhmr_pipeline.py:74`)
   — no timesteps, no noise schedule. The "noise" is augmented 2D-keypoint *input*. It trains from scratch
   (AdamW 2e-4, batch 128, fp16, 2 GPUs, 500 epochs; `gvhmr/configs/exp/gvhmr/mixed/mixed.yaml`).
2. **The image backbone is decoupled.** `f_imgseq` (the HMR2 ViT feature, `(L,1024)`) is **precomputed
   offline and cached as `.pt`** per dataset; **HMR2 is never imported or run in the training loop**
   (`gvhmr/dataset/bedlam/bedlam.py:120`, `threedpw/threedpw_motion_train.py:89`; the only training-time
   `f_imgseq` code is a commented-out aug block, `gvhmr/model/gvhmr/gvhmr_pl.py:132`). Conditions are added
   with **zero-initialized** projectors (`relative_transformer.py:103-107`) — ControlNet-style, so a new
   condition/backbone contributes 0 at init and learns in gracefully.
3. **2D pose is inference-only.** During training, `kp2d`/`obs` is **synthesized on-GPU** from GT joints +
   WHAM-style noise (`gvhmr_pl.py:105-122`); the cached ViTPose file is unused in the `mixed` recipe. So a
   2D-pose model swap has **zero training coupling**.

**Consequence:** swapping the feature backbone = (a) re-extract cached features with the new extractor,
(b) set `imgseq_dim` to the new width (already a ctor arg, `relative_transformer.py:27`), (c) retrain the
small core. Everything else (EnDecoder, losses, FK) is backbone-agnostic. Your **2× RTX 6000 Ada (96 GB)**
box exceeds the original 2×4090 training rig.

---

## 2. The three tiers

| Tier | Models | Swap mechanism | Training? |
|---|---|---|---|
| **A — plug-in preprocessing** | detector (YOLO), 2D pose (ViTPose→COCO-17), camera/SLAM, scene-aware (DUSt3R+Depth-Anything) | emit the same output format; select via config | none |
| **C — body-model file** | SMPL / SMPL-X | swap the *file* within the same topology (v1.0↔v1.1); can't change betas/topology | none (decode is co-trained) |
| **B — trainable core (keystone)** | HMR2 feature backbone + GVHMR checkpoint | re-extract features + set `imgseq_dim` + **retrain** the small core | yes — but tractable |

The per-model swappability contracts are detailed in §6.

---

## 3. The pluggability architecture (one pattern everywhere)

The camera slot already shows the target pattern: a Hydra config group + a `--slam {simplevo|dpvo|dust3r}`
selector (`gvhmr/cli/__init__.py:50`), each backend writing the same `paths.slam` cache, normalized to a
common contract (`R_w2c`) in `load_data_dict` (`gvhmr/cli/demo.py:289`). We generalize that to every
swappable stage:

```
stage          interface (cache contract)                       selector           today
─────          ──────────────────────────                       ────────           ─────
detector    →  bbx_xyxy (L,4) → bbx_xys (L,3)                    --detector    ✅   group; yolo, yolo11
pose2d      →  COCO-17 (L,17,3) [x,y,conf]                       --pose2d      ✅   group; vitpose, rtmpose
camera      →  R_w2c (L,3,3) [+ metric T_w2c for world]          --camera      ✅   group (was --slam)
scene       →  T_w2c (L,4,4) metric                              --camera dust3r ✅
backbone    →  f_imgseq (L, D) + declared D                      --backbone    ✅   group; extract-features + retrain
body model  →  SMPL-X npz/pkl (fixed topology)                   $GVHMR_BODY_MODELS ✅
```

**This is now built** — see [`docs/CONFIGURATION.md`](CONFIGURATION.md) for the user-facing guide. Each
stage is a config group under `gvhmr/configs/{detector,pose2d,backbone,camera}/`, selectable by name
(`--detector`/`--pose2d`/`--backbone`/`--camera`), bundlable into a `--recipe`, and tweakable with `--set`.
Every stage has a **Protocol** declaring its `extract(...) -> <contract>`, a registry mapping the selector
name → implementation, and knobs in the config (not Python). The COCO-17 / `imgseq_dim` invariants stay as
explicit guardrails. Defaults are the released models, so the golden path is byte-identical.

---

## 4. Phased roadmap

Ordered by dependency and risk — earliest phases are self-contained and need no GPU/data.

### Phase A1 — pluggable detector + 2D pose  *(start here; no training/data)*
- ✅ **Done:** `Detector`/`Pose2D` protocols + a lazy registry (`gvhmr/utils/preproc/base.py`); current
  YOLO tracker + ViTPose wrapped as the defaults; hard-coded weights/knobs lifted to ctor args
  (`tracker.py` `DEFAULT_YOLO_CKPT`, `vitpose.py` `DEFAULT_VITPOSE_*`) — **byte-identical defaults**;
  guarded `preproc/__init__.py` so the registry imports on the base/CI install; demo wired to build via
  `make_detector`/`make_pose2d` with `cfg.detector` / `cfg.detector_ckpt` / `cfg.pose2d` / `cfg.pose2d_ckpt`
  overrides; `tests/test_preproc_pluggable.py` pins the contract (241 tests green, golden intact).
  → You can already point at a different weight file via config (e.g. a `yolov11x.pt`).
- ✅ **CLI weight-swap:** `gvhmr demo … --detector-ckpt yolov11x.pt` / `--pose2d-ckpt …` thread through to the
  registry (`+detector_ckpt`/`+pose2d_ckpt` config overrides), so a newer weight is reachable from the CLI.
- ✅ **Config groups + name selectors (done):** detector/pose2d/backbone/camera are first-class Hydra groups
  (`gvhmr/configs/{detector,pose2d,backbone,camera}/`), selected by `--detector`/`--pose2d`/`--backbone`/
  `--camera`, bundlable into `--recipe`, tweakable with `--set`. `--camera` subsumes `--slam`/`--use-dpvo`
  (kept as aliases). One source of truth, shared with `train`. See [`docs/CONFIGURATION.md`](CONFIGURATION.md).
- ✅ **First non-default implementation (done):** `pose2d=rtmpose` — RTMPose via rtmlib/ONNXRuntime (ungated,
  no mmcv), a genuinely different architecture emitting COCO-17. Optional dep `uv sync --extra rtmpose`;
  verified end-to-end. The COCO-17 assert (`relative_transformer.py`) stays the contract guard.
- Drop-ins enabled: YOLOv9/10/11/12 (ultralytics, same `.track()` API, `--detector yolo11`); RTMPose landed,
  and RTMO / Sapiens / DWPose / MoveNet fit **iff configured to COCO-17**.
- **Acceptance:** ✅ `gvhmr demo … --detector yolo11 --pose2d rtmpose --camera dust3r` composes; golden green;
  `test_preproc_pluggable` + `test_config_smoke` assert the registry, group composition, and name-swap.

### Phase A2 — newer scene-aware backend  ✅ **Done**
- ✅ **`--camera vggt`** (`gvhmr/utils/preproc/vggt_slam.py`): VGGT (CVPR 2025) predicts camera + depth in
  **one feed-forward pass** — replacing DUSt3R's pairwise inference + global-alignment optimizer (faster,
  often more robust). Still scale-ambiguous, so it reuses the Depth-Anything-V2 metric-scale fix and the
  slerp/lerp interpolation from the DUSt3R backend, returning the same metric `T_w2c (L,4,4)`. The
  world-compose path fires for `dust3r`/`vggt` alike (unchanged).
- **Verified end-to-end on CUDA:** VGGT API (extrinsic `(K,3,4)` w2c, depth `(K,H,W)`), and the full
  `run_vggt_slam` on a real clip → `T_w2c (L,4,4)`, metric scale from Depth-Anything, orthonormal rotations.
- **Install:** `scripts/setup_scene_aware.sh` clones VGGT into `third-party/vggt` (imported via sys.path;
  weights auto-download `facebook/VGGT-1B`). **Do not `pip install -e` it** — its `numpy<2` pin downgrades
  numpy and breaks scipy (`np.long`); VGGT runs fine on numpy 2.x and its runtime deps are already in the
  base env (timm → safetensors). Documented in `docs/CONFIGURATION.md` + `docs/INSTALL.md`.
- ⏭ Optional future: a `mast3r` backend the same way; bump Depth-Anything-V2 → V3.

### Phase C — body-model file flexibility  ✅ **Done**
- The SMPL/SMPL-X root is now a single `BODY_MODEL_ROOT` constant (`gvhmr/utils/smplx_utils.py`), absolute
  (cwd-independent — fixes the old relative-string paths) and **`$GVHMR_BODY_MODELS`-overridable** (relocate
  to a shared datasets dir, or point at a v1.1 install). All 11 hard-coded `inputs/checkpoints/body_models`
  paths route through it. 244 tests green, golden inference byte-identical (default resolves to the same path).
- The hard constraint is documented at the constant: the 151-d decode + regressors are co-trained to
  **SMPL-X neutral, 10 betas, 22-joint** topology — you may swap the *file* (v1.0↔v1.1, same 10475-vert /
  54-joint topology) but **not** the model family / β-count without retraining the decode.

### Phase B1 — make training runnable + documented  *(keystone, de-risk first)*
- ✅ **Done:** `docs/TRAINING.md` written; a **smoke `fit` runs end-to-end** (validated on macOS **CPU**) via
  a new `gvhmr/configs/exp/gvhmr/mixed/smoke_3dpw.yaml` (3DPW-only, needs just the 3DPW pack + body models).
  Getting there required real **training cleanup**, all behaviour-preserving on CUDA (241 tests green, golden
  intact): (a) **device-aware trainer** (`gvhmr/cli/train.py` honours `$GVHMR_DEVICE`; off-CUDA → fp32 + 1
  device); (b) **torch ≥ 2.6 `weights_only=False`** on all 32 dataset/model `torch.load` of trusted packs;
  (c) **`.cuda()` → device** in `gvhmr_pl.training_step` + a **device-dispatching `get_wham_aug_kp3d`**
  (CUDA path byte-preserved, CPU/MPS uses the CPU variants).
- ✅ **RNG-order landmine pinned** (`tests/test_augment_rng.py`): snapshots the CPU augmentation stream for a
  fixed seed (wham-kp3d sum, visible-mask count, determinism), so reordering the augmentors (`docs/BEHAVIOR.md`)
  fails CI instead of silently shifting training.
- **Acceptance:** ✅ a few-step `fit` completes off-GPU; TRAINING.md merged; RNG snapshot test added.

### Phase B2 — backbone-pluggable offline feature extractor
- ✅ **Done (framework):** `FeatureBackbone` protocol + `make_backbone` registry (`base.py`), mirroring the
  detector/pose2d pattern; the HMR2 `Extractor` registered as `hmr2` with a declared `feat_dim = 1024`; demo
  wired to build via `make_backbone(cfg.backbone)` (default `hmr2`, byte-identical). Added the **`imgseq_dim`
  consistency guard** in the network forward (`relative_transformer.py`) so a feature/checkpoint dim mismatch
  is a clear error pointing at the backbone/retrain, not a cryptic `LayerNorm` failure or a silently-dropped
  condition. Test extended (241 green, golden intact).
- ✅ **Alternative backbone landed & verified:** `DINOv2Backbone` (`dinov2_backbone.py`, `backbone=dinov2`,
  vits14/vitb14/vitl14/vitg14 → 384/768/1024/1536-d) — ungated via torch.hub, reuses 4D-Humans' ImageNet-
  normalized crop. **Verified on the GPU box:** produces `(F, 384)` CLS-token features. This is the concrete
  proof the feature-swap works end-to-end (extract with a non-HMR2 backbone).
- ✅ **Offline-extraction tool (done):** `gvhmr extract-features VIDEOS OUT --backbone <name>` writes the
  training cache format (`imgfeats/<ds>_<backbone>/<vid>.pt = {features (N,D), bbx_xys, img_wh}`). Two box
  sources: run the detector (BYOD) or `--bbx-from` an existing cache (exact re-extraction on the released
  boxes). Resolves the backbone through the same `configs/backbone` group as the demo (consistent feature
  width). Verified end-to-end (dinov2 → (F,384)); CI-safe schema/resolution tests. This was the remaining
  plumbing before B3 — a real retrain is now one `extract-features` + one `train` away (given raw frames).

### Phase B3 — retrain the core on a new backbone
- ✅ **Swapped-backbone training PLUMBING proven:** the 3DPW dataset takes an `imgfeat_subdir` (a `dinov2`
  variant → `imgfeats/3dpw_train_dinov2/`) and skips vids without a feature file; `smoke_3dpw_dinov2.yaml`
  sets `network.imgseq_dim=384` and a CPU `fit` **trained 2 steps end-to-end on 384-d features** (the
  re-inited `imgseq_embedder(384)` + the dim guard + all losses compose and run). `DINOv2Backbone` loads
  offline from the torch.hub cache. 244 tests green, golden intact.
- ⚠️ **Blocked for a *real* retrain:** the smoke used **synthetic** 384-d features because the raw 3DPW
  **frames are gated and absent** (the `videos/` dir is empty — only cached HMR2 features shipped). A real
  DINOv2 retrain needs the 3DPW images (or another dataset with frames + GT): re-extract with
  `make_backbone("dinov2_vits14")`, then a full `fit`. The offline-extraction path is otherwise ready.
- **Ready-to-run recipe** (once raw frames are available):
1. **Co-locate** DINOv2 + data: the torch.hub cache (`~/.cache/torch/hub/facebookresearch_dinov2_main` +
   `checkpoints/dinov2_vits14_pretrain.pth`, ~85 MB) downloads only where there's network; the 3DPW pack is
   on the Mac. Copy the cache to wherever the data lives (or stage 3DPW on the box).
2. **Parameterize** the 3DPW train dataset's feature dir (`threedpw_motion_train.py:41`
   `imgfeats/3dpw_train_smplx_refit`) → a ctor arg, and register a `dinov2` variant.
3. **Re-extract** a few 3DPW vids with `make_backbone("dinov2_vits14")` (reuse the bbx from the existing
   cache), saving `imgfeats/3dpw_train_dinov2/<vid>.pt` in the same `{features, bbx_xys, …}` schema.
4. **Fine-tune** with `network.imgseq_dim=384` + the dinov2 feature dir (from scratch, or from the released
   ckpt — its 1024-d `imgseq_embedder` is dropped by `strict=False` and re-learns from zero-init). A
   `smoke_3dpw_dinov2` config + `GVHMR_DEVICE=cpu` proves the loop; the box does the real run.
- **Acceptance:** a checkpoint trained on DINOv2 features, with an eval number — the first swapped-backbone GVHMR.

### Phase B3 — retrain the core on a new backbone
- Re-extract features for the available training data with the chosen backbone, set `imgseq_dim`, and train
  the small core (start from scratch or fine-tune from `gvhmr_siga24_release.ckpt` via `ckpt_path`).
- Validate against EMDB-2 / 3DPW (the `tools/eval/eval_world.py` + golden harness) and compare to the
  released model.
- **Acceptance:** a trained checkpoint on a new backbone with eval numbers; a new golden fingerprint recorded
  for that config (the released-model golden stays untouched).

### Phase B4 — bring-your-own-data fine-tuning recipe
- A documented path: custom videos + SMPL(-X) GT (or pseudo-GT) → cache features + camera → a thin dataset
  adapter (mirroring `gvhmr/dataset/threedpw/threedpw_motion_train.py`) → fine-tune for a domain.
- **Acceptance:** a worked example fine-tuning on a small custom set end-to-end on the box.

---

## 5. Data & compute prerequisites (the real gate for Tier B)

Training/eval data is **registration-gated** and large; GVHMR distributes the precomputed `hmr4d_support`
bundles (the `.pt`/`.pth` packs the datasets load) via the project **Google Drive** (`docs/INSTALL.md`).

| Dataset | Role | Provides | Gated |
|---|---|---|---|
| **AMASS** | train | motion only (no images; `f_imgseq` zeros) | ✅ sign-up |
| **BEDLAM** | train | motion + **cached ViT feats** (`imgfeats/bedlam_*`) | ✅ |
| **H36M** | train | motion + cached feats (`vitfeat_h36m.pt`, loaded **fully in RAM**) | ✅ |
| **3DPW** | train+eval | refit SMPL-X + cached feats | ✅ |
| **EMDB / RICH** | eval | world-traj / SMPL-X + feats | ✅ (institutional email for EMDB) |
| **SMPL / SMPL-X** | body models | required by training *and* inference | ✅ |

- **Compute:** the box (2× RTX 6000 Ada, 96 GB) > the original 2×4090 rig — sufficient for the small core.
  Training is **GPU-only** (`gvhmr/cli/train.py:61`; no CPU/MPS). H36M ViT feats must fit in RAM.
- **The expensive part of a backbone swap** is **re-extracting features**, which needs the *raw images*
  (BEDLAM/H36M/3DPW videos — large + gated). Mitigations: start on a **subset** (debug configs), or on
  **custom data** (Phase B4), before committing to a full re-extraction.
- **Disk/RAM are not quantified upstream** — measure when staging the first dataset and record here.

---

## 6. Per-model swappability reference (contracts + anchors)

**Plug-in preprocessing (no GVHMR weight depends on internals — swap freely if the format matches):**
- **Detector** → `bbx_xyxy (L,4)` → `bbx_xys (L,3)` (`tracker.py:43-99`, `geo/hmr_cam.py:295`). Path hard-coded.
- **2D pose** → **COCO-17** `(L,17,3)` (`vitpose.py:74`); `J==17` hard-asserted (`relative_transformer.py:128`);
  **inference-only** (training synthesizes kp2d). Path hard-coded.
- **Camera** → `R_w2c (L,3,3)` for the net (translation only feeds the optional world-compose); already
  pluggable via `--slam` (`demo.py:253-298`, `postproc_world.py:30`).
- **Scene-aware** → metric `T_w2c (L,4,4)` (`dust3r_slam.py:56`); two vendored models behind one function.

**Learned conditioning / retrain-locked:**
- **HMR2 features `f_imgseq` (1024-d)** — the trained `imgseq_embedder = LayerNorm(1024)+Linear(1024→512)`
  (`relative_transformer.py:103-107`) is fit to HMR2.0a's token distribution; wrong dim → silently dropped,
  wrong backbone → miscalibrated. **Swap ⇒ retrain.**
- **GVHMR checkpoint** — pins the NetworkEncoderRoPE arch + 151-d SMPL-X decode + EnDecoder stats
  `MM_V1_AMASS_LOCAL_BEDLAM_CAM` (`gvhmr_pl_demo.py:119`, `endecoder.py:201`).
- **SMPL-X "supermotion"** — neutral / 10 betas / 22-joint topology baked into the output head, FK, losses,
  and `smplx2smpl`/`J_regressor` assets (`smplx_utils.py:29`, `demo.py:48`).

---

## 7. Risks & landmines

- **RNG-order is load-bearing** (`docs/BEHAVIOR.md:66`, 5 augmentation layers mixing `np.random`/`torch`/CUDA).
  Don't reorder augmentors; add the snapshot test (Phase B1) before touching them.
- **Golden inference must stay green** for the released path through all of Tier A/C (no behavior change).
- **`imgseq_dim` mismatch silently disables conditioning** (`strict=False` load) — add the hard guard (B2).
- **151-d latent layout + EnDecoder stats** are byte-exact contracts (`docs/BEHAVIOR.md`, landmine #1).
- **`siga24_release.yaml` references a non-existent `NetworkEncoderRoPEV2`** — the live path is
  `relative_transformer.py`/`demo.yaml`; don't wire training to the broken yaml (`docs/BEHAVIOR.md` #8).

---

## 8. Milestones

- **M1** — Phase A1 (config groups + name selectors + RTMPose) + Phase C (body-model config). ✅ **done.**
- **M2** — Phase A2 (VGGT scene backend). ✅ **done** — `--camera vggt`, verified end-to-end on CUDA.
- **M3** — Phase B1 (training runnable + `docs/TRAINING.md` + smoke run + RNG test). ✅ **done.**
- **M4** — Phase B2 (backbone-pluggable feature extractor + `gvhmr extract-features` + dim guard). ✅ **done.**
- **M5** — Phase B3/B4 (retrain on a new backbone; bring-your-own-data fine-tuning). ⏳ plumbing proven
  (synthetic-feature smoke); a *real* retrain is blocked only on gated raw frames.

**Remaining:** only the *real* M5 retrain (needs gated raw training frames) — the plumbing is proven, so
it's a data problem, not a code one. Everything else (the full config/extensibility framework, all four
pluggable stages with real alternatives, the offline extractor, both scene-aware cameras) is **done**: see
[`docs/CONFIGURATION.md`](CONFIGURATION.md).
