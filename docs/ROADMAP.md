# ROADMAP — toward the best possible GVHMR given current models

Two concrete plans for pushing GVHMR past the 2024 paper. **Plan A ("Refit")** upgrades the frozen
components and world-grounding while keeping the architecture — highest ROI, and what this repo's
swappable-stage design was built for. **Plan B ("Rebuild")** is the end-to-end moonshot for when A
saturates.

## Diagnosis — where the ceiling is set

GVHMR's accuracy is bottlenecked by its **frozen 2023-era evidence extractors**, not its motion model:

- The dominant signal is **HMR2's ViT-Huge features** (frozen, 2023), ~85–95% of runtime.
- The **2D-pose stage is inference-only** — training synthesizes 2D from GT (`gvhmr_pl.py` `training_step`,
  `get_wham_aug_kp3d`), so ViTPose/RTMPose never enter training. Detector likewise only gates the crop.
- The network **never sees camera translation** — it consumes rotation (`cam_angvel`) only; all world
  translation is post-network (`utils/postproc_world.py::compose_world_from_dust3r`).
- Training consumes **cached features** (no ViT in the loop), so retraining the 12-layer motion head is
  *cheap*; the one-time cost is feature extraction.

Implication: upgrade the evidence (backbone) and the world-grounding stack, retrain the small head, and
measure with the (now bug-fixed) `gvhmr eval` / `gvhmr sweep` / `tools/eval/eval_world.py` harness.

---

## Plan A — "Refit": modernize components, keep the architecture

### A1 — Backbone bake-off (do first: biggest lever, cheapest to test)

The feature backbone is a swappable stage (`gvhmr/configs/backbone/`, `make_backbone`,
`gvhmr extract-features`). Additive late fusion means only one layer depends on feature width — the
`imgseq_embedder` (`LayerNorm(D)→Linear(D→512)` in `network/gvhmr/relative_transformer.py`), sized by
`network.imgseq_dim`; a runtime guard enforces consistency.

Steps:
1. **Add the candidate backbone** as a `FeatureBackbone` (a class with `feat_dim` + `extract_video_features(video, bbx_xys)→(F,D)`),
   register it in `make_backbone`/`BACKBONES`, add `configs/backbone/<name>.yaml`. *Sapiens is scaffolded —
   see "Status" below.* Candidates: **Sapiens** (human-specialized, top pick), DINOv2-ViT-L (already present).
2. **Fix the placeholder dim** (done): AMASS's masked-off `f_imgseq` now follows `network.imgseq_dim`
   (`pure_motion/base_dataset.py`, `pure_motion/amass.py`) so batches collate under any backbone.
3. **Extract training features** for the three datasets with real features — BEDLAM
   (`imgfeats/bedlam_*`), H36M (`vitfeat_h36m.pt`), 3DPW-train (`imgfeats/3dpw_train_<name>`) — via
   `gvhmr extract-features … --backbone <name> --bbx-from <existing>` (hold boxes fixed). Register the new
   `imgfeat_subdir` dataset variants (see the `dinov2`/`sapiens` variants in `threedpw_motion_train.py`).
   **This is the only expensive step (~1–3 GPU-days, BEDLAM-dominated); one-time.**
4. **Retrain the head**: `gvhmr train exp=gvhmr/mixed/mixed network.imgseq_dim=<D>` on the new feature
   dirs. Cheap (small transformer on cached features, IO-bound) — a full 500-epoch run is on the order of a
   few days on 2× RTX 6000 Ada; iterate the head cheaply once features are cached.
5. **Compare**: gate on `gvhmr eval 3dpw,emdb` vs `PAPER_REFERENCE`, then grid with `gvhmr sweep`.

**Bonus (additive fusion makes it clean):** add Sapiens depth/normal/seg as *new conditioning streams*
(a new `_build_condition_embedder` summed into the token) if raw features saturate.

### A2 — World-grounding upgrade (biggest headroom vs paper, no retrain)

The network only consumes rotation, so this is entirely preproc + the compose:
1. **Modern metric depth** (UniDepth / Metric3D-v2) in the scale step — directly improves RTE / WA-MPJPE.
   *The scale step is now a swappable `MetricDepth` seam (see "Status") — implement a class + a
   `make_metric_depth` branch, no backend edits.*
2. **Stronger feed-forward geometry** (VGGT present; add MASt3R-class) behind the same `{T_w2c, scale}`
   contract.
3. **Replace the heuristic frequency-graft** (`compose_world_from_dust3r:46–47`) with a learned residual or
   a small joint human-trajectory + metric-camera optimizer.
4. **Gate with `tools/eval/eval_world.py`**: `gt-cam` isolates the compose from SLAM error, `dust3r`
   measures end-to-end.

### A3 — Physics / contact realism (cheap retrain)

Contact-consistency + non-penetration + velocity losses on the predicted world motion. Targets `fs`,
`jitter`, world realism. **Done (all three wired, default-off)** — `model/gvhmr/physics_losses.py`, gated by
`weights.{foot_slide,penetration,transl_w_accel}`. The predicted world joints are FK'd fast + differentiably
(roll out the predicted translation with the GT world orientation — teacher-forced, as `transl_w` does —
then `fk_v2`; no inference for-loop). Enable the weights and retrain; validate on `fs`/`jitter`.

### A4 — Inference-only levers (no retrain)

Detector (a per-detector **box-distribution normalization adapter**) and 2D pose (RTMW / Sapiens-pose).
Sweep, ship what wins in the wild.

**Measured caveat (yolo26x):** the box-adapter only helps a detector with a *systematic* framing bias. On
3DPW, calibrating yolo26x→baseline gives the **identity** transform (pooled median size-ratio 1.0000, offset
0) — yolo26x has no systematic bias, so a global adapter can't recover its −19% PA-MPJPE. That penalty is
**per-frame** box variation (61% of frames differ, ±3px center / ±14px size, zero-mean), which no fixed
transform fixes. The adapter remains valid for a detector that *is* systematically tighter/looser/offset.

**Do not retrain for A4** — those stages don't affect training.

---

## Plan B — "Rebuild": end-to-end video-native

Only after A saturates. The staged frozen-feature design bottlenecks information; B removes it.

- **Fine-tune the backbone jointly** (LoRA → full). The ViT now enters the training loop → you need raw
  image crops, not cached features (the major cost jump).
- **Feed camera translation into the network** as a *new conditioning stream* (additive fusion → one
  embedder) so the model jointly reasons about human + camera + scene, and supervise `transl_w` against
  real trajectories — attacking world metrics directly instead of the post-hoc compose.
- **Keep** gravity-view canonicalization (`hmr_global.py`, EnDecoder 151-latent). Consider scaling the
  transformer, a flow-matching / diffusion prior, whole-clip context, physics.
- **Data**: needs images (BEDLAM frames) + scaled synthetic + pseudo-labeled in-the-wild (the packs ship
  features, not frames — a data-engineering effort).

**Migration (de-risks B with A's harness):** land A1+A2 → add backbone LoRA to the *same* pipeline
(measured on `gvhmr eval`) → add the metric-camera conditioning + `transl_w` (measured on `eval_world.py`)
→ only then commit to full joint training. Each rung is independently measurable; stop where marginal gain
stops paying for compute.

---

## Validation ladder (applies to both)

1. `tests/test_golden_inference.py` stays green — a new backbone is a *new* checkpoint; the released path is
   untouched.
2. `gvhmr eval 3dpw,emdb` vs `PAPER_REFERENCE` deltas — the accuracy gate.
3. `gvhmr sweep` — grid across trained backbones/stages (harness bug-fixed).
4. `tools/eval/eval_world.py` (`prior`/`gt-cam`/`dust3r`) — world grounding.

**Landmines** (see `docs/BEHAVIOR.md`): the 151-latent layout + `avgbeta` slice coupling, dataset RNG
order, `weights_only=False` loads, the `imgseq_dim` guard.

---

## Status

**A1 step 1 is scaffolded** (Plan A1.1/A1.2), CI-green, behavior-preserving:

- `gvhmr/utils/preproc/sapiens_backbone.py` — `SapiensBackbone` (`FeatureBackbone`), sapiens-lite
  TorchScript loader, high-res crop + GAP pooling, `feat_dim` re-verified at first forward.
- `gvhmr/configs/backbone/sapiens.yaml`; registered in `make_backbone`/`BACKBONES`.
- `configs/exp/gvhmr/mixed/smoke_3dpw_sapiens.yaml` + the `imgfeat_3dpw/sapiens` dataset variant.
- AMASS placeholder width now follows `network.imgseq_dim` (`pure_motion/{base_dataset,amass}.py`).
- `tests/test_backbone_sapiens.py` pins the plumbing (registry, config compose, interpolated AMASS dim).

**Backbone validated with real weights.** Downloaded `facebook/sapiens-pretrain-0.3b-torchscript` and ran
`SapiensBackbone` on a real 3DPW video → finite `(F, 1024)` features. Running it corrected two scaffold
assumptions: the pretrain encoders are traced at a **fixed 1024²** (not 1024×768), and they return a
**1-tuple** wrapping the `(B, C, 64, 64)` feature map (both fixed in `sapiens_backbone.py`). So the extractor
works end-to-end; the retrain is now just compute:

**To make it a full run:** (1) `gvhmr extract-features --backbone sapiens --set backbone.checkpoint=<pt2>`
over BEDLAM/H36M/3DPW-train (note: 1024² ViT is heavy — plan the GPU-days); (2) `gvhmr train
exp=gvhmr/mixed/mixed network.imgseq_dim=<D>` on the new feature dirs; (3) `gvhmr eval` / `gvhmr sweep`.

**A2 metric-depth seam landed** (Plan A2.1), CI-green, behavior-preserving:

- `gvhmr/utils/preproc/metric_depth.py` — `MetricDepth` protocol + `make_metric_depth` registry +
  `metric_scale_from_depths` (the median-ratio scale math, byte-identical). Default `depth_anything_v2`
  wraps the released DA-V2; `unidepth`/`metric3d` are declared stubs that raise until implemented.
- `dust3r_slam.py` / `vggt_slam.py` refactored onto the shared seam (dedup'd; the released scale step is
  unchanged), with a `depth_model` kwarg; `configs/camera/{dust3r,vggt}.yaml` expose `depth_model` and the
  demo threads it.
- `tests/test_metric_depth.py` pins the registry, the seam, and the scale math.

**UniDepth-V2 landed + validated with real weights — but the result is NUANCED, not the win I first claimed.**
`UniDepthMetric` (2024, predicts metric depth + intrinsics) is implemented behind the seam
(`make_metric_depth("unidepth")`; cloned into `third-party/UniDepth`, sys.path-imported like dust3r/vggt — no
env changes). Two real A/Bs on 3DPW, both models through the seam:
- **Person depth** (n=18): DA-V2 MAE 6.82 m (bias +6.8), **UniDepth MAE 1.31 m — 5× better**. The released
  DA-V2 is the *VKITTI outdoor-driving* model, badly miscalibrated for human-scale depth.
- **Scale fix** (the thing A2 actually targets — VGGT recon → metric-depth median ratio → camera travel vs GT,
  n=1 video): **DA-V2 22% error vs UniDepth 76%.** DA-V2 wins.

So **better foreground depth ≠ better scene scale**: the scale fix uses the median ratio over the *whole
image*, where DA-V2's VKITTI scene-depth calibration happens to match the VGGT recon better, despite being
worse on people. My earlier "UniDepth improves world grounding" claim is **not supported** — running it for
real corrected it. UniDepth is a strong *foreground*-depth model, not (on this test) a better *scale* model.
Caveats: the scale test is 1 video and depends on VGGT recon quality — a proper multi-video `eval_world.py`
A/B (needs EMDB videos) is the real arbiter. `camera.depth_model=unidepth` is available to A/B further.

**To add another metric-depth model:** implement a `MetricDepth` class (emit metres-valued depth), add a
`make_metric_depth` branch, set `camera.depth_model=<name>`, and A/B with `tools/eval/eval_world.py`.

**A3 physics losses + A4 box-adapter landed** (CI-green, behavior-preserving):

- **A3** — `gvhmr/model/gvhmr/physics_losses.py`: `velocity_smoothness_loss` (wired into
  `compute_extra_global_loss`, `weights.transl_w_accel` default-off → training byte-identical),
  `foot_contact_loss` + `ground_penetration_loss` (written + tested, **not yet wired**: they need predicted
  **world** joints, which the pipeline FKs only at inference via the slow for-loop `get_smpl_params_w_Rt_v2`
  — the concrete A3 follow-on is a fast, differentiable training-time world-joint FK). Enable the physics
  retrain by adding weights and, once wired, the contact/penetration terms; validate on `fs`/`jitter`.
- **A4** — `gvhmr/utils/preproc/box_adapter.py`: `BoxAdapter` (normalized affine on `(cx,cy,size)`, default
  identity) + `fit_box_adapter` (calibrate new→baseline from paired boxes). Wired into the demo behind
  `box_adapt` (default null → skipped → golden-identical). **Validated on real data (negative for yolo26x):**
  the yolo26x→baseline calibration on 3DPW is the identity transform, so the adapter cannot recover yolo26x's
  penalty — it's per-frame, not a systematic framing bias (see the A4 caveat above). The mechanism stands for
  detectors with a real systematic bias; yolo26x isn't one.

## Regime B status

B is a research program, not a scaffold — its value is a multi-week training effort (joint backbone
fine-tuning, a metric-camera conditioning stream, whole-clip context, physics), not code that can be
"stubbed". The migration path above is the concrete entry: land the A-series (backbone + world stack +
physics), then take B one measurable rung at a time (backbone LoRA → camera conditioning → full joint
training), each gated on `gvhmr eval` / `eval_world.py`. The additive-fusion architecture means the new
conditioning streams B needs are one-embedder changes, and the seams from A1/A2 are the plug points.
