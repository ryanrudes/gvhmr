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

**Real A/B run (reduced) — Sapiens LOST, refuting the "backbone is the biggest win" hypothesis.** Trained the
motion head on AMASS(capped)+3DPW-train with Sapiens-0.3b vs HMR2 features, identical recipe, evaluated on
3DPW-test:

| 3DPW-test | HMR2 | Sapiens-0.3b |
|---|---|---|
| **PA-MPJPE** | **42.8** | **74.5** |
| MPJPE | 69.6 | 121.2 |
| PVE | 82.0 | 145.2 |
| Accel | 9.8 | 7.9 |

Sapiens is **~2× worse**. The reason is real: HMR2's feature is the **SMPL-head token — task-trained for mesh
recovery**; Sapiens's is a *generic* MAE-pretrain feature, and I fed it naively. This refutes "Sapiens is a
drop-in win" but NOT "Sapiens can't help" — the integration was weak in three ways that likely each cost a
lot: (1) smallest model (0.3b), (2) **global-average-pooling** the `(C,64,64)` map to one vector (throwing away
spatial structure HMR2's learned token keeps — probably the biggest culprit), (3) the pretrain encoder, not
Sapiens's pose head. Matches this session's theme: a "better" generic model doesn't beat task-specific
features without careful integration. Both numbers are low (vs the released 36.2) because it's a reduced
AMASS+3DPW train — the A/B is controlled, so the relative verdict holds.

**A fair Sapiens test would need:** a larger variant, a *learned* pooling / the pose-head features (not GAP),
possibly fine-tuning — i.e. a real feature-design effort, not a drop-in. **To retry the full run:**
`gvhmr extract-features --backbone sapiens …` over BEDLAM/H36M/3DPW-train (1024² ViT — GPU-days) then
`gvhmr train … network.imgseq_dim=<D>`.

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

- **A3** — `gvhmr/model/gvhmr/physics_losses.py`: **all three losses now wired** into
  `compute_extra_global_loss` (f1b85bf), weight-gated + default-off so released training is byte-identical —
  `velocity_smoothness_loss` (`weights.transl_w_accel`), `foot_contact_loss` (`weights.foot_slide`),
  `ground_penetration_loss` (`weights.penetration`). The earlier blocker is resolved: the predicted **world**
  joints are FK'd fast + differentiably in training by rolling out the predicted local translation velocity
  with the GT world orientation (teacher-forced, exactly as `transl_w` does — a vectorized cumsum) then
  `fk_v2`; no inference-time for-loop (`get_smpl_params_w_Rt_v2`) needed.

  ### A3 — the DEFINITIVE result (full recipe, 2026-07-14): a smoothness/accuracy TRADE, not a free win

  > **This supersedes, and substantially retracts, the "A3 CONFIRMED" claim from the reduced model below.**

  Two matched 500-epoch runs on the **full released recipe** (AMASS+BEDLAM+H36M+3DPW), the paper's true
  effective batch (**256** = the recipe's `devices=2 × 128`), same seed (42), `torch.compile` on both arms,
  differing in *nothing* but the three physics weights. Trained on H200s via `scripts/slurm/submit.sh`
  (W&B `armA_off` / `armB_light`). TF32 off, so the derivative metrics are trustworthy (06e3922).

  | metric | A: off | B: light | Δ | (reduced model said) |
  |---|---|---|---|---|
  | **EMDB-2 Jitter** | 16.24 | **14.58** | **−10.2%** ✓ | −9% ✓ |
  | **RICH Jitter** | 12.76 | **11.12** | **−12.9%** ✓ | — |
  | **EMDB-1 Accel** | 3.58 | **3.38** | **−5.6%** ✓ | −5% ✓ |
  | **RICH Accel** | 4.12 | **3.95** | −4.1% ✓ | — |
  | **EMDB-2 Foot-slide** | 3.51 | 3.51 | **0.00 — nothing** ✗ | −15% ✓ |
  | **EMDB-2 WA-MPJPE** | 112.32 | 113.66 | **+1.34 WORSE** ✗ | **−41.07** ✓ |
  | **EMDB-2 W-MPJPE₁₀₀** | 279.17 | 286.14 | **+6.97 WORSE** ✗ | — |
  | EMDB-2 RTE | 1.93 | 1.85 | −0.08 | −0.71 ✓ |
  | **EMDB-1 MPJPE** | 74.92 | 77.26 | **+2.34 WORSE** | — |
  | 3DPW PA-MPJPE *(guardrail)* | 35.94 | 36.12 | +0.18 | +0.16 |

  **What survives:** the losses genuinely and consistently **cut jitter 10–13%** (EMDB-2 *and* RICH) and
  **accel 4–6%**. That is a real effect, reproduced on a paper-grade model.

  **What is REFUTED:** the reduced model's headline — "improves *everything* including world MPJPE
  (WAA −41) and RTE, at ~no accuracy cost" — does **not** replicate.
  - **The world-grounding gain inverted**: WAA **−41.07 → +1.34**. The sign flipped. That −41 was the
    regularizer tidying up a *weak* model's sloppy trajectories (baseline WAA 320 vs the paper's 109);
    a model that already reproduces the paper has no such slack to recover.
  - **Foot-sliding — the loss's most direct target — does exactly nothing** on EMDB-2 (3.51 → 3.51).
  - It now **costs** real accuracy: EMDB-1 MPJPE +2.34mm, PVE +2.36mm; EMDB-2 W-MPJPE +6.97mm.

  So A3 is a **smoothness-for-accuracy trade**, worth taking when temporal smoothness is the product
  (animation, retargeting) and not otherwise. The weights stay **default-off**. The general lesson is the
  one this A/B exists to catch: *a regularizer that rescues a weak model can do nothing — or harm — on a
  strong one, and a reduced-model A/B will happily tell you otherwise.*

  **Obvious follow-up (not run):** ablate the three losses separately. `foot_slide` moved EMDB-2
  foot-sliding by 0.00 while the world metrics got worse, so the jitter win is plausibly *all* from
  `transl_w_accel` (velocity smoothness) and the contact/penetration terms may be pure cost. A
  `transl_w_accel`-only arm would test that in one run.

  ### The original reduced-model A/B (kept for the record — its world-grounding claim did not hold up)

  Same reduced recipe + seed as the A1 HMR2 baseline (`a3_physics_hmr2.yaml`), evaluated on 3DPW +
  EMDB. Weights were calibrated from a probe (raw magnitudes: `foot_slide`~3e-4, `penetration`~5e-4,
  `transl_w_accel`~5e-7) and *bracketed* — a **light** arm (`foot_slide`=200, `penetration`=100,
  `transl_w_accel`=1e4) and a **strong** arm (1000/500/1e5) — to separate "physics helps" from "weights
  too hot":

  Numbers below are the **TF32-free re-measurement (2026-07-13)** — the original table was scored while the
  TF32 regression was live (see the note under the table), so every arm was re-evaluated from its saved
  checkpoint with TF32 off:

  | metric | paper | off (baseline) | light | strong |
  |---|---|---|---|---|
  | 3DPW PA-MPJPE *(guardrail)* | 36.2 | 42.80 | 42.96 (+0.16) | 43.01 (+0.21) |
  | 3DPW Accel | 5.0 | 9.67 | **9.47** (−0.20) | **9.34** (−0.33) |
  | EMDB WAA-MPJPE | 109.1 | 320.60 | **279.54** (−41.1) | 341.47 (+20.9) |
  | EMDB RTE | 1.9 | 6.92 | **6.21** (−0.71) | 7.64 (+0.72) |
  | EMDB Jitter | 16.5 | 57.50 | **52.03** (−5.5) | 53.20 (−4.3) |
  | EMDB Foot-Slide | 3.5 | 11.13 | **9.42** (−1.70) | 10.60 (−0.52) |
  | EMDB-1 Accel | 3.6 | 11.66 | **11.03** (−0.63) | 10.89 (−0.76) |

  On *this* model every physics target dropped in both arms with PA-MPJPE held to +0.4%, and the **light**
  arm appeared to improve *everything* — including world MPJPE (WAA −41) and RTE (−0.71) — while the
  **strong** arm over-regularized the trajectory (WAA +21, RTE +0.72). That reading is what the full-recipe
  run above **overturns**: on a paper-grade model the WAA gain inverts and the foot-slide gain vanishes, so
  the world-grounding half of this conclusion was an artifact of the weak baseline (WAA 320 vs paper 109).
  The jitter/accel half held. Config `exp=gvhmr/mixed/a3_physics_hmr2` (weights default 0 → identical to the
  baseline arm); the full-recipe arm is `exp=gvhmr/mixed/mixed_physics_light`.

  **Why the reduced A/B misled, worth internalizing:** its baseline was *far* from the paper on exactly the
  metrics being judged (WAA 320 vs 109, jitter 57 vs 16, foot-slide 11.1 vs 3.5). A regularizer has enormous
  room to improve a model that bad, and none of that transfers. A reduced-model A/B can establish that a loss
  *does something*; it cannot establish that the something is *useful on a good model*.

  > **Re-measured after the TF32 regression (06e3922).** The first scoring of this A/B ran while TF32 was
  > globally enabled, which corrupts exactly the metrics A3 is judged on (it inflated the baseline's EMDB-1
  > accel by 71%: 11.66 → 19.92, and jitter 57.50 → 60.59). The conclusion nevertheless **survives intact**:
  > TF32's error was *common-mode*, shifting all three arms almost equally, so while the absolute numbers
  > moved a lot the A/B *deltas* are unchanged to ~0.1 (e.g. WAA −41.10 → −41.07, foot-slide −1.71 → −1.70).
  > Worth remembering both ways: a shared artifact can leave a *ranking* valid while making every absolute
  > number wrong — and the artifact here (−3.1 jitter) was the same size as the claimed effect (−5.0), so it
  > had to be checked rather than assumed.
- **A4** — `gvhmr/utils/preproc/box_adapter.py`: `BoxAdapter` (normalized affine on `(cx,cy,size)`, default
  identity) + `fit_box_adapter` (calibrate new→baseline from paired boxes). Wired into the demo behind
  `box_adapt` (default null → skipped → golden-identical). **Validated on real data (negative for yolo26x):**
  the yolo26x→baseline calibration on 3DPW is the identity transform, so the adapter cannot recover yolo26x's
  penalty — it's per-frame, not a systematic framing bias (see the A4 caveat above). The mechanism stands for
  detectors with a real systematic bias; yolo26x isn't one.

**Full-recipe reproduce — DONE (2026-07-13).** A from-scratch 500-epoch retrain on all four datasets
(`exp=gvhmr/mixed/mixed`, single GPU, W&B `mixed_reproduce_4ds_500e`) **reproduces the paper**, which
retires the "reduced model" caveat that qualified every A-series result above:

| | this retrain | paper |
|---|---|---|
| 3DPW PA-MPJPE / MPJPE / Accel | 36.4 / 55.7 / 4.8 | 36.2 / 55.6 / 5.0 |
| EMDB-1 PA-MPJPE / Accel | **42.4** / **3.5** | 42.7 / 3.6 |
| EMDB-2 WA-MPJPE / RTE / Jitter | 111.9 / 2.0 / **14.7** | 109.1 / 1.9 / 16.5 |
| RICH PA-MPJPE / MPJPE | 41.0 / 69.9 | 39.5 / 66.0 |

It matches on 3DPW, **beats the paper on EMDB-1 PA-MPJPE and on jitter across all three benchmarks**, and
trails ~3-4 mm on RICH pose and EMDB-2 world translation. The most likely cause of that shortfall is a
recipe deviation on my side, not the code: `mixed.yaml` specifies `devices: 2 × batch_size: 128` (effective
**256** under DDP) and this ran `devices=1` (effective **128**) at the same LR, to leave the second GPU free
on a shared box.

**Confirmed — and fixed (2026-07-14).** Re-run at the paper's true effective batch of **256** (1× H200,
`scripts/slurm/submit.sh`), the reproduce is even closer, and **beats the paper on 3DPW**:

| | batch 256 (correct) | batch 128 (first try) | paper |
|---|---|---|---|
| 3DPW PA-MPJPE / MPJPE | **35.9** / 55.5 | 36.4 / 55.7 | 36.2 / 55.6 |
| EMDB-1 PA-MPJPE / Accel | 43.3 / 3.6 | 42.4 / 3.5 | 42.7 / 3.6 |
| EMDB-2 RTE / Jitter / FS | 1.9 / 16.2 / 3.5 | 2.0 / 14.7 / 3.8 | 1.9 / 16.5 / 3.5 |
| RICH PA-MPJPE / MPJPE | 40.7 / 68.4 | 41.0 / 69.9 | 39.5 / 66.0 |

The batch fix moved exactly what it was predicted to: 3DPW PA-MPJPE 36.4 → **35.9** (now *better* than the
paper), RICH MPJPE gap halved (+3.9 → +2.4), EMDB-2 W-MPJPE gap halved (+8.8 → +4.3). It costs a little on
EMDB-1 pose and jitter, so it is not strictly dominant — but the pose metrics that were lagging now match or
beat the paper. **This (`armA_off`) is the baseline the definitive A3 A/B above is measured against.**

## Regime B status

B is a research program, not a scaffold — its value is a multi-week training effort (joint backbone
fine-tuning, a metric-camera conditioning stream, whole-clip context, physics), not code that can be
"stubbed". The migration path above is the concrete entry: land the A-series (backbone + world stack +
physics), then take B one measurable rung at a time (backbone LoRA → camera conditioning → full joint
training), each gated on `gvhmr eval` / `eval_world.py`. The additive-fusion architecture means the new
conditioning streams B needs are one-embedder changes, and the seams from A1/A2 are the plug points.
