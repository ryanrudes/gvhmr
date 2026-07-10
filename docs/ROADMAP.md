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
1. **Modern metric depth** (UniDepth / Metric3D-v2) in the scale step (`dust3r_slam.py:123–140`, shared by
   `vggt_slam.py`) — directly improves RTE / WA-MPJPE.
2. **Stronger feed-forward geometry** (VGGT present; add MASt3R-class) behind the same `{T_w2c, scale}`
   contract.
3. **Replace the heuristic frequency-graft** (`compose_world_from_dust3r:46–47`) with a learned residual or
   a small joint human-trajectory + metric-camera optimizer.
4. **Gate with `tools/eval/eval_world.py`**: `gt-cam` isolates the compose from SLAM error, `dust3r`
   measures end-to-end.

### A3 — Physics / contact realism (cheap retrain)

Wire the existing `static_conf` foot/wrist contact prediction (joints `[7,10,8,11,20,21]`,
`static_conf_bce`) into a contact-consistency + non-penetration + velocity loss. Targets `fs`, `jitter`,
world realism.

### A4 — Inference-only levers (no retrain)

Detector (a per-detector **box-distribution normalization adapter** — the measured yolo26x −19% PA-MPJPE
is a recoverable distribution mismatch, not fundamental) and 2D pose (RTMW / Sapiens-pose). Sweep, ship
what wins in the wild.

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

**To make it a real run:** (1) download a sapiens-lite TorchScript encoder → set `backbone.checkpoint`;
(2) confirm the encoder's input size / feature tap against `sapiens_backbone.py`'s notes; (3)
`gvhmr extract-features` over BEDLAM/H36M/3DPW-train; (4) `gvhmr train exp=gvhmr/mixed/mixed
network.imgseq_dim=<D>` on the new feature dirs; (5) `gvhmr eval` / `gvhmr sweep`.
