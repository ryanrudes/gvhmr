# Training

How to train / retrain GVHMR. The trainable core is **small** (a 12-layer RoPE transformer, latent 512 —
*not* a diffusion model; see `docs/EXTENSIBILITY.md`) and consumes **offline-cached** image features, so a
retrain is far cheaper than the "SIGGRAPH model" framing suggests. Training is Hydra-driven
(`gvhmr train exp=…`) and now **device-aware**: real runs use multi-GPU CUDA, but a smoke `fit` runs on
CPU/MPS.

## Quick smoke test (no full datasets)

Proves the whole `fit` loop end-to-end. Needs only the **3DPW** support pack + the **SMPL-X/SMPL** body
models (no AMASS/BEDLAM/H36M):

```bash
GVHMR_DEVICE=cpu uv run gvhmr train exp=gvhmr/mixed/smoke_3dpw
```

It runs 2 optimizer steps on a tiny 3DPW slice and prints per-loss values
(`cr_j3d_loss`, `cr_vert_loss`, `j2d_loss`, `simple_loss`, …) then `End of script.` — validated on macOS
CPU. On the GPU box, drop `GVHMR_DEVICE=cpu` to smoke-test the real CUDA/fp16 path. The config is
`gvhmr/configs/exp/gvhmr/mixed/smoke_3dpw.yaml` (3DPW-only, batch 2, 2 batches, no logger/checkpoint).

## Real training

```bash
uv run gvhmr train exp=gvhmr/mixed/mixed        # from scratch: 2 GPUs, fp16-mixed, batch 128, 500 epochs
```

- **Model:** trains **from scratch** (no pretrained init). `NetworkEncoderRoPE` + `EnDecoder` (151-d latent)
  + the loss pipeline. AdamW 2e-4, MultiStepLR (halve at epochs 200/350), gradient-clip 0.5. Fine-tune from
  the released checkpoint with `ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`.
- **Losses** (weights in `mixed.yaml`): `cr_j3d` 500, `cr_verts` 500, `j2d` 1000, `verts2d` 1000,
  `simple` 1 (151-d latent MSE), `transl_c`/`transl_w`/`static_conf_bce` 1.
- **Compute:** GPU-only for real runs (fp16-mixed + multi-device are CUDA-only; off-CUDA the trainer forces
  fp32 + 1 device). The original release was 2×4090 for ~420 epochs; a 2× RTX 6000 Ada box is sufficient.
- **Validation** runs every 10 epochs on EMDB/RICH/3DPW (metric callbacks), batch size 1.

Reproduce the paper eval:
```bash
uv run gvhmr train global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed \
  ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
```

## Data layout & gated downloads

All training/eval data lives under `inputs/<DATASET>/hmr4d_support/` as precomputed `.pt`/`.pth` packs
(motion params + **cached ViT features** + labels). GVHMR distributes these via the project
**[Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD)**; the underlying
raw datasets are registration-gated. Extract so the `*/hmr4d_support/` dirs sit under `inputs/`.

| Dataset | Role | Provides | Gated |
|---|---|---|---|
| **AMASS** | train | motion only (image features are zeros) | ✅ sign-up |
| **BEDLAM** | train | motion + cached ViT feats (`imgfeats/bedlam_*`) | ✅ |
| **H36M** | train | motion + cached feats (`vitfeat_h36m.pt`, held in RAM) | ✅ |
| **3DPW** | train + eval | refit SMPL-X + cached feats | ✅ |
| **EMDB / RICH** | eval | world-traj / SMPL-X + feats | ✅ (EMDB: institutional email) |
| **SMPL / SMPL-X** | body models | `inputs/checkpoints/body_models/{smplx,smpl}/…` (neutral required) | ✅ |

The mixed recipe needs AMASS + BEDLAM + H36M + 3DPW-train for `fit`, and EMDB + RICH + 3DPW-test for val.
The **smoke** config needs only 3DPW + body models. EnDecoder normalization stats and all joint regressors
are **shipped in-repo** (`stats_compose.py`, `gvhmr/utils/body_model/`) — no download.

## Notes & landmines

- **Device-aware** (`gvhmr/cli/train.py`): honours `$GVHMR_DEVICE` (cuda→mps→cpu). CUDA behaviour is
  unchanged; off-CUDA forces fp32 + 1 device. The augmentation (`get_wham_aug_kp3d`) dispatches by device —
  the CUDA path is byte-preserved (cached device-pinned variants), CPU/MPS uses the equivalent CPU variants.
- **torch ≥ 2.6 compat:** all dataset/model `torch.load` of trusted local packs now pass `weights_only=False`
  (the default flipped to `True` in torch 2.6 and refuses numpy-bearing checkpoints).
- **RNG order is load-bearing** (`docs/BEHAVIOR.md`): the augmentors mix `np.random`/`torch`/CUDA RNG across
  five layers; **don't reorder them** or the training stream shifts. (A snapshot test for this is planned —
  `docs/EXTENSIBILITY.md` Phase B1.)
- **Retraining on a new feature backbone** (Tier B keystone): re-extract the cached features with the new
  extractor, set `network.imgseq_dim`, and retrain. See `docs/EXTENSIBILITY.md` Phases B2–B4.
- The broken `siga24_release.yaml` (references a non-existent `NetworkEncoderRoPEV2`) is **not** the training
  path — `mixed` composes the valid `relative_transformer` (`docs/BEHAVIOR.md`).
