# Architecture

GVHMR recovers world-grounded SMPL/SMPL-X human motion from video. It is a single
package `gvhmr/`, driven by Hydra config composition and exposed through the `gvhmr`
CLI (`gvhmr/cli/`, a Typer + Rich app).

## Inference data flow

```
raw video
  │
  ├─ run_preprocess (disk-cached)                         gvhmr/utils/preproc/
  │     ├─ Tracker (YOLOv8)            → bbx
  │     ├─ VitPoseExtractor            → kp2d (COCO-17)
  │     ├─ Extractor (HMR2 ViT)        → f_imgseq (1024-d tokens)
  │     └─ SimpleVO (pycolmap) / DPVO  → camera rotation
  │
  ├─ load_data_dict → {length, bbx_xys, kp2d, K_fullimg, cam_angvel, f_imgseq}
  │
  ├─ DemoPL.predict                                       gvhmr/model/gvhmr/gvhmr_pl_demo.py
  │     └─ Pipeline.forward                               gvhmr/model/gvhmr/pipeline/
  │           ├─ NetworkEncoderRoPE (RoPE transformer)    gvhmr/network/gvhmr/
  │           │     → pred_x (151-d latent), pred_cam, static_conf
  │           ├─ EnDecoder.decode (151-d ⇄ SMPL)          gvhmr/model/gvhmr/utils/endecoder.py
  │           └─ postprocess (static-joint + CCD IK)      gvhmr/model/gvhmr/utils/postprocess.py
  │
  └─ render_incam / render_global (pytorch3d)             gvhmr/utils/vis/renderer.py
```

## Package layout

| Path | Contents |
|---|---|
| `gvhmr/configs/` | Hydra root YAMLs (`demo`, `train`, `siga24_release`) + `store_gvhmr.py`. `register_store_gvhmr()` imports the modules that self-register options into `MainStore` (a `hydra.ConfigStore`). |
| `gvhmr/network/gvhmr/` | `relative_transformer.py::NetworkEncoderRoPE` — the trained denoiser. |
| `gvhmr/network/base_arch/` | RoPE attention + embeddings the denoiser is built from. |
| `gvhmr/network/hmr2/` | **vendored** 4D-Humans ViT, used (frozen) as the per-frame feature extractor. |
| `gvhmr/model/gvhmr/` | Lightning modules (`gvhmr_pl`, `gvhmr_pl_demo`), `pipeline/`, `utils/` (endecoder, postprocess, stats), metric `callbacks/`. |
| `gvhmr/model/common_utils/` | optimizer + LR-scheduler config factories. |
| `gvhmr/dataset/`, `gvhmr/datamodule/` | training/eval datasets (AMASS, BEDLAM, H36M, 3DPW, EMDB, RICH). |
| `gvhmr/utils/geo*`, `gvhmr/utils/matrix.py` | geometry: transforms, quaternions, camera math. |
| `gvhmr/utils/geo/rotations.py` | **facade** re-exporting the vendored pytorch3d rotation conversions (the single import point for rotation math). |
| `gvhmr/utils/_vendor/pytorch3d/` | frozen, byte-identical pytorch3d `rotation_conversions`/`so3`/`math`. |
| `gvhmr/utils/body_model/` | SMPL/SMPL-X wrappers + committed `.pt` regressor assets. |
| `gvhmr/utils/device.py` | device selection (cuda/mps/cpu) + tensor movement. |
| `gvhmr/utils/{eval,vis,ik,kpts,comm}/`, `*_utils.py` | metrics, rendering, IK, IO, logging. |
| `gvhmr/cli/` | the **`gvhmr` CLI** (Typer + Rich): `demo`, `demo-folder`, `train`, `bench`, `info`. The demo pipeline lives here (`cli/demo.py`). |
| `gvhmr/utils/console.py` | the single shared Rich console — all logging (`pylogger`), progress (`track`), and prints flow through it. |
| `tools/` | thin backward-compat shims (`demo/demo.py`, `train.py`, …) that forward to the CLI; plus `video/`. |
| `tests/` | CPU/MPS characterization suite (see `docs/BEHAVIOR.md`). |

## Config system

Config is **inversion-of-control**: ~20 leaf modules call
`MainStore.store(name=…, node=builds(Cls, …), group=…)` at import time. `tools/` then
`compose()` a `DictConfig` and `hydra.utils.instantiate(..., _recursive_=False)` it.
`register_store_gvhmr()` triggers all the self-registrations by importing
`gvhmr/configs/store_gvhmr.py`.

`gvhmr.PROJ_ROOT` (= the repo root, the parent of the `gvhmr/` package) is the anchor for
`inputs/`, `outputs/`, and committed package-data paths.

## The 151-dim latent

`NetworkEncoderRoPE` predicts a 151-d per-frame vector that `EnDecoder` maps to SMPL:
`[0:126]` body pose (6D × 21 joints), `[126:136]` betas, `[136:142]` global orient
(camera frame), `[142:148]` global orient (gravity-view frame), `[148:151]` local
translation velocity. This layout is a hard contract — see `docs/BEHAVIOR.md`.
