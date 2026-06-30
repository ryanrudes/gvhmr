# AGENTS.md — GVHMR

Guide for AI agents (and humans) working in this repo. `CLAUDE.md` is a symlink to
this file. Read this before making changes.

## What this is

GVHMR — *World-Grounded Human Motion Recovery via Gravity-View Coordinates*
(SIGGRAPH Asia 2024). Given a video, it recovers SMPL/SMPL-X human motion in both
camera and world frames. PyTorch + PyTorch-Lightning + Hydra.

This repository is a **modernized fork** of [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR).
The goal of the modernization was cleaner code, modern packaging, a real test
suite, Apple-Silicon (MPS) support, and agent-friendliness — **without changing
what the model computes**. Behaviour preservation is the prime directive.

## Architecture (data flow)

```
video ─► preproc ─► load_data_dict ─► DemoPL.predict ─► Pipeline.forward ─► render
         │           (bbx, kp2d,        (gvhmr/model)    │  NetworkEncoderRoPE
         │            f_imgseq, cam)                      │  (the RoPE denoiser)
         │                                                ▼  EnDecoder: 151-dim latent
         ▼                                                   ⇄ in-cam + world SMPL
   YOLO track → ViTPose 2D kp → HMR2 ViT features → SimpleVO/DPVO camera
```

| Area | Path | Role |
|---|---|---|
| Package | `gvhmr/` | all library code (import name `gvhmr`; distribution name `gvhmr`) |
| Configs | `gvhmr/configs/` | Hydra + hydra-zen. `register_store_gvhmr()` populates the `MainStore` by importing modules that self-register via `MainStore.store(...)` |
| Network | `gvhmr/network/` | `gvhmr/relative_transformer.py::NetworkEncoderRoPE` (trained denoiser); `hmr2/` (vendored ViT feature extractor) |
| Model | `gvhmr/model/gvhmr/` | Lightning modules, `pipeline/`, `utils/endecoder.py` (151-dim latent ⇄ SMPL), `utils/postprocess.py`, metric `callbacks/` |
| Core utils | `gvhmr/utils/` | geometry, body models, eval, vis, device, IO |
| CLI/scripts | `tools/` | `demo/demo.py`, `train.py`, etc. (Hydra/argparse wiring only) |
| Vendored | `gvhmr/utils/_vendor/`, `gvhmr/network/hmr2/`, `gvhmr/utils/preproc/vitpose_pytorch/` | frozen third-party code (see Provenance) |

See `docs/ARCHITECTURE.md` for the full map.

## Environment & dev commands

Uses **uv** + a **hatchling** build backend. Base install runs on CPU / Apple-Silicon
MPS; heavy/GPU-only pieces are optional extras.

```bash
uv sync --extra dev          # base + test/lint/typecheck tooling (works on macOS)
uv run gvhmr --help          # the CLI (Typer + Rich); `gvhmr info` for a diagnostic
uv run gvhmr demo VIDEO -s   # run the demo (needs --extra preproc; render needs pytorch3d)
uv run pytest                # run the test suite (CPU/MPS, no GPU/checkpoints/datasets)
uv run ruff check gvhmr tools tests
uv run ruff format gvhmr tools tests
uv run pyright               # type-check (vendored trees excluded)
```

## CLI & console output

The CLI is a **Typer** app in `gvhmr/cli/` (`gvhmr demo`/`demo-folder`/`train`/`bench`/`info`);
the `tools/` scripts are thin backward-compat shims. **All console output goes through the
one shared Rich console in `gvhmr/utils/console.py`** — use `Log` (Rich-backed logging),
`track(...)` for progress bars (drop-in for tqdm), `status(...)` for spinners, `rule(...)`
for section dividers, and `console.print(...)`. Don't use bare `print()` or `tqdm` in
first-party code (vendored trees keep theirs). Command bodies in `gvhmr/cli/__init__.py`
lazy-import their heavy implementations so `--help`/`info` stay instant.

**Skeleton overlays.** Besides the in-cam/world *mesh* videos, `gvhmr demo` can export the SMPL
24-joint skeleton (`gvhmr/utils/vis/skeleton.py` → spheres-at-joints + cylinders-at-bones, a normal
mesh the moderngl renderer draws): `--skeleton` (world-frame skeleton-only video), `--skeleton-overlay`
(mesh videos with the skeleton drawn *on top* — `extra_meshes` + `extra_on_top` on the GL renderer's
`render_mesh`/`render_with_ground`), and `--skeleton-joints` for a subset (groups like `legs`/`left_arm`,
or joint names/indices; a bone draws only when both endpoints are kept). Left side warm, right cool.

Extras: `preproc` (YOLO/ViTPose/pycolmap), `vis` (wis3d/viser), `notebook`, `render`
(optional pytorch3d fallback). Mesh rendering works out of the box (moderngl is a base dep).

**CUDA torch (Linux).** uv can't auto-pick a CUDA build for `uv sync` (`--torch-backend=auto` is
`uv pip`-only) and a lock can't gate wheels on CUDA version — so the CUDA build is an explicit, mutually-
exclusive **extra**: `cpu` / `cu124` / `cu126` / `cu128` (route torch/torchvision to a PyTorch index;
declared `conflicts` in `[tool.uv]`). `uv sync --extra cu128` (pick nearest ≤ `nvidia-smi`'s CUDA;
cu128 covers 12.8–13.x via back-compat). They pin **torch < 2.8** — newer wheels have a broken `nvshmem`
dep that won't import. macOS uses bare `uv sync` (MPS). CI uses `--extra cpu`.

**DPVO** (CUDA-only SLAM) is installed by `scripts/setup_dpvo.sh` (not a uv extra — it compiles CUDA
extensions): the script detects the CUDA version → `uv sync --extra cuXXX` → builds DPVO from a thin fork
(`ryanrudes/DPVO`) that vendors Eigen 3.4.0 + carries modern-PyTorch build patches (`.scalar_type()`
dispatch, `loop_closure` packaging, `torch.amp`). The vendored `gvhmr/utils/preproc/dpvo_default.yaml`
lets the pip-installed `dpvo` find its config. DPVO lives outside the lock, so use `UV_NO_SYNC=1` (or
pass `--extra cuXXX` consistently) to keep a bare `uv sync` from pruning it. See `docs/INSTALL.md`.

## Device / MPS

Never hard-code `.cuda()`. Use `gvhmr/utils/device.py`:
`get_device()` (honours `$GVHMR_DEVICE`, else cuda→mps→cpu), `to_device(obj, device)`,
`device_name(device)`, `synchronize(device)`. In LightningModules use `self.device`.
Core inference + geometry run on MPS; **mesh rendering runs on the GPU** via a moderngl
(Metal/OpenGL) renderer (`gvhmr/utils/vis/renderer_gl.py`, picked by `make_renderer`); only
**DPVO is CUDA-only**. The legacy pytorch3d renderer is a CPU/CUDA fallback.

**Scene-aware camera on Mac.** DPVO (the only camera backend that recovers *translation*) is CUDA-only,
so `gvhmr demo --slam dust3r` provides a device-agnostic alternative: `gvhmr/utils/preproc/dust3r_slam.py`
reconstructs the scene with vendored **DUSt3R** (pure-PyTorch, MPS) and fixes the global scale against a
**Depth-Anything-V2** metric-depth prediction → a per-frame *metric* `T_w2c`. For a moving camera the demo
then composes `world = T_c2w_metric · in-cam` (the in-cam carry through the metric camera, gravity-aligned
+ frame-aligned-fused with the prior's local motion) — recovering the global traversal a following camera
induces, which the velocity prior misses. Weights live in `~/Datasets/GVHMR/{dust3r,depth_anything}/`.

## Debugging

`gvhmr/utils/debug.py`: `describe(tensor)`, `decompose_latent(pred_x)` (split the 151-dim
latent by name), `nan_hooks(model)` (locate non-finite outputs), `count_parameters`.
3D viz via `gvhmr/utils/wis3d_utils.py` (`vis` extra). See `docs/DEBUGGING.md`.

## Accuracy

Test-time accuracy levers (no retraining) live behind opt-in flags so the default
`predict` path stays **byte-identical** (golden-guarded): `--flip-test` (mirror-averaging
TTA, ported from the eval into `DemoPL.predict`), `--f_mm` / metadata focal, and the
automatic SimpleVO carry-forward. Iterate with the 2D-reprojection + jitter proxies (no GT
needed) — but reprojection is in-cam/depth-ambiguous and **gameable** (drop-imgseq, SMPLify
refinement), so always pair it with jitter. Full evidence + rejected ideas in `docs/ACCURACY.md`.

## Performance

Profile/bench with `gvhmr bench`. Optimizations must keep
`tests/test_golden_inference.py` green (golden output from the released checkpoint).
Note: the GVHMR model `predict` is **faster on CPU** than MPS (latency-bound on the IK's
small ops); MPS helps the batched preproc models. See `docs/PERFORMANCE.md`.

## Conventions

- **Python ≥ 3.11, target 3.13.** Modern typing: PEP 585/604 (`list[int]`, `X | None`),
  `from __future__ import annotations` where helpful. No `from typing import List/Optional`.
- **Line length 120**, formatter is `ruff format`.
- **Behaviour preservation is non-negotiable.** Before touching numeric/model code,
  read `docs/BEHAVIOR.md` and the landmines below. Pin behaviour with a test *first*.
- **Vendored code is frozen.** Don't refactor `gvhmr/network/hmr2`,
  `gvhmr/utils/preproc/vitpose_pytorch`, or `gvhmr/utils/_vendor/**`. Minimal compat
  patches are allowed but must be tagged `# [GVHMR vendor patch]` and documented.
- **Rotations** go through the facade `gvhmr.utils.geo.rotations` (a frozen, byte-identical
  copy of pytorch3d's rotation conversions), *not* `pytorch3d.transforms`.

## Behaviour landmines (read before refactoring numeric code)

1. **151-dim latent layout** in `EnDecoder.decode`: `[0:126]body_r6d / 126:136 betas /
   136:142 global_orient_c / 142:148 global_orient_gv / 148:151 transl_vel`. `stats_compose`
   vectors must keep this order. Reordering silently corrupts output.
2. **Checkpoints load `strict`** on module-attribute names. Renaming a module attribute or
   changing a ctor default (`output_dim=151`, `latent_dim=512`, `num_layers=12`, …) breaks loading.
3. **Byte-exact constants** matter: `pred_cam_mean/std`, clamp 0.25, rotary base 10000,
   GELU `approximate='tanh'`, eval foot verts `[3216,3387,6617,6787]`, crops `[:,:,:,32:224]`/`[:,:,:,32:-32]`.
4. **`torch.svd` → `torch.linalg.svd` returns Vᴴ, not V** — transpose, or rotations invert.
5. **`autocast(enabled=False)` must stay disabled** (fp32 for FK/IK/rotary). Modern form:
   `torch.amp.autocast("cuda", enabled=False)`.
6. **`siga24_release.yaml` references `NetworkEncoderRoPEV2`, which doesn't exist** (only V1).
   The live demo path uses `demo.yaml`. Don't assume that yaml loads as-is.
7. **Dataset RNG order is load-bearing** — don't reorder `np.random` calls in augmentors.

## Tests

`tests/` is a **characterization / regression net**: it pins the pure math the model
depends on (quaternions, rotary embeddings, eval/SVD alignment, camera intrinsics, the
151-dim stats layout, config composition, MPS parity) on CPU/MPS — no GPU, checkpoints,
or datasets needed. When you change behaviour-sensitive code, the relevant test must
stay green; when you fix a bug, add/De-xfail a test. Markers: `gpu`, `checkpoint`,
`dataset`, `pytorch3d` (auto-skip).

## Syncing from the original repo

This fork renamed the package and vendored some code, so don't `git merge` upstream blindly.
```bash
git remote add upstream https://github.com/zju3dv/GVHMR   # one time
python scripts/upstream_sync.py                            # maps upstream changes → local paths
python scripts/upstream_sync.py --since <ref> --show-diff  # see the patches
```
The mapping lives in `docs/upstream_sync.yaml` — **update it whenever you move/split/rename code.**
See `docs/PROVENANCE.md`.
