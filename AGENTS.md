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
   YOLO track → ViTPose 2D kp → HMR2 ViT features → SimpleVO/DPVO/DUSt3R/VGGT camera
```

Each preprocessing stage (detector / 2D-pose / feature-backbone / camera) is a **swappable Hydra config
group** — pick an implementation by name, or bundle choices into a recipe (see CLI, and `docs/CONFIGURATION.md`).

| Area | Path | Role |
|---|---|---|
| Package | `gvhmr/` | all library code (import name `gvhmr`; distribution name `gvhmr`) |
| Configs | `gvhmr/configs/` | Hydra + hydra-zen. `register_store_gvhmr()` populates the `MainStore` (Python-defined model/network groups); swappable **preproc stages are YAML config groups** under `configs/{detector,pose2d,backbone,camera}/` (+ `recipe/`) |
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
scripts/install.sh           # user-facing one-command install (platform/GPU detect → uv sync → records [env]); --dev adds tooling
uv sync --extra dev          # base + test/lint/typecheck tooling (works on macOS)
bin/gvhmr --help             # the CLI (Typer + Rich) WITHOUT uv's auto-re-sync; `gvhmr info` for a diagnostic
bin/gvhmr demo VIDEO -s      # run the demo (needs --extra preproc; rendering is base — moderngl)
bin/gvhmr env sync           # replay the recorded env through uv (--inexact — never prunes DPVO/extras)
make check                   # the REQUIRED CI gates locally (ruff format --check + pytest) — run before pushing
make fmt                     # format the WHOLE tree; make lint / typecheck / test are the rest
uv run pre-commit install    # (or `make hooks`) once — auto-runs `ruff format` on commit, pinned to CI's ruff
```

**Users should never need raw uv.** `scripts/install.sh` bootstraps then hands off to the
`gvhmr config init` wizard, which walks every optional component (the registries
`EXTRA_COMPONENTS`/`SCRIPT_COMPONENTS` in `gvhmr/cli/envcmd.py` — keep them in sync with pyproject
extras and setup scripts; a test enforces it) and records the box's torch build + extras + dpvo/scene
in the config file's `[env]` table. `gvhmr env sync` replays them with `--inexact` so a sync never
prunes the out-of-band DPVO or a CUDA torch. `bin/gvhmr` is the no-re-sync wrapper (plain `uv run`
re-syncs to the lock's defaults first — the classic trap; the Makefile uses `uv run --no-sync` for the
same reason). `gvhmr info` detects env drift and points at `gvhmr env sync`.

**Formatting must match CI.** CI's one required style gate is `ruff format --check gvhmr tools tests` (the
lint/pyright jobs are advisory). Format the **whole tree** (`make fmt`), not just the files you touched —
formatting per-file is how the gate drifted red. The pre-commit hook (`.pre-commit-config.yaml`, pinned to
the same ruff version) enforces this on commit; `make check` reproduces the full required gate locally.

## CLI & console output

The CLI is a **Typer** app in `gvhmr/cli/`
(`gvhmr demo`/`demo-folder`/`train`/`eval`/`bench`/`info`/`download`/`extract-features`/`config`/`env`);
the `tools/` scripts are thin backward-compat shims. **All console output goes through the
one shared Rich console in `gvhmr/utils/console.py`** — use `Log` (Rich-backed logging),
`track(...)` for progress bars (drop-in for tqdm), `status(...)` for spinners, `rule(...)`
for section dividers, and `console.print(...)`. Don't use bare `print()` or `tqdm` in
first-party code (vendored trees keep theirs). Command bodies in `gvhmr/cli/__init__.py`
lazy-import their heavy implementations so `--help`/`info` stay instant.

**Swappable preprocessing (config groups).** The detector, 2D-pose, feature backbone, and camera are each a
Hydra config **group** (`gvhmr/configs/{detector,pose2d,backbone,camera}/`), shared with `train` — the same
mechanism as the model/network groups. Choose an implementation by name (`--detector`/`--pose2d`/
`--backbone`/`--camera`), bundle a set of choices into a committable `--recipe` (`configs/recipe/`), or tweak
any knob with `--set key=val` (precedence: recipe → name flag → `--set`). Defaults are the released models,
so the default `predict` path stays **golden-byte-identical**. Real alternatives today: `--detector yolo26x`
(every YOLO family×size is a preset; any other via `--detector-ckpt`),
`--pose2d rtmpose` (needs `--extra rtmpose`), `--camera dust3r|vggt`, `--backbone dinov2` (needs a retrain).
`gvhmr extract-features VIDEOS OUT --backbone <name>` writes the training feature cache for a backbone swap.
Full guide `docs/CONFIGURATION.md`; roadmap/rationale `docs/EXTENSIBILITY.md`; training `docs/TRAINING.md`.

**Skeleton overlays.** Besides the in-cam/world *mesh* videos, `gvhmr demo` can export the SMPL
24-joint skeleton (`gvhmr/utils/vis/skeleton.py` → spheres-at-joints + cylinders-at-bones, a normal
mesh the moderngl renderer draws): `--skeleton` (world-frame skeleton-only video), `--skeleton-overlay`
(mesh videos with the skeleton drawn *on top* — `extra_meshes` + `extra_on_top` on the GL renderer's
`render_mesh`/`render_with_ground`), and `--skeleton-joints` for a subset (groups like `legs`/`left_arm`,
or joint names/indices; a bone draws only when both endpoints are kept). Left side warm, right cool.

Extras: `preproc` (YOLO/ViTPose/pycolmap — **`gvhmr demo` needs it**), `rtmpose` (rtmlib/onnxruntime —
the alt 2D-pose backend), `train` (wandb + tensorboard loggers), `dpvo` (numba/pypose — DPVO's locked,
torch-ABI-free runtime deps; see DPVO below), `vis` (wis3d/viser), `notebook`, `render` (optional
pytorch3d fallback). Mesh rendering works out of the box (moderngl is a base dep).

**CUDA torch (Linux).** uv can't auto-pick a CUDA build for `uv sync` (`--torch-backend=auto` is
`uv pip`-only) and a lock can't gate wheels on CUDA version — so the CUDA build is an explicit, mutually-
exclusive **extra**: `cpu` / `cu124` / `cu126` / `cu128` (route torch/torchvision to a PyTorch index;
declared `conflicts` in `[tool.uv]`). `uv sync --extra cu128` (pick nearest ≤ `nvidia-smi`'s CUDA;
cu128 covers 12.8–13.x via back-compat; **V100/P100 need cu126** — cu128 dropped sm_70/sm_60). They pin
**torch < 2.8** — newer wheels have a broken `nvshmem` dep that won't import. macOS uses bare `uv sync`
(MPS). CI uses `--extra cpu`. `scripts/install.sh` picks all of this automatically and records it in
`[env]`, so `gvhmr env sync` can replay it.

**DPVO** (CUDA-only SLAM) is set up by `scripts/setup_dpvo.sh`: it detects the CUDA version →
`uv sync --extra cuXXX --extra preproc --extra dpvo` → compiles the CUDA pieces (dpvo + torch-scatter,
which can't live in the lock) from a thin fork (`ryanrudes/DPVO`) that vendors Eigen 3.4.0 + carries
modern-PyTorch build patches (`.scalar_type()` dispatch, `loop_closure` packaging, `torch.amp`) → records
`dpvo = true` in `[env]`. The torch-ABI-free runtime deps (numba/pypose) ARE locked, via the `dpvo`
extra — that's what pins numpy where numba needs it (a sync used to float numpy past numba's cap and
break every DPVO box). The vendored `gvhmr/utils/preproc/dpvo_default.yaml` lets the pip-installed
`dpvo` find its config. The compiled bits still live outside the lock, so avoid bare `uv sync` / plain
`uv run` — `bin/gvhmr` + `gvhmr env sync` (`--inexact`) are the safe paths, and re-running the script
recovers a pruned DPVO (idempotent). See `docs/INSTALL.md`.

**Asset roots, config file & fetching.** Machine-local settings — where large assets live (`checkpoints` /
`data` / `body_models` / `scene`), the default model version per stage, and the recorded environment
(`[env]`) — live in one readable TOML file (**`<repo>/gvhmr.toml`**, gitignored; lookup `$GVHMR_CONFIG`
(authoritative when set) → `./gvhmr.toml` → `<repo>/gvhmr.toml` → legacy `~/.config/gvhmr/config.toml`),
managed with **`gvhmr config`** (`init` wizard / `show` / `set`). Everything
resolves through `gvhmr/utils/localconfig.py` (`resolve()`) + `gvhmr/utils/assets.py` (`ROOTS`), with
precedence **env var / CLI flag > config file > default** — so the file is the friendly baseline and the
`$GVHMR_*` env vars (`GVHMR_CHECKPOINTS` / `GVHMR_BODY_MODELS` / `GVHMR_DATA_ROOT` / `GVHMR_DATA`) still win
for CI / one-offs. The test suite is hermetic to it (conftest points `$GVHMR_CONFIG` at a non-file). `gvhmr download [demo|slam|all]` fetches checkpoints and `gvhmr download --data <DS,…>`
the packs (HF mirror `camenduru/GVHMR`) into those roots; `gvhmr demo` auto-fetches missing checkpoints and
reads the config `[models]` defaults; `gvhmr info` / `gvhmr config show` show what's present + where. Body
models are registration-gated (can't auto-download — the tooling prints the sign-up + target path).

## Python library (`gvhmr/inference/`)

A HuggingFace-style API over the demo pipeline, re-exported at the top level and **lazy-loaded** via
`gvhmr/__init__.py::__getattr__` (so `import gvhmr` stays torch-free). It **reuses the exact `gvhmr demo`
code paths** (`gvhmr/cli/demo.py`: `build_demo_cfg`/`run_preprocess`/`load_data_dict`/`recover_motion`/
`_render`) — behaviour-preserving, so `tests/test_golden_inference.py` guards it. Surface:
`gvhmr.pipeline(task, *, model, device)` (task aliases in `pipelines.TASKS`) → `GVHMRPipeline`;
`gvhmr.recover(video, **call_kwargs)` one-liner (caches a pipeline per `(model, device)`);
`gvhmr.GVHMR`/`GVHMRPipeline`/`MotionResult`. `GVHMRPipeline.__call__(video, *, static_camera, camera,
detector/pose2d/backbone, f_mm, flip_test, world_from_incam, render, output_dir, recipe, set_overrides,
progress, …)` returns a `MotionResult` (`result.py`) — `smpl_params_world`/`_camera`, `intrinsics`, and
**lazy** `vertices_*`/`joints_*`/`faces` (need body models via `_smpl.py` → `smplx2smpl` map + J
regressor, byte-identical to the renderer); `.save()`/`.save_npz()`/`.render(view=…)`. The "power path"
`GVHMR.from_pretrained(...).predict(data, …)` is tensor-level (bring-your-own preproc). Hub I/O in
`hub.py` (`DEFAULT_REPO=ryanrudes/gvhmr`, mirror fallback `camenduru/GVHMR`) — checkpoints only. **Body
models are never re-hosted**: gated MPI SMPL/SMPL-X fetched per-user via `gvhmr auth smpl` / `$SMPLX_USER`
/`$SMPLX_PW` (`utils/mpi_download.py`, `cli/hubcmd.py`). Publish: `gvhmr publish-hub`/`publish-space`.
Full guide `docs/LIBRARY.md`.

## Device / MPS

Never hard-code `.cuda()`. Use `gvhmr/utils/device.py`:
`get_device()` (honours `$GVHMR_DEVICE`, else cuda→mps→cpu), `to_device(obj, device)`,
`device_name(device)`, `synchronize(device)`. In LightningModules use `self.device`.
Core inference + geometry run on MPS; **mesh rendering runs on the GPU** via a moderngl
(Metal/OpenGL) renderer (`gvhmr/utils/vis/renderer_gl.py`, picked by `make_renderer`); only
**DPVO is CUDA-only**. The legacy pytorch3d renderer is a CPU/CUDA fallback.

**Scene-aware camera on Mac.** DPVO (the only *built-in* backend that recovers *translation*) is CUDA-only,
so `gvhmr demo --camera dust3r` (or `--camera vggt`) provides device-agnostic alternatives that run on
Apple-Silicon MPS: `dust3r_slam.py` reconstructs the scene with vendored **DUSt3R** + a global-alignment
optimizer, while `vggt_slam.py` uses **VGGT** (one feed-forward pass, often faster/more robust). Both are
scale-ambiguous, so **Depth-Anything-V2** metric depth fixes the global scale → a per-frame *metric* `T_w2c`.
For a moving camera the demo then composes `world = T_c2w_metric · in-cam` (the in-cam carry through the
metric camera, gravity-aligned + frame-aligned-fused with the prior's local motion) — recovering the global
traversal a following camera induces, which the velocity prior misses. Set both up with
`scripts/setup_scene_aware.sh` (clones into `third-party/`; DUSt3R/Depth-Anything weights →
`~/Datasets/GVHMR/{dust3r,depth_anything}/`, VGGT auto-downloads `facebook/VGGT-1B`). **Note:** don't
`pip install -e` VGGT — its `numpy<2` pin breaks scipy; it's imported via sys.path and runs on numpy 2.x.
`--slam` / `--use-dpvo` remain as deprecated aliases for `--camera`.

## Debugging

`gvhmr/utils/debug.py`: `describe(tensor)`, `decompose_latent(pred_x)` (split the 151-dim
latent by name), `nan_hooks(model)` (locate non-finite outputs), `count_parameters`.
3D viz via `gvhmr/utils/wis3d_utils.py` (`vis` extra). See `docs/DEBUGGING.md`.

## Accuracy & evaluation

Test-time accuracy levers (no retraining) live behind opt-in flags so the default
`predict` path stays **byte-identical** (golden-guarded): `--flip-test` (mirror-averaging
TTA, ported from the eval into `DemoPL.predict`), `--f_mm` / metadata focal, and the
automatic SimpleVO carry-forward. Iterate with the 2D-reprojection + jitter proxies (no GT
needed) — but reprojection is in-cam/depth-ambiguous and **gameable** (drop-imgseq, SMPLify
refinement), so always pair it with jitter. Full evidence + rejected ideas in `docs/ACCURACY.md`.

**The paper benchmarks are `gvhmr eval`** (`gvhmr/cli/evalcmd.py` → the Lightning test tasks
`configs/global/task/gvhmr/test_*` → `callbacks/metric_{3dpw,emdb,rich}.py` →
`utils/eval/eval_utils.py`). Auto-fetches packs/ckpt, preflights the gated body models, and prints
results next to `PAPER_REFERENCE` (arXiv 2409.06662 — keep in sync if metrics change). Verified
2026-07-02: the released ckpt reproduces the paper (3DPW/EMDB-1 exact; world metrics within ~1%).
The callbacks stash `pl_module.metrics_summary` → `train.LAST_TEST_METRICS` for the summary table;
they are device-agnostic (`pl_module.device`); metric math + the golden test resolve assets through
`gvhmr.utils.assets` (never hard-code `inputs/checkpoints`). See `docs/EVAL.md`.

**Preprocessing swaps on the benchmarks:** the packs' preproc is frozen, so `gvhmr eval --detector/
--pose2d` regenerates it (`gvhmr/utils/eval/preproc_variants.py`) into `preproc_variants/<slug>/`
(canonical files untouched) and the test loaders read it via the root `preproc_variant` config key —
interpolated, because hydra's override grammar can't address the `3dpw` node (digit-leading key).
3DPW/EMDB only (RICH has no ungated videos); needs the raw videos once (`--raw-dir`); multi-person
identity is IoU-guarded against the canonical track (mismatch ⇒ canonical boxes kept + reported).
`gvhmr sweep` (`gvhmr/cli/sweepcmd.py`) grids these combos through the real W&B sweep API
(wandb.sweep/agent; trials log `<DS>/<metric>` + `…_vs_paper`; 'canonical' is the baseline value).

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
