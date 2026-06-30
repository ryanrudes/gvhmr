# Install

GVHMR uses [**uv**](https://docs.astral.sh/uv/) and installs in **layers**: a base
install that runs on CPU / Apple-Silicon **MPS** (core inference, geometry, configs,
tests), plus optional extras for the heavy/GPU-only pieces.

```bash
git clone https://github.com/zju3dv/GVHMR && cd GVHMR
uv sync                       # base install (CPU/MPS) — works on macOS out of the box
uv run gvhmr info             # device, installed features, and checkpoint status
```

`uv sync` creates `.venv/` and installs from the locked `uv.lock`. Prefix commands with
`uv run` (e.g. `uv run python …`, `uv run pytest`). No conda required.

> Coming from the old setup? `pip install -r requirements.txt && pip install -e .` is
> replaced by `uv sync`. There is no `setup.py`/`requirements.txt` anymore — everything
> is in `pyproject.toml`.

## Extras

```bash
uv sync --extra dev           # tests + ruff + pyright
uv sync --extra preproc       # YOLO + ViTPose + pycolmap (per-video preprocessing)
uv sync --extra vis           # wis3d / viser (interactive 3D debugging)
uv sync --extra notebook      # jupyter / ipython / ipdb
uv sync --all-extras          # everything pip-resolvable
```

| Extra | Provides | Notes |
|---|---|---|
| `dev` | pytest, ruff, pyright | runs the full test suite on CPU/MPS |
| `preproc` | ultralytics, cython_bbox, lapx, pycolmap | needs model weights (below); SimpleVO uses pycolmap |
| `vis` | wis3d, viser | research visualization |
| `notebook` | jupyter, ipython, ipdb | interactive work |

### Mesh rendering — works out of the box

The demo's **overlay/world videos** render on the **GPU via moderngl** (OpenGL/Metal), which
is a **base dependency** — so rendering just works after `uv sync`, no extra build, real-time
on Apple Silicon. It needs an **SMPL** body model under
`inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl` (legacy `.pkl` files load via `chumpy`
in the base deps; GVHMR auto-applies the Python-3.13 / NumPy-2 shims in
`gvhmr/utils/_smpl_compat.py`).

**pytorch3d** is only a *fallback* (CUDA boxes, or a headless machine with no GL context) and
is otherwise unnecessary. If you want it, it's wired into the `render` extra so `uv` builds it
from source and never prunes it:

```bash
uv sync --extra render        # optional fallback; MACOSX_DEPLOYMENT_TARGET=11.0 prefix if the build asks
```

> `make_renderer` (`gvhmr/utils/vis/renderer_gl.py`) prefers the GPU renderer and falls back to
> pytorch3d automatically. Everything upstream (tracking, pose, features, motion recovery) runs
> on MPS; only the optional pytorch3d *fallback* rasterizer is CPU/CUDA-only.
- **DPVO** (optional SLAM; SimpleVO is the default) — **CUDA only**. One script on a box with a
  CUDA toolchain (`nvcc`); no manual Eigen download, and it adapts to whatever CUDA the box has:
  ```bash
  scripts/setup_dpvo.sh         # syncs the project, fits torch to the box, builds DPVO
  ```
  It uses uv's `--torch-backend=auto` to install the torch wheel matching this box's driver (the
  default PyPI wheel targets one CUDA — currently 13.x — and mismatches most toolkits), then builds
  DPVO from a [thin fork](https://github.com/ryanrudes/DPVO) that vendors Eigen 3.4.0 and carries the
  minimal modern-PyTorch build patches. Why a script and not `uv sync --extra slam`: uv (0.11) can't
  auto-select the CUDA torch for the project/lock workflow (`--torch-backend` is a `uv pip` feature
  only), so a committed lock would pin one CUDA for everyone.

  Then fetch the weight to `inputs/checkpoints/dpvo/dpvo.pth` (see *Weights & data*) and run via the
  venv (so `uv run`'s auto-sync doesn't swap the matched torch back):
  ```bash
  source .venv/bin/activate && gvhmr demo VIDEO --use-dpvo
  ```
  On Mac/MPS, where DPVO can't build, use `gvhmr demo VIDEO --slam dust3r` instead.

## Apple Silicon (MPS)

Base install runs core GVHMR inference and all geometry on the Apple-Silicon GPU.
Selection is automatic (cuda → mps → cpu); override with `GVHMR_DEVICE=mps|cpu|cuda`.
**Caveats:** mesh rendering (pytorch3d) and DPVO are CUDA-only; some preprocessing
models hard-code CUDA. So on a Mac you can run the model and the geometry; full
end-to-end video rendering still needs a CUDA box.

## Weights & data

```bash
mkdir -p inputs/checkpoints outputs
```

**Body models** — sign up for [SMPL](https://smpl.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/), then place:

```
inputs/checkpoints/body_models/
├── smplx/SMPLX_{GENDER}.npz
└── smpl/SMPL_{GENDER}.pkl
```

**Pretrained models** — from the project's
[Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD):

```
inputs/checkpoints/
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
├── yolo/yolov8x.pt
└── dpvo/dpvo.pth            # only if using DPVO
```

**Training/eval data** — see the project Drive; extract under `inputs/` so the
`*/hmr4d_support/` directories sit alongside (these dataset dir names are unchanged).
