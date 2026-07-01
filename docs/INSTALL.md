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

## CUDA / GPU (Linux)

On **macOS you're done** — bare `uv sync` installs the Apple-Silicon **MPS** build. On **Linux/CUDA**,
bare `uv sync` installs the default PyPI torch wheel, which targets one recent CUDA (13.x) and **won't
see the GPU on most drivers**. uv can't auto-detect your GPU for `uv sync` (`--torch-backend=auto` is
`uv pip`-only) and a lockfile can't gate wheels on CUDA version — so the CUDA build is an explicit
one-of choice. Check `nvidia-smi` and add the matching extra (nearest one ≤ your CUDA version):

```bash
nvidia-smi                              # → e.g. "CUDA Version: 12.8"
uv sync --extra cu128                   # torch for CUDA 12.8 — covers 12.8 … 13.x (driver back-compat)
uv sync --extra cu128 --extra preproc   # combine with other extras as usual
```

| Your CUDA (`nvidia-smi`) | Extra | torch |
|---|---|---|
| 12.0 – 12.5 | `cu124` | 2.6.x |
| 12.6 / 12.7 | `cu126` | 2.7.x |
| 12.8 / 12.9 / 13.x | `cu128` | 2.7.x |
| CPU-only Linux / CI | `cpu` | CPU build |

The CUDA extras are mutually exclusive and pin torch **< 2.8** (newer wheels carry a broken `nvshmem`
dependency that fails to import). `gvhmr info` shows the resolved torch and `cuda ✓` once synced.

> On a GPU box, **pass your extra every time** — `uv sync --extra cu128`, `uv run --extra cu128 …` — or
> set `UV_NO_SYNC=1`; a bare `uv sync` reverts torch to the PyPI default.

## Extras

```bash
uv sync --extra dev           # tests + ruff + pyright
uv sync --extra preproc       # YOLO + ViTPose + pycolmap (per-video preprocessing)
uv sync --extra rtmpose       # RTMPose 2D-pose backend (`--pose2d rtmpose`; rtmlib + ONNXRuntime)
uv sync --extra vis           # wis3d / viser (interactive 3D debugging)
uv sync --extra notebook      # jupyter / ipython / ipdb
uv sync --all-extras          # everything pip-resolvable
```

| Extra | Provides | Notes |
|---|---|---|
| `dev` | pytest, ruff, pyright | runs the full test suite on CPU/MPS |
| `preproc` | ultralytics, cython_bbox, lapx, pycolmap | needs model weights (below); SimpleVO uses pycolmap |
| `rtmpose` | rtmlib, onnxruntime | alternative 2D-pose backend (`--pose2d rtmpose`); model auto-downloads |
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
  CUDA toolchain (`nvcc`); no manual Eigen download:
  ```bash
  scripts/setup_dpvo.sh         # detects your CUDA → syncs the matching torch → builds DPVO
  ```
  It reads your toolkit version, runs `uv sync --extra cuXXX` for the matching torch (see *CUDA / GPU*
  above), then builds DPVO from a [thin fork](https://github.com/ryanrudes/DPVO) that vendors Eigen
  3.4.0 and carries the minimal modern-PyTorch build patches (`.scalar_type()` dispatch, `loop_closure`
  packaging, `torch.amp`). Then fetch the weight to `inputs/checkpoints/dpvo/dpvo.pth` (see *Weights &
  data*) and run with `--use-dpvo`.

  > DPVO is installed out-of-band (it's CUDA-only, so it can't live in the lock), which means a bare
  > `uv sync` / plain `uv run` would prune it and revert torch. On a GPU box set `export UV_NO_SYNC=1`
  > once (e.g. in `~/.bashrc`), then `uv run gvhmr demo VIDEO --use-dpvo` works; or
  > `source .venv/bin/activate && gvhmr demo VIDEO --use-dpvo`. When you *do* re-sync, keep your
  > backend (`uv sync --extra cuXXX`). Re-running `scripts/setup_dpvo.sh` recovers it (idempotent).

  On Mac/MPS, where DPVO can't build, use `gvhmr demo VIDEO --slam dust3r` instead.

## Apple Silicon (MPS)

Base install runs core GVHMR inference and all geometry on the Apple-Silicon GPU.
Selection is automatic (cuda → mps → cpu); override with `GVHMR_DEVICE=mps|cpu|cuda`.
**Caveats:** mesh rendering (pytorch3d) and DPVO are CUDA-only; some preprocessing
models hard-code CUDA. So on a Mac you can run the model and the geometry; full
end-to-end video rendering still needs a CUDA box.

## Weights & data

**Fetch the checkpoints with one command** — they land in the right place automatically (no manual
download / placement):

```bash
uv run gvhmr download            # demo checkpoints: gvhmr + hmr2 + vitpose + yolo
uv run gvhmr download slam       # + DPVO
uv run gvhmr download all        # every checkpoint
uv run gvhmr info                # verify what's present / missing
```

Files go to `inputs/checkpoints/` by default; set **`$GVHMR_CHECKPOINTS`** to keep large weights
elsewhere (e.g. a shared `~/Datasets` dir) — one env var, no symlinks. Every checkpoint path in the code
resolves through that root (`gvhmr/utils/assets.py`). Source: the HuggingFace mirror
[`camenduru/GVHMR`](https://huggingface.co/camenduru/GVHMR) (resumable, checksummed).

**Body models are registration-gated** and can't be auto-fetched — sign up for
[SMPL](https://smpl.is.tue.mpg.de/) + [SMPL-X](https://smpl-x.is.tue.mpg.de/), then place them under
`$GVHMR_BODY_MODELS` (default `inputs/checkpoints/body_models/`):

```
body_models/
├── smplx/SMPLX_{GENDER}.npz
└── smpl/SMPL_{GENDER}.pkl
```

`gvhmr download` prints these instructions when they're missing.

**Training/eval data packs** (the `hmr4d_support` bundles) are on the same mirror:

```bash
uv run gvhmr download --data 3dpw,amass,h36m    # downloads + extracts under inputs/<DS>/hmr4d_support/
```

Available: `3dpw amass h36m emdb rich bedlam` (BEDLAM is ~21 GB). See [`docs/TRAINING.md`](TRAINING.md)
for the full matrix + which raw datasets are registration-gated.
