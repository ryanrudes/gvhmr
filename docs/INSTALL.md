# Install

GVHMR uses [**uv**](https://docs.astral.sh/uv/) — no conda, no `requirements.txt`. One command sets
up everything the demo needs:

```bash
git clone https://github.com/ryanrudes/gvhmr && cd gvhmr
scripts/install.sh          # detects your platform/GPU, picks the right torch build, installs the demo deps
```

The script detects macOS (Apple-Silicon MPS) vs Linux+NVIDIA (matching CUDA wheel, with a V100/P100
architecture guard) vs CPU-only, runs the right `uv sync`, **records the choices in the config file**,
offers to fetch the model checkpoints (~5.5 GB), and finishes with `gvhmr info` so you can see the
result. `scripts/install.sh --help` lists the flags (`--dev`, `--cpu`, `--cuda cu126`, `--dpvo`, `-y`,
`--no-download`).

**After that, you never need to touch uv.** Run everything through the wrapper (or the activated venv):

```bash
bin/gvhmr demo docs/example_video/tennis.mp4 -s   # the CLI, without uv's auto-re-sync
bin/gvhmr config init                             # wizard: asset locations + models + managed env
bin/gvhmr env sync                                # repair/re-apply the environment anytime
```

`bin/gvhmr` matters because a plain `uv run` first re-syncs the env to the lockfile's *defaults* —
silently reverting the CUDA torch build and pruning DPVO. The wrapper calls the venv directly, and
`gvhmr env sync` replays this box's **recorded** setup (`[env]` in the config file) with `--inexact`,
so nothing is ever pruned. `gvhmr info` detects drift and tells you when to run it.

Prefer to do it by hand (or want to understand what the script decided)? Everything below.

## Manual install

```bash
uv sync --extra preproc       # macOS / any box: base env + the demo's preprocessing models
uv run gvhmr info             # device, installed features, and checkpoint status
```

`uv sync` creates `.venv/` and installs from the locked `uv.lock`; run commands with `bin/gvhmr` (or
activate the venv — `uv run gvhmr` works too, with the caveat in the yellow box below). The **base**
install (bare `uv sync`) is enough to import the package, run the model on CPU/MPS, and run the test
suite; `--extra preproc` adds the per-video preprocessing (YOLO tracking, SimpleVO) that `gvhmr demo`
needs. If you run a command whose dependencies are missing, it tells you the exact fix — nothing fails
with a bare traceback. After a manual install, `gvhmr env record --extras preproc` (plus
`--torch cuXXX` on a CUDA box) makes `gvhmr env sync` able to restore your setup from then on.

## CUDA / GPU (Linux)

On **macOS you're done** — bare `uv sync` installs the Apple-Silicon **MPS** build. On **Linux**, bare
`uv sync` installs the default PyPI torch wheel, which targets one recent CUDA (13.x) and **won't see
the GPU on most drivers**. uv can't auto-detect your GPU for `uv sync` (`--torch-backend=auto` is
`uv pip`-only) and a lockfile can't gate wheels on CUDA version — so the CUDA build is an explicit
one-of choice (this is exactly what `scripts/install.sh` automates). Check `nvidia-smi` and add the
matching extra (nearest one ≤ your CUDA version):

```bash
nvidia-smi                              # → e.g. "CUDA Version: 12.8"
uv sync --extra preproc --extra cu128   # torch for CUDA 12.8 — covers 12.8 … 13.x (driver back-compat)
```

| Your CUDA (`nvidia-smi`) | Extra | torch |
|---|---|---|
| 12.0 – 12.5 | `cu124` | 2.6.x |
| 12.6 / 12.7 | `cu126` | 2.7.x |
| 12.8 / 12.9 / 13.x | `cu128` | 2.7.x |
| CPU-only Linux / CI | `cpu` | CPU build |

The CUDA extras are mutually exclusive and pin torch **< 2.8** (newer wheels carry a broken `nvshmem`
dependency that fails to import). `gvhmr info` shows the resolved torch and `cuda ✓` once synced — and
if it detects an NVIDIA driver that your torch build can't use, it prints the exact extra to install.

> **Older datacenter GPUs (V100 / P100) → use `cu126`, not `cu128`.** The CUDA version above is only half
> the story — the **GPU architecture** matters too. The `cu128` wheel is compiled for `sm_75+` (Turing
> onward) plus Blackwell, and **dropped Volta `sm_70` (V100) and Pascal `sm_60` (P100)**: on those it fails
> with *"CUDA-capable device(s) is/are busy or unavailable"* (or a `sm_70 is not compatible` warning) even
> with a brand-new driver. The `cu126` wheel still ships `sm_50…sm_90` (P100 → A100 → H100), so it's the
> right pick on clusters that have V100/P100 nodes (common on HPC). `scripts/install.sh` detects this and
> picks `cu126` automatically. Verify on a GPU node with
> `python -c "import torch; print(torch.cuda.get_arch_list())"`.

### ⚠ uv syncs exactly — let `gvhmr env` remember your extras

`uv sync` (and plain `uv run`, which re-syncs first) removes anything not implied by the extras you pass
*this time*. On a CUDA box that means a bare `uv sync` **reverts torch to the PyPI default** and prunes
your preproc/DPVO setup. The built-in answer — record once, replay forever:

```bash
gvhmr env record --torch cu128 --extras preproc   # done automatically by install.sh / the wizard
gvhmr env sync                                    # re-sync from the record (--inexact: prunes nothing)
bin/gvhmr …                                       # day-to-day: the wrapper never triggers a re-sync
```

If you prefer raw uv anyway: repeat your extras on every sync (`uv sync --extra preproc --extra cu128`),
use `uv run --no-sync`, or `export UV_NO_SYNC=1` in your shell profile.

## Extras

```bash
uv sync --extra preproc       # YOLO + ViTPose + SimpleVO (per-video preprocessing — the demo needs this)
uv sync --extra dev           # tests + ruff + pyright
uv sync --extra rtmpose       # RTMPose 2D-pose backend (`--pose2d rtmpose`; rtmlib + ONNXRuntime)
uv sync --extra train         # experiment logging for `gvhmr train` (wandb default, tensorboard alt)
uv sync --extra vis           # wis3d / viser (interactive 3D debugging)
uv sync --extra notebook      # jupyter / ipython / ipdb
```

| Extra | Provides | Notes |
|---|---|---|
| `preproc` | ultralytics, cython_bbox, lapx, pycolmap | **needed by `gvhmr demo`**; weights auto-fetch |
| `dev` | pytest, ruff, pyright, pre-commit | runs the full test suite on CPU/MPS |
| `rtmpose` | rtmlib, onnxruntime | alternative 2D-pose backend; model auto-downloads |
| `train` | wandb, tensorboard | training loggers (`logger=wandb` is the default) |
| `vis` | wis3d, viser | research visualization |
| `notebook` | jupyter, ipython, ipdb | interactive work |
| `render` | pytorch3d (built from source) | **optional fallback only** — see below |

### Mesh rendering — works out of the box

The demo's overlay/world videos render on the **GPU via moderngl** (OpenGL / Metal / headless EGL),
which is a **base dependency** — rendering just works after `uv sync`, real-time even on Apple
Silicon, no extra build. It needs the **SMPL** body model (`smpl/SMPL_NEUTRAL.pkl`, see *Weights &
data*); legacy `.pkl` files load via `chumpy` in the base deps, with the Python-3.13/NumPy-2 shims
applied automatically.

**pytorch3d is only a fallback** (for a box with no usable GL/EGL context) and is otherwise
unnecessary — don't build it just because you saw it in a traceback. If you do want it:

```bash
uv sync --extra render        # builds pytorch3d from source; MACOSX_DEPLOYMENT_TARGET=11.0 prefix if asked
```

`make_renderer` (`gvhmr/utils/vis/renderer_gl.py`) prefers the GPU renderer and falls back
automatically.

## Optional camera backends (moving cameras)

The default camera is **SimpleVO** (in `preproc`; rotation only). For world **translation** on a
moving camera, two optional setups:

- **Scene-aware metric cameras — any device (recommended on Mac).** `--camera dust3r` reconstructs the
  scene with DUSt3R + a global aligner; `--camera vggt` uses VGGT in one feed-forward pass (often
  faster/more robust). Both fix the metric scale with Depth-Anything-V2 and run on Apple-Silicon
  MPS / CPU / CUDA. One script sets both up (clones into `third-party/`, fetches weights):

  ```bash
  scripts/setup_scene_aware.sh
  gvhmr demo VIDEO --camera vggt        # or --camera dust3r
  ```

- **DPVO — CUDA only.** Classic deep patch visual odometry. It compiles CUDA extensions, so it can't
  live in the lock; one script on a box with a CUDA toolchain (`nvcc`) does everything (detects your
  CUDA → syncs the matching torch → builds DPVO from a thin fork that vendors Eigen and the
  modern-PyTorch build patches):

  ```bash
  scripts/setup_dpvo.sh                 # or: scripts/install.sh --dpvo
  uv run gvhmr download slam            # the DPVO weight
  gvhmr demo VIDEO --camera dpvo
  ```

  Because DPVO is installed out-of-band, a bare `uv sync` prunes it — see *Keep your extras on every
  sync* above. Re-running the script recovers it (idempotent).

## Apple Silicon (MPS)

`uv sync --extra preproc` and you're done: core inference, geometry, the preprocessing models, and
**mesh rendering (moderngl → Metal)** all run on the Apple-Silicon GPU. Device selection is automatic
(cuda → mps → cpu); override with `GVHMR_DEVICE=mps|cpu|cuda`. The only CUDA-only piece is **DPVO** —
use `--camera dust3r|vggt` instead (above). Note the GVHMR model's own `predict` is latency-bound and
actually faster on CPU; MPS accelerates the batched preprocessing models (see `docs/PERFORMANCE.md`).

## Weights & data

**Fetch the checkpoints with one command** — they land in the right place automatically (no manual
download / placement). `gvhmr demo` also auto-fetches anything missing on first run.

```bash
uv run gvhmr download            # demo checkpoints: gvhmr + hmr2 + vitpose + yolo  (~5.5 GB)
uv run gvhmr download slam       # + DPVO
uv run gvhmr download all        # every checkpoint
uv run gvhmr info                # verify what's present / missing
```

Files go to `inputs/checkpoints/` by default. To keep large assets on a **high-storage volume**, run
**`gvhmr config init`** once — a wizard that writes one readable config file (`<repo>/gvhmr.toml`,
gitignored: asset locations + default model versions + the managed env); `$GVHMR_CHECKPOINTS` etc. still
override it for CI/one-offs. See [docs/CONFIGURATION.md](CONFIGURATION.md). Source: the HuggingFace
mirror [`camenduru/GVHMR`](https://huggingface.co/camenduru/GVHMR) (resumable, checksummed).

**Body models are registration-gated** and can't be auto-fetched — sign up for
[SMPL](https://smpl.is.tue.mpg.de/) (rendering) + [SMPL-X](https://smpl-x.is.tue.mpg.de/) (motion
recovery), then place them under `$GVHMR_BODY_MODELS` (default `inputs/checkpoints/body_models/`):

```
body_models/
├── smplx/SMPLX_{GENDER}.npz
└── smpl/SMPL_{GENDER}.pkl
```

`gvhmr download` prints these instructions (with the resolved target path) whenever they're missing.

**Training/eval data packs** (the `hmr4d_support` bundles) are on the same mirror:

```bash
uv run gvhmr download --data 3dpw,emdb,rich     # downloads + extracts under inputs/<DS>/hmr4d_support/
```

Available: `3dpw amass h36m emdb rich bedlam` (BEDLAM is ~21 GB). They extract under `inputs/` by
default; set `$GVHMR_DATA_ROOT` (or the config file) to keep them elsewhere — the download and every
dataset loader honor it. See [`docs/TRAINING.md`](TRAINING.md).

## Troubleshooting

**`gvhmr info` says `cuda ✗` but the box has an NVIDIA GPU.** You synced without a CUDA extra, so you
have the default PyPI torch wheel. `gvhmr info` detects this and prints the fix — `gvhmr env sync` when
the env is recorded, else the matching `uv sync --extra cuXXX` (table above). (Forcing
`GVHMR_DEVICE=cuda` can't fix a CPU-only wheel.)

**torch/DPVO "disappeared" after a `uv sync`.** uv syncs exactly: any extra (or out-of-band install like
DPVO) you don't re-pass gets pruned. Run `gvhmr env sync` to restore the recorded setup (and re-run
`scripts/setup_dpvo.sh` if DPVO itself vanished); day-to-day, prefer `bin/gvhmr` over `uv run`.

**"CUDA-capable device(s) is/are busy or unavailable" / `sm_70 is not compatible` on V100/P100.** You
have the `cu128` wheel, which dropped those GPU architectures — re-sync with `--extra cu126` (see the
CUDA section).

**Rendering skipped.** The overlay renderer needs the registration-gated **SMPL** body model
(`smpl/SMPL_NEUTRAL.pkl`) and a working GL context; headless GPU nodes fall back to EGL automatically.
`gvhmr info` shows exactly which piece is missing. pytorch3d is only the last-resort fallback.

**`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` on download (HPC/clusters).**
Some login shells export a **misconfigured** `SSL_CERT_DIR` (or `SSL_CERT_FILE`) — e.g. `SSL_CERT_DIR`
pointing at a bundle *file* instead of a hash *directory*. OpenSSL's issuer lookup then fails and
`huggingface_hub` / `ultralytics` downloads die, even though the network is fine (`git` and `urllib` may
still work, which makes it baffling). `gvhmr` **auto-repairs this** before every command
(`gvhmr/utils/net.py::ensure_ca_bundle` falls back to [`certifi`](https://pypi.org/project/certifi/)'s
bundle, respecting a *valid* `SSL_CERT_FILE` so corporate roots keep working). If you still hit it in a
non-CLI context, set it yourself:

```bash
export SSL_CERT_FILE=$(uv run python -m certifi)
```

> Coming from the original repo's setup? `pip install -r requirements.txt && pip install -e .` is
> replaced by `uv sync` — there is no `setup.py`/`requirements.txt` here; everything is in
> `pyproject.toml`.
