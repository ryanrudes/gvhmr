#!/usr/bin/env bash
# One-command DPVO (CUDA SLAM) setup that ADAPTS to whatever CUDA the box has — no hard-wired torch.
#
# DPVO is the only built-in camera backend that recovers translation, but it's CUDA-only and builds
# custom CUDA extensions, so it can't be a base dependency. The friction historically was: a manual
# Eigen download + matching the torch CUDA build to the local toolkit. This script removes both:
#
#   1. uv's `--torch-backend=auto` selects the torch wheel matching THIS box's NVIDIA driver (the
#      default PyPI wheel is built for one CUDA — currently 13.x — and mismatches most toolkits).
#   2. DPVO is pulled from a thin fork (ryanrudes/DPVO) that vendors Eigen 3.4.0 and carries the
#      minimal modern-PyTorch build patches (.scalar_type() in dispatch, loop_closure packaging).
#
# Why a script and not `uv sync --extra slam`: uv (0.11) can't auto-select the CUDA torch for the
# project workflow — `--torch-backend` is a `uv pip` feature only — so a committed lock would pin one
# CUDA for everyone. This installs the project normally, then fits torch + DPVO to the box.
#
# Prereqs: a CUDA toolkit (nvcc) + uv. Run from the repo root. On Mac/MPS (no CUDA) use the
# device-agnostic `gvhmr demo --slam dust3r` instead (scripts/setup_scene_aware.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "[setup] uv not found — see https://docs.astral.sh/uv/"; exit 1; }
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
  echo "[setup] No 'nvcc' on PATH — DPVO needs a CUDA toolkit. On Mac/MPS use: gvhmr demo --slam dust3r"
  exit 1
fi
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"
echo "[setup] CUDA_HOME=$CUDA_HOME  ($("$NVCC" --version | tail -1))"

# 1) Project + preprocessing models (DPVO is used by the demo, which needs YOLO/ViTPose/HMR2).
echo "[setup] syncing project (base + preproc)…"
uv sync --extra preproc

# 2) Fit torch to this box's CUDA. uv's --torch-backend=auto reads the driver and picks the matching
#    wheel index — so this adapts to any CUDA, no pinned version. Only (re)install if the current torch
#    can't see the GPU (a plain `uv sync` may have pulled a default wheel built for a different CUDA).
if uv run --no-sync python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[setup] torch already sees the GPU ($(uv run --no-sync python -c 'import torch;print(torch.__version__)'))"
else
  echo "[setup] installing a CUDA-matched torch (uv --torch-backend=auto)…"
  uv pip install --torch-backend=auto --reinstall-package torch --reinstall-package torchvision torch torchvision
fi

# 3) Build DPVO (Eigen-vendored fork) + its deps against that torch. --no-build-isolation so the CUDA
#    extensions compile against the installed torch (uv pip is all-or-nothing on isolation, so we install
#    just these packages here rather than the whole project).
echo "[setup] building DPVO + slam deps (CUDA compile — a few minutes)…"
uv pip install --no-build-isolation \
  "dpvo @ git+https://github.com/ryanrudes/DPVO.git" torch-scatter numba pypose yacs

# 4) Weight (not redistributable here) — same location the SLAM model expects.
mkdir -p inputs/checkpoints/dpvo
if [ ! -f inputs/checkpoints/dpvo/dpvo.pth ]; then
  echo "[setup] NOTE: place the DPVO weight at inputs/checkpoints/dpvo/dpvo.pth (from the GVHMR Google Drive)."
fi

# 5) Sanity check the build.
uv run --no-sync python -c "
import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'
import cuda_corr, cuda_ba, lietorch_backends  # the compiled CUDA extensions
from gvhmr.utils.preproc.slam import SLAMModel
print(f'[setup] OK — torch {torch.__version__}, DPVO CUDA extensions import, SLAMModel ready.')
"

cat <<'NOTE'
[setup] DPVO ready. Run via the project venv so `uv run`'s auto-sync doesn't swap the matched torch back:
    source .venv/bin/activate && gvhmr demo VIDEO --use-dpvo
  (equivalently: UV_NO_SYNC=1 uv run gvhmr demo VIDEO --use-dpvo)
NOTE
