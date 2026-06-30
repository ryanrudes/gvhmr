#!/usr/bin/env bash
# One-command DPVO (CUDA SLAM) setup. Detects the box's CUDA version, syncs the matching torch via the
# cuXXX extra, and builds DPVO against it — no manual Eigen download, no torch version juggling.
#
# DPVO is the only built-in camera backend that recovers translation, but it's CUDA-only and compiles
# custom CUDA extensions, so it can't be a base dependency. This script:
#   1. picks the `cuXXX` extra matching this box's CUDA toolkit, so `uv sync --extra cuXXX` installs a
#      torch whose CUDA build works here (see the CUDA-backend extras in pyproject.toml);
#   2. builds DPVO from a thin fork (ryanrudes/DPVO) that vendors Eigen 3.4.0 and carries the minimal
#      modern-PyTorch build patches (.scalar_type() dispatch, loop_closure packaging, torch.amp).
#
# Prereqs: a CUDA toolkit (nvcc) + uv. Run from the repo root. On Mac/MPS (no CUDA) DPVO can't build —
# use the device-agnostic `gvhmr demo --slam dust3r` instead (scripts/setup_scene_aware.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "[setup] uv not found — see https://docs.astral.sh/uv/"; exit 1; }
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
  echo "[setup] No 'nvcc' on PATH — DPVO needs a CUDA toolkit. On Mac/MPS use: gvhmr demo --slam dust3r"
  exit 1
fi
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"

# 1) Map the local CUDA toolkit version to a torch backend extra. CUDA wheels are minor-version
#    compatible, so cu128 also covers CUDA 13.x boxes (driver back-compat) — no cu130 extra needed.
CUDA_VER="$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
case "$CUDA_VER" in
  12.0|12.1|12.2|12.3|12.4|12.5) EXTRA=cu124 ;;
  12.6|12.7)                     EXTRA=cu126 ;;
  12.8|12.9|13.*)                EXTRA=cu128 ;;
  *)                             EXTRA=cu126 ;;  # safe, widely-compatible default
esac
echo "[setup] CUDA_HOME=$CUDA_HOME  (toolkit $CUDA_VER → torch extra: $EXTRA)"

# 2) Project + the matching CUDA torch + preprocessing models (the demo needs YOLO/ViTPose/HMR2).
echo "[setup] syncing project (torch $EXTRA + preproc)…"
uv sync --extra "$EXTRA" --extra preproc

# 3) Build DPVO (Eigen-vendored fork) + its CUDA-compiled deps STRICTLY against the synced torch.
#    --no-build-isolation: compile against the env's torch (not an isolated build env).
#    --no-deps: critical — without it, uv resolves the torch-linked packages fresh and briefly pulls a
#      newer torch to satisfy their unpinned `torch` requirement, compiling the extensions against THAT
#      and leaving them ABI-mismatched with the synced torch ("undefined symbol" at import).
#    --no-cache + --reinstall: a re-run after a torch change recompiles (uv's wheel cache is keyed by
#      git commit, not torch version, so a stale wheel would otherwise be reused).
echo "[setup] building DPVO + CUDA slam deps (compile — a few minutes)…"
uv pip install --no-build-isolation --no-cache --no-deps \
  --reinstall-package dpvo --reinstall-package torch-scatter \
  "dpvo @ git+https://github.com/ryanrudes/DPVO.git" torch-scatter pypose
# DPVO's remaining runtime deps don't touch the torch ABI — install normally (numba pulls llvmlite).
uv pip install numba yacs

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

cat <<NOTE

  ┌───────────────────────────────────────────────────────────────────────────────────────┐
  │  DPVO is installed out-of-band (it can't live in the lock — CUDA-only), so a bare        │
  │  \`uv sync\` / plain \`uv run\` would prune it and revert torch to the PyPI default. On a    │
  │  GPU box, pin uv to your env so it never reverts:                                         │
  │      echo 'export UV_NO_SYNC=1' >> ~/.bashrc && export UV_NO_SYNC=1                       │
  │  Then run normally:   uv run gvhmr demo VIDEO --use-dpvo                                  │
  │  Or just use the venv: source .venv/bin/activate && gvhmr demo VIDEO --use-dpvo          │
  │  (When you *do* want to re-sync, keep your backend: \`uv sync --extra $EXTRA\`.)            │
  │  If DPVO ever gets pruned, just re-run this script — it's idempotent and recovers.        │
  └───────────────────────────────────────────────────────────────────────────────────────┘
NOTE
