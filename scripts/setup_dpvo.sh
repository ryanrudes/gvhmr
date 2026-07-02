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
# use the device-agnostic `gvhmr demo --camera dust3r|vggt` instead (scripts/setup_scene_aware.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "[setup] uv not found — see https://docs.astral.sh/uv/"; exit 1; }
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
  echo "[setup] No 'nvcc' on PATH — DPVO needs a CUDA toolkit. On Mac/MPS use: gvhmr demo --camera dust3r"
  exit 1
fi
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"

# 1) Pick the torch backend extra: the box's RECORDED choice ([env].torch in gvhmr.toml — what the
#    installer/wizard/user selected) wins, so a rebuild compiles against the torch that's actually in
#    use; a fresh box (no venv/record yet) falls back to mapping the local CUDA toolkit version.
#    CUDA wheels are minor-version compatible, so cu128 also covers CUDA 13.x boxes — no cu130 needed.
RECORDED="$(uv run --no-sync python -c 'from gvhmr.utils.localconfig import env_torch; print(env_torch() or "")' 2>/dev/null || true)"
CUDA_VER="$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
case "$RECORDED" in
  cu*) EXTRA="$RECORDED"
       echo "[setup] CUDA_HOME=$CUDA_HOME  (recorded [env].torch → extra: $EXTRA; toolkit $CUDA_VER)" ;;
  *)   case "$CUDA_VER" in
         12.0|12.1|12.2|12.3|12.4|12.5) EXTRA=cu124 ;;
         12.6|12.7)                     EXTRA=cu126 ;;
         12.8|12.9|13.*)                EXTRA=cu128 ;;
         *)                             EXTRA=cu126 ;;  # safe, widely-compatible default
       esac
       echo "[setup] CUDA_HOME=$CUDA_HOME  (toolkit $CUDA_VER → torch extra: $EXTRA)" ;;
esac

# 2) Project + the matching CUDA torch + preprocessing models (the demo needs YOLO/ViTPose/HMR2) + DPVO's
#    torch-ABI-free runtime deps (the `dpvo` extra: numba, pypose — locked, so numpy stays numba-compatible).
#    --inexact: never remove anything already in the env (e.g. dev tooling) — this script only adds.
echo "[setup] syncing project (torch $EXTRA + preproc + dpvo runtime deps)…"
uv sync --inexact --extra "$EXTRA" --extra preproc --extra dpvo

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
  "dpvo @ git+https://github.com/ryanrudes/DPVO.git" torch-scatter
# (numba/pypose/yacs — the torch-ABI-free runtime deps — came from the `dpvo`/`preproc` extras above.)

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

# 6) Record the setup so `gvhmr env sync` knows this box's torch build and that DPVO belongs here
#    (sync uses --inexact, so it won't prune DPVO — and it warns to re-run this script if DPVO vanishes).
uv run --no-sync gvhmr env record --torch "$EXTRA" --dpvo || true

cat <<NOTE

  ┌───────────────────────────────────────────────────────────────────────────────────────┐
  │  DPVO is installed out-of-band (it can't live in the lock — CUDA-only), so a bare        │
  │  \`uv sync\` / plain \`uv run\` would prune it and revert torch to the PyPI default.         │
  │  You don't need uv day-to-day — use the wrapper (or the venv):                            │
  │      bin/gvhmr demo VIDEO --camera dpvo                                                   │
  │      source .venv/bin/activate && gvhmr demo VIDEO --camera dpvo                          │
  │  Need to re-sync (new deps / extras)?  \`bin/gvhmr env sync\` replays this box's recorded   │
  │  setup without pruning anything. If DPVO ever vanishes, re-run this script (idempotent).  │
  └───────────────────────────────────────────────────────────────────────────────────────┘
NOTE
