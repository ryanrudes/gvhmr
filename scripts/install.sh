#!/usr/bin/env bash
# One-command GVHMR install. Detects your platform and GPU, picks the right torch build, and syncs
# everything the demo needs — the step-by-step version of this script is docs/INSTALL.md.
#
#   scripts/install.sh                 # detect platform/GPU → uv sync with the right extras
#   scripts/install.sh --dev           # + test/lint tooling
#   scripts/install.sh --cpu           # force the CPU torch build (skip GPU detection)
#   scripts/install.sh --cuda cu126    # force a specific CUDA torch build
#   scripts/install.sh --dpvo          # + build DPVO (CUDA SLAM) via scripts/setup_dpvo.sh
#   scripts/install.sh -y              # non-interactive: fetch demo checkpoints without asking
#   scripts/install.sh --no-download   # non-interactive: skip the checkpoint fetch
#
# What it decides for you (all overridable):
#   * macOS            → bare PyPI torch (Apple-Silicon MPS) — no extra needed
#   * Linux + NVIDIA   → the cuXXX extra matching `nvidia-smi`'s CUDA version; V100/P100 boxes are
#                        forced to cu126 (the cu128 wheel dropped sm_70/sm_60 — see docs/INSTALL.md)
#   * Linux, no GPU    → the cpu extra
# It always includes --extra preproc (YOLO + ViTPose + SimpleVO) so `gvhmr demo` works out of the box.
set -euo pipefail
cd "$(dirname "$0")/.."

say()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
run()  { printf '\033[1;32m$\033[0m %s\n' "$*"; "$@"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

TORCH_EXTRA="" DEV=0 DPVO=0 DOWNLOAD=ask
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)          TORCH_EXTRA=cpu ;;
    --cuda)         shift; TORCH_EXTRA="${1:?--cuda needs an argument (cu124|cu126|cu128)}" ;;
    --dev)          DEV=1 ;;
    --dpvo)         DPVO=1 ;;
    -y|--download)  DOWNLOAD=yes ;;
    --no-download)  DOWNLOAD=no ;;
    -h|--help)      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "unknown option '$1' (see --help)" ;;
  esac
  shift
done

# 0) uv is the one prerequisite.
command -v uv >/dev/null || die "uv not found — install it first:
    curl -LsSf https://astral.sh/uv/install.sh | sh
  (or: brew install uv / pipx install uv — https://docs.astral.sh/uv/)"

# 1) Pick the torch build for this box (unless forced by --cpu/--cuda).
OS="$(uname -s)"
if [ -z "$TORCH_EXTRA" ]; then
  if [ "$OS" = "Darwin" ]; then
    say "macOS detected → default PyPI torch (Apple-Silicon MPS), no CUDA extra needed"
  elif command -v nvidia-smi >/dev/null 2>&1 && SMI="$(nvidia-smi 2>/dev/null)"; then
    CUDA_VER="$(printf '%s' "$SMI" | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    case "$CUDA_VER" in
      12.0|12.1|12.2|12.3|12.4|12.5) TORCH_EXTRA=cu124 ;;
      12.6|12.7)                     TORCH_EXTRA=cu126 ;;
      12.*|13.*)                     TORCH_EXTRA=cu128 ;;  # cu128 covers 12.8 … 13.x (driver back-compat)
      *)                             TORCH_EXTRA=cu126 ;;  # unparsable → the widely-compatible choice
    esac
    # The GPU architecture matters too: cu128 wheels dropped Volta (V100, sm_70) and Pascal (P100, sm_60).
    GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if [ "$TORCH_EXTRA" = "cu128" ] && printf '%s' "$GPUS" | grep -qiE 'V100|P100'; then
      say "V100/P100 GPU detected → using cu126 (the cu128 wheel dropped sm_70/sm_60 support)"
      TORCH_EXTRA=cu126
    fi
    say "NVIDIA driver CUDA $CUDA_VER ($(printf '%s' "$GPUS" | head -1)) → torch extra: $TORCH_EXTRA"
  else
    TORCH_EXTRA=cpu
    say "Linux without an NVIDIA driver → CPU torch build"
  fi
fi

# 2) Sync the environment: base + preproc (the demo's detector/2D-pose/SimpleVO) + the torch build.
SYNC=(uv sync --extra preproc)
[ -n "$TORCH_EXTRA" ] && SYNC+=(--extra "$TORCH_EXTRA")
[ "$DEV" = 1 ] && SYNC+=(--extra dev)
run "${SYNC[@]}"

# 2b) Record the choices in the config file, so `gvhmr env sync` can always restore this exact setup —
#     users never need to remember uv flags (or fear a bare sync pruning things) again.
EXTRAS="preproc"; [ "$DEV" = 1 ] && EXTRAS="preproc,dev"
run uv run --no-sync gvhmr env record --torch "${TORCH_EXTRA:-none}" --extras "$EXTRAS"

# 3) The configuration wizard: asset locations + optional components (RTMPose, DPVO, DUSt3R/VGGT scene
#    cameras, visualization, …) + checkpoint fetch, all interactive. Skipped when non-interactive.
#    --no-sync everywhere below: a plain `uv run` re-syncs to the lock's defaults, which would undo the
#    cuXXX torch we just installed (that's also why day-to-day usage goes through bin/gvhmr).
WIZARD_RAN=0
if [ -t 0 ] && [ "$DOWNLOAD" = ask ]; then
  printf '\033[1;36m[install]\033[0m run the configuration wizard now? (asset locations + optional components: RTMPose, DPVO, scene cameras, …) [Y/n] '
  read -r reply || reply=""
  case "$reply" in [nN]*) ;; *) run uv run --no-sync gvhmr config init && WIZARD_RAN=1 ;; esac
fi

# 4) Demo checkpoints (~5.5 GB from the HuggingFace mirror; `gvhmr demo` also auto-fetches on first run).
#    The wizard offers this itself, so only ask here when it didn't run.
if [ "$DOWNLOAD" = ask ] && [ "$WIZARD_RAN" = 0 ] && [ -t 0 ]; then
  printf '\033[1;36m[install]\033[0m fetch the demo checkpoints now (~5.5 GB)? [Y/n] '
  read -r reply || reply=""
  case "$reply" in [nN]*) DOWNLOAD=no ;; *) DOWNLOAD=yes ;; esac
fi
if [ "$DOWNLOAD" = yes ]; then
  run uv run --no-sync gvhmr download demo
elif [ "$WIZARD_RAN" = 0 ]; then
  say "skipping checkpoint fetch — \`bin/gvhmr download\` later, or let the demo auto-fetch"
fi

# 5) Optional: DPVO via the --dpvo flag (the wizard also offers it). CUDA-only; compiles the kernels.
if [ "$DPVO" = 1 ]; then
  case "$TORCH_EXTRA" in
    cu*) run scripts/setup_dpvo.sh ;;
    *)   say "skipping --dpvo: DPVO is CUDA-only (use \`gvhmr demo --camera dust3r|vggt\` instead)" ;;
  esac
fi

# 6) Show what we ended up with + the next steps.
run uv run --no-sync gvhmr info || true
say "done. From here on you never need uv directly — use the wrapper (or activate the venv):"
say "    bin/gvhmr demo docs/example_video/tennis.mp4 -s     # try it on the bundled example"
say "    bin/gvhmr config init                               # wizard: asset locations + components + env"
say "    bin/gvhmr env sync                                  # repair the env if it ever drifts"
say "body models (registration-gated, can't be auto-fetched): sign up at https://smpl.is.tue.mpg.de/"
say "and https://smpl-x.is.tue.mpg.de/ — \`bin/gvhmr download\` prints exactly where to put them."
case "$TORCH_EXTRA" in
  cu*) say "NOTE (CUDA box): avoid bare \`uv sync\`/\`uv run\` — they revert torch to the PyPI wheel."
       say "bin/gvhmr (or \`source .venv/bin/activate\`) sidesteps that; \`bin/gvhmr env sync\` restores it." ;;
esac
