#!/usr/bin/env bash
# Fetch the world-frame eval datasets used by `tools/eval/eval_world.py`:
#
#   SLOPER4D     real outdoor, 200 m–1.3 km world trajectories — public, NO registration.
#                Tests the full --slam dust3r pipeline end-to-end under real SLAM/depth noise.
#   WHAC-A-Mole  synthetic, EXACT camera + SMPL-X GT — the clean control (gt-cam mode) that
#                isolates whether the composition math is right. HuggingFace, large (~50 GB).
#
# Data lands under $GVHMR_DATA/{sloper4d,whac} (default ~/Datasets/GVHMR). Override $GVHMR_DATA to
# relocate. Both are big; this script only fetches what you ask for. Safe to re-run (skips existing).
#
#   scripts/setup_eval_datasets.sh sloper4d        # just SLOPER4D
#   scripts/setup_eval_datasets.sh whac            # just WHAC-A-Mole (needs `huggingface-cli`)
#   scripts/setup_eval_datasets.sh sloper4d whac   # both
set -euo pipefail

DATA="${GVHMR_DATA:-$HOME/Datasets/GVHMR}"
WANT="${*:-}"
[ -z "$WANT" ] && { echo "usage: $0 [sloper4d] [whac]"; exit 1; }

for name in $WANT; do
  case "$name" in
    sloper4d)
      DST="$DATA/sloper4d"; mkdir -p "$DST"
      echo "[setup] SLOPER4D → $DST"
      echo "  SLOPER4D is distributed per-sequence from the project page (no registration):"
      echo "    http://www.lidarhumanmotion.net/data-sloper4d/"
      echo "  Download the sequence packages (each has <seq>_labels.pkl + RGB frames) into:"
      echo "    $DST/<seq>/"
      echo "  The eval globs '*_labels.pkl' recursively, so any subfolder layout works."
      echo "  (Direct links are gated behind a click-through form, so this step is manual.)"
      ;;
    whac)
      DST="$DATA/whac"; mkdir -p "$DST"
      echo "[setup] WHAC-A-Mole → $DST"
      if command -v huggingface-cli >/dev/null 2>&1; then
        echo "  downloading annotations (humandata .npz) — images are ~50 GB, fetch separately if needed…"
        huggingface-cli download waanqii/WHAC-A-Mole --repo-type dataset \
          --include "*.npz" --local-dir "$DST" || {
            echo "  [warn] HF download failed (auth? quota?). Manual: https://huggingface.co/datasets/waanqii/WHAC-A-Mole"; }
      else
        echo "  huggingface-cli not found. Install it (uv pip install huggingface_hub) or download manually:"
        echo "    https://huggingface.co/datasets/waanqii/WHAC-A-Mole/tree/main  → $DST"
      fi
      echo "  After download, confirm the npz schema once:  uv run python tools/eval/eval_world.py --probe <file.npz>"
      ;;
    *) echo "[setup] unknown dataset '$name' (expected: sloper4d | whac)"; exit 1 ;;
  esac
done

echo "[setup] done. Then:  uv run python tools/eval/eval_world.py --dataset <name> --modes prior gt-cam"
