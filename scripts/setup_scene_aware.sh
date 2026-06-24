#!/usr/bin/env bash
# Set up the optional scene-aware camera backend used by `gvhmr demo --slam dust3r`.
#
# DPVO (the only built-in backend that recovers camera *translation*) is CUDA-only. The dust3r
# backend is the device-agnostic alternative (Apple-Silicon MPS / CPU / CUDA): it reconstructs the
# scene with DUSt3R and fixes the metric scale with Depth-Anything-V2. Neither is vendored into the
# repo (they are large and carry their own licenses), so this script clones them into third-party/
# and downloads their weights into ~/Datasets/GVHMR (override with $GVHMR_DATA). Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third-party"
DATA="${GVHMR_DATA:-$HOME/Datasets/GVHMR}"
mkdir -p "$TP" "$DATA/dust3r" "$DATA/depth_anything"

# 1) DUSt3R — pure-PyTorch reconstruction + global aligner (no custom CUDA; the slow PyTorch RoPE
#    fallback engages automatically when the optional curope CUDA kernel is absent).
if [ ! -d "$TP/dust3r" ]; then
  echo "[setup] cloning DUSt3R (+ croco submodule)…"
  git clone --recursive --depth 1 https://github.com/naver/dust3r "$TP/dust3r"
fi
# torch>=2.6 defaults weights_only=True, which refuses DUSt3R's checkpoint; relax it (trusted file).
MODEL_PY="$TP/dust3r/dust3r/model.py"
if ! grep -q "weights_only=False" "$MODEL_PY"; then
  echo "[setup] patching DUSt3R checkpoint loader for torch>=2.6…"
  perl -pi -e "s/torch\.load\(model_path, map_location='cpu'\)/torch.load(model_path, map_location='cpu', weights_only=False)/" "$MODEL_PY"
fi

# 2) Depth-Anything-V2 (metric) — standalone PyTorch, no transformers dependency.
if [ ! -d "$TP/depth_anything_v2_repo" ]; then
  echo "[setup] cloning Depth-Anything-V2…"
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 "$TP/depth_anything_v2_repo"
fi

# 3) weights
DUST3R_CKPT="$DATA/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "$DUST3R_CKPT" ]; then
  echo "[setup] downloading DUSt3R weights (~2.5 GB)…"
  curl -L -o "$DUST3R_CKPT" \
    https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
fi
# VKITTI = outdoor (max_depth 80); for indoor scenes grab the Hypersim model instead (max_depth 20).
DA_CKPT="$DATA/depth_anything/depth_anything_v2_metric_vkitti_vitb.pth"
if [ ! -f "$DA_CKPT" ]; then
  echo "[setup] downloading Depth-Anything-V2 metric (outdoor) weights (~370 MB)…"
  curl -L -o "$DA_CKPT" \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth?download=true"
fi

echo "[setup] done. Try:  gvhmr demo VIDEO --slam dust3r"
