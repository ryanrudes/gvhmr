#!/usr/bin/env bash
# Set up the optional scene-aware camera backends used by `gvhmr demo --camera dust3r|vggt`.
#
# DPVO (the only built-in backend that recovers camera *translation*) is CUDA-only. The dust3r/vggt
# backends are the device-agnostic alternatives (Apple-Silicon MPS / CPU / CUDA): they reconstruct the
# scene (DUSt3R's global-aligner, or VGGT's single forward pass) and fix the metric scale with
# Depth-Anything-V2. None is vendored into the repo (large, own licenses), so this script clones them
# into third-party/ and downloads weights into ~/Datasets/GVHMR (override with $GVHMR_DATA). Safe to
# re-run. VGGT weights auto-download from the HuggingFace hub on first use.
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

# 2) Depth-Anything-V2 (metric) — standalone PyTorch, no transformers dependency. Used by BOTH scene
#    backends to fix the metric scale.
if [ ! -d "$TP/depth_anything_v2_repo" ]; then
  echo "[setup] cloning Depth-Anything-V2…"
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 "$TP/depth_anything_v2_repo"
fi

# 2b) VGGT — single feed-forward camera+depth pass (used by `--camera vggt`). Pure-PyTorch. Cloned only
#     and imported via sys.path (vggt_slam.py adds third-party/vggt). Do NOT `pip install -e` it: VGGT's
#     requirements pin numpy<2, which downgrades numpy and breaks scipy (needs >=2 for np.long) — and the
#     slerp pose-interpolation depends on scipy. VGGT runs fine on numpy 2.x (verified), and its runtime
#     deps (safetensors, huggingface_hub, einops, Pillow) are already provided by GVHMR's base env
#     (timm → safetensors). VGGT weights auto-download from the HF hub (facebook/VGGT-1B) on first use.
if [ ! -d "$TP/vggt" ]; then
  echo "[setup] cloning VGGT…"
  git clone --depth 1 https://github.com/facebookresearch/vggt "$TP/vggt"
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

# Record it in [env] so `gvhmr env sync` / `gvhmr env show` know this box has the scene cameras
# (best-effort: the clones+weights above don't need the venv, so it may not exist yet).
uv run --no-sync gvhmr env record --scene 2>/dev/null || true

echo "[setup] done. Try:  gvhmr demo VIDEO --camera dust3r   (or --camera vggt)"
