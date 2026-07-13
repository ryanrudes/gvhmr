#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# The definitive A3 physics-loss A/B, on a fresh CUDA box (e.g. a Lambda H100 instance).
#
#   git clone https://github.com/ryanrudes/gvhmr.git && cd gvhmr
#   SMPLX_USER=you@example.com SMPLX_PW='...' WANDB_API_KEY='...' bash scripts/lambda_experiment.sh
#
# Runs TWO matched 500-epoch trainings on the full released recipe (AMASS+BEDLAM+H36M+3DPW) —
#   arm A  physics OFF  (exp=gvhmr/mixed/mixed)                — the baseline
#   arm B  physics LIGHT (exp=gvhmr/mixed/mixed_physics_light) — the ONLY difference is the three weights
# — then scores both on 3DPW / EMDB / RICH and prints them side by side.
#
# Two things it fixes relative to the runs on the dev box:
#   * effective batch 256 (devices x per-GPU batch), which is the PAPER's recipe. The dev-box reproduce
#     ran at 128 (half) and trails the paper ~3-4mm on RICH / EMDB-2 world translation.
#   * torch.compile on the denoiser (~1.9x; verified faithful to fp32 rounding) — ON FOR BOTH ARMS,
#     because compiling one arm and not the other would put a difference in the A/B that isn't physics.
#
# Resumable: re-run the same command after a preemption and each arm picks up from its last checkpoint.
# Run it inside tmux/screen — these are multi-hour trainings.
#
# Requires: SMPLX_USER + SMPLX_PW (free registration at https://smpl-x.is.tue.mpg.de/). SMPL is NOT
# needed (that one is only for mesh-overlay rendering). WANDB_API_KEY is optional (offline without it).
# ---------------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

say()  { printf '\033[1;36m[exp]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[exp]\033[0m %s\n' "$*" >&2; exit 1; }

# EVERYTHING heavy goes under this one directory — point it at your big disk:
#     GVHMR_EXP_ROOT=/mnt/nvme/gvhmr bash scripts/lambda_experiment.sh
# It holds the data packs, the checkpoints, the HF + uv download caches, and the training outputs.
# ~115 GB peak. The only heavy thing NOT under it is the repo's own .venv (~10 GB, CUDA torch).
DATA_ROOT="${GVHMR_EXP_ROOT:-$HOME/gvhmr-data}"
EPOCHS="${EPOCHS:-500}"
EFF_BATCH="${EFF_BATCH:-256}"   # the paper's effective batch (its recipe is devices=2 x batch 128)
KEEP_TARS="${KEEP_TARS:-0}"     # the packs arrive as ~27 GB of tarballs; deleted after extraction by default

# --- 0. preflight: fail fast, before anything expensive ------------------------------------------
command -v nvidia-smi >/dev/null || die "no nvidia-smi — this script wants a CUDA box"
[ -n "${SMPLX_USER:-}" ] && [ -n "${SMPLX_PW:-}" ] || die \
  "set SMPLX_USER and SMPLX_PW (free signup: https://smpl-x.is.tue.mpg.de/). SMPL-X is required to train."

NGPU=$(nvidia-smi -L | wc -l)
# per-GPU batch must divide EFF_BATCH evenly, so use the largest power-of-2 <= the GPUs we give an arm.
pow2() { local n=1; while [ $((n * 2)) -le "$1" ]; do n=$((n * 2)); done; echo "$n"; }
if [ "$NGPU" -ge 8 ]; then
  ARM_GPUS=4; PARALLEL=1          # 8 GPUs: run BOTH arms at once, 4 each -> 4 x 64 = 256 effective
else
  ARM_GPUS=$(pow2 "$NGPU"); PARALLEL=0
fi
PER_GPU_BATCH=$((EFF_BATCH / ARM_GPUS))

# Check the disk that DATA_ROOT actually lands on — not $HOME, which may be a different (small) volume.
mkdir -p "$DATA_ROOT"
FREE_GB=$(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge 130 ] || die "need ~130 GB free on $DATA_ROOT (have ${FREE_GB}G).
  packs ~27 GB tarballs + ~50 GB extracted, checkpoints ~5.5 GB, training outputs ~16 GB, caches ~15 GB.
  Point it elsewhere with:  GVHMR_EXP_ROOT=/your/big/disk bash scripts/lambda_experiment.sh"

say "GPUs: $NGPU  |  per arm: $ARM_GPUS GPU(s) x batch $PER_GPU_BATCH = $EFF_BATCH effective  |  parallel arms: $PARALLEL"
say "everything heavy -> $DATA_ROOT (${FREE_GB}G free)   epochs: $EPOCHS"

# --- 1. toolchain + deps --------------------------------------------------------------------------
# Keep the big caches off the root disk too: uv's wheel cache (CUDA torch is multi-GB) and anything
# huggingface_hub decides to stage. Otherwise a small / or /home quietly fills up mid-download.
export UV_CACHE_DIR="$DATA_ROOT/.uv-cache"
export HF_HOME="$DATA_ROOT/.hf"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

command -v uv >/dev/null || { say "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
case "$CUDA_VER" in
  12.[0-5]*) CU=cu124 ;;
  12.[6-7]*) CU=cu126 ;;
  *)         CU=cu128 ;;   # 12.8+ / 13.x — H100 (sm_90) is covered by all three
esac
say "CUDA $CUDA_VER -> torch extra: $CU"
uv sync --extra "$CU" --extra preproc --extra train --extra dev

# --- 2. asset locations + gated body models --------------------------------------------------------
mkdir -p "$DATA_ROOT"/{data,checkpoints,body_models}
export GVHMR_CONFIG="$REPO/gvhmr.toml"
bin/gvhmr config set data          "$DATA_ROOT/data"
bin/gvhmr config set checkpoints   "$DATA_ROOT/checkpoints"
bin/gvhmr config set body_models   "$DATA_ROOT/body_models"
bin/gvhmr auth smpl --smplx-username "$SMPLX_USER" --smplx-password "$SMPLX_PW" \
  ${SMPL_USER:+--smpl-username "$SMPL_USER"} ${SMPL_PW:+--smpl-password "$SMPL_PW"}

say "fetching the gated SMPL-X body model (fails NOW if the credentials are wrong, not 20 min in)"
uv run --no-sync python -c "from gvhmr.utils.assets import ensure_body_models; ensure_body_models()"

# --- 3. data packs (~27 GB compressed, straight from the HF mirror — no upload from anywhere) -------
say "fetching training + eval packs (3dpw, amass, h36m, bedlam, emdb, rich) — this is the long part"
bin/gvhmr download demo --data 3dpw,amass,h36m,bedlam,emdb,rich

# The packs arrive as tarballs and are extracted next to themselves — hf_hub_download keeps the .tar.gz,
# so ~27 GB just sits there. Re-running is unaffected: fetch_data_pack skips any pack already extracted.
if [ "$KEEP_TARS" = 0 ]; then
  freed=$(find "$DATA_ROOT" -maxdepth 2 -name '*.tar.gz' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s/1073741824}')
  find "$DATA_ROOT" -maxdepth 2 -name '*.tar.gz' -delete 2>/dev/null || true
  say "deleted the pack tarballs (~${freed:-0} GB reclaimed; KEEP_TARS=1 to keep them)"
fi
du -sh "$DATA_ROOT"/data/* 2>/dev/null | sed 's/^/  /' || true

# --- 4. the two arms -------------------------------------------------------------------------------
[ -n "${WANDB_API_KEY:-}" ] || { export WANDB_MODE=offline; say "no WANDB_API_KEY -> W&B offline"; }

# Training checkpoints are ~16 GB (50 saves/arm x 163 MB x 2 arms) — keep them on the big disk too,
# not in the repo. `output_dir` is the single key everything (incl. the ckpt saver) hangs off.
OUT_A="$DATA_ROOT/outputs/armA_off"
OUT_B="$DATA_ROOT/outputs/armB_light"

common=(
  "pl_trainer.max_epochs=$EPOCHS"
  "pl_trainer.devices=$ARM_GPUS"
  "data.loader_opts.train.batch_size=$PER_GPU_BATCH"
  "model.compile_denoiser=true"          # ~1.9x; BOTH arms, or the A/B is confounded
  "resume_mode=last"                     # re-run this script after a preemption to continue
  "pl_trainer.check_val_every_n_epoch=1000"
  "+pl_trainer.limit_val_batches=0"
  "pl_trainer.num_sanity_val_steps=0"
)

run_arm() {  # $1 = exp, $2 = name suffix, $3 = CUDA_VISIBLE_DEVICES, $4 = output_dir
  CUDA_VISIBLE_DEVICES="$3" bin/gvhmr train \
    "exp=$1" "exp_name_var=_$2" "output_dir=$4" "${common[@]}" 2>&1 | tee "$DATA_ROOT/$2.log"
}

gpus_a=$(seq -s, 0 $((ARM_GPUS - 1)))
if [ "$PARALLEL" = 1 ]; then
  gpus_b=$(seq -s, "$ARM_GPUS" $((2 * ARM_GPUS - 1)))
  say "training BOTH arms in parallel (A on GPUs $gpus_a, B on GPUs $gpus_b)"
  run_arm gvhmr/mixed/mixed               armA_off   "$gpus_a" "$OUT_A" &  pid_a=$!
  run_arm gvhmr/mixed/mixed_physics_light armB_light "$gpus_b" "$OUT_B" &  pid_b=$!
  wait $pid_a || die "arm A (physics off) failed — see $DATA_ROOT/armA_off.log"
  wait $pid_b || die "arm B (physics light) failed — see $DATA_ROOT/armB_light.log"
else
  say "training the arms sequentially on GPUs $gpus_a (8+ GPUs would run them in parallel)"
  run_arm gvhmr/mixed/mixed               armA_off   "$gpus_a" "$OUT_A"
  run_arm gvhmr/mixed/mixed_physics_light armB_light "$gpus_a" "$OUT_B"
fi

# --- 5. score both on the paper benchmarks ---------------------------------------------------------
# NB TF32 is off by default (it costs 4x the EMDB accel error — docs/PERFORMANCE.md). Don't "optimize" it back on.
last_ckpt() { ls -v "$1"/checkpoints/*.ckpt 2>/dev/null | tail -1; }
A=$(last_ckpt "$OUT_A")
B=$(last_ckpt "$OUT_B")
[ -n "$A" ] && [ -n "$B" ] || die "couldn't find both checkpoints under $DATA_ROOT/outputs — see the .log files"

say "scoring arm A (physics off):  $A"
CUDA_VISIBLE_DEVICES=0 bin/gvhmr eval all --ckpt "$A" --json "$DATA_ROOT/armA_off.json"
say "scoring arm B (physics light): $B"
CUDA_VISIBLE_DEVICES=0 bin/gvhmr eval all --ckpt "$B" --json "$DATA_ROOT/armB_light.json"

uv run --no-sync python - "$DATA_ROOT/armA_off.json" "$DATA_ROOT/armB_light.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ref = a["paper_reference"]
print(f"\n{'':<12}{'metric':<22}{'A: off':>9}{'B: light':>10}{'Δ (B-A)':>10}{'paper':>9}")
print("-" * 72)
for ds in a["metrics"]:
    for k, va in a["metrics"][ds].items():
        vb = b["metrics"][ds][k]
        print(f"{ds:<12}{k:<22}{va:>9.2f}{vb:>10.2f}{vb - va:>+10.2f}{ref[ds].get(k, float('nan')):>9.1f}")
print("\nPhysics should lower jitter / foot-sliding / accel while holding PA-MPJPE (the guardrail).")
PY

say "done."
say "  arm A ckpt : $A"
say "  arm B ckpt : $B"
say "  metrics    : $DATA_ROOT/armA_off.json  |  $DATA_ROOT/armB_light.json"
say "  logs       : $DATA_ROOT/armA_off.log   |  $DATA_ROOT/armB_light.log"
du -sh "$DATA_ROOT" 2>/dev/null | sed 's/^/  total on disk: /'
