#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# The definitive A3 physics-loss A/B on ONE fat box (a Lambda on-demand H100, a workstation, …).
# For a Slurm cluster use scripts/slurm/submit.sh instead — same experiment, same shared library.
#
#   git clone https://github.com/ryanrudes/gvhmr.git && cd gvhmr
#   GVHMR_EXP_ROOT=/mnt/nvme/gvhmr SMPLX_USER=you@example.com SMPLX_PW='...' WANDB_API_KEY='...' \
#     bash scripts/lambda_experiment.sh
#
# Trains BOTH arms (physics OFF baseline / physics LIGHT), scores them on 3DPW/EMDB/RICH, prints the
# delta table. See scripts/_exp_lib.sh for what the experiment is and why the knobs are what they are.
#
# Run it in tmux — these are multi-hour trainings. Resumable: re-run the same command after a crash.
# Needs ~130 GB under GVHMR_EXP_ROOT and an SMPL-X login (https://smpl-x.is.tue.mpg.de/).
#
#   SETUP_ONLY=1   stop after the venv + packs (useful to stage data before booking the GPUs)
# ---------------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_exp_lib.sh

command -v nvidia-smi >/dev/null || die "no nvidia-smi — this wants a CUDA box"
NGPU=$(nvidia-smi -L | wc -l)

# 8+ GPUs: train both arms at once (half the GPUs each). Otherwise one after the other.
if [ "$NGPU" -ge 8 ]; then ARM_GPUS=4; PARALLEL=1; else ARM_GPUS=$(exp_pow2 "$NGPU"); PARALLEL=0; fi
PER_GPU_BATCH=$((EFF_BATCH / ARM_GPUS))

say "GPUs: $NGPU | per arm: $ARM_GPUS x batch $PER_GPU_BATCH = $EFF_BATCH effective | parallel: $PARALLEL"
exp_check_disk
exp_install
exp_assets

if [ "${SETUP_ONLY:-0}" = 1 ]; then say "SETUP_ONLY — stopping before training"; exit 0; fi

[ -n "${WANDB_API_KEY:-}" ] || { export WANDB_MODE=offline; say "no WANDB_API_KEY -> W&B offline"; }
mapfile -t OVERRIDES < <(exp_overrides "$ARM_GPUS" "$PER_GPU_BATCH")

run_arm() {  # $1 exp, $2 name suffix, $3 CUDA_VISIBLE_DEVICES, $4 output_dir
  CUDA_VISIBLE_DEVICES="$3" bin/gvhmr train \
    "exp=$1" "exp_name_var=_$2" "output_dir=$4" "${OVERRIDES[@]}" 2>&1 | tee "$DATA_ROOT/$2.log"
}

gpus_a=$(seq -s, 0 $((ARM_GPUS - 1)))
if [ "$PARALLEL" = 1 ]; then
  gpus_b=$(seq -s, "$ARM_GPUS" $((2 * ARM_GPUS - 1)))
  say "training BOTH arms in parallel (A on GPUs $gpus_a, B on GPUs $gpus_b)"
  run_arm "$EXP_A" armA_off   "$gpus_a" "$OUT_A" & pid_a=$!
  run_arm "$EXP_B" armB_light "$gpus_b" "$OUT_B" & pid_b=$!
  wait $pid_a || die "arm A (physics off) failed — see $DATA_ROOT/armA_off.log"
  wait $pid_b || die "arm B (physics light) failed — see $DATA_ROOT/armB_light.log"
else
  say "training the arms sequentially on GPUs $gpus_a"
  run_arm "$EXP_A" armA_off   "$gpus_a" "$OUT_A"
  run_arm "$EXP_B" armB_light "$gpus_a" "$OUT_B"
fi

exp_score
du -sh "$DATA_ROOT" 2>/dev/null | sed 's/^/  total on disk: /'
