#!/bin/bash
# Step 2/3 — the two arms, as a 2-task array (0 = physics OFF baseline, 1 = physics LIGHT).
# They are independent, so Slurm runs them concurrently if the cluster has the GPUs, else back to back.
#
# THE SLURM/LIGHTNING CONTRACT (get this wrong and it hangs or silently trains one rank):
#   PyTorch-Lightning detects Slurm (SLURM_* in the env) and uses SLURMEnvironment — it then expects
#   *srun* to have already spawned one task per GPU, and takes each task's global rank from SLURM_PROCID.
#   It does NOT spawn its own subprocesses. So:
#       --ntasks-per-node MUST equal --gres=gpu:N MUST equal pl_trainer.devices
#   And do NOT set --gpus-per-task=1: that pins each task to a single visible GPU, after which
#   Lightning's devices=N would look for cuda:0..N-1 inside a task that can only see cuda:0.
# NOTE: the resource flags below are only DEFAULTS — submit.sh overrides them on the command line
# (PARTITION / GRES / CPUS_PER_TASK / TIME / SBATCH_EXTRA), and CLI flags beat #SBATCH directives.
# If sbatch says "Requested node configuration is not available", it is one of these four, and
# `sinfo -o "%20P %12G %6D %6c %10m %12l %N"` tells you what the cluster actually has.
#SBATCH --job-name=gvhmr-train
#SBATCH --output=%x-%A_%a.out
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_exp_lib.sh
exp_load_secrets   # WANDB_API_KEY etc. from the 0600 file, not from the Slurm job record

GPUS="${SLURM_GPUS_ON_NODE:-${GPUS_PER_ARM:-4}}"
read -r BATCH ACCUM <<<"$(exp_split_batch "$GPUS")"

# The array index selects an arm out of $ARMS (submit.sh sizes the array to match).
read -r -a ARM_LIST <<<"$ARMS"
ARM="${ARM_LIST[${SLURM_ARRAY_TASK_ID:-0}]:-}"
[ -n "$ARM" ] || die "array index ${SLURM_ARRAY_TASK_ID} has no arm in ARMS='$ARMS'"
EXP="$(exp_arm_exp "$ARM")"
OUT="$(exp_arm_out "$ARM")"
VAR="_arm_$ARM"

# Interconnect: fine on NVLink/NVSwitch nodes. On PCIe boxes NCCL can hang on init (GPUs pinned at
# 100%, zero output) — set NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 in that case.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

say "arm '$ARM' (index ${SLURM_ARRAY_TASK_ID:-0} of ARMS='$ARMS'): $EXP"
say "  $GPUS GPU(s) x batch $BATCH x accum $ACCUM = $EFF_BATCH effective  ->  $OUT"

mapfile -t OVERRIDES < <(exp_overrides "$GPUS" "$BATCH" "$ACCUM")

# One GPU needs no srun/DDP at all — and Lightning would otherwise see the SLURM_* vars and try to
# build a distributed group for a single process.
if [ "$GPUS" -eq 1 ]; then
  env -u SLURM_NTASKS -u SLURM_NTASKS_PER_NODE -u SLURM_PROCID -u SLURM_LOCALID \
    bin/gvhmr train "exp=$EXP" "exp_name_var=$VAR" "output_dir=$OUT" "${OVERRIDES[@]}"
else
  srun --ntasks-per-node="$GPUS" \
    bin/gvhmr train "exp=$EXP" "exp_name_var=$VAR" "output_dir=$OUT" "${OVERRIDES[@]}"
fi
