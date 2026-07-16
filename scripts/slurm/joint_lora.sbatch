#!/bin/bash
# Detached, resumable launcher for a joint LoRA backbone run (ROADMAP Regime B stage 3).
#
# Interactive `srun` dies when your SSH session closes (SIGHUP) — this is a BATCH job, so it survives
# logout and keeps running on the node. Resumes from the last checkpoint by default, so a preempted or
# disconnected run costs time, not progress.
#
#   cd <repo> && git pull
#   GVHMR_DATA_ROOT=/resnick/groups/ARCL/rrudes/gvhmr/data sbatch scripts/slurm/joint_lora.sbatch
#
# Knobs (env or `sbatch --flag` overrides):
#   EXP     experiment config (default gvhmr/mixed/joint_lora_3dpw; use *_frozen for the baseline arm)
#   RESUME  last | none  (default last — resume the newest checkpoint; `none` starts fresh)
#   plus the usual --partition / --gres / --time if your cluster differs from the defaults below.
#SBATCH --job-name=gvhmr-joint
#SBATCH --output=%x-%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${GVHMR_DATA_ROOT:?set GVHMR_DATA_ROOT to the pack root (…/gvhmr/data)}"

# Single-GPU, single-process: keep Lightning from seeing SLURM_* and trying to stand up a DDP group.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID 2>/dev/null || true

EXP="${EXP:-gvhmr/mixed/joint_lora_3dpw}"
RESUME="${RESUME:-last}"
args=(train "exp=$EXP")
[ "$RESUME" != none ] && args+=("resume_mode=$RESUME")

echo "[joint] $(date '+%F %H:%M')  exp=$EXP  resume=$RESUME  data=$GVHMR_DATA_ROOT"
bin/gvhmr "${args[@]}"
echo "[joint] DONE rc=$? $(date '+%F %H:%M')"
