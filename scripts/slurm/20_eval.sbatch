#!/bin/bash
# Step 3/3 — score both arms on 3DPW / EMDB / RICH and print the delta table.
# Single GPU, single task: eval is a few minutes and does not use DDP.
#SBATCH --job-name=gvhmr-eval
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Each arm's eval (3DPW + EMDB-1/2 + RICH, flip-test on) runs ~1h. Scores are cached per arm in
# arm_<name>.json and reused, so a normal run scores only what's new — but a cold RESCORE=1 has to
# get through every arm, hence the headroom.
#SBATCH --time=06:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_exp_lib.sh

# Lightning would otherwise see the Slurm env vars and try to set up a distributed group for a
# single-process eval. This is one task on one GPU — tell it so.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID 2>/dev/null || true

exp_score
