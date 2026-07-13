#!/bin/bash
# Step 1/3 — CPU-only setup: venv + gated body model + ~27 GB of data packs.
#
# Deliberately NOT a GPU job: this is an hour of downloading, and GPU-hours are the expensive thing.
# The training array depends on this finishing, so nothing touches a GPU until the data is on disk.
#
# Needs outbound internet on this node. If your compute nodes are air-gapped, run the same step on the
# login node instead:  source scripts/_exp_lib.sh && exp_check_disk && exp_install && exp_assets
#SBATCH --job-name=gvhmr-setup
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/_exp_lib.sh
exp_load_secrets   # credentials from the 0600 file, not from the Slurm job record

exp_check_disk
exp_install
exp_assets
say "setup done — packs + body model are on $DATA_ROOT, the training array can start"
