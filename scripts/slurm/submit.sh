#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Submit the definitive A3 physics A/B to Slurm as a dependency chain:
#
#     setup (CPU)  ->  train (2-task array: physics OFF | physics LIGHT)  ->  eval (1 GPU)
#
#   git clone https://github.com/ryanrudes/gvhmr.git && cd gvhmr
#   GVHMR_EXP_ROOT=/shared/gvhmr SMPLX_USER=you@example.com SMPLX_PW='...' \
#     bash scripts/slurm/submit.sh
#
# Nothing touches a GPU until the ~27 GB of packs are on disk (setup is a CPU job), and eval only runs
# if BOTH arms succeed (afterok). Re-submitting after a failure resumes each arm from its last
# checkpoint (resume_mode=last), so a preempted job costs you time, not progress.
#
# GVHMR_EXP_ROOT must be on a filesystem EVERY node can see — it holds the packs, the checkpoints and
# the outputs (~115 GB). Defaults to ~/gvhmr-data, which on most clusters is shared; on a cluster where
# $HOME is node-local that default is wrong, so set it.
#
# Knobs:  GPUS_PER_ARM (default 4)   EPOCHS (500)   EFF_BATCH (256)   PARTITION / ACCOUNT / TIME
# Extra sbatch flags:  SBATCH_EXTRA='--qos=high --constraint=h100'
# ---------------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/_exp_lib.sh

command -v sbatch >/dev/null || die "no sbatch on PATH — this is the Slurm path; for one box use scripts/lambda_experiment.sh"
[ -n "${SMPLX_USER:-}" ] && [ -n "${SMPLX_PW:-}" ] || die \
  "set SMPLX_USER and SMPLX_PW (free signup: https://smpl-x.is.tue.mpg.de/) before submitting"

GPUS_PER_ARM="${GPUS_PER_ARM:-4}"
BATCH=$((EFF_BATCH / GPUS_PER_ARM))
[ $((BATCH * GPUS_PER_ARM)) -eq "$EFF_BATCH" ] || die \
  "GPUS_PER_ARM=$GPUS_PER_ARM must divide EFF_BATCH=$EFF_BATCH (use 1, 2, 4 or 8)"

# Slurm strips most of the environment on some clusters; --export=ALL is the default but be explicit
# about the things the jobs genuinely need, and never put the password on a command line.
EXPORTS="ALL,GVHMR_EXP_ROOT=$DATA_ROOT,EPOCHS=$EPOCHS,EFF_BATCH=$EFF_BATCH,GPUS_PER_ARM=$GPUS_PER_ARM,KEEP_TARS=$KEEP_TARS"
EXPORTS="$EXPORTS,SMPLX_USER=$SMPLX_USER,SMPLX_PW=$SMPLX_PW"
[ -n "${WANDB_API_KEY:-}" ] && EXPORTS="$EXPORTS,WANDB_API_KEY=$WANDB_API_KEY" || EXPORTS="$EXPORTS,WANDB_MODE=offline"

COMMON=(--export="$EXPORTS")
[ -n "${PARTITION:-}" ] && COMMON+=(--partition="$PARTITION")
[ -n "${ACCOUNT:-}" ]   && COMMON+=(--account="$ACCOUNT")
# shellcheck disable=SC2206
[ -n "${SBATCH_EXTRA:-}" ] && COMMON+=(${SBATCH_EXTRA})

say "root        : $DATA_ROOT   (must be visible from every node)"
say "per arm     : $GPUS_PER_ARM GPU(s) x batch $BATCH = $EFF_BATCH effective (the paper's recipe)"
say "epochs      : $EPOCHS   |   W&B: ${WANDB_API_KEY:+online}${WANDB_API_KEY:-offline}"

jid_setup=$(sbatch --parsable "${COMMON[@]}" scripts/slurm/00_setup.sbatch)
say "1/3 setup  -> job $jid_setup  (CPU: venv + body model + ~27 GB packs)"

jid_train=$(sbatch --parsable "${COMMON[@]}" --dependency=afterok:"$jid_setup" \
  --gres=gpu:"$GPUS_PER_ARM" --ntasks-per-node="$GPUS_PER_ARM" \
  ${TIME:+--time="$TIME"} scripts/slurm/10_train.sbatch)
say "2/3 train  -> job $jid_train  (array 0=physics OFF, 1=physics LIGHT; after setup)"

jid_eval=$(sbatch --parsable "${COMMON[@]}" --dependency=afterok:"$jid_train" scripts/slurm/20_eval.sbatch)
say "3/3 eval   -> job $jid_eval  (after BOTH arms succeed)"

cat <<EOF

  watch:    squeue -u \$USER
  logs:     tail -f gvhmr-train-${jid_train}_0.out    # arm A (physics off)
            tail -f gvhmr-train-${jid_train}_1.out    # arm B (physics light)
  result:   gvhmr-eval-${jid_eval}.out               # the side-by-side table
  cancel:   scancel $jid_setup $jid_train $jid_eval

  If a job dies, re-run this same command: each arm resumes from its last checkpoint.
EOF
