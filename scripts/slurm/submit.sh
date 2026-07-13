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
# CLUSTER KNOBS — these are the ones you actually have to get right. Find your cluster's values with:
#     sinfo -o "%20P %12G %6D %6c %10m %12l %N"      # partition, GRES, nodes, cores, mem, maxtime
#
#   PARTITION      the GPU partition (REQUIRED on most clusters; the default one usually has no GPUs)
#   GRES           GPU request. Default "gpu:<GPUS_PER_ARM>". Many clusters need a TYPE: "gpu:a100:4"
#   CPUS_PER_TASK  cores per rank (default 8). ntasks-per-node x this must FIT ON A NODE — that
#                  product is the most common cause of "Requested node configuration is not available"
#   TIME           walltime (default 24:00:00) — must be <= the partition's limit
#   ACCOUNT        billing account, if your cluster requires one
#   SBATCH_EXTRA   anything else, e.g. '--qos=high --constraint=h100'
#
# Other knobs:  GPUS_PER_ARM (4)   EPOCHS (500)   EFF_BATCH (256)
#   SETUP_JOBID=<id>   chain onto an ALREADY-SUBMITTED setup job instead of submitting another
#   SKIP_SETUP=1       assets are already staged; submit training immediately
# ---------------------------------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/_exp_lib.sh

command -v sbatch >/dev/null || die "no sbatch on PATH — this is the Slurm path; for one box use scripts/lambda_experiment.sh"

# SMPL-X credentials are ONLY needed by the setup step (it fetches the gated body model). With
# SKIP_SETUP / SETUP_JOBID the model is already on disk and the login is already persisted by
# `gvhmr auth smpl`, so demanding them again just to submit two GPU jobs is wrong — and forces the
# password back onto a command line for no reason.
RUNS_SETUP=1
{ [ "${SKIP_SETUP:-0}" = 1 ] || [ -n "${SETUP_JOBID:-}" ]; } && RUNS_SETUP=0
if [ "$RUNS_SETUP" = 1 ] && { [ -z "${SMPLX_USER:-}" ] || [ -z "${SMPLX_PW:-}" ]; }; then
  # Already authenticated on this machine? Then setup needs nothing from the environment either.
  if ! bin/gvhmr auth smpl --help >/dev/null 2>&1 || ! grep -qs 'smplx' "${XDG_CONFIG_HOME:-$HOME/.config}/gvhmr/smpl_credentials.toml"; then
    die "setup needs SMPL-X credentials (free signup: https://smpl-x.is.tue.mpg.de/):
  SMPLX_USER=you@example.com SMPLX_PW='...' bash scripts/slurm/submit.sh
  or run \`bin/gvhmr auth smpl\` once, or pass SKIP_SETUP=1 if the packs are already staged."
  fi
  say "SMPL-X login already saved — not re-asking"
fi

# If we're NOT running setup, prove the assets are actually there. SKIP_SETUP only *asserts* they are;
# without this the arms queue for hours and then die on startup — the worst possible failure mode.
if [ "$RUNS_SETUP" = 0 ]; then
  missing=""
  for ds in AMASS BEDLAM H36M 3DPW EMDB RICH; do
    [ -d "$DATA_ROOT/data/$ds/hmr4d_support" ] || missing="$missing $ds"
  done
  [ -f "$DATA_ROOT/body_models/smplx/SMPLX_NEUTRAL.npz" ] || missing="$missing SMPL-X-body-model"
  [ -z "$missing" ] || die "assets missing under $DATA_ROOT:$missing
  Setup has not completed. Run it (on the login node, ~1h):
    SMPLX_USER=... SMPLX_PW=... bash -c 'source scripts/_exp_lib.sh && exp_check_disk && exp_install && exp_assets'
  …then re-submit. (Don't pass SKIP_SETUP=1 until this check passes — the jobs would queue, then fail.)"
  say "assets      : all 6 packs + SMPL-X present [ok]"
fi

GPUS_PER_ARM="${GPUS_PER_ARM:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
GRES="${GRES:-gpu:$GPUS_PER_ARM}"
TIME="${TIME:-24:00:00}"
BATCH=$((EFF_BATCH / GPUS_PER_ARM))
[ $((BATCH * GPUS_PER_ARM)) -eq "$EFF_BATCH" ] || die \
  "GPUS_PER_ARM=$GPUS_PER_ARM must divide EFF_BATCH=$EFF_BATCH (use 1, 2, 4 or 8)"

if [ -z "${PARTITION:-}" ]; then
  say "WARNING: no PARTITION set — submitting to the default partition, which on most clusters has no GPUs."
  say "         if this fails with 'Requested node configuration is not available', that's why."
  say "         sinfo -o \"%20P %12G %6D %6c %10m %12l %N\"   # shows the GPU partitions"
fi

# Slurm strips most of the environment on some clusters; --export=ALL is the default but be explicit
# about the things the jobs genuinely need, and never put the password on a command line.
# Credentials go to a 0600 file on the shared root, which each job sources — NOT through --export,
# which would write them into the Slurm job record (readable via `scontrol show job -d` / accounting).
exp_write_secrets

EXPORTS="ALL,GVHMR_EXP_ROOT=$DATA_ROOT,EPOCHS=$EPOCHS,EFF_BATCH=$EFF_BATCH,GPUS_PER_ARM=$GPUS_PER_ARM,KEEP_TARS=$KEEP_TARS"
[ -n "${TORCH_CU:-}" ] && EXPORTS="$EXPORTS,TORCH_CU=$TORCH_CU"

# W&B, best source first:
#   1. ~/.netrc  — what `wandb login` writes (0600). $HOME is shared on a cluster, so every compute
#      node picks it up: no key in the environment, nothing in the Slurm job record, nothing to pass.
#      THIS IS THE ONE TO USE. Set it up once:  bin/gvhmr wandb-login   (or: uvx wandb login)
#   2. $WANDB_API_KEY — stashed in the 0600 secrets file by exp_write_secrets.
#   3. neither -> offline (runs still log locally; `wandb sync <dir>` uploads them later).
if grep -qs 'api\.wandb\.ai' "$HOME/.netrc"; then
  WANDB_STATUS="online (~/.netrc)"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_STATUS="online (key -> 0600 secrets file)"
else
  EXPORTS="$EXPORTS,WANDB_MODE=offline"
  WANDB_STATUS="offline (no ~/.netrc, no WANDB_API_KEY)"
fi

COMMON=(--export="$EXPORTS")
[ -n "${ACCOUNT:-}" ] && COMMON+=(--account="$ACCOUNT")
# shellcheck disable=SC2206
[ -n "${SBATCH_EXTRA:-}" ] && COMMON+=(${SBATCH_EXTRA})

# The GPU jobs need the GPU partition; setup is CPU-only and is happiest on the default (usually
# bigger, always-idle) partition — so it does NOT inherit PARTITION unless you set SETUP_PARTITION.
GPUJOB=("${COMMON[@]}")
[ -n "${PARTITION:-}" ] && GPUJOB+=(--partition="$PARTITION")
SETUPJOB=("${COMMON[@]}")
[ -n "${SETUP_PARTITION:-}" ] && SETUPJOB+=(--partition="$SETUP_PARTITION")

say "root        : $DATA_ROOT   (must be visible from every node)"
say "per arm     : $GPUS_PER_ARM GPU(s) x batch $BATCH = $EFF_BATCH effective (the paper's recipe)"
say "resources   : partition=${PARTITION:-<default>} gres=$GRES cpus/task=$CPUS_PER_TASK time=$TIME"
# Print a WORD, never the variable. `${VAR:-offline}` substitutes the fallback only when VAR is UNSET —
# when it IS set it expands to the value, so the obvious-looking
#     "${WANDB_API_KEY:+online}${WANDB_API_KEY:-offline}"
# printed "online<your-api-key>" to stdout. It only looks right in the offline case. Never interpolate
# a secret into a status line. ($WANDB_STATUS is computed above and is a description, not a key.)
say "epochs      : $EPOCHS   |   W&B: $WANDB_STATUS"

# 1/3 setup — skip it if the assets are staged, or chain onto one you already submitted.
DEP=""
if [ -n "${SETUP_JOBID:-}" ]; then
  jid_setup="$SETUP_JOBID"; DEP="--dependency=afterok:$jid_setup"
  say "1/3 setup  -> reusing job $jid_setup (SETUP_JOBID)"
elif [ "${SKIP_SETUP:-0}" = 1 ]; then
  jid_setup="(skipped)"
  say "1/3 setup  -> SKIPPED (SKIP_SETUP=1) — assuming packs + body model are already on $DATA_ROOT"
else
  jid_setup=$(sbatch --parsable "${SETUPJOB[@]}" scripts/slurm/00_setup.sbatch)
  DEP="--dependency=afterok:$jid_setup"
  say "1/3 setup  -> job $jid_setup  (CPU: venv + body model + ~27 GB packs)"
fi

# shellcheck disable=SC2086
jid_train=$(sbatch --parsable "${GPUJOB[@]}" $DEP \
  --gres="$GRES" --ntasks-per-node="$GPUS_PER_ARM" \
  --cpus-per-task="$CPUS_PER_TASK" --time="$TIME" scripts/slurm/10_train.sbatch)
say "2/3 train  -> job $jid_train  (array 0=physics OFF, 1=physics LIGHT)"

jid_eval=$(sbatch --parsable "${GPUJOB[@]}" --dependency=afterok:"$jid_train" \
  --gres="${EVAL_GRES:-gpu:1}" scripts/slurm/20_eval.sbatch)
say "3/3 eval   -> job $jid_eval  (after BOTH arms succeed)"

cat <<EOF

  watch:    squeue -u \$USER
  logs:     tail -f gvhmr-train-${jid_train}_0.out    # arm A (physics off)
            tail -f gvhmr-train-${jid_train}_1.out    # arm B (physics light)
  result:   gvhmr-eval-${jid_eval}.out               # the side-by-side table
  cancel:   scancel $jid_train $jid_eval

  If a job dies, re-submit with SKIP_SETUP=1 — each arm resumes from its last checkpoint.
EOF
