# shellcheck shell=bash
# Shared pieces of the definitive A3 physics A/B, used by BOTH entry points so they cannot drift:
#   scripts/lambda_experiment.sh   — one fat box (Lambda on-demand, a dev workstation, …)
#   scripts/slurm/                 — a Slurm cluster (setup job -> training array -> eval job)
#
# The experiment: two matched 500-epoch trainings on the full released recipe (AMASS+BEDLAM+H36M+3DPW),
#   arm A  physics OFF   exp=gvhmr/mixed/mixed                 (the baseline)
#   arm B  physics LIGHT exp=gvhmr/mixed/mixed_physics_light   (identical but for the three weights)
# scored on 3DPW / EMDB / RICH. Both arms share seed 42, effective batch 256 (the PAPER's recipe — the
# dev-box reproduce ran at 128 and trails the paper ~3-4mm on RICH), and torch.compile (~1.9x). Compile
# must be on for BOTH arms or it becomes a difference in the A/B that isn't physics.

say() { printf '\033[1;36m[exp]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[exp]\033[0m %s\n' "$*" >&2; exit 1; }

# --- knobs (env-overridable) ----------------------------------------------------------------------
# EVERYTHING heavy lives under one root — packs, checkpoints, HF/uv caches, training outputs (~115 GB
# peak). On a cluster this MUST be on a shared filesystem every node can see.
DATA_ROOT="${GVHMR_EXP_ROOT:-$HOME/gvhmr-data}"
EPOCHS="${EPOCHS:-500}"
EFF_BATCH="${EFF_BATCH:-256}"   # the paper's effective batch = devices x per-GPU batch
KEEP_TARS="${KEEP_TARS:-0}"     # packs arrive as ~27 GB of tarballs; deleted after extraction by default

# --- the arms ---------------------------------------------------------------------------------------
#   off     physics OFF                      — the baseline every other arm is judged against
#   light   all three physics losses         — the definitive A/B (jitter -10..13%, but world/pose WORSE)
#   vel     velocity smoothness ONLY         — ablation: RAN, refuted. transl_w_accel gives ~0 jitter and
#                                              the MOST world harm (EMDB-2 W-MPJPE +17.9). Pure cost.
#   contact foot_slide + penetration ONLY    — the mirror ablation: vel showed the jitter win is all in
#                                              the contact terms; this arm tests if dropping transl_w_accel
#                                              keeps that win without vel's world regression (see ROADMAP A3)
# ARMS selects which to TRAIN (default: the original pair). Scoring always covers every arm that has a
# checkpoint, so you can train one new arm and still get the full comparison table.
ARMS="${ARMS:-off light}"

exp_arm_exp() {  # arm -> exp config
  case "$1" in
    off)     echo "gvhmr/mixed/mixed" ;;
    light)   echo "gvhmr/mixed/mixed_physics_light" ;;
    vel)     echo "gvhmr/mixed/mixed_physics_vel" ;;
    contact) echo "gvhmr/mixed/mixed_physics_contact" ;;
    *)       die "unknown arm '$1' (known: off light vel contact)" ;;
  esac
}
exp_arm_out() { echo "$DATA_ROOT/outputs/arm_$1"; }

# The first two arms were trained before this generalization, under their original directory names.
exp_arm_out() {
  case "$1" in
    off)   echo "$DATA_ROOT/outputs/armA_off" ;;
    light) echo "$DATA_ROOT/outputs/armB_light" ;;
    *)     echo "$DATA_ROOT/outputs/arm_$1" ;;
  esac
}

OUT_A="$(exp_arm_out off)"
OUT_B="$(exp_arm_out light)"
EXP_A="$(exp_arm_exp off)"
EXP_B="$(exp_arm_exp light)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_ROOT/.uv-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf}"

# --- helpers --------------------------------------------------------------------------------------
exp_repo() { cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd; }

# per-GPU batch must divide EFF_BATCH, so an arm gets the largest power-of-2 <= its GPU count
exp_pow2() { local n=1; while [ $((n * 2)) -le "$1" ]; do n=$((n * 2)); done; echo "$n"; }

exp_check_disk() {
  mkdir -p "$DATA_ROOT"
  local free
  free=$(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc '0-9')
  [ "$free" -ge 130 ] || die "need ~130 GB free on $DATA_ROOT (have ${free}G).
  packs ~27 GB tarballs + ~50 GB extracted, checkpoints ~5.5 GB, outputs ~16 GB, caches ~15 GB.
  Point it elsewhere:  GVHMR_EXP_ROOT=/your/big/disk ..."
  say "root: $DATA_ROOT (${free}G free)"
}

exp_install() {  # torch build matching the box's CUDA, + the extras training/eval need
  command -v uv >/dev/null || { say "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  local cuda="" cu
  # NB: must GUARD on nvidia-smi existing, not just redirect its stderr. On a CPU setup node the
  # binary is absent, so `nvidia-smi | sed | head` exits 127 — and under `set -euo pipefail` that
  # kills the whole job before the "no GPU here" branch below can ever run. (It did: exit 127, 2s.)
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1) || cuda=""
  fi
  if [ -n "${TORCH_CU:-}" ]; then
    cu="$TORCH_CU"; say "torch extra: $cu (TORCH_CU)"
  else
    case "$cuda" in
      12.[0-5]*) cu=cu124 ;;
      12.[6-7]*) cu=cu126 ;;
      # On a Slurm cluster this runs on a CPU *setup* node, which cannot see the GPU nodes' driver.
      # cu124 is the conservative pick: it supports H100 (sm_90) and needs the oldest driver of the
      # three, so it can't be too new for the compute nodes. Override with TORCH_CU=cu128 if you know
      # the GPU nodes are on a recent driver — check with:
      #   srun -p <gpu-partition> --gres=gpu:1 -t 5 nvidia-smi
      "")        cu=cu124; say "no GPU on this node (CPU setup job?) — defaulting to cu124 (safe for H100; TORCH_CU= to override)" ;;
      *)         cu=cu128 ;;   # 12.8+ / 13.x
    esac
    say "CUDA ${cuda:-unknown} -> torch extra: $cu"
  fi
  uv sync --extra "$cu" --extra preproc --extra train --extra dev
}

exp_assets() {  # gated body model first (fail fast on bad creds), then the ~27 GB of packs
  [ -n "${SMPLX_USER:-}" ] && [ -n "${SMPLX_PW:-}" ] || die \
    "set SMPLX_USER and SMPLX_PW (free signup: https://smpl-x.is.tue.mpg.de/). SMPL-X is required to train.
  SMPL is NOT needed — that one is only for mesh-overlay rendering."

  mkdir -p "$DATA_ROOT"/{data,checkpoints,body_models}
  export GVHMR_CONFIG="$(exp_repo)/gvhmr.toml"
  bin/gvhmr config set data        "$DATA_ROOT/data"
  bin/gvhmr config set checkpoints "$DATA_ROOT/checkpoints"
  bin/gvhmr config set body_models "$DATA_ROOT/body_models"
  bin/gvhmr auth smpl --smplx-username "$SMPLX_USER" --smplx-password "$SMPLX_PW" \
    ${SMPL_USER:+--smpl-username "$SMPL_USER"} ${SMPL_PW:+--smpl-password "$SMPL_PW"}

  say "fetching the gated SMPL-X body model (fails NOW on bad credentials, not 20 min in)"
  uv run --no-sync python -c "from gvhmr.utils.assets import ensure_body_models; ensure_body_models()"

  say "fetching packs (3dpw, amass, h36m, bedlam, emdb, rich) — the long part, ~27 GB"
  bin/gvhmr download demo --data 3dpw,amass,h36m,bedlam,emdb,rich

  # hf_hub_download KEEPS the .tar.gz next to the extracted pack, so ~27 GB would just sit there.
  # Safe to delete: fetch_data_pack skips any pack already extracted, so a re-run won't re-download.
  if [ "$KEEP_TARS" = 0 ]; then
    find "$DATA_ROOT" -maxdepth 2 -name '*.tar.gz' -delete 2>/dev/null || true
    say "deleted the pack tarballs (KEEP_TARS=1 to keep them)"
  fi
}

# The overrides shared by both arms. $1 = devices, $2 = per-GPU batch.
#
# num_workers is per RANK, and the recipe's default (12) assumes one process per node. Under DDP that
# becomes devices x 12 processes competing for the node's cores — on a 28-core V100/P100 node, 4 ranks
# x 12 = 48 workers thrash. Derive it from the CPUs Slurm actually gave this rank instead.
exp_overrides() {
  local devices="$1" batch="$2" accum="${3:-1}"
  local workers="${NUM_WORKERS:-}"
  if [ -z "$workers" ]; then
    local cpus="${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK:-12}}"
    workers=$((cpus - 2))                      # leave the rank itself a couple of cores
    [ "$workers" -gt 12 ] && workers=12        # 12 is the recipe's value; more buys nothing
    [ "$workers" -lt 2 ] && workers=2
  fi
  printf '%s\n' \
    "pl_trainer.max_epochs=$EPOCHS" \
    "pl_trainer.devices=$devices" \
    "data.loader_opts.train.batch_size=$batch" \
    "data.loader_opts.train.num_workers=$workers" \
    "+pl_trainer.accumulate_grad_batches=$accum" \
    "model.compile_denoiser=true" \
    "resume_mode=last" \
    "pl_trainer.check_val_every_n_epoch=1000" \
    "+pl_trainer.limit_val_batches=0" \
    "pl_trainer.num_sanity_val_steps=0"
}

# Split EFF_BATCH into devices x micro-batch x accumulation.
#   effective batch is the ONLY thing the experiment cares about; how you reach it is a scheduling
#   choice. The net is LayerNorm-only (no BatchNorm), so accumulation is mathematically equivalent to
#   the bigger batch — which means a 1-GPU job (schedules in minutes) can run the SAME experiment as a
#   4-GPU node (can queue for days).
#   MICRO_BATCH caps what one GPU holds at once: 256 needs ~17 GB, 128 ~9 GB, 64 ~5 GB.
exp_split_batch() {  # $1 = devices  ->  echoes "<micro_batch> <accum>"
  local devices="$1" micro accum
  micro="${MICRO_BATCH:-$((EFF_BATCH / devices))}"
  [ "$micro" -lt 1 ] && micro=1
  accum=$((EFF_BATCH / (devices * micro)))
  [ "$accum" -lt 1 ] && accum=1
  if [ $((devices * micro * accum)) -ne "$EFF_BATCH" ]; then
    die "devices($devices) x MICRO_BATCH($micro) x accum($accum) != EFF_BATCH($EFF_BATCH).
  Pick a MICRO_BATCH that divides $((EFF_BATCH / devices)) evenly (e.g. 64, 128, 256)."
  fi
  echo "$micro $accum"
}

# `|| true`: with no match `ls` exits non-zero, which pipefail+set -e would turn into a cryptic abort
# instead of the actionable "couldn't find both checkpoints" message below.
# --- secrets ---------------------------------------------------------------------------------------
# NEVER pass credentials through `sbatch --export`: they land in the Slurm job record, where
# `scontrol show job -d` and the accounting DB can expose them to admins and other users. Stash them
# in a 0600 file on the shared root instead, and have each job source it.
EXP_SECRETS="$DATA_ROOT/.exp_secrets"

exp_write_secrets() {
  mkdir -p "$DATA_ROOT"
  local old
  old=$(umask); umask 077                       # create 0600 from the outset — never briefly world-readable
  : >"$EXP_SECRETS"
  [ -n "${SMPLX_USER:-}" ]    && printf 'export SMPLX_USER=%q\n'    "$SMPLX_USER"    >>"$EXP_SECRETS"
  [ -n "${SMPLX_PW:-}" ]      && printf 'export SMPLX_PW=%q\n'      "$SMPLX_PW"      >>"$EXP_SECRETS"
  [ -n "${SMPL_USER:-}" ]     && printf 'export SMPL_USER=%q\n'     "$SMPL_USER"     >>"$EXP_SECRETS"
  [ -n "${SMPL_PW:-}" ]       && printf 'export SMPL_PW=%q\n'       "$SMPL_PW"       >>"$EXP_SECRETS"
  [ -n "${WANDB_API_KEY:-}" ] && printf 'export WANDB_API_KEY=%q\n' "$WANDB_API_KEY" >>"$EXP_SECRETS"
  umask "$old"
  chmod 600 "$EXP_SECRETS"
  say "credentials -> $EXP_SECRETS (0600, off the Slurm job record)"
}

# Optional by design: an older chain that exported its secrets still works, and the jobs stay usable
# by hand. Only loads what isn't already in the environment.
exp_load_secrets() { [ -f "$EXP_SECRETS" ] && . "$EXP_SECRETS" || true; }

exp_last_ckpt() { ls -v "$1"/checkpoints/*.ckpt 2>/dev/null | tail -1 || true; }

# Score EVERY arm that has a checkpoint (not just the ones we trained this time) and print one table
# with `off` as the reference column. That's what lets you train a single new arm and still compare it
# against the baseline you already paid for. NB TF32 stays OFF (it costs 4x the EMDB accel error).
#
# An arm's scores are CACHED in $DATA_ROOT/arm_<name>.json and reused, so adding a 4th arm costs one
# eval, not four. Without this the job re-scored every arm from scratch and blew its wall-clock before
# reaching the new arm (it did: the `contact` arm's eval timed out mid-`light`). RESCORE=1 forces a
# fresh eval of every arm; SCORE_ARMS limits which arms are considered at all.
exp_score() {
  local scored=()
  for arm in ${SCORE_ARMS:-off light vel contact}; do
    local out ck json
    out="$(exp_arm_out "$arm")"
    json="$DATA_ROOT/arm_$arm.json"
    ck=$(exp_last_ckpt "$out")
    [ -n "$ck" ] || continue
    if [ -s "$json" ] && [ "${RESCORE:-0}" != 1 ]; then
      say "arm '$arm': reusing cached scores ($json; RESCORE=1 to redo)"
      scored+=("$arm")
      continue
    fi
    say "scoring arm '$arm': $ck"
    bin/gvhmr eval all --ckpt "$ck" --json "$json"
    scored+=("$arm")
  done
  [ ${#scored[@]} -gt 0 ] || die "no arm has a checkpoint under $DATA_ROOT/outputs — see the logs"

  uv run --no-sync python - "$DATA_ROOT" "${scored[@]}" <<'PY'
import json, sys
root, arms = sys.argv[1], sys.argv[2:]
data = {a: json.load(open(f"{root}/arm_{a}.json")) for a in arms}
base = data.get("off")
ref = next(iter(data.values()))["paper_reference"]
others = [a for a in arms if a != "off"]

hdr = f"\n{'':<10}{'metric':<20}" + (f"{'off':>9}" if base else "")
for a in others:
    hdr += f"{a:>10}{'Δ':>9}"
hdr += f"{'paper':>8}"
print(hdr)
print("-" * (len(hdr) - 1))
for ds in next(iter(data.values()))["metrics"]:
    for k in next(iter(data.values()))["metrics"][ds]:
        row = f"{ds:<10}{k:<20}"
        b = base["metrics"][ds][k] if base else None
        if b is not None:
            row += f"{b:>9.2f}"
        for a in others:
            v = data[a]["metrics"][ds][k]
            row += f"{v:>10.2f}" + (f"{v - b:>+9.2f}" if b is not None else " " * 9)
        row += f"{ref[ds].get(k, float('nan')):>8.1f}"
        print(row)
print("\nPhysics targets: jitter / foot-sliding / accel should DROP. Guardrails: PA-MPJPE, and the")
print("world metrics (WA-MPJPE / W-MPJPE / RTE) — the full A/B found those got WORSE, so watch them.")
PY
  say "metrics: ${scored[*]} -> $DATA_ROOT/arm_<name>.json"
}
