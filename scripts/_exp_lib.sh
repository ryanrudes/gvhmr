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

OUT_A="$DATA_ROOT/outputs/armA_off"
OUT_B="$DATA_ROOT/outputs/armB_light"
EXP_A="gvhmr/mixed/mixed"
EXP_B="gvhmr/mixed/mixed_physics_light"

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
  local cuda cu
  cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
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
exp_overrides() {
  printf '%s\n' \
    "pl_trainer.max_epochs=$EPOCHS" \
    "pl_trainer.devices=$1" \
    "data.loader_opts.train.batch_size=$2" \
    "model.compile_denoiser=true" \
    "resume_mode=last" \
    "pl_trainer.check_val_every_n_epoch=1000" \
    "+pl_trainer.limit_val_batches=0" \
    "pl_trainer.num_sanity_val_steps=0"
}

exp_last_ckpt() { ls -v "$1"/checkpoints/*.ckpt 2>/dev/null | tail -1; }

exp_score() {  # eval both arms and print the delta table. NB TF32 stays OFF (4x EMDB accel error).
  local a b
  a=$(exp_last_ckpt "$OUT_A"); b=$(exp_last_ckpt "$OUT_B")
  [ -n "$a" ] && [ -n "$b" ] || die "couldn't find both checkpoints under $DATA_ROOT/outputs — see the logs"
  say "scoring arm A (physics off):   $a"
  bin/gvhmr eval all --ckpt "$a" --json "$DATA_ROOT/armA_off.json"
  say "scoring arm B (physics light): $b"
  bin/gvhmr eval all --ckpt "$b" --json "$DATA_ROOT/armB_light.json"

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
  say "metrics: $DATA_ROOT/armA_off.json | $DATA_ROOT/armB_light.json"
}
