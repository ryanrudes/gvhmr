"""Trainer-free **world-frame** eval on real-video datasets with ground-truth global motion.

Mirrors ``tools/eval/eval_3dpw.py`` (device-agnostic, no Lightning trainer) but scores the EMDB-2
*global* protocol — W-MPJPE / WA-MPJPE / RTE / jitter / foot-sliding — and, crucially, A/B's the
ways GVHMR can place the human in the world:

    prior    stock velocity-prior world trajectory (the baseline)
    gt-cam   carry the in-cam human through the dataset's *ground-truth* camera (isolates whether the
             composition math is right — zero SLAM/depth error) ← what WHAC-A-Mole stresses
    dust3r   carry it through the full DUSt3R + Depth-Anything-V2 metric camera (real end-to-end)
             ← what SLOPER4D stresses

The model is run **once** per sequence (camera *rotation* fed from GT ``T_w2c``, exactly as the EMDB
loader does), then each mode only re-composes the world translation, so the comparison is controlled.

    uv run python tools/eval/eval_world.py --dataset sloper4d --modes prior gt-cam
    uv run python tools/eval/eval_world.py --dataset whac --modes prior gt-cam dust3r --limit 3

Needs the ``preproc`` extra (YOLO/ViTPose/HMR2) and the dataset downloaded (see
``scripts/setup_eval_datasets.sh``); the ``dust3r`` mode additionally needs ``scripts/setup_scene_aware.sh``.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_module

from gvhmr.configs import register_store_gvhmr
from gvhmr.utils.console import console, rule, track
from gvhmr.utils.device import get_device
from gvhmr.utils.eval.eval_utils import as_np_array
from gvhmr.utils.eval.world_eval import GLOBAL_METRIC_KEYS, BodyModels, score_global, world_verts_j3d
from gvhmr.utils.geo_transform import compute_cam_angvel
from gvhmr.utils.net_utils import detach_to_cpu
from gvhmr.utils.postproc_world import compose_world_from_dust3r
from gvhmr.utils.pylogger import Log
from gvhmr.utils.smplx_utils import make_smplx

# Adapter registry: dataset name → zero-arg-ish loader(data_root, limit) yielding WorldSeqGT.
# emdb is the field-standard global benchmark (what GVHMR reports) — the right arbiter for A2.
ADAPTERS = {"sloper4d": "gvhmr.dataset.sloper4d.sloper4d_eval:iter_sequences",
            "whac": "gvhmr.dataset.whac.whac_eval:iter_sequences",
            "emdb": "gvhmr.dataset.emdb.emdb_world_eval:iter_sequences"}  # fmt: skip


def _load_adapter(name: str):
    module_path, fn = ADAPTERS[name].split(":")
    import importlib

    return getattr(importlib.import_module(module_path), fn)


def build_model(device):
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=eval", "static_cam=False"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    return model.eval().to(device), cfg.ckpt_path


def build_body_models(device) -> BodyModels:
    smplx = make_smplx("supermotion").to(device)
    smpl = {g: make_smplx("smpl", gender=g).to(device) for g in ("male", "female", "neutral")}
    smplx2smpl = torch.load("gvhmr/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to(device)
    J_reg = torch.load("gvhmr/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).to(device)
    return BodyModels(smplx=smplx, smpl=smpl, smplx2smpl=smplx2smpl, J_regressor=J_reg, device=device)


def predict_sequence(gt, model, device):
    """Run preproc + model once for a sequence; returns the raw prediction (in-cam + prior global)."""
    from gvhmr.cli.demo import build_demo_cfg, run_preprocess

    # static_cam=True only tells preproc to skip its own SLAM — we feed camera rotation from GT below.
    cfg = build_demo_cfg(
        gt.frames_mp4,
        output_root=f"outputs/eval_world/{gt.vid}",
        static_cam=True,
        use_dpvo=False,
        f_mm=None,
        verbose=False,
        render_scale=None,
    )
    run_preprocess(cfg, flip_test=False)

    L = gt.length
    K = gt.K_fullimg
    K = K if K.dim() == 3 else K[None].repeat(L, 1, 1)
    data = {
        "length": torch.tensor(L),
        "bbx_xys": torch.load(cfg.paths.bbx, weights_only=False)["bbx_xys"],
        "kp2d": torch.load(cfg.paths.vitpose, weights_only=False),
        "f_imgseq": torch.load(cfg.paths.vit_features, weights_only=False),
        "cam_angvel": compute_cam_angvel(gt.T_w2c[:, :3, :3]),  # GT camera rotation, as EMDB does
        "K_fullimg": K,
    }
    return detach_to_cpu(model.predict(data, static_cam=False)), cfg


def dust3r_T_w2c(cfg, device, depth_model: str = "depth_anything_v2"):
    """DUSt3R reconstruction, scaled to metres by ``depth_model``.

    The metric-depth model IS the A2 question: DUSt3R (like VGGT) recovers the scene only up to scale,
    and this model is what fixes it. ROADMAP A2 compared Depth-Anything-V2 against UniDepth on a
    *single* video and found DA-V2 won on scale despite being 5x worse on person depth — a result that
    plainly needed more than n=1. Hence the switch.
    """
    from gvhmr.utils.preproc.dust3r_slam import run_dust3r_slam

    result = run_dust3r_slam(cfg.video_path, max_depth=80.0, depth_model=depth_model, device=device)
    Log.info(f"  DUSt3R camera ({depth_model}): scale {result['scale']:.1f}, conf {result['conf']:.2f}")
    return torch.as_tensor(result["T_w2c"]).float()


def main() -> None:
    p = argparse.ArgumentParser(description="World-frame eval: prior vs GT-camera vs DUSt3R composition.")
    p.add_argument("--dataset", required=True, choices=list(ADAPTERS), help="Which benchmark to evaluate.")
    p.add_argument("--data-root", default=None, help="Dataset root (default: $GVHMR_DATA/<dataset>).")
    p.add_argument("--modes", nargs="+", default=["prior", "gt-cam"],
                   choices=["prior", "gt-cam", "dust3r"], help="World-trajectory strategies to score.")  # fmt: skip
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sequences.")
    p.add_argument("--depth-model", default="depth_anything_v2",
                   help="Metric-depth model that fixes DUSt3R's scale (depth_anything_v2 | unidepth). "
                        "This is the ROADMAP A2 knob — the released default is depth_anything_v2.")  # fmt: skip
    p.add_argument("--probe", default=None, help="WHAC only: dump a HumanData npz's key tree and exit.")
    args = p.parse_args()

    if args.probe is not None:
        from gvhmr.dataset.whac.whac_eval import probe

        probe(args.probe)
        return

    device = get_device()
    console.print(f"[gvhmr]world-eval[/] · {args.dataset} · modes={args.modes} · {device}")
    model, _ = build_model(device)
    bm = build_body_models(device)

    sequences = list(_load_adapter(args.dataset)(args.data_root, args.limit))
    console.print(f"  {len(sequences)} sequence(s)")

    # agg[mode][metric] -> list of per-frame numpy arrays (concatenated at the end).
    agg: dict[str, dict[str, list]] = {m: defaultdict(list) for m in args.modes}

    for gt in track(sequences, desc=f"{args.dataset} world-eval"):
        try:
            pred, cfg = predict_sequence(gt, model, device)
        except Exception as e:  # noqa: BLE001 — one bad sequence shouldn't sink the whole run
            Log.warning(f"[warn]skip[/] {gt.vid}: {type(e).__name__}: {e}")
            continue

        gt_verts, gt_j3d = world_verts_j3d(bm, gt.smpl_params, body_type=gt.body_type, gender=gt.gender)
        mask = gt.mask

        for mode in args.modes:
            pm = copy.deepcopy(pred)
            if mode == "gt-cam":
                compose_world_from_dust3r(pm, gt.T_w2c)
            elif mode == "dust3r":
                try:
                    compose_world_from_dust3r(pm, dust3r_T_w2c(cfg, device, args.depth_model))
                except Exception as e:  # noqa: BLE001
                    Log.warning(f"[warn]dust3r skip[/] {gt.vid}: {type(e).__name__}: {e}")
                    continue
            pv, pj = world_verts_j3d(bm, pm["smpl_params_global"], body_type="smplx", gender="neutral")
            m = score_global(pv, pj, gt_verts, gt_j3d, mask=mask)
            for k in GLOBAL_METRIC_KEYS:
                agg[mode][k].append(as_np_array(m[k]).reshape(-1))

    # COVERAGE IS PART OF THE RESULT. Every mode is wrapped in a per-sequence `except … continue` so one
    # bad sequence can't sink a long run — but that means a mode can fail on EVERY sequence and still
    # print a confident-looking row, averaged over whatever survived. That is exactly what happened: a
    # missing `roma` made dust3r fail 21/21 while the run completed "successfully" and tabulated a row
    # from the 2 sequences that slipped through. Silence looked like success. So: show n, and refuse to
    # let a partial row masquerade as a full one.
    total = len(sequences)
    rule(f"{args.dataset} world-frame results (mm / %, lower is better) · {total} sequences")
    console.print("  mode        " + "".join(f"{k:>12}" for k in GLOBAL_METRIC_KEYS) + f"{'n':>6}")
    incomplete = []
    for mode in args.modes:
        n = len(agg[mode][GLOBAL_METRIC_KEYS[0]])
        cells = []
        for k in GLOBAL_METRIC_KEYS:
            vals = agg[mode][k]
            cells.append(f"{np.concatenate(vals).mean():>12.2f}" if vals else f"{'—':>12}")
        flag = "" if n == total else "  [warn]← PARTIAL[/]"
        console.print(f"  {mode:12}" + "".join(cells) + f"{n:>6}" + flag)
        if n != total:
            incomplete.append((mode, n))

    if incomplete:
        console.print()
        for mode, n in incomplete:
            console.print(
                f"[warn]{mode}: only {n}/{total} sequences contributed[/] — this row is NOT comparable to a "
                f"full one. Scroll up for the per-sequence '{mode} skip' warnings and fix the cause."
            )
        raise SystemExit(1)  # a partial row is a failed run, not a result


if __name__ == "__main__":
    main()
