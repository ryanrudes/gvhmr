"""Measure GVHMR's in-camera depth (tz) against EMDB ground truth.

Motivation. A downstream consumer (multi-view rig, calibrated intrinsics) reports that GVHMR's in-camera
body_pose / betas / orientation / lateral bearing are excellent and cross-view-consistent, but the
*along-ray depth* is biased FAR in some views. That bias is invisible to every metric we ship:
`mpjpe`/`pa_mpjpe` are pelvis-aligned or Procrustes-aligned, so they subtract the very error in question,
and 2D reprojection cannot see it either (an inflated body placed farther reprojects to the same pixels).

`tz = 2*f / (s*b)` (hmr_cam.py) — f = K[0,0], s = the network's crop-relative scale, b = bbox size in px.
So depth is exactly linear in the focal, and any error in s (e.g. from an out-of-distribution crop) lands
straight on depth. This probe measures the thing directly:

  1. signed depth error vs GT, bucketed by crop size  -> is the FAR bias crop-size dependent?
  2. estimated K (our default) vs EMDB's GT K         -> how much of it is just the focal heuristic?
  3. does `MotionResult.depth_reliability()`'s weight actually predict |depth error|?  (shipped in
     v1.6.0 to that consumer with its correlation UNMEASURED — this is the check that should have
     preceded the release)

NB the EMDB test dataset deliberately throws away GT intrinsics (`emdb_motion_test.py:90` — "We use
estimated K") and uses the image-diagonal heuristic, so every EMDB depth number we have ever reported
carries a focal error. `--k gt` swaps the GT intrinsics back in to separate the two mechanisms.

Runs the DEFAULT user path: no flip-test (matching `gvhmr demo`'s default), released checkpoint.

    python scripts/depth_bias_probe.py --split 1 --k est
    python scripts/depth_bias_probe.py --split 1 --k gt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from gvhmr import PROJ_ROOT
from gvhmr.configs import register_store_gvhmr
from gvhmr.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
from gvhmr.utils.assets import CHECKPOINT_ROOT
from gvhmr.utils.geo.hmr_cam import normalize_kp2d
from gvhmr.utils.geo_transform import apply_T_on_points
from gvhmr.utils.net_utils import load_pretrained_model
from gvhmr.utils.smplx_utils import make_smplx


def build_model(ckpt: str, device: torch.device):
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="train", overrides=["exp=gvhmr/mixed/mixed"])
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, ckpt)
    return model.eval().to(device)


@torch.no_grad()
def run_sequence(model, data, device, use_gt_K: bool, gt_K):
    """One EMDB sequence through the default (no-flip-test) predict path -> pred/GT in-cam root depth."""
    L = data["length"]
    K = gt_K[None].repeat(L, 1, 1) if use_gt_K else data["K_fullimg"]

    batch = {
        "length": torch.tensor([L]).to(device),  # builds the padding mask — must be on-device
        "obs": normalize_kp2d(data["kp2d"][None], data["bbx_xys"][None]).to(device),
        "bbx_xys": data["bbx_xys"][None].to(device),
        "K_fullimg": K[None].to(device),
        "cam_angvel": data["cam_angvel"][None].to(device),
        "f_imgseq": data["f_imgseq"][None].to(device),
    }
    mask = data["mask"]
    batch["obs"][0, ~mask] = 0
    return model.pipeline.forward(batch, train=False, postproc=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=int, default=1, choices=[1, 2])
    ap.add_argument("--k", choices=["est", "gt"], default="est", help="est = our default heuristic; gt = EMDB's K")
    ap.add_argument("--out", default=None, help="write per-frame arrays to this .npz")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or str(CHECKPOINT_ROOT / "gvhmr/gvhmr_siga24_release.ckpt")
    model = build_model(ckpt, device)

    ds = EmdbSmplFullSeqDataset(split=args.split, flip_test=False)
    smplx = make_smplx("supermotion").to(device)
    smpl_model = {g: make_smplx("smpl", gender=g).to(device) for g in ("male", "female")}
    _bm = PROJ_ROOT / "gvhmr/utils/body_model"
    J_regressor = torch.load(_bm / "smpl_neutral_J_regressor.pt", weights_only=False).to(device)
    smplx2smpl = torch.load(_bm / "smplx2smpl_sparse.pt", weights_only=False).to(device)

    rows = {k: [] for k in ("pred_z", "gt_z", "bbx_px", "betas_mag", "betas_std", "vid_idx", "f_px")}
    for i in range(len(ds)):
        data = ds[i]
        vid = ds.idx2meta[i][0]
        out = run_sequence(model, data, device, args.k == "gt", ds.labels[vid]["K_fullimg"])

        # GT: world SMPL -> in-cam joints (the exact path metric_emdb.py uses)
        gt_params = {k: v.to(device) for k, v in data["smpl_params"].items()}
        gt_w_verts = smpl_model[data["gender"]](**gt_params).vertices
        gt_w_j3d = torch.matmul(J_regressor, gt_w_verts)
        gt_c_j3d = apply_T_on_points(gt_w_j3d, data["T_w2c"].to(device))

        # Pred: in-cam SMPL-X -> SMPL topology -> joints (same as the metric callback)
        pred_params = {k: v[0] for k, v in out["pred_smpl_params_incam"].items()}
        pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx(**pred_params).vertices])
        pred_c_j3d = einsum(J_regressor, pred_c_verts, "j v, l v i -> l j i")

        m = data["mask"].to(device)
        betas = pred_params["betas"]  # (L, 10) per-frame
        rows["pred_z"].append(pred_c_j3d[m, 0, 2].cpu().numpy())  # pelvis depth
        rows["gt_z"].append(gt_c_j3d[m, 0, 2].cpu().numpy())
        rows["bbx_px"].append(data["bbx_xys"][m.cpu(), 2].numpy())
        rows["betas_mag"].append(betas[m].norm(dim=-1).cpu().numpy())
        rows["betas_std"].append((betas[m] - betas[m].mean(0, keepdim=True)).norm(dim=-1).cpu().numpy())
        rows["f_px"].append(np.full(int(m.sum()), float(data["K_fullimg"][0, 0, 0])))
        rows["vid_idx"].append(np.full(int(m.sum()), i))
        print(f"  [{i + 1:2d}/{len(ds)}] {vid:34s} frames={int(m.sum()):5d}", flush=True)

    d = {k: np.concatenate(v) for k, v in rows.items()}
    if args.out:
        np.savez(args.out, **d)
    report(d, args)


def report(d: dict[str, np.ndarray], args) -> None:
    err = d["pred_z"] - d["gt_z"]  # signed: >0 = predicted TOO FAR
    rel = err / d["gt_z"]
    print(f"\n{'=' * 78}\nEMDB-{args.split}  K={args.k}  n={len(err)} frames, {len(np.unique(d['vid_idx']))} seqs")
    print(f"{'=' * 78}")
    print(f"signed depth error (m):  mean {err.mean():+.3f}   median {np.median(err):+.3f}   std {err.std():.3f}")
    print(f"relative depth error:    mean {rel.mean():+.2%}   median {np.median(rel):+.2%}")
    print(f"|error|:                 mean {np.abs(err).mean():.3f} m   median {np.median(np.abs(err)):.3f} m")
    print(f"fraction predicted FAR:  {(err > 0).mean():.1%}")

    # 1. is the bias crop-size dependent?
    print(f"\n--- signed error by crop size (bbx px) — the FAR-bias hypothesis {'-' * 22}")
    qs = np.quantile(d["bbx_px"], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"{'bbx px':>16}  {'n':>6}  {'mean err':>10}  {'rel err':>9}  {'betas |b|':>10}")
    for lo, hi in zip(qs[:-1], qs[1:], strict=True):
        s = (d["bbx_px"] >= lo) & (d["bbx_px"] <= hi)
        if s.sum() == 0:
            continue
        print(
            f"{lo:6.0f}–{hi:<6.0f}    {s.sum():6d}  {err[s].mean():+10.3f}  {rel[s].mean():+8.1%}  "
            f"{d['betas_mag'][s].mean():10.2f}"
        )

    # 2. does the shipped reliability proxy predict the error? (weight = high -> should be RELIABLE)
    med = np.median(d["bbx_px"])
    weight = np.clip(d["bbx_px"] / max(med, 1e-6), None, 1.0) / (1.0 + d["betas_std"])
    for name, x in (("bbx_px", d["bbx_px"]), ("betas_mag", d["betas_mag"]), ("depth_reliability weight", weight)):
        r = np.corrcoef(x, np.abs(err))[0, 1]
        rs = np.corrcoef(x, err)[0, 1]
        print(f"corr({name:24s}, |err|) = {r:+.3f}   corr(·, signed err) = {rs:+.3f}")


if __name__ == "__main__":
    main()
