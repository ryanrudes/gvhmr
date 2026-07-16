"""Measure GVHMR's in-camera depth (tz) against ground truth, on EMDB or 3DPW.

Motivation. A downstream consumer (multi-view rig, calibrated intrinsics) reports that GVHMR's in-camera
body_pose / betas / orientation / lateral bearing are excellent and cross-view-consistent, but the
*along-ray depth* is biased in some views. That bias is invisible to every metric we ship: `mpjpe`/`pa_mpjpe`
are pelvis-aligned or Procrustes-aligned, so they subtract the very error in question, and 2D reprojection
cannot see it either (an inflated body placed farther reprojects to the same pixels).

`tz = 2*f / (s*b)` (hmr_cam.py) — f = K[0,0], s = the network's crop-relative scale, b = bbox size in px.
So depth is exactly linear in the focal. This probe separates the two candidate error sources:

  1. estimated K (the `estimate_K` diagonal-FOV heuristic) vs the dataset's GT K -> how much is the focal?
  2. with GT K, the residual per-video depth SCALE (`pred_z ~= a * gt_z`) -> what a consumer must model.

The decisive statistic is the WITHIN- vs BETWEEN-video slope of pred_z on gt_z. A pooled fit blends them
and will misreport a between-video relationship as a within-video law (this happened: see the retraction in
docs/CAMERA_METADATA.md). Fixed-prior shrinkage `pred = (1-k)*gt + k*prior` requires within == between.

Dataset notes:
  - EMDB's test loader deliberately DISCARDS GT intrinsics (`emdb_motion_test.py:90` — "We use estimated K"),
    so every EMDB depth number this project has reported carries a focal error (fx ~1435 GT vs ~2400 est).
  - 3DPW's loader uses GT K already (`threedpw_motion_test.py:80`), and its focal error is far milder
    (fx ~1962 GT vs ~2203 est), so it is a genuinely independent test of both claims.

Runs the DEFAULT user path: no flip-test (matching `gvhmr demo`'s default), released checkpoint.

    python scripts/depth_bias_probe.py --dataset emdb --split 1 --k est
    python scripts/depth_bias_probe.py --dataset emdb --split 1 --k gt
    python scripts/depth_bias_probe.py --dataset 3dpw --k gt
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
from gvhmr.dataset.threedpw.threedpw_motion_test import ThreedpwSmplFullSeqDataset
from gvhmr.utils.assets import CHECKPOINT_ROOT
from gvhmr.utils.geo.hmr_cam import estimate_K, normalize_kp2d
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


def make_dataset(name: str, split: int):
    if name == "emdb":
        return EmdbSmplFullSeqDataset(split=split, flip_test=False)
    return ThreedpwSmplFullSeqDataset(flip_test=False)


def vid_of(ds, i: int, name: str) -> str:
    return ds.idx2meta[i][0] if name == "emdb" else ds.idx2meta[i]


def sequence_K(ds, i: int, name: str, kind: str) -> torch.Tensor:
    """The (3,3) intrinsics to run this sequence with: the dataset's GT, or the estimate_K heuristic."""
    vid = vid_of(ds, i, name)
    label = ds.labels[vid]
    if kind == "gt":
        return label["K_fullimg"]
    if name == "emdb":  # mirrors emdb_motion_test._load_data
        wh = (1440, 1920) if vid != "P0_09_outdoor_walk" else (720, 960)
    else:
        wh = tuple(int(x) for x in label["img_wh"])
    return estimate_K(*wh)


# SMPL 24-joint kinematic tree. Bone lengths are invariant to pose AND to the rigid camera transform,
# so summing them gives a body-size scalar comparable between GT and pred (both reach the same 24-joint
# SMPL skeleton via J_regressor), without depending on the SMPL vs SMPL-X shape bases being commensurate.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def skeleton_size(j3d: torch.Tensor) -> torch.Tensor:
    """(L, 24, 3) joints -> (L,) total skeleton bone length. Pose- and camera-invariant."""
    child = torch.arange(1, 24, device=j3d.device)
    parent = torch.tensor(SMPL_PARENTS[1:], device=j3d.device)
    return (j3d[:, child] - j3d[:, parent]).norm(dim=-1).sum(-1)


@torch.no_grad()
def run_sequence(model, data, device, K: torch.Tensor):
    """One sequence through the default (no-flip-test) predict path -> pred in-cam SMPL params."""
    L = data["length"]
    batch = {
        "length": torch.tensor([L]).to(device),  # builds the padding mask — must be on-device
        "obs": normalize_kp2d(data["kp2d"][None], data["bbx_xys"][None]).to(device),
        "bbx_xys": data["bbx_xys"][None].to(device),
        "K_fullimg": K[None, None].repeat(1, L, 1, 1).to(device),
        "cam_angvel": data["cam_angvel"][None].to(device),
        "f_imgseq": data["f_imgseq"][None].to(device),
    }
    mask = data["mask"]
    batch["obs"][0, ~mask] = 0
    return model.pipeline.forward(batch, train=False, postproc=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["emdb", "3dpw"], default="emdb")
    ap.add_argument("--split", type=int, default=1, choices=[1, 2], help="EMDB only")
    ap.add_argument("--k", choices=["est", "gt"], default="est", help="est = the estimate_K heuristic; gt = dataset K")
    ap.add_argument("--out", default=None, help="write per-frame arrays to this .npz")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or str(CHECKPOINT_ROOT / "gvhmr/gvhmr_siga24_release.ckpt")
    model = build_model(ckpt, device)

    ds = make_dataset(args.dataset, args.split)
    smplx = make_smplx("supermotion").to(device)
    smpl_model = {g: make_smplx("smpl", gender=g).to(device) for g in ("male", "female", "neutral")}
    _bm = PROJ_ROOT / "gvhmr/utils/body_model"
    J_regressor = torch.load(_bm / "smpl_neutral_J_regressor.pt", weights_only=False).to(device)
    smplx2smpl = torch.load(_bm / "smplx2smpl_sparse.pt", weights_only=False).to(device)

    rows = {
        k: [] for k in ("pred_z", "gt_z", "bbx_px", "betas_mag", "betas_std", "vid_idx", "f_px", "pred_size", "gt_size")
    }
    for i in range(len(ds)):
        data = ds[i]
        vid = vid_of(ds, i, args.dataset)
        K = sequence_K(ds, i, args.dataset, args.k)
        out = run_sequence(model, data, device, K)

        # GT: world SMPL -> in-cam joints (the exact path the metric callbacks use)
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
        rows["pred_size"].append(skeleton_size(pred_c_j3d[m]).cpu().numpy())
        rows["gt_size"].append(skeleton_size(gt_c_j3d[m]).cpu().numpy())
        rows["f_px"].append(np.full(int(m.sum()), float(K[0, 0])))
        rows["vid_idx"].append(np.full(int(m.sum()), i))
        print(f"  [{i + 1:2d}/{len(ds)}] {vid:34s} frames={int(m.sum()):5d}", flush=True)

    d = {k: np.concatenate(v) for k, v in rows.items()}
    if args.out:
        np.savez(args.out, **d)
    report(d, args)


def within_between(pred: np.ndarray, gt: np.ndarray, vid: np.ndarray) -> tuple[float, float, float]:
    """Slope of pred on gt, decomposed. Fixed-prior shrinkage requires within == between == (1-k).

    A pooled fit is a blend of the two and will misreport between-video structure as a within-video law.
    """
    num = den = 0.0
    for v in np.unique(vid):
        s = vid == v
        gc, pc = gt[s] - gt[s].mean(), pred[s] - pred[s].mean()
        num += gc @ pc
        den += gc @ gc
    w_within = num / den

    gm = np.array([gt[vid == v].mean() for v in np.unique(vid)])
    pm = np.array([pred[vid == v].mean() for v in np.unique(vid)])
    w_between = np.linalg.lstsq(np.stack([gm, np.ones_like(gm)], 1), pm, rcond=None)[0][0]
    w_pooled = np.linalg.lstsq(np.stack([gt, np.ones_like(gt)], 1), pred, rcond=None)[0][0]
    return float(w_within), float(w_between), float(w_pooled)


def report(d: dict[str, np.ndarray], args) -> None:
    pred, gt, vid = d["pred_z"], d["gt_z"], d["vid_idx"]
    err = pred - gt  # signed: >0 = predicted TOO FAR
    rel = err / gt
    vids = np.unique(vid)
    tag = f"EMDB-{args.split}" if args.dataset == "emdb" else "3DPW"
    print(f"\n{'=' * 78}\n{tag}  K={args.k}  n={len(err)} frames, {len(vids)} seqs\n{'=' * 78}")
    print(f"focal (px):              mean {d['f_px'].mean():.1f}")
    print(f"signed depth error (m):  mean {err.mean():+.3f}   median {np.median(err):+.3f}   std {err.std():.3f}")
    print(f"relative depth error:    mean {rel.mean():+.2%}   median {np.median(rel):+.2%}")
    print(f"fraction predicted FAR:  {(err > 0).mean():.1%}")

    # Per-video scale — the thing a consumer must model.
    scales = np.array([(pred[vid == v] @ gt[vid == v]) / (gt[vid == v] @ gt[vid == v]) for v in vids])
    dists = np.array([gt[vid == v].mean() for v in vids])
    n_near = int((scales < 1).sum())
    print(f"\nper-video scale a (pred_z ~= a*gt_z): {scales.min():.4f} - {scales.max():.4f}   std {scales.std():.4f}")
    print(f"  videos biased NEAR (a<1): {n_near}/{len(vids)}")

    # The decisive test: fixed-prior shrinkage requires within == between.
    w_in, w_bt, w_pool = within_between(pred, gt, vid)
    print(f"\n--- within/between decomposition (slope of pred_z on gt_z) {'-' * 19}")
    print(f"  within-video  : {w_in:.4f}   (~1.0 => the network tracks depth changes inside a take)")
    print(f"  between-video : {w_bt:.4f}   (< 1 => videos get an overall scale set by something else)")
    print(f"  pooled        : {w_pool:.4f}   (a BLEND — do not interpret as a law)")

    # THE scale-ambiguity test. tz = 2f/(s*b): depth is inferred from apparent size, so a body-size error
    # must be paid for in depth. Predicts scale ~= pred_size/gt_size with SLOPE 1 — far sharper than a corr.
    if "pred_size" in d:
        ratio = np.array([(d["pred_size"][vid == v] / d["gt_size"][vid == v]).mean() for v in vids])
        slope = np.linalg.lstsq(np.stack([ratio, np.ones_like(ratio)], 1), scales, rcond=None)[0]
        print(f"\n--- scale-ambiguity test: does a body-size error explain the depth scale? {'-' * 5}")
        print(f"  body-size ratio (pred/gt): {ratio.min():.4f} - {ratio.max():.4f}   mean {ratio.mean():.4f}")
        print(f"  corr(size ratio, depth scale) = {np.corrcoef(ratio, scales)[0, 1]:+.3f}")
        print(f"  fit  scale = {slope[0]:+.3f}*ratio {slope[1]:+.3f}    (theory: slope +1.0, intercept 0)")
        print(
            f"  residual of scale after size ratio: {np.std(scales - (slope[0] * ratio + slope[1])):.4f} "
            f"(raw scale std {scales.std():.4f})"
        )

    # Does anything predict the per-video scale? (n = #videos; treat weakly — this is where we got burned)
    print(f"\n--- what predicts the per-video scale? (n={len(vids)} videos) {'-' * 21}")
    crops = np.array([d["bbx_px"][vid == v].mean() for v in vids])
    betas = np.array([d["betas_mag"][vid == v].mean() for v in vids])
    for name, x in (("gt_z (distance)", dists), ("1/gt_z", 1 / dists), ("bbx_px (crop)", crops), ("betas_mag", betas)):
        print(f"  corr({name:16s}, scale) = {np.corrcoef(x, scales)[0, 1]:+.3f}")

    # What a correction can actually buy.
    a_glob = (pred @ gt) / (gt @ gt)
    r_pv = np.mean(
        [np.abs((pred[vid == v] - scales[i] * gt[vid == v]) / gt[vid == v]).mean() for i, v in enumerate(vids)]
    )
    print(
        f"\nmean |rel err|:  raw {np.abs(rel).mean():.2%}  ->  global scale {a_glob:.4f}: "
        f"{np.abs((pred - a_glob * gt) / gt).mean():.2%}  ->  per-video scale: {r_pv:.2%}"
    )

    # The shipped proxy, against true error.
    med = np.median(d["bbx_px"])
    weight = np.clip(d["bbx_px"] / max(med, 1e-6), None, 1.0) / (1.0 + d["betas_std"])
    print(f"corr(depth_reliability weight, |err|) = {np.corrcoef(weight, np.abs(err))[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
