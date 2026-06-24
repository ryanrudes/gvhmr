"""Trainer-free 3DPW eval (in-cam PA-MPJPE / MPJPE), baseline vs. flip-test.

Replicates `MetricMocap` (gvhmr/model/gvhmr/callbacks/metric_3dpw.py) but runs device-agnostic
(CPU/MPS) without the Lightning trainer, so the accuracy levers can be A/B-tested locally.

    uv run python tools/eval/eval_3dpw.py            # all 37 sequences
    uv run python tools/eval/eval_3dpw.py --limit 5  # quick check

Needs the preprocessed 3DPW support at inputs/3DPW/hmr4d_support (see docs/ACCURACY.md).
"""

from __future__ import annotations

import argparse

import hydra
import numpy as np
import torch
from einops import einsum
from hydra import compose, initialize_config_module

from gvhmr.configs import register_store_gvhmr
from gvhmr.dataset.threedpw.threedpw_motion_test import ThreedpwSmplFullSeqDataset
from gvhmr.utils.console import console, track
from gvhmr.utils.device import get_device, to_device
from gvhmr.utils.eval.eval_utils import as_np_array, compute_camcoord_metrics
from gvhmr.utils.geo_transform import apply_T_on_points
from gvhmr.utils.net_utils import detach_to_cpu
from gvhmr.utils.smplx_utils import make_smplx


def main() -> None:
    p = argparse.ArgumentParser(description="3DPW in-cam eval: baseline vs flip-test.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sequences.")
    args = p.parse_args()
    torch.set_num_threads(torch.get_num_threads())
    device = get_device()

    # Eval body models / regressors — exactly as MetricMocap.__init__.
    smplx = make_smplx("supermotion_EVAL3DPW").to(device)
    smpl = {g: make_smplx("smpl", gender=g).to(device) for g in ("male", "female")}
    J_reg = torch.load("gvhmr/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt", weights_only=False).to_dense().to(device)
    smplx2smpl = torch.load("gvhmr/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to(device)

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=eval", "static_cam=False"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().to(device)

    ds = ThreedpwSmplFullSeqDataset(flip_test=True)
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    console.print(f"[gvhmr]3DPW[/] test: evaluating {n}/{len(ds)} sequences on {device}")

    def cam_metrics(pred_incam, gt):
        out = smplx(**to_device(pred_incam, device))
        verts = torch.stack([torch.matmul(smplx2smpl, v) for v in out.vertices])
        j3d = einsum(J_reg, verts, "j v, l v i -> l j i")
        batch = {"pred_j3d": j3d, "target_j3d": gt["j3d"], "pred_verts": verts, "target_verts": gt["verts"]}
        return compute_camcoord_metrics(batch, mask=gt["mask"], pelvis_idxs=[2, 3])

    agg = {"base": {"pa_mpjpe": [], "mpjpe": []}, "flip": {"pa_mpjpe": [], "mpjpe": []}}
    for idx in track(range(n), desc="3DPW eval"):
        d = ds[idx]
        L = d["length"]
        gt_out = smpl[d["gender"]](**to_device(d["smpl_params"], device))
        gt_c_verts = apply_T_on_points(gt_out.vertices, d["T_w2c"].to(device))
        gt = {"verts": gt_c_verts, "j3d": torch.matmul(J_reg, gt_c_verts), "mask": d["mask"].to(device)}
        Kf = d["K_fullimg"] if d["K_fullimg"].dim() == 3 else d["K_fullimg"][None].repeat(L, 1, 1)
        data = {k: d[k] for k in ("kp2d", "bbx_xys", "cam_angvel", "f_imgseq")} | {
            "length": torch.tensor(L),
            "K_fullimg": Kf,
        }
        ft = d["flip_test"]
        flip_data = {k: ft[k] for k in ("kp2d", "bbx_xys", "cam_angvel", "f_imgseq")} | {
            "length": torch.tensor(L),
            "K_fullimg": Kf,
        }
        p0 = detach_to_cpu(model.predict(data, static_cam=False))
        p1 = detach_to_cpu(model.predict(data, static_cam=False, flip_test_data=flip_data))
        for tag, pred in (("base", p0), ("flip", p1)):
            m = cam_metrics(pred["smpl_params_incam"], gt)
            for k in ("pa_mpjpe", "mpjpe"):
                agg[tag][k].append(as_np_array(m[k]))

    console.rule("3DPW results (mm, lower is better)")
    for k in ("pa_mpjpe", "mpjpe"):
        b = np.concatenate(agg["base"][k]).mean()
        f = np.concatenate(agg["flip"][k]).mean()
        console.print(f"  {k:10}: baseline [ok]{b:.2f}[/]  flip-test [ok]{f:.2f}[/]  Δ [stage]{f - b:+.2f}[/]")


if __name__ == "__main__":
    main()
