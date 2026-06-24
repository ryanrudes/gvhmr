#!/usr/bin/env python3
"""Benchmark GVHMR demo inference (model forward + decode + postprocess).

Builds the released model and times ``DemoPL.predict`` on seeded synthetic inputs at
several sequence lengths. Excludes preprocessing/rendering — it measures the core
pipeline that the perf work targets. Requires the checkpoint + SMPL-X body models
(see docs/INSTALL.md).

    uv run python tools/bench/bench_inference.py
    GVHMR_DEVICE=mps uv run python tools/bench/bench_inference.py
"""

from __future__ import annotations

import argparse
import time

import torch

from gvhmr.utils.device import device_name, get_device, synchronize, to_device


def build_model(device):
    import hydra
    from hydra import compose, initialize_config_module

    import gvhmr.model.gvhmr.gvhmr_pl_demo  # noqa: F401  (registers the demo model)
    from gvhmr.configs import register_store_gvhmr

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=bench", "static_cam=True"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    return model.eval().to(device)


def synthetic_data(length: int, device):
    torch.manual_seed(1234)
    data = {
        "length": torch.tensor(length),
        "kp2d": torch.randn(length, 17, 3),
        "bbx_xys": torch.rand(length, 3) * 200 + 100,
        "K_fullimg": torch.tensor([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]]).repeat(length, 1, 1),
        "cam_angvel": torch.zeros(length, 6),
        "f_imgseq": torch.randn(length, 1024),
    }
    return to_device(data, device)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    device = get_device()
    print(f"[Device] {device_name(device)} ({device})")
    model = build_model(device)
    print(f"[Model]  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params\n")
    print(f"{'frames':>8} {'sec_video':>10} {'ms/call':>10} {'realtime_x':>12}")
    for L in args.lengths:
        data = synthetic_data(L, device)
        model.predict(data, static_cam=True)  # warmup
        synchronize(device)
        t = time.time()
        for _ in range(args.repeats):
            model.predict(data, static_cam=True)
        synchronize(device)
        ms = (time.time() - t) / args.repeats * 1000
        sec_video = L / 30.0
        print(f"{L:>8} {sec_video:>9.1f}s {ms:>9.0f} {sec_video / (ms / 1000):>11.0f}x")


if __name__ == "__main__":
    main()
