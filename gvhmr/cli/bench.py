"""``gvhmr bench`` — time core inference (model forward + decode + postprocess).

Excludes preprocessing/rendering. Requires the released checkpoint + SMPL-X body models.
"""

from __future__ import annotations

import time

import torch
from rich.table import Table

from gvhmr.utils.console import console, status
from gvhmr.utils.device import device_name, get_device, synchronize, to_device


def _build_model(device):
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


def _synthetic_data(length: int, device) -> dict:
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
def run(*, lengths: list[int], repeats: int) -> None:
    device = get_device()
    console.print(f"[gvhmr]Benchmark[/] on [ok]{device_name(device)}[/] ([gvhmr]{device}[/])")
    with status("Loading model + checkpoint"):
        model = _build_model(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    table = Table(title=f"inference latency · {n_params:.1f}M params")
    table.add_column("frames", justify="right")
    table.add_column("video", justify="right", style="muted")
    table.add_column("ms / call", justify="right")
    table.add_column("realtime", justify="right", style="ok")
    for length in lengths:
        data = _synthetic_data(length, device)
        model.predict(data, static_cam=True)  # warmup
        synchronize(device)
        t = time.time()
        for _ in range(repeats):
            model.predict(data, static_cam=True)
        synchronize(device)
        ms = (time.time() - t) / repeats * 1000
        sec_video = length / 30.0
        table.add_row(str(length), f"{sec_video:.1f}s", f"{ms:.0f}", f"{sec_video / (ms / 1000):.0f}×")
    console.print(table)
