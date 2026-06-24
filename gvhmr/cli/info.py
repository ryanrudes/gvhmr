"""``gvhmr info`` — a Rich diagnostic of the environment, extras, and assets."""

from __future__ import annotations

import importlib.util
import platform

from rich.table import Table

from gvhmr import PROJ_ROOT
from gvhmr.utils.console import console


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _yn(ok: bool) -> str:
    return "[ok]✓[/]" if ok else "[err]✗[/]"


def run() -> None:
    import torch

    from gvhmr.utils.device import device_name, get_device

    # --- Environment ---
    env = Table(title="[gvhmr]GVHMR[/] · environment", show_header=False, expand=False)
    env.add_column(style="muted")
    env.add_column()
    device = get_device()
    env.add_row("python", platform.python_version())
    env.add_row("platform", f"{platform.system()} {platform.machine()}")
    env.add_row("torch", torch.__version__)
    env.add_row("device", f"{device_name(device)} ([gvhmr]{device}[/])")
    env.add_row("cuda", _yn(torch.cuda.is_available()))
    env.add_row("mps", _yn(torch.backends.mps.is_available()))
    env.add_row("project root", str(PROJ_ROOT))
    console.print(env)

    # --- Optional features (extras) ---
    feats = Table(title="optional features", expand=False)
    feats.add_column("feature")
    feats.add_column("provided by", style="muted")
    feats.add_column("status", justify="center")
    for name, module in [
        ("preprocessing (YOLO)", "ultralytics"),
        ("camera / SimpleVO", "pycolmap"),
        ("body models", "smplx"),
        ("mesh rendering", "pytorch3d"),
        ("3D visualization", "wis3d"),
        ("SMPL .pkl loader", "chumpy"),
    ]:
        feats.add_row(name, module, _yn(_has(module)))
    console.print(feats)

    # --- Checkpoints / assets ---
    ckpt = Table(title="checkpoints & body models", expand=False)
    ckpt.add_column("asset")
    ckpt.add_column("path", style="muted")
    ckpt.add_column("status", justify="center")
    base = PROJ_ROOT / "inputs/checkpoints"
    for label, rel in [
        ("GVHMR", "gvhmr/gvhmr_siga24_release.ckpt"),
        ("HMR2", "hmr2/epoch=10-step=25000.ckpt"),
        ("ViTPose", "vitpose/vitpose-h-multi-coco.pth"),
        ("YOLO", "yolo/yolov8x.pt"),
        ("SMPL-X", "body_models/smplx/SMPLX_NEUTRAL.npz"),
        ("SMPL", "body_models/smpl/SMPL_NEUTRAL.pkl"),
    ]:
        ckpt.add_row(label, f"inputs/checkpoints/{rel}", _yn((base / rel).exists()))
    console.print(ckpt)
