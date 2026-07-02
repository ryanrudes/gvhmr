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
        ("camera / DPVO (CUDA SLAM)", "dpvo"),  # scripts/setup_dpvo.sh
        ("body models", "smplx"),
        ("mesh rendering", "pytorch3d"),
        ("3D visualization", "wis3d"),
        ("SMPL .pkl loader", "chumpy"),
    ]:
        feats.add_row(name, module, _yn(_has(module)))
    console.print(feats)

    # --- Checkpoints / assets (from the manifest; fetch with `gvhmr download`) ---
    from gvhmr.utils import assets

    ckpt = Table(title=f"checkpoints  [dim]({assets.CHECKPOINT_ROOT})[/]", expand=False)
    ckpt.add_column("asset")
    ckpt.add_column("group", style="muted")
    ckpt.add_column("status", justify="center")
    for name, a in assets.ASSETS.items():
        ckpt.add_row(name, a.group, _yn(assets.is_present(a)))
    body_ok = (assets.BODY_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz").exists()
    ckpt.add_row("body_models", "demo (gated)", _yn(body_ok))
    console.print(ckpt)

    gap = assets.missing() + ([] if body_ok else ["body_models"])
    if gap:
        console.print(f"[warn]missing[/]: {', '.join(gap)} — run [gvhmr]gvhmr download[/] (body models are gated)")

    # --- Training / eval data packs (fetch with `gvhmr download --data`; relocate with $GVHMR_DATA_ROOT) ---
    data = Table(title=f"data packs  [dim]({assets.DATA_ROOT}  ·  $GVHMR_DATA_ROOT)[/]", expand=False)
    data.add_column("pack")
    data.add_column("role", style="muted")
    data.add_column("status", justify="center")
    roles = {"amass": "train", "bedlam": "train", "h36m": "train", "3dpw": "train+eval", "emdb": "eval", "rich": "eval"}
    for name, (_, ds_dir, _) in assets.DATA_PACKS.items():
        present = (assets.DATA_ROOT / ds_dir / "hmr4d_support").exists()
        data.add_row(name, roles.get(name, ""), _yn(present))
    console.print(data)

    from gvhmr.utils import localconfig

    cfg = localconfig.config_file()
    where = f"[ok]{cfg}[/]" if cfg else "[muted]none[/] (using defaults / env vars)"
    console.print(
        f"\nconfig file: {where} — manage asset locations + default model versions with [gvhmr]gvhmr config[/]"
    )
