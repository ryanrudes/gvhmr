"""``gvhmr info`` — a Rich diagnostic of the environment, extras, and assets."""

from __future__ import annotations

import importlib.util
import platform

from rich.table import Table

from gvhmr import PROJ_ROOT
from gvhmr.cli.envcmd import SYSTEM_COMPONENTS, system_component_hint, system_component_installed
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

    # The classic Linux trap: bare `uv sync` installs the default PyPI torch wheel, which won't see the
    # GPU on most drivers. Detect it (driver present, torch blind) and print the exact fix — the recorded
    # env's one-command re-sync when available, else the raw uv command.
    from gvhmr.cli.envcmd import detect_torch_extra
    from gvhmr.utils import localconfig

    recorded_torch = localconfig.env_torch()
    if (
        not torch.cuda.is_available()
        and recorded_torch != "cpu"  # an explicitly-recorded CPU build is a choice, not drift
        and (extra := detect_torch_extra()) not in (None, "cpu")
    ):
        fix = (
            "[gvhmr]gvhmr env sync[/]"
            if recorded_torch == extra
            else f"[gvhmr]gvhmr config set torch {extra}[/] then [gvhmr]gvhmr env sync[/]  "
            f"[dim](or: uv sync --extra {extra} — see docs/INSTALL.md)[/]"
        )
        console.print(f"[warn]NVIDIA GPU found but this torch build can't use it[/] — run {fix}")

    # --- Optional features (extras / setup scripts) ---
    third_party = PROJ_ROOT / "third-party"
    feats = Table(title="optional features", expand=False)
    feats.add_column("feature")
    feats.add_column("provided by", style="muted")
    feats.add_column("status", justify="center")
    feats.add_column("get it with", style="muted")
    for name, provider, ok, how in [
        ("preprocessing (YOLO detector)", "ultralytics", _has("ultralytics"), "uv sync --extra preproc"),
        ("camera / SimpleVO", "pycolmap", _has("pycolmap"), "uv sync --extra preproc"),
        ("camera / DPVO (CUDA SLAM)", "dpvo", _has("dpvo"), "scripts/setup_dpvo.sh"),
        (
            "camera / DUSt3R (scene-aware)",
            "third-party/dust3r",
            (third_party / "dust3r").is_dir(),
            "scripts/setup_scene_aware.sh",
        ),
        (
            "camera / VGGT (scene-aware)",
            "third-party/vggt",
            (third_party / "vggt").is_dir(),
            "scripts/setup_scene_aware.sh",
        ),
        ("2D-pose / RTMPose (alt backend)", "rtmlib", _has("rtmlib"), "uv sync --extra rtmpose"),
        *(
            # System tools: no extra can provide these, so `gvhmr env sync` can only advise. Missing
            # exiftool silently costs ~65%-200% on in-cam depth — see docs/CAMERA_METADATA.md.
            (label, f"{binary} (system)", system_component_installed(key), system_component_hint(key))
            for key, (label, _desc, binary, _cmds) in SYSTEM_COMPONENTS.items()
        ),
        ("mesh renderer (GPU, moderngl)", "moderngl", _has("moderngl"), "base install"),
        ("render fallback (pytorch3d)", "pytorch3d", _has("pytorch3d"), "uv sync --extra render (optional)"),
        ("3D visualization", "wis3d", _has("wis3d"), "uv sync --extra vis"),
    ]:
        feats.add_row(name, provider, _yn(ok), "" if ok else how)
    console.print(feats)

    # --- Checkpoints / assets (from the manifest; fetch with `gvhmr download`) ---
    from gvhmr.utils import assets

    ckpt = Table(title=f"checkpoints  [dim]({assets.CHECKPOINT_ROOT})[/]", expand=False)
    ckpt.add_column("asset")
    ckpt.add_column("group", style="muted")
    ckpt.add_column("status", justify="center")
    for name, a in assets.ASSETS.items():
        ckpt.add_row(name, a.group, _yn(assets.is_present(a)))
    smplx_ok = (assets.BODY_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz").exists()
    smpl_ok = (assets.BODY_MODEL_ROOT / "smpl/SMPL_NEUTRAL.pkl").exists()
    ckpt.add_row("body_models/smplx", "demo (gated)", _yn(smplx_ok))
    ckpt.add_row("body_models/smpl", "render (gated)", _yn(smpl_ok))
    console.print(ckpt)

    gated_gap = [n for n, ok in [("body_models/smplx", smplx_ok), ("body_models/smpl", smpl_ok)] if not ok]
    gap = assets.missing() + gated_gap
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
    if not assets.missing("demo") and smplx_ok and _has("ultralytics") and _has("pycolmap"):
        console.print("[dim]Ready — try: [gvhmr]uv run gvhmr demo docs/example_video/tennis.mp4 -s[/][/]")
