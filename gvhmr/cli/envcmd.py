"""``gvhmr env`` — let GVHMR manage its own Python environment, so users never run uv by hand.

uv syncs *exactly*: a bare ``uv sync`` (or plain ``uv run``) reverts the machine-specific torch build and
prunes the preproc extras and out-of-band DPVO. The fix is to **record** this box's choices once — in the
same readable config file as everything else (``[env]``: torch build, extras, dpvo) — and replay them:

- ``gvhmr env record``  — write the choices (done automatically by ``scripts/install.sh`` and the wizard)
- ``gvhmr env sync``    — run uv with the full recorded flag set (``--inexact``, so nothing gets pruned)
- ``gvhmr env show``    — what's recorded + the exact command ``sync`` would run

The wizard (``gvhmr config init``) records this for you; ``gvhmr info`` points at ``gvhmr env sync``
whenever it detects drift (e.g. torch lost CUDA after a bare sync).
"""

from __future__ import annotations

import typer

from gvhmr import PROJ_ROOT
from gvhmr.utils.console import console

app = typer.Typer(help="Record & re-sync this box's Python environment (so you never run uv by hand).")

#: The torch-backend choices (the cuXXX/cpu extras in pyproject.toml); "none" = the default PyPI wheel.
TORCH_CHOICES = ("none", "cpu", "cu124", "cu126", "cu128")


def _nvidia_cuda_version() -> str | None:
    """The driver's max supported CUDA version from ``nvidia-smi`` (e.g. ``"12.8"``), or None."""
    import re
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return m.group(1) if m else None


def _gpu_names() -> str:
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _extra_for_cuda(ver: str) -> str:
    """Map a driver CUDA version to the matching torch extra (mirrors docs/INSTALL.md's table)."""
    major, minor = (int(x) for x in ver.split(".")[:2])
    if (major, minor) >= (12, 8):
        return "cu128"
    if (major, minor) >= (12, 6):
        return "cu126"
    return "cu124"


def detect_torch_extra() -> str | None:
    """This box's recommended torch backend extra: ``None`` on macOS / when no NVIDIA driver is found
    (= the default PyPI wheel), else the cuXXX matching the driver — with the V100/P100 guard (the
    cu128 wheel dropped ``sm_70``/``sm_60``, so those boxes get ``cu126``)."""
    import platform
    import re

    if platform.system() == "Darwin":
        return None
    ver = _nvidia_cuda_version()
    if ver is None:
        return "cpu" if platform.system() == "Linux" else None
    extra = _extra_for_cuda(ver)
    if extra == "cu128" and re.search(r"V100|P100", _gpu_names(), re.IGNORECASE):
        extra = "cu126"
    return extra


def sync_args(torch_extra: str | None, extras: list[str], dpvo: bool = False) -> list[str]:
    """The ``uv`` argument vector that realizes the recorded env.

    ``--inexact`` is the key: it syncs the requested extras **without removing** anything else, so the
    out-of-band DPVO install (and any user pip-installs) survive. ``dpvo=true`` adds the ``dpvo`` extra —
    DPVO's locked, torch-ABI-free runtime deps (numba, pypose), whose constraints keep numpy where numba
    needs it. Pure — unit-tested.
    """
    args = ["sync", "--inexact"]
    wanted = list(extras) + (["dpvo"] if dpvo else [])
    args += [f"--extra={e}" for e in dict.fromkeys(wanted)]  # de-dup, keep order
    if torch_extra and torch_extra not in ("none", "default"):
        args.append(f"--extra={torch_extra}")
    return args


def record_env(torch: str | None = None, extras: str | None = None, dpvo: bool | None = None) -> dict:
    """Merge the given fields into the config file's ``[env]`` table (None = leave unchanged).
    Returns the resulting table. Used by the CLI command, the wizard, and the setup scripts."""
    from gvhmr.cli.config import write_settings
    from gvhmr.utils import localconfig

    env = {str(k): str(v) for k, v in localconfig.env_table().items()}
    if torch is not None:
        env["torch"] = torch
    if extras is not None:
        env["extras"] = ",".join(e.strip() for e in extras.split(",") if e.strip())
    if dpvo is not None:
        env["dpvo"] = "true" if dpvo else "false"
    write_settings(env=env)
    return env


@app.command()
def record(
    torch: str | None = typer.Option(None, "--torch", help=f"Torch backend extra ({'/'.join(TORCH_CHOICES)})."),
    extras: str | None = typer.Option(None, "--extras", help="Comma-separated extras (e.g. preproc,dev)."),
    dpvo: bool | None = typer.Option(None, "--dpvo/--no-dpvo", help="DPVO (out-of-band CUDA SLAM) installed."),
) -> None:
    """Record this box's environment choices in the config file [dim](the installer/wizard do this)[/]."""
    from gvhmr.utils import localconfig

    if torch is not None and torch not in TORCH_CHOICES:
        console.print(f"[err]'{torch}' is not a torch backend[/] — choose from: {', '.join(TORCH_CHOICES)}")
        raise typer.Exit(1)
    env = record_env(torch=torch, extras=extras, dpvo=dpvo)
    console.print(f"recorded env: [gvhmr]{env}[/] → [muted]{localconfig.target_config_path()}[/]")


@app.command()
def sync(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the uv command without running it."),
) -> None:
    """Re-sync the environment from the recorded choices [dim](never prunes; run after any drift)[/]."""
    import importlib.util
    import shutil
    import subprocess

    from gvhmr.utils import localconfig

    if not localconfig.env_table():
        console.print(
            "[err]No environment recorded yet.[/] Run [gvhmr]gvhmr config init[/] (wizard) or record directly, e.g. "
            "[gvhmr]gvhmr env record --torch cu128 --extras preproc[/]."
        )
        raise typer.Exit(1)
    args = sync_args(localconfig.env_torch(), localconfig.env_extras(), dpvo=localconfig.env_dpvo())
    console.print(f"$ [gvhmr]uv {' '.join(args)}[/]  [dim](cwd {PROJ_ROOT})[/]")
    if dry_run:
        return
    uv = shutil.which("uv")
    if uv is None:
        console.print("[err]uv not found on PATH[/] — install it: https://docs.astral.sh/uv/")
        raise typer.Exit(1)
    code = subprocess.run([uv, *args], cwd=PROJ_ROOT).returncode
    if code != 0:
        raise typer.Exit(code)
    if localconfig.env_dpvo() and importlib.util.find_spec("dpvo") is None:
        console.print(
            "[warn]DPVO is recorded for this box but missing[/] — re-run [gvhmr]scripts/setup_dpvo.sh[/] "
            "(it compiles against the synced torch; idempotent)."
        )
    console.print("[ok]environment synced[/] — [dim]`gvhmr info` shows the result[/]")


@app.command()
def show() -> None:
    """Show the recorded environment and the exact command [gvhmr]sync[/] would run."""
    import importlib.util

    from gvhmr.utils import localconfig

    env = localconfig.env_table()
    if not env:
        console.print("recorded env: [muted]none[/] — run [gvhmr]gvhmr config init[/] or [gvhmr]gvhmr env record[/]")
        detected = detect_torch_extra()
        console.print(f"detected for this box: torch backend [gvhmr]{detected or 'none (PyPI wheel)'}[/]")
        return
    console.print(f"recorded env ([muted]{localconfig.config_file()}[/]):")
    console.print(f"  torch  = [gvhmr]{localconfig.env_torch() or 'none (PyPI wheel)'}[/]")
    console.print(f"  extras = [gvhmr]{', '.join(localconfig.env_extras()) or '(none)'}[/]")
    dpvo = localconfig.env_dpvo()
    dpvo_now = importlib.util.find_spec("dpvo") is not None
    console.print(f"  dpvo   = [gvhmr]{str(dpvo).lower()}[/]" + ("" if dpvo_now == dpvo else "  [warn](drifted)[/]"))
    console.print(
        f"sync runs: [gvhmr]uv {' '.join(sync_args(localconfig.env_torch(), localconfig.env_extras(), dpvo=dpvo))}[/]"
    )
