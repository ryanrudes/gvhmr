"""Device selection and movement for GVHMR (CUDA / Apple-Silicon MPS / CPU).

The original code hard-coded ``.cuda()`` everywhere. These helpers make the
inference/demo path device-agnostic so GVHMR runs on an Apple-Silicon GPU (MPS)
or CPU as well as CUDA. Selection order (first available wins):

1. an explicit ``prefer`` argument,
2. the ``GVHMR_DEVICE`` environment variable (e.g. ``GVHMR_DEVICE=mps``),
3. CUDA, then MPS, then CPU.

Note: mesh rendering (pytorch3d) and DPVO SLAM remain CUDA-only; on MPS those
features are unavailable, but core GVHMR inference and the geometry math run.
"""

from __future__ import annotations

import os

import torch


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


_BACKENDS_CONFIGURED = False


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def tf32_enabled() -> bool:
    """Whether TF32 tensor-core matmuls should be enabled. **Off unless explicitly opted in.**

    TF32 is *not* free here: it truncates fp32 matmul mantissas to 10 bits, and on a full-length
    sequence that error accumulates through the RoPE denoiser's attention. Measured on EMDB with the
    released checkpoint, it costs **+3.3mm PA-MPJPE (42.7 → 46.0) and 4x the acceleration error
    (3.6 → 14.2 m/s²)** — the derivative metrics amplify per-frame noise by fps² / fps³. 3DPW and RICH
    move <0.3mm, which is why it went unnoticed. It also buys nothing: the hot stages already run bf16
    (the ViTs, faster than TF32 anyway), fp16 (YOLO) or CPU (``predict_device``), so an EMDB eval times
    the same either way. Opt in with ``GVHMR_ENABLE_TF32=1`` for an fp32-heavy workload that can afford it.
    """
    if _env_true("GVHMR_DISABLE_TF32"):  # explicit off always wins (back-compat)
        return False
    return _env_true("GVHMR_ENABLE_TF32")


def configure_torch_backends() -> None:
    """Configure the CUDA backends (idempotent; called from :func:`get_device` on a CUDA box).

    - **cuDNN benchmark**: our conv/ViT inputs are fixed-shape per stage, so autotuning the kernels once
      pays off across the (many) frames. Kernel *selection* only — it cannot touch the denoiser, which
      has no convs, so the benchmarks are unaffected.
    - **TF32**: off unless opted in — see :func:`tf32_enabled` for the accuracy evidence.

    Note ``torch.amp.autocast(enabled=False)`` does NOT gate TF32: it suppresses bf16/fp16 autocasting
    only, while TF32 is a separate switch that applies to every fp32 matmul. The fp32 FK/IK guards
    therefore never protected the rotary/attention path from it.
    """
    global _BACKENDS_CONFIGURED
    if _BACKENDS_CONFIGURED or not torch.cuda.is_available():
        return
    _BACKENDS_CONFIGURED = True
    torch.backends.cudnn.benchmark = True
    if tf32_enabled():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def predict_device() -> torch.device:
    """Device for the GVHMR network's ``predict()`` (the RoPE denoiser + EnDecoder FK/IK).

    Verified ~6.7x FASTER on CPU than CUDA (1989→299 ms @ L=256) and faster than MPS too: it's a small
    model dominated by many tiny sequential ops (rotation recurrences, per-frame IK), so per-kernel launch
    overhead on an accelerator dwarfs the compute. Preproc (the heavy ViTs) and rendering stay on the GPU
    — only the network moves here. Default CPU; override with ``$GVHMR_PREDICT_DEVICE`` (e.g. ``cuda``).
    """
    choice = os.environ.get("GVHMR_PREDICT_DEVICE")
    if choice:
        return get_device(choice)
    return torch.device("cpu")


def get_device(prefer: str | torch.device | None = None) -> torch.device:
    """Return the best available device, honouring ``prefer`` / ``$GVHMR_DEVICE``.

    If the requested type is unavailable (e.g. ``cuda`` on CPU-only hardware), falls back
    through cuda → mps → cpu instead of raising at ``.to(device)`` time.
    """
    choice = prefer if prefer is not None else os.environ.get("GVHMR_DEVICE")
    if choice:
        dev = torch.device(choice)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = _auto_device()
        elif dev.type == "mps" and not _mps_available():
            dev = _auto_device()
    else:
        dev = _auto_device()
    if dev.type == "cuda":
        configure_torch_backends()  # every entry point (demo/eval/bench/inference) funnels through here
    return dev


def to_device(data, device: str | torch.device):
    """Recursively move tensors in a (nested) dict/list/tuple to ``device``.

    Non-tensor leaves are returned unchanged. This is the device-agnostic
    counterpart to the legacy ``net_utils.to_cuda``.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data


def device_name(device: str | torch.device) -> str:
    """A human-readable name for logging (e.g. the CUDA model, or 'Apple Silicon GPU (MPS)')."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def synchronize(device: str | torch.device | None = None) -> None:
    """Synchronize the active accelerator (for accurate timing); a no-op on CPU."""
    device = get_device() if device is None else torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
