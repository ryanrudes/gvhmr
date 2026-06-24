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


def get_device(prefer: str | torch.device | None = None) -> torch.device:
    """Return the best available device, honouring ``prefer`` / ``$GVHMR_DEVICE``."""
    choice = prefer if prefer is not None else os.environ.get("GVHMR_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
