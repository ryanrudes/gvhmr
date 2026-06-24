"""Research-style debugging & inspection helpers for GVHMR.

Small, dependency-light utilities for poking at tensors and the model's internal
representations during research/debugging — none of which affect model behaviour.

Examples
--------
>>> from gvhmr.utils.debug import describe, decompose_latent, nan_hooks
>>> describe(pred_x)                       # shape/dtype/device/stats/nan-count
>>> parts = decompose_latent(pred_x)       # the 151-dim latent, split by name
>>> handles = nan_hooks(model)             # warn on the first NaN-producing module

For 3D motion/mesh visualization see :mod:`gvhmr.utils.wis3d_utils`
(``make_wis3d``, ``add_motion_as_lines``) — install the ``vis`` extra.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

# The 151-dim per-frame latent that NetworkEncoderRoPE predicts and EnDecoder decodes.
# (start, end, name). This is the behaviour contract documented in docs/BEHAVIOR.md —
# keep it in sync if the layout ever changes.
LATENT_LAYOUT: tuple[tuple[int, int, str], ...] = (
    (0, 126, "body_pose_r6d"),  # 21 joints x 6D
    (126, 136, "betas"),  # 10 shape params
    (136, 142, "global_orient_c"),  # 6D, camera frame
    (142, 148, "global_orient_gv"),  # 6D, gravity-view frame
    (148, 151, "local_transl_vel"),  # 3D velocity
)
LATENT_DIM = 151


@dataclass
class TensorStats:
    shape: tuple[int, ...]
    dtype: str
    device: str
    min: float
    max: float
    mean: float
    n_nan: int
    n_inf: int

    def __str__(self) -> str:
        flags = ""
        if self.n_nan:
            flags += f"  ⚠ {self.n_nan} NaN"
        if self.n_inf:
            flags += f"  ⚠ {self.n_inf} Inf"
        return (
            f"Tensor{tuple(self.shape)} {self.dtype} [{self.device}] "
            f"min={self.min:.4g} max={self.max:.4g} mean={self.mean:.4g}{flags}"
        )


def describe(x: torch.Tensor, name: str | None = None, *, print_it: bool = True) -> TensorStats:
    """Summarize a tensor (shape/dtype/device/stats/NaN-Inf counts) for quick inspection."""
    xf = x.detach().float()
    stats = TensorStats(
        shape=tuple(x.shape),
        dtype=str(x.dtype).replace("torch.", ""),
        device=str(x.device),
        min=float(xf.min()) if x.numel() else float("nan"),
        max=float(xf.max()) if x.numel() else float("nan"),
        mean=float(xf.mean()) if x.numel() else float("nan"),
        n_nan=int(torch.isnan(xf).sum()),
        n_inf=int(torch.isinf(xf).sum()),
    )
    if print_it:
        print(f"{name + ': ' if name else ''}{stats}")
    return stats


def decompose_latent(latent: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a ``(..., 151)`` latent into its named components (see ``LATENT_LAYOUT``).

    Useful for inspecting what the denoiser predicted without going through the
    full ``EnDecoder``. Raises if the last dim is not 151.
    """
    if latent.shape[-1] != LATENT_DIM:
        raise ValueError(f"expected last dim {LATENT_DIM}, got {latent.shape[-1]}")
    return {name: latent[..., start:end] for start, end, name in LATENT_LAYOUT}


def summarize_latent(latent: torch.Tensor) -> None:
    """Print a per-component summary of a 151-dim latent."""
    for name, part in decompose_latent(latent).items():
        describe(part, name=name)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Total number of parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def nan_hooks(module: nn.Module, *, warn: bool = True) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks that flag the first module producing NaN/Inf output.

    Returns the hook handles; call ``.remove()`` on each (or use ``remove_hooks``)
    when done. Great for locating where a forward pass goes non-finite.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(qualified_name: str):
        def hook(_mod, _inp, out):
            tensors = out if isinstance(out, (tuple, list)) else [out]
            for t in tensors:
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    msg = f"[nan_hooks] non-finite output from {qualified_name} ({_mod.__class__.__name__})"
                    if warn:
                        warnings.warn(msg, stacklevel=2)
                    else:
                        raise FloatingPointError(msg)

        return hook

    for name, sub in module.named_modules():
        if name:  # skip the root
            handles.append(sub.register_forward_hook(make_hook(name)))
    return handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove hooks returned by :func:`nan_hooks`."""
    for h in handles:
        h.remove()
