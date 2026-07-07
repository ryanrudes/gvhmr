from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities.memory import recursive_detach
from scipy.ndimage._filters import _gaussian_kernel1d

from gvhmr.utils.pylogger import Log

# The in-place initializers modules call via dynamic ``torch.nn.init.<name>`` lookups
# (nn.Linear/Conv reset_parameters use kaiming_uniform_/uniform_, LayerNorm uses ones_/zeros_, ...).
_TORCH_INIT_FNS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    "dirac_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "sparse_",
)


@contextlib.contextmanager
def skip_torch_init():
    """Temporarily no-op ``torch.nn.init.*`` (and timm's ``trunc_normal_``).

    Wrap the *construction* of a model whose weights are immediately overwritten by a strict
    ``load_state_dict`` — the random init is pure wasted work there (seconds for a ViT-Huge).
    Only safe when the subsequent checkpoint load covers every parameter and persistent buffer
    (i.e. ``strict=True``); skipped tensors hold uninitialized memory until the load.
    """

    def _noop(tensor, *args, **kwargs):
        return tensor

    saved = {name: getattr(torch.nn.init, name) for name in _TORCH_INIT_FNS}
    try:
        # The vendored ViTs (hmr2, vitpose) initialize pos_embed via timm's trunc_normal_, which fills
        # tensors directly instead of going through torch.nn.init — patch its shared internal helper.
        from timm.layers import weight_init as _timm_weight_init

        saved_timm = _timm_weight_init._trunc_normal_
    except ImportError:  # pragma: no cover
        _timm_weight_init = None
        saved_timm = None
    try:
        for name in _TORCH_INIT_FNS:
            setattr(torch.nn.init, name, _noop)
        if _timm_weight_init is not None:
            _timm_weight_init._trunc_normal_ = _noop
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)
        if _timm_weight_init is not None:
            _timm_weight_init._trunc_normal_ = saved_timm


def torch_load_mmap(path: str | Path, **kwargs):
    """``torch.load`` with ``mmap=True``, falling back to a regular load for legacy-format saves.

    mmap avoids reading the whole file eagerly (and re-uses the OS page cache across loads), but
    requires the zipfile serialization format — torch.load raises on old pickle-format checkpoints,
    hence the fallback. Keyword args (map_location, weights_only, ...) pass through unchanged.
    """
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (RuntimeError, ValueError):
        return torch.load(path, **kwargs)


def load_pretrained_model(model: nn.Module, ckpt_path: str | Path) -> None:
    """
    Load ckpt to model with strategy
    """
    assert Path(ckpt_path).exists()
    # use model's own load_pretrained_model method
    if hasattr(model, "load_pretrained_model"):
        model.load_pretrained_model(ckpt_path)
    else:
        Log.info(f"Loading ckpt: {ckpt_path}")
        ckpt = torch.load(ckpt_path, "cpu", weights_only=False)
        model.load_state_dict(ckpt, strict=True)


def find_last_ckpt_path(dirpath: str | Path) -> Path | None:
    """
    Assume ckpt is named as e{}* or last*, following the convention of pytorch-lightning.
    """
    assert dirpath is not None
    dirpath = Path(dirpath)
    assert dirpath.exists()
    # Priority 1: last.ckpt
    auto_last_ckpt_path = dirpath / "last.ckpt"
    if auto_last_ckpt_path.exists():
        return auto_last_ckpt_path

    # Priority 2
    model_paths = []
    for p in sorted(list(dirpath.glob("*.ckpt"))):
        if "last" in p.name:
            continue
        model_paths.append(p)
    if len(model_paths) > 0:
        return model_paths[-1]
    else:
        Log.info("No checkpoint found, set model_path to None")
        return None


def get_resume_ckpt_path(resume_mode: str | Path, ckpt_dir: str | Path | None = None) -> str | Path | None:
    if Path(resume_mode).exists():  # This is a path
        return resume_mode
    assert resume_mode == "last"
    return find_last_ckpt_path(ckpt_dir)


def select_state_dict_by_prefix(state_dict: dict, prefix: str, new_prefix: str = "") -> dict:
    """
    For each weight that start with {old_prefix}, remove the {old_prefic} and form a new state_dict.
    Args:
        state_dict: dict
        prefix: str
        new_prefix: str, if exists, the new key will be {new_prefix} + {old_key[len(prefix):]}
    Returns:
        state_dict_new: dict
    """
    state_dict_new = {}
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            new_key = new_prefix + k[len(prefix) :]
            state_dict_new[new_key] = state_dict[k]
    return state_dict_new


def detach_to_cpu(in_dict):
    return recursive_detach(in_dict, to_cpu=True)


def to_cuda(data):
    """Move data in the batch to cuda(), carefully handle data that is not tensor"""
    if isinstance(data, torch.Tensor):
        return data.cuda()
    elif isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v) for v in data]
    else:
        return data


def get_valid_mask(max_len: int, valid_len: int, device: str | torch.device = "cpu") -> torch.Tensor:
    mask = torch.zeros(max_len, dtype=torch.bool).to(device)
    mask[:valid_len] = True
    return mask


def length_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Returns: (B, max_len)
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def repeat_to_max_len(x: torch.Tensor, max_len: int, dim: int = 0) -> torch.Tensor:
    """Repeat last frame to max_len along dim"""
    assert isinstance(x, torch.Tensor)
    if x.shape[dim] == max_len:
        return x
    elif x.shape[dim] < max_len:
        x = x.clone()
        x = x.transpose(0, dim)
        x = torch.cat([x, repeat(x[-1:], "b ... -> (b r) ...", r=max_len - x.shape[0])])
        x = x.transpose(0, dim)
        return x
    else:
        raise ValueError(f"Unexpected length v.s. max_len: {x.shape[0]} v.s. {max_len}")


def repeat_to_max_len_dict(x_dict: dict, max_len: int, dim: int = 0) -> dict:
    for k, v in x_dict.items():
        x_dict[k] = repeat_to_max_len(v, max_len, dim=dim)
    return x_dict


class Transpose(nn.Module):
    def __init__(self, dim1: int, dim2: int) -> None:
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)


class GaussianSmooth(nn.Module):
    def __init__(self, sigma: float = 3, dim: int = -1) -> None:
        super().__init__()
        kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
        kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None]  # (1, 1, K)
        self.register_buffer("kernel_smooth", kernel_smooth, persistent=False)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., f, ...) f at dim"""
        rad = self.kernel_smooth.size(-1) // 2

        x = x.transpose(self.dim, -1)
        x_shape = x.shape[:-1]
        x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
        x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
        x = F.conv1d(x, self.kernel_smooth)
        x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
        x = x.transpose(-1, self.dim)
        return x


def gaussian_smooth(x: torch.Tensor, sigma: float = 3, dim: int = -1) -> torch.Tensor:
    kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
    kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None].to(x)  # (1, 1, K)
    rad = kernel_smooth.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
    x = x.transpose(-1, dim)
    return x


def moving_average_smooth(x: torch.Tensor, window_size: int = 5, dim: int = -1) -> torch.Tensor:
    kernel_smooth = torch.ones(window_size).float() / window_size
    kernel_smooth = kernel_smooth[None, None].to(x)  # (1, 1, window_size)
    rad = kernel_smooth.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
    x = x.transpose(-1, dim)
    return x
