"""LoRA adapters for opt-in joint backbone training (ROADMAP Regime B, rung 1).

GVHMR trains on *cached* HMR2 features — the ViT is never in the loop, which is what makes training
cheap. To let the backbone learn *this* task (the real source of HMR2's edge over a frozen swap; see
ROADMAP A1), we re-insert the ViT + SMPL-head decoder with **low-rank adapters** so only a few M params
train and the ~600 M base stays frozen.

Design constraints this file exists to satisfy:

* **The vendored `hmr2/` tree is frozen** (don't refactor it). So LoRA is applied by *replacing* the target
  `nn.Linear` instances by their stable named-module paths *after* `load_hmr2()` — the base `state_dict`
  keys are untouched, so the checkpoint still loads `strict=True`.
* **Behaviour preservation.** `LoRALinear` initializes ``B`` to zero, so at init the delta is exactly zero
  and the wrapped module is numerically identical to the base ``nn.Linear``. Enabling LoRA on a fresh model
  changes nothing until the adapters train — the default (cached-feature) path is never affected because it
  never constructs these wrappers.

Nothing here imports torch at module load beyond the annotations; it's a plain nn.Module helper.
"""

from __future__ import annotations

import math

import torch
from torch import nn

#: The HMR2 Linear layers LoRA targets, by the *suffix* of their named-module path. Attention qkv/proj and
#: the MLP fc1/fc2 in every ViT block, plus the SMPL-head decoder's attention/ff Linears. These paths are
#: upstream-stable (renaming would already break the strict checkpoint load), so matching by suffix is safe.
HMR2_LORA_TARGETS: tuple[str, ...] = (
    ".attn.qkv",
    ".attn.proj",
    ".mlp.fc1",
    ".mlp.fc2",
    ".to_qkv",
    ".to_q",
    ".to_kv",
)


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank update ``(alpha/rank) · B @ A``.

    ``forward(x) = base(x) + scaling · dropout(x) @ Aᵀ @ Bᵀ``. ``A`` is Kaiming-init, ``B`` is **zero**, so
    the initial delta is zero and the wrapper matches ``base`` bit-for-bit until trained. The base weight
    (and bias) are frozen; only ``lora_A`` / ``lora_B`` require grad, so the existing optimizer (which
    collects ``requires_grad`` params) picks up exactly the adapters.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays zero -> zero delta at init
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (…,in) @ (in,rank) @ (rank,out) — computed low-rank-first so the (in,out) product is never formed.
        delta = self.dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return self.base(x) + self.scaling * delta

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.3g}"


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    mod = root
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def apply_lora(
    model: nn.Module,
    *,
    targets: tuple[str, ...] = HMR2_LORA_TARGETS,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> list[str]:
    """Replace every ``nn.Linear`` whose module path ends with one of ``targets`` by a :class:`LoRALinear`.

    Returns the list of replaced module paths (so callers can log / assert the count). With
    ``freeze_base`` the non-LoRA parameters are set ``requires_grad_(False)`` first, so after this call the
    *only* trainable parameters are the adapters. Idempotent guard: a module already wrapped is skipped.
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)

    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in targets):
            to_replace.append((name, module))

    replaced: list[str] = []
    for name, linear in to_replace:
        parent_path, _, attr = name.rpartition(".")
        parent = _get_submodule(model, parent_path) if parent_path else model
        setattr(parent, attr, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    return replaced


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Every LoRA adapter parameter in ``model`` (``lora_A`` / ``lora_B``) — the trainable set."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The LoRA-only state dict — what to checkpoint (the frozen ~600 M base is not re-saved)."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))
