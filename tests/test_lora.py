"""LoRA adapters for opt-in joint backbone training (ROADMAP Regime B). CPU-only, no weights/GPU.

Pins the behaviour-preservation contract the joint-training path depends on: a freshly-wrapped model is
numerically identical to the base (zero-init delta), only the adapters train, and `apply_lora` swaps
exactly the intended Linears by their stable named-module paths (so the frozen hmr2 tree is untouched and
the checkpoint still loads strict).
"""

from __future__ import annotations

import torch
from torch import nn

from gvhmr.network.lora import (
    HMR2_LORA_TARGETS,
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    lora_parameters,
    lora_state_dict,
)


def test_zero_init_is_identity():
    """B is zero at init -> the wrapper matches the base Linear bit-for-bit. This is the behaviour-
    preservation guarantee: enabling LoRA changes nothing until the adapters actually train."""
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    wrapped = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 5, 16)
    assert torch.equal(wrapped(x), base(x))


def test_delta_math():
    """Once B is non-zero the output is exactly base(x) + (alpha/rank) · x Aᵀ Bᵀ."""
    torch.manual_seed(1)
    base = nn.Linear(8, 8)
    w = LoRALinear(base, rank=2, alpha=4.0)  # scaling = 2.0
    with torch.no_grad():
        w.lora_B.copy_(torch.randn_like(w.lora_B))
    x = torch.randn(7, 8)
    expected = base(x) + 2.0 * (x @ w.lora_A.T @ w.lora_B.T)
    assert torch.allclose(w(x), expected, atol=1e-6)


def test_only_adapters_train():
    base = nn.Linear(10, 10)
    w = LoRALinear(base, rank=4, alpha=8.0)
    assert not w.base.weight.requires_grad
    assert w.base.bias is not None and not w.base.bias.requires_grad
    assert w.lora_A.requires_grad and w.lora_B.requires_grad


def test_grad_flows_to_adapters_only():
    torch.manual_seed(2)
    base = nn.Linear(12, 12)
    w = LoRALinear(base, rank=4, alpha=8.0)
    with torch.no_grad():  # break the zero-init so B receives a gradient
        w.lora_B.copy_(torch.randn_like(w.lora_B) * 0.01)
    w(torch.randn(4, 12)).pow(2).sum().backward()
    assert w.lora_A.grad is not None and w.lora_B.grad is not None
    assert w.base.weight.grad is None


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.fc2 = nn.Linear(dim * 4, dim)


class _MiniViT(nn.Module):
    """Mimics the hmr2 named-module paths (backbone.blocks.{i}.attn.qkv, …) apply_lora matches on."""

    def __init__(self, dim=16, depth=2):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([_Block(dim) for _ in range(depth)])
        self.backbone.patch_embed = nn.Linear(dim, dim)  # NOT a target — must be left alone


def test_apply_lora_swaps_exactly_the_targets():
    model = _MiniViT(dim=16, depth=2)
    replaced = apply_lora(model, targets=HMR2_LORA_TARGETS, rank=4, alpha=8.0)
    # 2 blocks x {qkv, proj, fc1, fc2} = 8 swaps; patch_embed untouched
    assert len(replaced) == 8
    assert all(isinstance(m, LoRALinear) for name, m in model.named_modules() if name.endswith(".attn.qkv"))
    assert isinstance(model.backbone.patch_embed, nn.Linear)
    assert not isinstance(model.backbone.patch_embed, LoRALinear)


def test_apply_lora_freezes_base_and_is_idempotent():
    model = _MiniViT(dim=16, depth=2)
    apply_lora(model, rank=4, alpha=8.0)
    # base frozen, adapters trainable
    assert not model.backbone.patch_embed.weight.requires_grad
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    # a second pass finds nothing new to wrap (already LoRALinear)
    assert apply_lora(model, rank=4, alpha=8.0) == []


def test_lora_helpers():
    model = _MiniViT(dim=16, depth=2)
    apply_lora(model, rank=4, alpha=8.0)
    # rank 4: A is (4,in), B is (out,4). Count matches lora_parameters sum.
    assert count_lora_parameters(model) == sum(p.numel() for p in lora_parameters(model))
    sd = lora_state_dict(model)
    assert sd and all("lora_A" in k or "lora_B" in k for k in sd)
    # exactly two tensors (A,B) per swapped module
    assert len(sd) == 8 * 2


def test_forward_shape_preserved_after_apply():
    model = _MiniViT(dim=16, depth=1)
    apply_lora(model, rank=4, alpha=8.0)
    block = model.backbone.blocks[0]
    x = torch.randn(2, 16)
    assert block.attn.qkv(x).shape == (2, 48)
    assert block.mlp.fc2(block.mlp.fc1(x)).shape == (2, 16)
