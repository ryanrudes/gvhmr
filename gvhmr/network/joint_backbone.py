"""HMR2 backbone forwardable IN the training loop, with LoRA adapters (ROADMAP Regime B, stage 2).

The preproc :class:`~gvhmr.utils.preproc.vitfeat_extractor.Extractor` runs the SAME HMR2 forward under
``@torch.no_grad()`` to *cache* features. This module is its trainable twin: it keeps grad and inserts
low-rank adapters (:mod:`gvhmr.network.lora`) so the backbone can learn the task while only a few-M
adapter params update and the ~600 M base stays frozen.

Faithfulness guarantee (pinned by ``tests/test_joint_backbone.py``): the adapters init to zero, so a
freshly-built ``JointBackbone`` produces the token **bit-identical** to the frozen HMR2 that the cached
``f_imgseq`` was extracted with — enabling the joint path on a fresh model is a no-op until it trains.
"""

from __future__ import annotations

import torch
from torch import nn

from gvhmr.network.hmr2 import HMR2, load_hmr2
from gvhmr.network.lora import HMR2_LORA_TARGETS, apply_lora, count_lora_parameters
from gvhmr.utils.net_utils import skip_torch_init


class JointBackbone(nn.Module):
    """HMR2 (ViT + SMPL-head decoder) + LoRA adapters, forwarding crops → the 1024-d ``f_imgseq`` token.

    ``forward(crops)`` mirrors ``Extractor.extractor({"img": crops})`` exactly (same ``HMR2.forward``,
    ``feat_mode=True``), but with grad enabled and the target Linears LoRA-wrapped. ``apply_lora`` freezes
    the base, so after construction the only trainable parameters are the adapters.
    """

    feat_dim = 1024  # the SMPL-head token width — must match the network's imgseq_dim

    def __init__(
        self,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        checkpoint: str | None = None,
        targets: tuple[str, ...] = HMR2_LORA_TARGETS,
    ):
        super().__init__()
        with skip_torch_init():  # random init is overwritten by load_hmr2's strict ckpt load
            self.hmr2: HMR2 = load_hmr2(checkpoint) if checkpoint else load_hmr2()
        self.replaced = apply_lora(self.hmr2, targets=targets, rank=rank, alpha=alpha, dropout=dropout)
        if not self.replaced:
            raise RuntimeError(f"LoRA matched no Linear in HMR2 for targets={targets} — check the paths")
        self.n_lora_params = count_lora_parameters(self.hmr2)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """``crops`` is ``(N, 3, 256, 256)`` (the preproc-normalized crop) → ``(N, 1024)`` token, with grad."""
        return self.hmr2({"img": crops})
