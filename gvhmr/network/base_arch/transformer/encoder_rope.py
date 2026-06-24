from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

from gvhmr.network.base_arch.embeddings.rotary_embedding import ROPE


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.rope = ROPE(self.head_dim, max_seq_len=4096)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, L, C)
        # attn_mask: (L, L)
        # key_padding_mask: (B, L)
        B, L, _ = x.shape
        xq, xk, xv = self.query(x), self.key(x), self.value(x)

        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)

        xq = self.rope.rotate_queries_or_keys(xq)  # B, N, L, C
        xk = self.rope.rotate_queries_or_keys(xk)  # B, N, L, C

        # Fused scaled-dot-product attention (replaces the manual qk^T/softmax/av einsums).
        # Build a boolean keep-mask (SDPA: True = attend) from the original mask-out masks.
        masked_out = None  # True = position is masked out
        if attn_mask is not None:
            masked_out = attn_mask.reshape(1, 1, L, L).expand(B, self.num_heads, L, L)
        if key_padding_mask is not None:
            kpm = key_padding_mask.reshape(B, 1, 1, L).expand(B, self.num_heads, L, L)
            masked_out = kpm if masked_out is None else (masked_out | kpm)
        keep_mask = None if masked_out is None else ~masked_out

        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=keep_mask, dropout_p=self.dropout.p if self.training else 0.0
        )  # B, N, L, C
        output = output.transpose(1, 2).reshape(B, L, -1)  # B, L, C
        output = self.proj(output)  # B, L, C
        return output


class EncoderRoPEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Zero-out adaLN modulation layers
        nn.init.constant_(self.gate_msa, 0)
        nn.init.constant_(self.gate_mlp, 0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.gate_msa * self._sa_block(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, L, C)
        x = self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x
