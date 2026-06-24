"""Characterization tests for :mod:`gvhmr.network.base_arch.embeddings.rotary_embedding`.

This pins rotary positional-embedding (RoPE) numerics so the reorg/rename/typing
passes can't silently change them. The invariants exercised are:

* ``rotate_half`` is a 90-degree rotation in each ``(x_{2j}, x_{2j+1})`` pair, so it
  has period 4 (``rotate_half`` applied twice negates, four times is identity).
* ``get_encoding`` returns ``(L, D)``, is deterministic, and follows the standard
  ``1/10000^(2i/d)`` inverse-frequency formula with each frequency duplicated.
* ``apply_rotary_emb`` preserves the per-pair norm of the rotated block, leaves the
  ``t_left`` / ``t_right`` slices untouched, and honours ``start_index``.
* ``ROPE.rotate_queries_or_keys`` preserves the ``(B, H, L, D)`` shape and takes the
  ``seq_len > max_seq_len`` recompute branch correctly.

The source imports the deprecated ``torch.cuda.amp.autocast``; that emits a
``FutureWarning`` on import but is intentionally left unmodified.
"""

from __future__ import annotations

import math

import pytest
import torch

from gvhmr.network.base_arch.embeddings.rotary_embedding import (
    ROPE,
    apply_rotary_emb,
    get_encoding,
    rotate_half,
)


# --------------------------------------------------------------------------- #
# rotate_half
# --------------------------------------------------------------------------- #
def test_rotate_half_golden() -> None:
    # [0,1,2,3,4,5] -> pairs (0,1),(2,3),(4,5) each mapped (a,b)->(-b,a).
    x = torch.arange(6).float().reshape(1, 6)
    expected = torch.tensor([[-1.0, 0.0, -3.0, 2.0, -5.0, 4.0]])
    assert torch.equal(rotate_half(x), expected)


def test_rotate_half_twice_negates() -> None:
    # rotate_half is a +90deg rotation per pair; applying it twice == -identity.
    v = torch.randn(3, 8)
    assert torch.allclose(rotate_half(rotate_half(v)), -v, atol=1e-6)


def test_rotate_half_has_period_four() -> None:
    # Four applications return to identity (it is *not* a self-inverse involution).
    v = torch.randn(4, 6)
    out = rotate_half(rotate_half(rotate_half(rotate_half(v))))
    assert torch.allclose(out, v, atol=1e-6)


def test_rotate_half_preserves_norm() -> None:
    v = torch.randn(5, 10)
    assert torch.allclose(rotate_half(v).norm(dim=-1), v.norm(dim=-1), atol=1e-6)


# --------------------------------------------------------------------------- #
# get_encoding
# --------------------------------------------------------------------------- #
def test_get_encoding_shape_is_L_by_D() -> None:
    enc = get_encoding(d_model=8, max_seq_len=16)
    assert tuple(enc.shape) == (16, 8)


def test_get_encoding_is_deterministic() -> None:
    a = get_encoding(8, 16)
    b = get_encoding(8, 16)
    assert torch.equal(a, b)


def test_get_encoding_position_zero_is_all_zero() -> None:
    # angle = position * freq, so position 0 yields all zeros.
    enc = get_encoding(8, 16)
    assert torch.equal(enc[0], torch.zeros(8))


def test_get_encoding_follows_inverse_freq_formula() -> None:
    # Row 1 (position == 1) equals the raw frequencies 1/10000^(2i/d), each
    # duplicated because of the `repeat(... r=2)` interleave.
    d_model = 8
    enc = get_encoding(d_model, 16)
    freqs = [1.0 / (10000 ** ((2 * i) / d_model)) for i in range(d_model // 2)]
    expected = torch.tensor([f for f in freqs for _ in range(2)])
    assert torch.allclose(enc[1], expected, atol=1e-7)


def test_get_encoding_scales_linearly_with_position() -> None:
    # angle(pos) = pos * angle(1); pin the linearity used by the cos/sin terms.
    enc = get_encoding(8, 16)
    assert torch.allclose(enc[2], 2.0 * enc[1], atol=1e-6)
    assert torch.allclose(enc[5], 5.0 * enc[1], atol=1e-6)


def test_get_encoding_duplicates_each_frequency() -> None:
    # Columns are paired: column 2j and 2j+1 carry the same angle.
    enc = get_encoding(8, 16)
    assert torch.equal(enc[:, 0::2], enc[:, 1::2])


# --------------------------------------------------------------------------- #
# apply_rotary_emb
# --------------------------------------------------------------------------- #
def test_apply_rotary_emb_preserves_shape() -> None:
    d_model = 8
    enc = get_encoding(d_model, 16)
    x = torch.randn(2, 3, 5, d_model)
    out = apply_rotary_emb(enc[:5], x, seq_dim=-2)
    assert tuple(out.shape) == (2, 3, 5, d_model)


def test_apply_rotary_emb_preserves_per_pair_norm() -> None:
    # RoPE is a rotation within each (x_{2j}, x_{2j+1}) pair, so each pair's
    # 2-norm is invariant.
    d_model = 8
    L = 5
    enc = get_encoding(d_model, 16)
    x = torch.randn(1, 1, L, d_model)
    out = apply_rotary_emb(enc[:L], x, seq_dim=-2)
    x_pairs = x.reshape(L, d_model // 2, 2)
    out_pairs = out.reshape(L, d_model // 2, 2)
    assert torch.allclose(x_pairs.norm(dim=-1), out_pairs.norm(dim=-1), atol=1e-5)


def test_apply_rotary_emb_preserves_total_norm() -> None:
    d_model = 8
    enc = get_encoding(d_model, 16)
    x = torch.randn(2, 3, 5, d_model)
    out = apply_rotary_emb(enc[:5], x, seq_dim=-2)
    assert math.isclose(out.norm().item(), x.norm().item(), rel_tol=1e-5)


def test_apply_rotary_emb_position_zero_is_identity() -> None:
    # cos(0)=1, sin(0)=0 => the first sequence position is unchanged.
    d_model = 8
    enc = get_encoding(d_model, 16)
    x = torch.randn(1, 1, 4, d_model)
    out = apply_rotary_emb(enc[:4], x, seq_dim=-2)
    assert torch.allclose(out[..., 0, :], x[..., 0, :], atol=1e-6)


def test_apply_rotary_emb_start_index_leaves_flanks_untouched() -> None:
    # With start_index=2 and rot_dim=4, only columns [2:6) are rotated.
    rot_dim = 4
    start = 2
    enc = get_encoding(rot_dim, 16)  # (16, 4)
    x = torch.randn(2, 3, 5, 10)
    out = apply_rotary_emb(enc[:5], x, start_index=start, seq_dim=-2)
    assert tuple(out.shape) == (2, 3, 5, 10)
    assert torch.equal(out[..., :start], x[..., :start])  # t_left
    assert torch.equal(out[..., start + rot_dim :], x[..., start + rot_dim :])  # t_right
    # The middle block actually moved.
    assert not torch.allclose(out[..., start : start + rot_dim], x[..., start : start + rot_dim])


def test_apply_rotary_emb_ndim3_slices_trailing_positions() -> None:
    # For 3-D inputs the code re-slices freqs to the trailing seq_len; passing the
    # full encoding must match passing only its last `seq_len` rows.
    d_model = 8
    enc = get_encoding(d_model, 16)
    t = torch.randn(4, 5, d_model)  # ndim == 3, seq_dim=-2 => seq_len 5
    out_full = apply_rotary_emb(enc, t, seq_dim=-2)
    out_sliced = apply_rotary_emb(enc[-5:], t, seq_dim=-2)
    assert tuple(out_full.shape) == (4, 5, d_model)
    assert torch.allclose(out_full, out_sliced, atol=1e-6)


def test_apply_rotary_emb_rejects_too_large_rot_dim() -> None:
    # rot_dim must fit inside the feature dim.
    enc = get_encoding(8, 16)  # rot_dim 8
    x = torch.randn(1, 1, 5, 4)  # feature dim only 4
    with pytest.raises(AssertionError):
        apply_rotary_emb(enc[:5], x, seq_dim=-2)


def test_apply_rotary_emb_golden() -> None:
    # Frozen numeric output for a fixed seed, to catch any silent algebra change.
    torch.manual_seed(0)
    d_model = 8
    enc = get_encoding(d_model, 16)
    x = torch.randn(2, 3, 5, d_model)
    out = apply_rotary_emb(enc[:5], x, seq_dim=-2)
    expected = torch.tensor([-1.12584, -1.15236, -0.250579, -0.433879, 0.84871, 0.692009, -0.316013, -2.11522])
    assert torch.allclose(out[0, 0, 0, :], expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# ROPE.rotate_queries_or_keys
# --------------------------------------------------------------------------- #
def test_rope_preserves_BHLD_shape() -> None:
    d_model = 8
    rope = ROPE(d_model, max_seq_len=16)
    x = torch.randn(2, 3, 5, d_model)
    out = rope.rotate_queries_or_keys(x)
    assert tuple(out.shape) == (2, 3, 5, d_model)


def test_rope_buffer_matches_get_encoding() -> None:
    # The pre-cached buffer is exactly get_encoding(d_model, max_seq_len).
    d_model = 8
    rope = ROPE(d_model, max_seq_len=16)
    # get_buffer is typed Tensor (rope.encoding narrows to Module | Tensor for pyright).
    assert torch.equal(rope.get_buffer("encoding"), get_encoding(d_model, 16))


def test_rope_uses_cached_buffer_within_max_seq_len() -> None:
    # Short sequences slice the cached buffer; result must equal apply_rotary_emb
    # on that slice.
    d_model = 8
    rope = ROPE(d_model, max_seq_len=16)
    x = torch.randn(1, 2, 7, d_model)
    out = rope.rotate_queries_or_keys(x)
    expected = apply_rotary_emb(rope.get_buffer("encoding")[:7], x, seq_dim=-2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rope_recompute_branch_when_seq_exceeds_max() -> None:
    # seq_len > max_seq_len takes the get_encoding(d_model, seq_len) recompute path.
    d_model = 8
    max_seq_len = 16
    seq_len = 20
    rope = ROPE(d_model, max_seq_len=max_seq_len)
    x = torch.randn(1, 2, seq_len, d_model)
    out = rope.rotate_queries_or_keys(x)
    assert tuple(out.shape) == (1, 2, seq_len, d_model)
    # Must match a fresh encoding computed for the longer length.
    fresh = get_encoding(d_model, seq_len).to(x)
    expected = apply_rotary_emb(fresh, x, seq_dim=-2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rope_recompute_branch_golden() -> None:
    torch.manual_seed(0)
    d_model = 8
    rope = ROPE(d_model, max_seq_len=16)
    x = torch.randn(1, 2, 20, d_model)
    out = rope.rotate_queries_or_keys(x)
    expected = torch.tensor([0.133183, 0.042312, 1.310093])
    assert torch.allclose(out[0, 0, 17, :3], expected, atol=1e-5)


def test_rope_rejects_wrong_d_model() -> None:
    rope = ROPE(d_model=8, max_seq_len=16)
    x = torch.randn(1, 1, 5, 6)  # d_model 6 != 8
    with pytest.raises(AssertionError):
        rope.rotate_queries_or_keys(x)
