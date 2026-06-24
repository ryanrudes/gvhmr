"""Characterization tests for :mod:`gvhmr.utils.net_utils`.

These pin the pure-CPU helpers — mask building, state-dict prefix filtering,
last-frame repetition, gaussian / moving-average smoothing, and last-checkpoint
discovery — so the rename/reorg/typing passes cannot change numerics or control
flow. GPU movers (``to_cuda``) and ``torch.load`` paths are intentionally not
exercised here. Golden values were captured from the current implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage._filters import _gaussian_kernel1d

from gvhmr.utils import net_utils as N

# ---------------------------------------------------------------------------
# length_to_mask / get_valid_mask
# ---------------------------------------------------------------------------


def test_length_to_mask_golden() -> None:
    lengths = torch.tensor([1, 3, 0, 4])
    mask = N.length_to_mask(lengths, 4)
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_length_to_mask_row_sums_equal_lengths() -> None:
    # Invariant: each row has exactly `length` True entries (clamped to max_len).
    lengths = torch.tensor([0, 2, 5, 7])
    max_len = 6
    mask = N.length_to_mask(lengths, max_len)
    expected_counts = torch.clamp(lengths, max=max_len)
    assert torch.equal(mask.sum(dim=1), expected_counts)


def test_length_to_mask_is_prefix_pattern() -> None:
    # Invariant: every True region is a contiguous prefix (no holes).
    lengths = torch.tensor([3, 1, 4])
    mask = N.length_to_mask(lengths, 5)
    # A prefix mask is monotonically non-increasing along the row.
    assert (mask[:, :-1].int() >= mask[:, 1:].int()).all()


def test_get_valid_mask_matches_length_to_mask_row() -> None:
    max_len, valid = 8, 5
    row = N.get_valid_mask(max_len, valid)
    assert row.shape == (max_len,)
    assert row.dtype == torch.bool
    assert int(row.sum()) == valid
    # Equivalent to a single row of length_to_mask.
    ref = N.length_to_mask(torch.tensor([valid]), max_len)[0]
    assert torch.equal(row, ref)


# ---------------------------------------------------------------------------
# select_state_dict_by_prefix
# ---------------------------------------------------------------------------


def test_select_state_dict_by_prefix_strips_prefix() -> None:
    sd = {"enc.a": 1, "enc.b": 2, "dec.c": 3}
    out = N.select_state_dict_by_prefix(sd, "enc.")
    assert out == {"a": 1, "b": 2}


def test_select_state_dict_by_prefix_rewrites_prefix() -> None:
    sd = {"enc.a": 1, "enc.b": 2, "dec.c": 3}
    out = N.select_state_dict_by_prefix(sd, "enc.", "E.")
    assert out == {"E.a": 1, "E.b": 2}


def test_select_state_dict_by_prefix_no_match_is_empty() -> None:
    sd = {"enc.a": 1}
    assert N.select_state_dict_by_prefix(sd, "missing.") == {}


def test_select_state_dict_by_prefix_preserves_values_identity() -> None:
    # Values are passed through by reference, not copied.
    t = torch.randn(3)
    sd = {"model.w": t}
    out = N.select_state_dict_by_prefix(sd, "model.")
    assert out["w"] is t


# ---------------------------------------------------------------------------
# repeat_to_max_len
# ---------------------------------------------------------------------------


def test_repeat_to_max_len_repeats_last_frame() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = N.repeat_to_max_len(x, 4, dim=0)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    assert torch.equal(out, expected)


def test_repeat_to_max_len_equal_returns_same_object() -> None:
    x = torch.randn(3, 2)
    assert N.repeat_to_max_len(x, 3, dim=0) is x


def test_repeat_to_max_len_along_dim1() -> None:
    y = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    out = N.repeat_to_max_len(y, 5, dim=1)
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 3.0, 3.0]]))


def test_repeat_to_max_len_preserves_prefix() -> None:
    # Invariant: the original frames are untouched; only the tail is duplicated.
    x = torch.randn(4, 3, 2)
    out = N.repeat_to_max_len(x, 9, dim=0)
    assert out.shape == (9, 3, 2)
    assert torch.equal(out[:4], x)
    assert torch.equal(out[4:], x[-1:].expand(5, 3, 2))


def test_repeat_to_max_len_too_long_raises() -> None:
    with pytest.raises(ValueError):
        N.repeat_to_max_len(torch.randn(5, 2), 3, dim=0)


def test_repeat_to_max_len_dict_applies_per_value() -> None:
    d = {"a": torch.tensor([[1.0]]), "b": torch.tensor([[2.0], [3.0]])}
    out = N.repeat_to_max_len_dict(d, 3, dim=0)
    assert torch.equal(out["a"], torch.tensor([[1.0], [1.0], [1.0]]))
    assert torch.equal(out["b"], torch.tensor([[2.0], [3.0], [3.0]]))


# ---------------------------------------------------------------------------
# gaussian_smooth / moving_average_smooth
# ---------------------------------------------------------------------------


def test_gaussian_kernel_is_normalized_and_symmetric() -> None:
    # The smoothing relies on a unit-sum, mirror-symmetric kernel.
    k = _gaussian_kernel1d(sigma=3, order=0, radius=int(4 * 3 + 0.5))
    assert np.isclose(k.sum(), 1.0, atol=1e-6)
    assert np.allclose(k, k[::-1], atol=1e-12)


def test_gaussian_smooth_constant_signal_stays_constant() -> None:
    # Unit-sum kernel + replicate padding => constant in, constant out.
    x = torch.full((1, 20), 5.0)
    out = N.gaussian_smooth(x, sigma=3, dim=-1)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)


def test_gaussian_smooth_preserves_shape() -> None:
    x = torch.randn(2, 11)
    assert N.gaussian_smooth(x, sigma=2.0).shape == x.shape


def test_gaussian_smooth_along_dim0() -> None:
    x = torch.full((7, 3), 2.0)
    out = N.gaussian_smooth(x, sigma=2.0, dim=0)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)


def test_gaussian_smooth_golden() -> None:
    sig = torch.arange(8, dtype=torch.float32)[None]
    out = N.gaussian_smooth(sig, sigma=1.0).flatten()
    expected = torch.tensor(
        [
            0.3637846112251282,
            1.0632563829421997,
            2.00469970703125,
            3.000133752822876,
            3.999866247177124,
            4.995300769805908,
            5.93674373626709,
            6.6362152099609375,
        ]
    )
    assert torch.allclose(out, expected, atol=1e-5)


def test_moving_average_smooth_constant_signal_stays_constant() -> None:
    x = torch.full((1, 20), 5.0)
    out = N.moving_average_smooth(x, window_size=5, dim=-1)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6)


def test_moving_average_smooth_interior_is_local_mean() -> None:
    # Away from the replicate-padded edges, each output is the centered mean.
    sig = torch.arange(8, dtype=torch.float32)[None]
    out = N.moving_average_smooth(sig, window_size=3).flatten()
    # Interior indices 1..6 are exact local means of a linear ramp.
    assert torch.allclose(out[1:7], sig.flatten()[1:7], atol=1e-6)


def test_moving_average_smooth_golden() -> None:
    sig = torch.arange(8, dtype=torch.float32)[None]
    out = N.moving_average_smooth(sig, window_size=3).flatten()
    expected = torch.tensor(
        [
            0.3333333432674408,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            6.6666669845581055,
        ]
    )
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# find_last_ckpt_path / get_resume_ckpt_path  (pure path logic, no torch.load)
# ---------------------------------------------------------------------------


def test_find_last_ckpt_path_empty_dir_returns_none(tmp_path: Path) -> None:
    assert N.find_last_ckpt_path(tmp_path) is None


def test_find_last_ckpt_path_picks_sorted_last(tmp_path: Path) -> None:
    for name in ("e01.ckpt", "e02.ckpt", "e10.ckpt"):
        (tmp_path / name).write_text("")
    out = N.find_last_ckpt_path(tmp_path)
    # Equal-width names sort lexicographically == numerically here.
    assert out.name == "e10.ckpt"


def test_find_last_ckpt_path_prioritizes_last_ckpt(tmp_path: Path) -> None:
    (tmp_path / "e10.ckpt").write_text("")
    (tmp_path / "last.ckpt").write_text("")
    out = N.find_last_ckpt_path(tmp_path)
    assert out.name == "last.ckpt"


def test_find_last_ckpt_path_skips_last_named_non_priority(tmp_path: Path) -> None:
    # A file that merely contains "last" (but isn't exactly last.ckpt) is skipped.
    (tmp_path / "epoch_last_v1.ckpt").write_text("")
    assert N.find_last_ckpt_path(tmp_path) is None


def test_get_resume_ckpt_path_returns_existing_path(tmp_path: Path) -> None:
    f = tmp_path / "foo.ckpt"
    f.write_text("")
    assert N.get_resume_ckpt_path(str(f)) == str(f)


def test_get_resume_ckpt_path_last_delegates_to_finder(tmp_path: Path) -> None:
    (tmp_path / "last.ckpt").write_text("")
    out = N.get_resume_ckpt_path("last", ckpt_dir=tmp_path)
    assert out == tmp_path / "last.ckpt"
