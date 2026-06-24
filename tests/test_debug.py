"""Tests for the research-debugging helpers in :mod:`gvhmr.utils.debug`."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from gvhmr.utils.debug import (
    LATENT_DIM,
    LATENT_LAYOUT,
    count_parameters,
    decompose_latent,
    describe,
    nan_hooks,
    remove_hooks,
)


def test_latent_layout_is_contiguous_and_sums_to_151() -> None:
    # The layout must tile [0, 151) with no gaps or overlaps (the decode contract).
    cursor = 0
    for start, end, _name in LATENT_LAYOUT:
        assert start == cursor, "non-contiguous latent layout"
        assert end > start
        cursor = end
    assert cursor == LATENT_DIM == 151


def test_decompose_latent_splits_by_name() -> None:
    x = torch.randn(2, 7, 151)
    parts = decompose_latent(x)
    assert set(parts) == {n for _, _, n in LATENT_LAYOUT}
    assert parts["body_pose_r6d"].shape == (2, 7, 126)
    assert parts["betas"].shape == (2, 7, 10)
    assert parts["local_transl_vel"].shape == (2, 7, 3)
    # Concatenating the parts back recovers the original exactly.
    recombined = torch.cat([parts[n] for _, _, n in LATENT_LAYOUT], dim=-1)
    assert torch.equal(recombined, x)


def test_decompose_latent_rejects_wrong_dim() -> None:
    import pytest

    with pytest.raises(ValueError):
        decompose_latent(torch.randn(4, 100))


def test_describe_reports_stats_and_nans() -> None:
    x = torch.tensor([1.0, 2.0, float("nan"), float("inf")])
    stats = describe(x, name="x", print_it=False)
    assert stats.shape == (4,)
    assert stats.n_nan == 1
    assert stats.n_inf == 1
    assert "NaN" in str(stats)


def test_count_parameters() -> None:
    model = nn.Linear(3, 4)  # 3*4 weights + 4 bias = 16
    assert count_parameters(model) == 16
    for p in model.parameters():
        p.requires_grad_(False)
    assert count_parameters(model, trainable_only=True) == 0


def test_nan_hooks_flag_non_finite_output() -> None:
    class Bad(nn.Module):
        def forward(self, x):
            return x * float("nan")

    model = nn.Sequential(Bad())
    handles = nan_hooks(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(torch.randn(2, 2))
    remove_hooks(handles)
    assert any("non-finite" in str(w.message) for w in caught)
