"""Characterization tests for :mod:`gvhmr.utils.seq_utils`.

These pin the mask <-> frame-id machinery, the linear interpolation helpers and
``find_top_k_span`` so the rename/reorg/typing passes can't silently change
numerics. The functions covered are pure (CPU torch/numpy, no GPU/pytorch3d):

    get_frame_id_list_from_mask / mask_to_frame_id / frame_id_to_mask
    get_frame_id_list_from_frame_id
    rearrange_by_mask
    linear_interpolate / linear_interpolate_frame_ids
    find_top_k_span

We prefer round-trip / invariant assertions, with a handful of golden values
captured by running the current code.
"""

from __future__ import annotations

import numpy as np
import torch

from gvhmr.utils.seq_utils import (
    find_top_k_span,
    frame_id_to_mask,
    get_frame_id_list_from_frame_id,
    get_frame_id_list_from_mask,
    linear_interpolate,
    linear_interpolate_frame_ids,
    mask_to_frame_id,
    rearrange_by_mask,
)

# --------------------------------------------------------------------------- #
# get_frame_id_list_from_mask                                                 #
# --------------------------------------------------------------------------- #


def test_get_frame_id_list_from_mask_golden() -> None:
    mask = torch.tensor([False, True, True, False, True, False, False, True, True, True])
    spans = get_frame_id_list_from_mask(mask)
    assert [t.tolist() for t in spans] == [[1, 2], [4], [7, 8, 9]]


def test_get_frame_id_list_from_mask_empty_returns_empty_list() -> None:
    assert get_frame_id_list_from_mask(torch.zeros(5, dtype=torch.bool)) == []


def test_get_frame_id_list_from_mask_all_true_single_span() -> None:
    spans = get_frame_id_list_from_mask(torch.ones(4, dtype=torch.bool))
    assert [t.tolist() for t in spans] == [[0, 1, 2, 3]]


def test_get_frame_id_list_from_mask_single_span_branch() -> None:
    # Exercises the starts.numel() == 1 reshape branch.
    mask = torch.tensor([False, True, True, True, False])
    spans = get_frame_id_list_from_mask(mask)
    assert [t.tolist() for t in spans] == [[1, 2, 3]]


def test_get_frame_id_list_from_mask_single_true_frame() -> None:
    # A length-1 span at the boundaries should still come back as a list of one tensor.
    assert [t.tolist() for t in get_frame_id_list_from_mask(torch.tensor([True, False, False]))] == [[0]]
    assert [t.tolist() for t in get_frame_id_list_from_mask(torch.tensor([False, False, True]))] == [[2]]


def test_get_frame_id_list_from_mask_partitions_true_indices() -> None:
    # Invariant: the union of all spans is exactly the set of True indices, in order,
    # and every span is a contiguous run.
    torch.manual_seed(1)
    mask = torch.rand(50) > 0.5
    spans = get_frame_id_list_from_mask(mask)
    recovered = torch.cat(spans) if spans else torch.empty(0, dtype=torch.long)
    assert torch.equal(recovered.long(), torch.where(mask)[0])
    for span in spans:
        assert torch.equal(span, torch.arange(span[0], span[-1] + 1))
        assert bool(mask[span].all())


# --------------------------------------------------------------------------- #
# frame_id_to_mask / mask_to_frame_id round-trip                              #
# --------------------------------------------------------------------------- #


def test_frame_id_to_mask_golden() -> None:
    mask = frame_id_to_mask(torch.tensor([0, 2, 4]), 6)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, True, False, True, False]


def test_frame_id_to_mask_then_mask_to_frame_id_roundtrips() -> None:
    frame_id = torch.tensor([0, 2, 4])
    mask = frame_id_to_mask(frame_id, 6)
    assert torch.equal(mask_to_frame_id(mask), frame_id)


def test_mask_to_frame_id_then_frame_id_to_mask_roundtrips() -> None:
    torch.manual_seed(2)
    mask = torch.rand(20) > 0.4
    frame_id = mask_to_frame_id(mask)
    assert torch.equal(frame_id_to_mask(frame_id, mask.numel()), mask)


def test_get_frame_id_list_from_frame_id_golden() -> None:
    spans = get_frame_id_list_from_frame_id(torch.tensor([1, 2, 5, 6, 7]))
    assert [t.tolist() for t in spans] == [[1, 2], [5, 6, 7]]


# --------------------------------------------------------------------------- #
# rearrange_by_mask                                                           #
# --------------------------------------------------------------------------- #


def test_rearrange_by_mask_scatters_rows_into_true_positions() -> None:
    x = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    mask = torch.tensor([False, True, False, True, False])
    out = rearrange_by_mask(x, mask)
    assert out.tolist() == [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]


def test_rearrange_by_mask_identity_when_full() -> None:
    # M == L short-circuits and returns x itself.
    x = torch.arange(6).reshape(3, 2).float()
    mask = torch.ones(3, dtype=torch.bool)
    out = rearrange_by_mask(x, mask)
    assert out is x


def test_rearrange_by_mask_preserves_rows_and_fills_zeros() -> None:
    torch.manual_seed(3)
    mask = torch.tensor([True, False, True, True, False, True])
    x = torch.randn(int(mask.sum()), 4)
    out = rearrange_by_mask(x, mask)
    assert out.shape == (mask.numel(), 4)
    assert torch.equal(out[mask], x)
    assert torch.equal(out[~mask], torch.zeros(int((~mask).sum()), 4))
    assert out.dtype == x.dtype


# --------------------------------------------------------------------------- #
# linear_interpolate                                                          #
# --------------------------------------------------------------------------- #


def test_linear_interpolate_golden() -> None:
    data = torch.tensor([[0.0, 0.0], [10.0, 20.0]])
    out = linear_interpolate(data, 3)
    expected = torch.tensor([[0.0, 0.0], [2.5, 5.0], [5.0, 10.0], [7.5, 15.0], [10.0, 20.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_linear_interpolate_endpoints_and_shape() -> None:
    data = torch.randn(2, 5)
    n_middle = 7
    out = linear_interpolate(data, n_middle)
    assert out.shape == (n_middle + 2, 5)
    # Endpoints are preserved exactly.
    assert torch.allclose(out[0], data[0], atol=1e-6)
    assert torch.allclose(out[-1], data[1], atol=1e-6)


def test_linear_interpolate_is_equispaced_on_the_segment() -> None:
    data = torch.randn(2, 3)
    out = linear_interpolate(data, 4)
    # Consecutive differences are constant for a straight-line interpolation.
    diffs = out[1:] - out[:-1]
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-5)


# --------------------------------------------------------------------------- #
# linear_interpolate_frame_ids                                                #
# --------------------------------------------------------------------------- #


def test_linear_interpolate_frame_ids_middle_span_golden() -> None:
    # Endpoints 10.0 (frame 1) and 60.0 (frame 5) bracket the invalid span [2,3,4].
    data = torch.tensor([0.0, 10.0, 99.0, 99.0, 99.0, 60.0, 70.0]).reshape(7, 1)
    out = linear_interpolate_frame_ids(data, [torch.tensor([2, 3, 4])])
    assert out.squeeze(-1).tolist() == [0.0, 10.0, 22.5, 35.0, 47.5, 60.0, 70.0]


def test_linear_interpolate_frame_ids_does_not_mutate_input() -> None:
    data = torch.tensor([0.0, 10.0, 99.0, 99.0, 99.0, 60.0, 70.0]).reshape(7, 1)
    before = data.clone()
    linear_interpolate_frame_ids(data, [torch.tensor([2, 3, 4])])
    assert torch.equal(data, before)


def test_linear_interpolate_frame_ids_leading_span_uses_next_value() -> None:
    data = torch.arange(10).float().reshape(10, 1)
    out = linear_interpolate_frame_ids(data, [torch.tensor([0, 1])])
    # A span at the start is filled with the first valid value (frame 2 == 2.0).
    assert out.squeeze(-1).tolist() == [2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_linear_interpolate_frame_ids_trailing_span_uses_prev_value() -> None:
    data = torch.arange(10).float().reshape(10, 1)
    out = linear_interpolate_frame_ids(data, [torch.tensor([8, 9])])
    # A span at the end is filled with the last valid value (frame 7 == 7.0).
    assert out.squeeze(-1).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0]


def test_linear_interpolate_frame_ids_matches_mask_workflow() -> None:
    # End-to-end: a mask of valid frames -> invalid spans -> interpolation.
    data = torch.tensor([0.0, 5.0, 100.0, 100.0, 25.0, 30.0]).reshape(6, 1)
    valid = torch.tensor([True, True, False, False, True, True])
    invalid_spans = get_frame_id_list_from_mask(~valid)
    out = linear_interpolate_frame_ids(data, invalid_spans)
    # Bracketed by 5.0 (frame 1) and 25.0 (frame 4): two interior points 11.666..., 18.333...
    expected = [0.0, 5.0, 5.0 + 20.0 / 3.0, 5.0 + 40.0 / 3.0, 25.0, 30.0]
    assert torch.allclose(out.squeeze(-1), torch.tensor(expected), atol=1e-5)


# --------------------------------------------------------------------------- #
# find_top_k_span                                                             #
# --------------------------------------------------------------------------- #


def test_find_top_k_span_golden_torch() -> None:
    mask = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=torch.bool)
    assert find_top_k_span(mask, k=3) == [(3, 6), (0, 2), (9, 11)]


def test_find_top_k_span_returns_all_when_k_large() -> None:
    mask = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=torch.bool)
    assert find_top_k_span(mask, k=10) == [(3, 6), (0, 2), (9, 11), (7, 8)]


def test_find_top_k_span_empty_mask() -> None:
    assert find_top_k_span(torch.zeros(5, dtype=torch.bool)) == []


def test_find_top_k_span_accepts_numpy() -> None:
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    assert find_top_k_span(mask, k=3) == [(3, 6), (0, 2), (9, 11)]


def test_find_top_k_span_spans_are_half_open_and_sorted_by_length() -> None:
    torch.manual_seed(4)
    mask = torch.rand(40) > 0.5
    spans = find_top_k_span(mask, k=3)
    lengths = [end - start for start, end in spans]
    # Sorted by descending length.
    assert lengths == sorted(lengths, reverse=True)
    # [start, end) is half-open and fully inside the mask, every covered frame True.
    for start, end in spans:
        assert 0 <= start < end <= mask.numel()
        assert bool(mask[start:end].all())
        # Boundaries are real edges: frame before start / at end is False or OOB.
        assert start == 0 or not bool(mask[start - 1])
        assert end == mask.numel() or not bool(mask[end])


def test_find_top_k_span_default_k_is_three() -> None:
    # All-ones mask with a single span: only one span regardless of default k=3.
    assert find_top_k_span(torch.ones(7, dtype=torch.bool)) == [(0, 7)]
