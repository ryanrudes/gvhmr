"""Characterization tests for :mod:`gvhmr.model.gvhmr.utils.stats_compose`.

``compose(targets, sources)`` is pure list/dict assembly: for each
``(target, source)`` pair it concatenates ``target[source]["mean"]`` and
``target[source]["std"]`` into a flat 151-dim normalization vector. A single
source is broadcast (``sources * len(targets)``) to all targets.

These tests pin the 151-dim layout and a handful of golden boundary values so
the reorg/rename/typing passes can't silently change the assembled stats. No
pytorch3d, no torch, no GPU - just Python list/dict assembly.
"""

from __future__ import annotations

import pytest

from gvhmr.model.gvhmr.utils import stats_compose as SC

# Documented 151-dim layout: the (length, attribute) of each contiguous block,
# in the order compose() concatenates them.
LAYOUT = [
    ("body_pose_r6d", 126),
    ("betas", 10),
    ("global_orient_c_r6d", 6),
    ("global_orient_gv_r6d", 6),
    ("local_transl_vel", 3),
]
TOTAL_DIM = 151

# The standard target ordering used by every MM_* model in the module.
TARGETS = [
    SC.body_pose_r6d,
    SC.betas,
    SC.global_orient_c_r6d,
    SC.global_orient_gv_r6d,
    SC.local_transl_vel,
]


def test_layout_blocks_sum_to_151() -> None:
    assert sum(length for _, length in LAYOUT) == TOTAL_DIM


def test_mm_v1_vector_lengths_are_151() -> None:
    assert len(SC.MM_V1["mean"]) == TOTAL_DIM
    assert len(SC.MM_V1["std"]) == TOTAL_DIM


def test_mm_v2_vector_lengths_are_151() -> None:
    assert len(SC.MM_V2["mean"]) == TOTAL_DIM
    assert len(SC.MM_V2["std"]) == TOTAL_DIM


@pytest.mark.parametrize("name", ["MM_V1", "MM_V2", "MM_V1_AMASS_LOCAL_BEDLAM_CAM", "MM_V2_1"])
def test_all_assembled_models_are_151(name: str) -> None:
    model = getattr(SC, name)
    assert len(model["mean"]) == TOTAL_DIM
    assert len(model["std"]) == TOTAL_DIM


def test_default_01_is_zero_mean_unit_std() -> None:
    assert SC.DEFAULT_01 == {"mean": [0.0], "std": [1.0]}


def test_source_block_lengths_match_layout() -> None:
    # Every source dict provides mean/std vectors of the documented block length.
    for name, length in LAYOUT:
        section = getattr(SC, name)
        # bedlam is present for every block and is the MM_V1 source.
        assert len(section["bedlam"]["mean"]) == length
        assert len(section["bedlam"]["std"]) == length


def test_compose_concatenates_in_target_order() -> None:
    # compose() must lay blocks out in the exact target order, mean and std
    # concatenated independently.
    expected_mean: list[float] = []
    expected_std: list[float] = []
    for section in TARGETS:
        expected_mean.extend(section["bedlam"]["mean"])
        expected_std.extend(section["bedlam"]["std"])
    out = SC.compose(TARGETS, ["bedlam"] * len(TARGETS))
    assert out["mean"] == expected_mean
    assert out["std"] == expected_std


def test_mm_v1_equals_all_bedlam_compose() -> None:
    # MM_V1 is the all-bedlam composition; pin it against a fresh assembly.
    assert SC.compose(TARGETS, ["bedlam"] * 5) == SC.MM_V1


def test_single_source_is_broadcast_over_targets() -> None:
    # A length-1 sources list is repeated len(targets) times.
    single = SC.compose([SC.body_pose_r6d, SC.betas], ["bedlam"])
    repeated = SC.compose([SC.body_pose_r6d, SC.betas], ["bedlam", "bedlam"])
    assert single == repeated


def test_compose_returns_new_lists_each_call() -> None:
    # Each call assembles fresh lists, so the result is independent.
    a = SC.compose([SC.betas], ["bedlam"])
    b = SC.compose([SC.betas], ["bedlam"])
    assert a == b
    assert a["mean"] is not b["mean"]
    a["mean"].append(99.0)
    assert len(b["mean"]) == 10


def test_compose_per_source_selection() -> None:
    # Heterogeneous sources select the matching sub-dict per block.
    sources = ["amass", "amass", "bedlam", "bedlam", "amass"]
    out = SC.compose(TARGETS, sources)
    expected_mean: list[float] = []
    for section, src in zip(TARGETS, sources):
        expected_mean.extend(section[src]["mean"])
    assert out["mean"] == expected_mean
    assert out == SC.MM_V1_AMASS_LOCAL_BEDLAM_CAM


def test_mm_v2_transl_vel_block_is_default_none() -> None:
    # MM_V2 uses the "none" transl-vel source: zero mean, unit std in the last 3 dims.
    assert SC.MM_V2["mean"][148:151] == [0.0, 0.0, 0.0]
    assert SC.MM_V2["std"][148:151] == [1.0, 1.0, 1.0]
    assert SC.local_transl_vel["none"] == {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}


def test_mm_v1_golden_block_boundary_means() -> None:
    # Golden values captured from the current code, probing each block boundary.
    mean = SC.MM_V1["mean"]
    assert mean[0] == 0.9772  # body_pose_r6d[0]
    assert mean[125] == 0.102  # body_pose_r6d[-1]
    assert mean[126] == 0.0378  # betas[0]
    assert mean[135] == 0.0936  # betas[-1]
    assert mean[136] == -4.9862e-3  # global_orient_c_r6d[0]
    assert mean[141] == -5.1653e-2  # global_orient_c_r6d[-1]
    assert mean[142] == 3.6018e-4  # global_orient_gv_r6d[0]
    assert mean[147] == 1.0021e-1  # global_orient_gv_r6d[-1]
    assert mean[148:151] == [7.3057e-05, -2.2142e-04, 3.2444e-03]  # local_transl_vel


def test_mm_v1_golden_transl_vel_std() -> None:
    assert SC.MM_V1["std"][148:151] == [0.0065, 0.0091, 0.0114]
