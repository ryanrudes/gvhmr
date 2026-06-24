"""Characterization tests for :mod:`gvhmr.model.common_utils.scheduler`.

Pins :class:`WarmupMultiStepLR.get_lr`: a linear warmup ramp of factor
``(epoch + 1) / (warmup + 1)`` for ``epoch < warmup``, followed by a multistep
decay of ``gamma ** bisect_right(milestones, epoch)`` at/after the warmup. These
golden sequences and invariants guard against numeric drift through the
reorg/rename/typing passes.

Quirk (see ``bugs_or_quirks``): under the installed torch (2.12.x) the real
``WarmupMultiStepLR.__init__`` is broken — it forwards ``verbose="deprecated"``
positionally to ``LRScheduler.__init__``, which no longer accepts it, so the
constructor raises ``TypeError`` before any lr is ever computed. The ``assert
warmup < milestones[0]`` guard runs *before* that ``super().__init__`` call, so
it is still reachable through the real constructor and is tested directly. The
warmup/decay behaviour is exercised via a faithful helper that runs the genuine
base-class initialisation (only the broken ``verbose`` arg is dropped), driving
the real ``get_lr`` and ``step``.
"""

from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import LRScheduler

from gvhmr.model.common_utils.scheduler import WarmupMultiStepLR


def _make_scheduler(
    milestones: list[int],
    warmup: int = 0,
    gamma: float = 0.1,
    lr: float = 1.0,
) -> tuple[torch.optim.Optimizer, WarmupMultiStepLR]:
    """Build a ``WarmupMultiStepLR`` over a single-param CPU optimizer.

    Replicates exactly what ``WarmupMultiStepLR.__init__`` does — set the
    ``milestones``/``warmup``/``gamma`` attributes, run the ``assert``, and run
    the base ``LRScheduler.__init__`` — while skipping only the ``verbose``
    positional that the installed torch's ``super().__init__`` rejects. This
    drives the genuine ``get_lr`` / ``step`` / ``_initial_step`` code paths.
    """
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=lr)
    sched = WarmupMultiStepLR.__new__(WarmupMultiStepLR)
    sched.milestones = milestones
    sched.warmup = warmup
    assert warmup < milestones[0]
    sched.gamma = gamma
    LRScheduler.__init__(sched, optimizer, -1)
    return optimizer, sched


def _run(
    milestones: list[int],
    warmup: int,
    gamma: float,
    n_epochs: int,
    lr: float = 1.0,
) -> list[float]:
    """Return the lr seen at the start of each of ``n_epochs`` epochs."""
    optimizer, sched = _make_scheduler(milestones, warmup, gamma, lr)
    lrs: list[float] = []
    for _ in range(n_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        sched.step()
    return lrs


def test_warmup_then_multistep_golden_sequence() -> None:
    # warmup=3, milestones=[5, 8], gamma=0.1, base lr=1.0.
    # epochs 0,1,2 -> (e+1)/4 = 0.25, 0.5, 0.75 (warmup ramp)
    # epochs 3,4   -> gamma**bisect_right([5,8],e) = 0.1**0 = 1.0
    # epochs 5,6,7 -> 0.1**1 = 0.1
    # epochs 8..   -> 0.1**2 = 0.01
    lrs = _run([5, 8], warmup=3, gamma=0.1, n_epochs=12)
    expected = [0.25, 0.5, 0.75, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01]
    assert lrs == pytest.approx(expected)


def test_no_warmup_golden_sequence() -> None:
    # warmup=0 -> the ramp never applies (epoch < 0 is never true); pure multistep.
    # milestones=[2, 4], gamma=0.5:
    # epochs 0,1 -> 0.5**0 = 1.0 ; epochs 2,3 -> 0.5**1 = 0.5 ; epochs 4.. -> 0.5**2 = 0.25
    lrs = _run([2, 4], warmup=0, gamma=0.5, n_epochs=7)
    assert lrs == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25])


def test_warmup_only_golden_sequence() -> None:
    # warmup=2, milestones=[10], gamma=0.1; first epochs are the ramp 1/3, 2/3, then 1.0.
    lrs = _run([10], warmup=2, gamma=0.1, n_epochs=5)
    expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0]
    assert lrs == pytest.approx(expected)


def test_warmup_ramp_factor_formula() -> None:
    # Invariant: during warmup, lr == base_lr * (epoch + 1) / (warmup + 1).
    warmup, base = 4, 2.5
    lrs = _run([10], warmup=warmup, gamma=0.1, n_epochs=warmup, lr=base)
    expected = [base * (e + 1) / (warmup + 1) for e in range(warmup)]
    assert lrs == pytest.approx(expected)


def test_warmup_ramp_is_strictly_increasing_then_flat_at_base() -> None:
    # The ramp is monotonically increasing and reaches exactly base lr at e == warmup.
    warmup, base = 5, 3.0
    lrs = _run([20], warmup=warmup, gamma=0.1, n_epochs=warmup + 1, lr=base)
    ramp = lrs[:warmup]
    assert all(ramp[i] < ramp[i + 1] for i in range(len(ramp) - 1))
    assert lrs[warmup] == pytest.approx(base)


def test_multistep_decay_uses_bisect_right_semantics() -> None:
    # Invariant: at/after warmup, factor == gamma ** bisect_right(milestones, epoch).
    # bisect_right is right-inclusive: the decay applies *at* the milestone epoch.
    from bisect import bisect_right

    milestones, gamma, base = [5, 8], 0.1, 1.0
    lrs = _run(milestones, warmup=0, gamma=gamma, n_epochs=10, lr=base)
    for epoch, lr in enumerate(lrs):
        assert lr == pytest.approx(base * gamma ** bisect_right(milestones, epoch))


def test_decay_applies_exactly_at_milestone_epoch() -> None:
    # Crossing a single milestone: lr drops by exactly gamma at epoch == milestone.
    lrs = _run([3], warmup=0, gamma=0.25, n_epochs=5)
    assert lrs == pytest.approx([1.0, 1.0, 1.0, 0.25, 0.25])


def test_base_lr_scales_the_whole_schedule_linearly() -> None:
    # get_lr is linear in base lr: doubling the optimizer lr doubles every step.
    unit = _run([4, 7], warmup=2, gamma=0.3, n_epochs=10, lr=1.0)
    scaled = _run([4, 7], warmup=2, gamma=0.3, n_epochs=10, lr=2.0)
    assert scaled == pytest.approx([2.0 * v for v in unit])


def test_get_lr_returns_one_value_per_param_group() -> None:
    _, sched = _make_scheduler([5, 8], warmup=3, gamma=0.1)
    out = sched.get_lr()
    assert isinstance(out, list)
    assert len(out) == 1
    # last_epoch starts at 0 after _initial_step, so this is the second warmup step.
    assert out[0] == pytest.approx(0.25)


def test_warmup_must_be_less_than_first_milestone_guard() -> None:
    # The real constructor's `assert warmup < milestones[0]` runs before the
    # (broken) super().__init__, so it is reachable directly.
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    with pytest.raises(AssertionError):
        WarmupMultiStepLR(optimizer, milestones=[3], warmup=3, gamma=0.1)
    with pytest.raises(AssertionError):
        WarmupMultiStepLR(optimizer, milestones=[3], warmup=5, gamma=0.1)


def test_constructor_succeeds_under_installed_torch() -> None:
    # Regression guard: __init__ must not forward the removed `verbose` arg to
    # LRScheduler.__init__ (that raised TypeError on torch >= 2.x).
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    sched = WarmupMultiStepLR(optimizer, milestones=[5, 8], warmup=3, gamma=0.1)
    assert sched.last_epoch == 0
