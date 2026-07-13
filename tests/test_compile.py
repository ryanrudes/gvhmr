"""``torch.compile`` on the denoiser: faithful math, and — critically — untouched checkpoint keys.

The denoiser is launch-overhead-bound (a small model, many tiny sequential kernels, ~7% MFU), so
fusion is a real win: 1.9x on fwd+bwd, measured on an RTX 6000 Ada. But it is opt-in and guarded,
because this repo has already shipped one "free" speed win that silently wrecked a benchmark (TF32
— see docs/PERFORMANCE.md). The two things that could go wrong are pinned here.
"""

from __future__ import annotations

import torch

from gvhmr.utils.net_utils import compile_forward


class _Tiny(torch.nn.Module):
    """Stands in for the denoiser: a couple of matmuls + a norm is enough to exercise the wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.norm = torch.nn.LayerNorm(32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.norm(self.fc1(x)), approximate="tanh"))


def test_compile_forward_keeps_state_dict_keys() -> None:
    """The whole point of compiling ``forward`` rather than the module.

    ``torch.compile(module)`` hands back an OptimizedModule whose state_dict keys all gain an
    ``_orig_mod.`` prefix. GVHMR loads checkpoints **strict** (AGENTS.md landmine #2), so that would
    break every checkpoint a compiled run saves — silently, and only when someone later tries to
    ``gvhmr eval`` it. Compiling the bound forward must leave the keys alone.
    """
    net = _Tiny()
    before = list(net.state_dict().keys())

    compile_forward(net)

    assert list(net.state_dict().keys()) == before, "compile must not rename/prefix checkpoint keys"
    assert not any(k.startswith("_orig_mod") for k in net.state_dict()), "OptimizedModule prefix leaked in"
    # and the compiled module still round-trips through a strict load
    net2 = _Tiny()
    net2.load_state_dict(net.state_dict(), strict=True)


def test_compile_forward_is_numerically_faithful() -> None:
    """Compiled == eager to fp32 rounding. Guard against fusion silently changing the math.

    Note dropout must be off (``eval()``) or the module is not even deterministic against *itself* —
    which is exactly the trap that makes a naive eager-vs-compiled diff look like a compile bug.
    """
    torch.manual_seed(0)
    net = _Tiny().eval()
    x = torch.randn(4, 7, 16)

    with torch.no_grad():
        eager = net(x).clone()
        # sanity: the baseline is deterministic, so any delta below is attributable to compile
        assert torch.equal(eager, net(x)), "eager is non-deterministic — the comparison would be meaningless"

        compile_forward(net)
        compiled = net(x)

    assert torch.allclose(eager, compiled, atol=1e-5, rtol=1e-5)
