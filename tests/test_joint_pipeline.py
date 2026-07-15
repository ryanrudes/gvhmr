"""The Pipeline `_img_features` opt-in branch (ROADMAP Regime B stage 3). CPU-only, no weights/GPU.

Pins the behaviour-preserving switch: with no joint backbone the cached ``f_imgseq`` is returned
*unchanged* (identity — the default path is byte-identical); with one, crops are flattened through it and
reshaped back to (B, L, 1024).
"""

from __future__ import annotations

import torch

from gvhmr.model.gvhmr.pipeline.gvhmr_pipeline import Pipeline


def _bare_pipeline():
    """A Pipeline instance without running the heavy __init__ (we only exercise _img_features)."""
    return Pipeline.__new__(Pipeline)


def test_default_returns_cached_feature_unchanged():
    p = _bare_pipeline()
    p.joint_backbone = None
    inp = {"f_imgseq": torch.randn(2, 5, 1024)}
    out = p._img_features(inp)
    assert out is inp["f_imgseq"]  # identity — not a copy, not recomputed


def test_no_crops_falls_back_even_when_backbone_present():
    p = _bare_pipeline()
    p.joint_backbone = object()  # present but unused: no crops in the batch (e.g. an AMASS-only batch)
    inp = {"f_imgseq": torch.randn(2, 5, 1024)}
    assert p._img_features(inp) is inp["f_imgseq"]


def test_crops_are_flattened_through_backbone_and_reshaped():
    p = _bare_pipeline()

    class _Stub:
        """(N,3,256,256) -> (N,1024); marks each row with its flat index so we can check the reshape."""

        def __call__(self, crops):
            n = crops.shape[0]
            return torch.arange(n, dtype=torch.float32)[:, None].expand(n, 1024).contiguous()

    p.joint_backbone = _Stub()
    inp = {"f_imgseq": torch.zeros(2, 3, 1024), "crops": torch.randn(2, 3, 3, 256, 256)}
    out = p._img_features(inp)
    assert out.shape == (2, 3, 1024)
    # flat index (B*L order) reshaped to (B,L): row [b,l] == b*3 + l
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 1 and out[1, 0, 0] == 3 and out[1, 2, 0] == 5
