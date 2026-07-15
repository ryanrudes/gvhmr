"""Crops-serving dataset faithfulness (ROADMAP Regime B stage 3). Needs GPU + HMR2 ckpt + 3DPW pack.

The make-or-break correctness check for joint training: the crops the dataset decodes on-the-fly must
reproduce the crops the extractor fed the ViT — otherwise the LoRA-0 joint model does not start where the
cached-feature model is, and the whole "learn the backbone from the same starting point" premise breaks.
We verify JointBackbone(dataset crops) ≈ the cached f_imgseq (bf16 cache vs fp32 forward → cosine, not eq).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.checkpoint, pytest.mark.dataset]


def test_dataset_crops_reproduce_cached_hmr2_features():
    from gvhmr.dataset.threedpw.threedpw_motion_train import ThreedpwSmplDataset
    from gvhmr.network.joint_backbone import JointBackbone
    from gvhmr.utils.assets import DATA_ROOT
    from gvhmr.utils.device import get_device

    if not (DATA_ROOT / "3DPW/hmr4d_support/train_refit_smplx.pt").exists():
        pytest.skip("3DPW train pack not present")

    ds = ThreedpwSmplDataset(serve_crops=True)  # default HMR2 (smplx_refit) cached features
    np.random.seed(0)  # the clip subset is sampled with np.random — pin it
    item = ds[0]
    length = int(item["length"])
    assert "crops" in item and item["crops"].shape[1:] == (3, 256, 256)
    crops = item["crops"][:length]  # (L,3,256,256) — drop the max_len padding
    cached = item["f_imgseq"][:length]  # (L,1024) cached HMR2 token
    assert cached.shape[-1] == 1024

    dev = get_device()
    joint = JointBackbone(rank=8, alpha=16.0).to(dev).eval()
    with torch.no_grad():
        feats = joint(crops.to(dev)).float().cpu()

    cos = F.cosine_similarity(feats, cached, dim=-1)
    # bf16-cached vs fp32 re-forward on the SAME crops → very high cosine if the crop path is faithful;
    # a frame misalignment or scale/bbx bug would tank the min.
    assert cos.mean() > 0.99, f"mean cosine {cos.mean():.4f} — crop path does not match the cached extractor"
    assert cos.min() > 0.95, f"min cosine {cos.min():.4f} — some frames misaligned"
