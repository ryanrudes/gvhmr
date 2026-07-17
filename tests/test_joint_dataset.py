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


def test_bedlam_dataset_crops_reproduce_cached_hmr2_features():
    """BEDLAM's crop contract differs from 3DPW's in two ways that both fail SILENTLY.

    BEDLAM's cache is fp32 and full-res, so a faithful crop path reproduces it *exactly* (cosine 1.00000).
    That is why this asserts 0.9999 rather than 3DPW's 0.99: decoding at 3DPW's img_ds=0.5 still scores
    ~0.9991 here — it would pass a loose threshold while feeding the ViT the wrong pixels, and the joint
    arm would then lose the A/B for a plumbing reason indistinguishable from a real negative result.
    """
    from gvhmr.dataset.bedlam.bedlam import BedlamDatasetV2
    from gvhmr.network.joint_backbone import JointBackbone
    from gvhmr.utils.assets import DATA_ROOT
    from gvhmr.utils.device import get_device

    if not (DATA_ROOT / "BEDLAM/hmr4d_support/videos").is_dir():
        pytest.skip("BEDLAM videos not present")

    ds = BedlamDatasetV2(mid_indices=["maxspan60"], serve_crops=True)  # maxspan60: ~14% have start_end[0]>0
    np.random.seed(0)  # the clip subset is sampled with np.random — pin it
    item = ds[0]
    length = int(item["length"])
    assert "crops" in item and item["crops"].shape[1:] == (3, 256, 256)
    crops = item["crops"][:length]
    cached = item["f_imgseq"][:length]
    assert cached.shape[-1] == 1024

    dev = get_device()
    joint = JointBackbone(rank=8, alpha=16.0).to(dev).eval()
    with torch.no_grad():
        feats = joint(crops.to(dev)).float().cpu()

    cos = F.cosine_similarity(feats, cached, dim=-1)
    assert cos.mean() > 0.9999, f"mean cosine {cos.mean():.5f} — wrong img_ds, or the frame offset is off"
    assert cos.min() > 0.999, f"min cosine {cos.min():.5f} — some frames misaligned"
