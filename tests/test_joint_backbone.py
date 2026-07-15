"""In-loop LoRA backbone (ROADMAP Regime B stage 2). Needs GPU + the HMR2 checkpoint → auto-skips in CI.

The go/no-go for joint backbone training:
  1. FAITHFULNESS — a freshly-built JointBackbone (LoRA zero-init) reproduces the frozen HMR2 token
     bit-for-bit, so enabling the joint path on a fresh model changes nothing until the adapters train.
  2. MECHANICS — a few optimizer steps reduce a real loss, gradients reach the adapters, and the ~600 M
     base stays frozen (no grad). If a task-trained backbone can't even move a tiny overfit, we stop
     before spending any cluster time.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.checkpoint]


@pytest.fixture(scope="module")
def crops():
    """A handful of real preproc-normalized 3DPW crops (also exercises the get_batch crop path)."""
    from gvhmr.utils.assets import DATA_ROOT

    feat_pt = DATA_ROOT / "3DPW/hmr4d_support/imgfeats/3dpw_test/downtown_arguing_00_0.pt"
    video = DATA_ROOT / "3DPW/hmr4d_support/videos/downtown_arguing_00.mp4"
    if not feat_pt.exists() or not video.exists():
        pytest.skip("3DPW pack (features + raw video) not present")
    from gvhmr.utils.preproc.vitfeat_extractor import get_batch

    cached = torch.load(feat_pt, map_location="cpu", weights_only=False)
    bbx_xys = cached["bbx_xys"][:6].float()
    imgs, _ = get_batch(str(video), bbx_xys, img_ds=0.5)  # (6,3,256,256)
    return imgs.float()


def test_lora_zero_init_matches_frozen_hmr2(crops):
    """JointBackbone with zero-init adapters == the frozen HMR2 the cached features were made with."""
    from gvhmr.network.hmr2 import load_hmr2
    from gvhmr.network.joint_backbone import JointBackbone
    from gvhmr.utils.device import get_device
    from gvhmr.utils.net_utils import skip_torch_init

    dev = get_device()
    x = crops.to(dev)

    with skip_torch_init():
        frozen = load_hmr2().to(dev).eval()
    with torch.no_grad():
        ref = frozen({"img": x}).float()

    joint = JointBackbone(rank=8, alpha=16.0).to(dev).eval()
    with torch.no_grad():
        mine = joint(x).float()

    assert mine.shape == ref.shape == (6, 1024)
    # zero-init delta is exactly zero and it's the same HMR2 forward → bit-identical (allow fp accumulation)
    assert torch.allclose(mine, ref, atol=1e-4, rtol=1e-4), f"max|Δ|={(mine - ref).abs().max().item():.2e}"


def test_only_adapters_train_and_loss_drops(crops):
    """A few steps of overfitting: loss falls, adapters get grads, the base ViT/head stay frozen."""
    from gvhmr.network.joint_backbone import JointBackbone
    from gvhmr.utils.device import get_device

    dev = get_device()
    x = crops.to(dev)
    joint = JointBackbone(rank=8, alpha=16.0).to(dev).train()

    # only the adapters should be trainable
    trainable = [n for n, p in joint.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert joint.n_lora_params == sum(p.numel() for n, p in joint.named_parameters() if p.requires_grad)

    target = torch.randn(6, 1024, device=dev)  # arbitrary fixed target → can the adapters move the token?
    opt = torch.optim.Adam([p for p in joint.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = (joint(x).float() - target).pow(2).mean()
        loss.backward()
        # grads reach adapters, never the frozen base
        a = joint.hmr2.backbone.blocks[0].attn.qkv
        assert a.lora_A.grad is not None and a.lora_B.grad is not None
        assert a.base.weight.grad is None
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.9, f"loss did not drop: {losses[0]:.4f} → {losses[-1]:.4f}"
