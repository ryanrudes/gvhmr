import os

import torch

from gvhmr.network.hmr2 import HMR2, load_hmr2
from gvhmr.utils.console import track
from gvhmr.utils.device import get_device
from gvhmr.utils.imgcrop import get_batch  # noqa: F401 — re-export; the crop path lives in the dpvo-free module
from gvhmr.utils.net_utils import skip_torch_init


def _fp32_forced() -> bool:
    """``$GVHMR_PREPROC_FP32=1`` forces the preproc ViTs back to full fp32 (disables bf16 autocast) —
    the accuracy-first escape hatch / a numerical A/B baseline. bf16's measured shift is below noise."""
    return os.environ.get("GVHMR_PREPROC_FP32", "").strip().lower() in ("1", "true", "yes")


class Extractor:
    """HMR2 (4D-Humans) ViT feature backbone. Satisfies the ``FeatureBackbone`` protocol (base.py)."""

    feat_dim = 1024  # HMR2.0a SMPL-head token width; must match the trained network's imgseq_dim

    def __init__(self, tqdm_leave=True, batch_size=32):
        self.device = get_device()
        with skip_torch_init():  # random init is overwritten by load_hmr2's strict ckpt load
            self.extractor: HMR2 = load_hmr2().to(self.device).eval()
        self.tqdm_leave = tqdm_leave
        self.batch_size = batch_size  # 16 was tuned for a 3090; bf16 halves memory so raise it

    @torch.no_grad()
    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        imgs = imgs.to(self.device)
        batch_size = self.batch_size
        use_amp = self.device.type == "cuda" and not _fp32_forced()  # bf16 autocast on CUDA (~2x)
        features = []
        for j in track(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                feature = self.extractor({"img": imgs_batch})
            features.append(feature.float().detach().cpu())  # fp32 for a dtype-stable .pt cache

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features
