import os

import cv2
import numpy as np
import torch

from gvhmr.network.hmr2 import HMR2, load_hmr2
from gvhmr.network.hmr2.utils.preproc import IMAGE_MEAN, IMAGE_STD, crop_and_resize
from gvhmr.utils.console import track
from gvhmr.utils.device import get_device
from gvhmr.utils.net_utils import skip_torch_init
from gvhmr.utils.video_io_utils import read_video_np


def _fp32_forced() -> bool:
    """``$GVHMR_PREPROC_FP32=1`` forces the preproc ViTs back to full fp32 (disables bf16 autocast) —
    the accuracy-first escape hatch / a numerical A/B baseline. bf16's measured shift is below noise."""
    return os.environ.get("GVHMR_PREPROC_FP32", "").strip().lower() in ("1", "true", "yes")


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256
    return imgs, bbx_xys


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
