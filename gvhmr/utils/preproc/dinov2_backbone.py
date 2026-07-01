"""DINOv2 image-feature backbone — an ungated (torch.hub) alternative to HMR2 for retraining.

The released GVHMR checkpoint's ``imgseq_embedder`` is fit to HMR2's 1024-d token, so DINOv2 features are
**not** a drop-in at inference — they require **retraining** the core with ``network.imgseq_dim = feat_dim``
(the guard in ``relative_transformer.py`` enforces this). This is the concrete alternative backbone the
Tier-B feature-swap was built for: extract features with it, then retrain. See ``docs/EXTENSIBILITY.md``.

4D-Humans' ``get_batch`` already ImageNet-normalizes the person crop (== DINOv2's expected normalization),
so we only resize the 256² crop to a patch-size (14) multiple and read the CLS token.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from gvhmr.utils.console import track
from gvhmr.utils.device import get_device

# DINOv2 ViT-14 variants → CLS-token width.
_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
_INPUT = 224  # 16 × 14 (patch size 14)


class DINOv2Backbone:
    """Per-frame DINOv2 CLS-token features. Satisfies the ``FeatureBackbone`` protocol (base.py)."""

    def __init__(self, model_name: str = "dinov2_vitb14", tqdm_leave: bool = True):
        assert model_name in _DIMS, f"unknown dinov2 model {model_name!r}; choose from {tuple(_DIMS)}"
        self.device = get_device()
        # Prefer the local torch.hub cache (offline-friendly: `source="local"` skips the GitHub ref check),
        # falling back to a fresh download. The cache can be copied between machines (~85 MB for vits14).
        cached = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if cached.exists():
            self.model = torch.hub.load(str(cached), model_name, source="local")
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(self.device).eval()
        self.feat_dim = _DIMS[model_name]
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract_video_features(self, video_path, bbx_xys, img_ds: float = 0.5) -> torch.Tensor:
        from .vitfeat_extractor import get_batch

        if isinstance(video_path, str):
            imgs, _ = get_batch(video_path, bbx_xys, img_ds=img_ds)  # (F,3,256,256), ImageNet-normalized
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path
        feats = []
        for j in track(range(0, len(imgs), 16), desc="DINOv2 Feature", leave=self.tqdm_leave):
            b = imgs[j : j + 16].to(self.device).float()
            b = F.interpolate(b, size=(_INPUT, _INPUT), mode="bicubic", align_corners=False)
            feats.append(self.model(b).detach().cpu())  # (B, feat_dim) CLS token
        return torch.cat(feats, dim=0).clone()  # (F, feat_dim)
