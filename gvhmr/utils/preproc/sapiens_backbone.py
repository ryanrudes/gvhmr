"""Sapiens image-feature backbone — Meta's human-centric ViT (2024) as an alternative to HMR2.

Like DINOv2, this is a **Tier-B** backbone: the released 1024-d GVHMR checkpoint's ``imgseq_embedder`` is
fit to HMR2's feature space, so Sapiens features are **not** a drop-in at inference — they require a
**retrain** with ``network.imgseq_dim = feat_dim`` (the guard in ``relative_transformer.py`` enforces it).

Why Sapiens: it is pretrained on 300M+ **human** images at high resolution, so its features are far more
human-specialized than HMR2's 2023 ViT — the single highest-leverage backbone swap in the bake-off (see
``docs/ROADMAP.md``, Plan A/A1).

Integration notes (verified against ``facebook/sapiens-pretrain-*-torchscript``):
- **Weights.** The "sapiens-lite" TorchScript encoders (``facebook/sapiens-pretrain-<size>-torchscript`` on
  the HF hub, file ``sapiens_<size>_epoch_1600_torchscript.pt2``) load with ``torch.jit.load`` and need no
  Sapiens code — the robust path used here. Point ``checkpoint`` at one (absolute, or a name/relative path
  under the configured ``checkpoints`` root).
- **Input.** The pretrain encoders are traced at a FIXED **1024×1024** (the positional embedding is a 64×64
  grid), with ImageNet normalization. We reuse 4D-Humans' ``get_batch`` (already ImageNet-normalized square
  crops) at ``img_dst_size=1024`` and pass 1024². ``input_hw`` must match the traced size.
- **Output.** The encoder returns a 1-tuple wrapping the patch **feature map** ``(B, C, 64, 64)`` (verified:
  0.3b → C=1024, 0.6b → 1280, …). We unwrap it and global-average-pool → ``(F, C)``. ``feat_dim`` (C) is
  seeded from ``_EMBED_DIMS`` and **re-verified from the first forward**, so a wrong table entry self-corrects.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from gvhmr.utils.console import track
from gvhmr.utils.device import get_device
from gvhmr.utils.pylogger import Log

# Sapiens pretrained-encoder hidden widths (patch size 16). Seed values — re-verified at first forward.
_EMBED_DIMS = {"sapiens_0.3b": 1024, "sapiens_0.6b": 1280, "sapiens_1b": 1536, "sapiens_2b": 1920}


def _resolve_ckpt(checkpoint: str | None, model_name: str) -> Path:
    """Resolve a TorchScript encoder path: absolute, or relative to the configured checkpoints root."""
    if checkpoint is None:
        raise FileNotFoundError(
            f"Sapiens backbone needs a TorchScript encoder checkpoint for {model_name!r}. Download a "
            "'sapiens-lite' pretrain encoder (*_torchscript.pt2) and pass it via the backbone config "
            "(`--set backbone.checkpoint=<path>`), an absolute path, or a name under the checkpoints root."
        )
    p = Path(checkpoint)
    if not p.is_absolute():
        try:
            from gvhmr.utils import assets

            p = Path(assets.CHECKPOINT_ROOT) / checkpoint
        except Exception:  # noqa: BLE001 — best-effort; fall back to the literal path
            p = Path(checkpoint)
    if not p.exists():
        raise FileNotFoundError(f"Sapiens checkpoint not found: {p} (set backbone.checkpoint to a real path)")
    return p


class SapiensBackbone:
    """Per-frame Sapiens encoder features (global-average-pooled). Satisfies ``FeatureBackbone`` (base.py)."""

    def __init__(
        self,
        model_name: str = "sapiens_0.6b",
        checkpoint: str | None = None,
        input_hw: tuple[int, int] = (1024, 1024),  # the pretrain encoders are traced at a fixed 1024²
        tqdm_leave: bool = True,
    ):
        self.device = get_device()
        self.model_name = model_name
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        # Provisional width; corrected from the first real forward (see extract_video_features).
        self.feat_dim = _EMBED_DIMS.get(model_name, 1280)
        self.model = self._load_model(_resolve_ckpt(checkpoint, model_name))
        self.tqdm_leave = tqdm_leave

    def _load_model(self, ckpt: Path):
        """Load a sapiens-lite TorchScript encoder. Swap this for non-lite ``.pth`` checkpoints."""
        model = torch.jit.load(str(ckpt), map_location="cpu")
        return model.to(self.device).eval()

    @torch.no_grad()
    def extract_video_features(self, video_path, bbx_xys, img_ds: float = 0.5) -> torch.Tensor:
        from .vitfeat_extractor import get_batch

        if isinstance(video_path, str):
            # High-res ImageNet-normalized square crops (Sapiens wants ~1024 px, not HMR2's 256).
            imgs, _ = get_batch(video_path, bbx_xys, img_ds=img_ds, img_dst_size=max(self.input_hw))
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path
        h, w = self.input_hw
        feats = []
        for j in track(range(0, len(imgs), 8), desc="Sapiens Feature", leave=self.tqdm_leave):
            b = imgs[j : j + 8].to(self.device).float()
            b = F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False)
            out = self.model(b)
            if isinstance(out, (tuple, list)):  # the pretrain encoder returns a 1-tuple
                out = out[0]
            v = out.mean(dim=(-2, -1)) if out.ndim == 4 else out  # GAP the (B,C,h,w) feature map → (B, C)
            feats.append(v.detach().float().cpu())
        out = torch.cat(feats, dim=0).clone()  # (F, C)
        if out.shape[-1] != self.feat_dim:  # trust the model, not the table
            Log.warning(
                f"[sapiens] feat_dim table said {self.feat_dim} but the encoder emits {out.shape[-1]}; "
                f"using {out.shape[-1]} (set network.imgseq_dim to match when you retrain)"
            )
            self.feat_dim = out.shape[-1]
        return out
