""":class:`GVHMR` — the core motion-recovery model, HuggingFace ``from_pretrained`` style.

This is the "power path": it wraps the trained ``DemoPL`` (NetworkEncoderRoPE denoiser + EnDecoder) and
exposes the tensor-level :meth:`GVHMR.predict` for callers who did their own preprocessing, plus a
convenience :meth:`GVHMR.recover` that runs the full video→motion pipeline. The model is built exactly the
way the golden test / ``gvhmr demo`` build it, so its outputs are byte-identical.
"""

from __future__ import annotations

from pathlib import Path

import torch

from gvhmr.inference import hub
from gvhmr.model.gvhmr.gvhmr_pl_demo import DemoPL


def _build_demo_pl(ckpt_path: str | Path) -> DemoPL:
    """Compose the demo config's model node, instantiate ``DemoPL``, and load the checkpoint (strict path)."""
    import hydra
    from hydra import compose, initialize_config_module

    import gvhmr.model.gvhmr.gvhmr_pl_demo  # noqa: F401 — registers the demo model in MainStore
    from gvhmr.configs import register_store_gvhmr

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=_gvhmr_lib_"])
    pl: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    pl.load_pretrained_model(str(ckpt_path))
    return pl


class GVHMR:
    """The trained GVHMR model. Load with :meth:`from_pretrained`; run end-to-end with :meth:`recover`."""

    def __init__(self, pl: DemoPL, *, device=None, repo_id: str | None = None, ckpt_path: str | Path | None = None):
        self.pl = pl
        self.repo_id = repo_id
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.device = torch.device(device) if device is not None else self._infer_device()

    def _infer_device(self):
        try:
            return next(self.pl.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        model: str = hub.DEFAULT_REPO,
        *,
        device=None,
        cache_dir=None,
        revision: str | None = None,
        token: str | None = None,
        ckpt_path: str | Path | None = None,
    ) -> GVHMR:
        """Load the GVHMR checkpoint from a Hub repo id (or a local ``ckpt_path``) onto ``device``.

        ``model`` is a HuggingFace repo id (default :data:`gvhmr.inference.hub.DEFAULT_REPO`, with mirror
        fallback). Pass ``ckpt_path=`` to load a local checkpoint directly and skip the download.
        """
        from gvhmr.utils.device import get_device

        dev = torch.device(device) if device is not None else get_device()
        if ckpt_path is None:
            ckpt_path = hub.ensure_gvhmr_ckpt(model, cache_dir=cache_dir, token=token, revision=revision)
        pl = _build_demo_pl(ckpt_path).eval().to(dev)
        return cls(pl, device=dev, repo_id=model, ckpt_path=ckpt_path)

    # ---- tensor-level API (bring your own preprocessing) -----------------------------------------
    @torch.no_grad()
    def predict(self, data: dict, *, static_cam: bool = False, flip_test_data=None, world_from_incam=False) -> dict:
        """Run the model on a prepared ``data`` dict (see ``DemoPL.predict``). Returns the raw ``pred`` dict."""
        return self.pl.predict(
            data, static_cam=static_cam, flip_test_data=flip_test_data, world_from_incam=world_from_incam
        )

    # ---- end-to-end convenience ------------------------------------------------------------------
    def recover(self, video, **kwargs):
        """Run the full video→motion pipeline with this model. See :meth:`GVHMRPipeline.__call__`."""
        from gvhmr.inference.pipelines import GVHMRPipeline

        return GVHMRPipeline(self, device=self.device)(video, **kwargs)

    def to(self, device) -> GVHMR:
        self.pl.to(device)
        self.device = torch.device(device)
        return self

    def eval(self) -> GVHMR:
        self.pl.eval()
        return self

    # ---- sharing ---------------------------------------------------------------------------------
    def save_pretrained(self, save_dir, *, repo_id: str = hub.DEFAULT_REPO):
        """Write a self-contained model folder (checkpoint + config + card)."""
        return hub.save_pretrained(self, save_dir, repo_id=repo_id)

    def push_to_hub(self, repo_id: str, *, token=None, private: bool = False) -> str:
        """Push this model's checkpoint + card to the Hub. Returns the repo URL."""
        return hub.push_to_hub(self, repo_id, token=token, private=private)

    def __repr__(self) -> str:
        return f"GVHMR(repo_id={self.repo_id!r}, device={str(self.device)!r})"
