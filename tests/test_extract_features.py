"""`gvhmr extract-features` (Tier B offline feature cache) — CI-safe: no real backbone / GPU / network.

Guards two things that matter for a retrain: (1) the backbone is resolved through the **same config
group** the demo/training configs use (so feature width is consistent — the fix for a silent 768-vs-384
mismatch), and (2) each ``<vid>.pt`` is written in the exact schema the datasets read
(``{features, bbx_xys, img_wh}``).
"""

from __future__ import annotations

import numpy as np
import torch

import gvhmr.cli.extract_features as ef
from gvhmr.utils.video_io_utils import get_writer


def test_backbone_resolves_via_config_group(monkeypatch):
    # _build_backbone must compose configs/backbone/<name>.yaml and pass its knobs to make_backbone,
    # so `dinov2` picks vits14 (384-d) — the same variant the training config expects, not the ctor
    # default (vitb14/768). make_backbone is faked so nothing is downloaded/constructed.
    captured = {}

    def fake_make_backbone(name, **kw):
        captured["name"], captured["kw"] = name, kw

        class _F:
            feat_dim = 0

        return _F()

    monkeypatch.setattr(ef, "make_backbone", fake_make_backbone)
    ef._build_backbone("dinov2", None)
    assert captured["name"] == "dinov2"
    assert captured["kw"].get("model_name") == "dinov2_vits14"  # from the group, not the ctor default


def test_writes_trainer_cache_schema(tmp_path, monkeypatch):
    # A 4-frame clip + a bbx cache (reuse path → no detector). A fake backbone emits (F, 8) so the test
    # needs no model. The written file must match how the datasets load it: features/bbx_xys/img_wh.
    video = tmp_path / "seqA.mp4"
    writer = get_writer(str(video), fps=30)
    for _ in range(4):
        writer.write_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.close()

    bbx_dir = tmp_path / "bbx"
    bbx_dir.mkdir()
    torch.save({"bbx_xys": torch.zeros(4, 3), "img_wh": torch.tensor([64, 64])}, bbx_dir / "seqA.pt")

    class FakeBackbone:
        feat_dim = 8

        def extract_video_features(self, video_path, bbx_xys):
            return torch.zeros(len(bbx_xys), self.feat_dim)

    monkeypatch.setattr(ef, "_build_backbone", lambda *a, **k: FakeBackbone())

    out = tmp_path / "imgfeats"
    ef.run(video, out, backbone="hmr2", bbx_from=bbx_dir)

    d = torch.load(out / "seqA.pt", weights_only=False)
    assert {"features", "bbx_xys", "img_wh"} <= set(d)
    assert d["features"].shape == (4, 8)
    assert d["bbx_xys"].shape == (4, 3)
    assert tuple(d["img_wh"].tolist()) == (64, 64)
