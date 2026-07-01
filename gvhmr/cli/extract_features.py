"""``gvhmr extract-features`` — offline image-feature extraction for (re)training (Tier B).

The bridge between a swappable feature backbone and a retrain: run any registered backbone
(``--backbone hmr2|dinov2|…``) over a set of videos and write the **training cache format** the datasets
read — one ``<vid>.pt`` per video holding ``{features (F,D), bbx_xys (F,3), img_wh (2,)}`` (the exact
schema in e.g. ``gvhmr/dataset/threedpw/threedpw_motion_train.py``). Point a dataset's ``imgfeat_subdir``
at the output dir, set ``network.imgseq_dim = D``, and train. See ``docs/EXTENSIBILITY.md`` (B2/B3/B4).

The backbone is resolved through the **same Hydra config group** the demo uses (``configs/backbone/``),
so ``--backbone dinov2`` picks the *identical* variant/knobs (and therefore the same feature width) the
demo and training configs expect — one source of truth. Override knobs with ``--set``
(e.g. ``--set backbone.model_name=dinov2_vitl14``).

Two ways to get the per-frame person box each backbone crops around:
- **detector** (default): run ``--detector`` (YOLO) to track the subject — the bring-your-own-data path.
- **reuse** (``--bbx-from DIR``): copy ``bbx_xys``/``img_wh`` from an existing feature cache, so a new
  backbone extracts on *exactly* the released boxes (the canonical re-extract-a-dataset path).
"""

from __future__ import annotations

from pathlib import Path

import torch

from gvhmr.utils.console import console, rule, track
from gvhmr.utils.preproc.base import make_backbone, make_detector
from gvhmr.utils.pylogger import Log


def _build_backbone(backbone: str, set_overrides: list[str] | None):
    """Resolve the backbone from the ``configs/backbone`` group (same as the demo) and construct it."""
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from gvhmr.configs import register_store_gvhmr

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        overrides = [f"backbone={backbone}", "video_name=_"] + list(set_overrides or [])
        node = compose(config_name="demo", overrides=overrides).backbone
    conf = OmegaConf.to_container(node, resolve=True)
    name = conf.pop("name")
    kwargs = {k: v for k, v in conf.items() if v is not None}
    return make_backbone(name, tqdm_leave=False, **kwargs)


def run(
    videos: Path,
    out: Path,
    *,
    backbone: str = "hmr2",
    detector: str = "yolo",
    bbx_from: Path | None = None,
    pattern: str = "*.mp4",
    overwrite: bool = False,
    set_overrides: list[str] | None = None,
) -> None:
    """Extract features for a video (or a folder of them) into the ``imgfeats`` training-cache format."""
    from gvhmr.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from gvhmr.utils.video_io_utils import get_video_lwh

    videos = Path(videos)
    vids = [videos] if videos.is_file() else sorted(videos.glob(pattern))
    if not vids:
        Log.warning(f"No videos matched [muted]{videos}/{pattern}[/]")
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rule(f"Extract features · backbone=[gvhmr]{backbone}[/] · {len(vids)} video(s)")
    extractor = _build_backbone(backbone, set_overrides)
    feat_dim = getattr(extractor, "feat_dim", None)
    Log.info(f"Backbone [ok]{backbone}[/]" + (f" → feat_dim [ok]{feat_dim}[/]" if feat_dim else ""))
    tracker = None if bbx_from is not None else make_detector(detector)

    written, skipped = 0, 0
    for video in track(vids, desc="Videos"):
        vid = video.stem
        dst = out / f"{vid}.pt"
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        width, height = get_video_lwh(video)[1:]

        if bbx_from is not None:  # reuse boxes from an existing feature cache (exact re-extraction)
            src = Path(bbx_from) / f"{vid}.pt"
            if not src.exists():
                Log.warning(f"[warn]no bbx for {vid}[/] in {bbx_from} — skipping")
                skipped += 1
                continue
            cached = torch.load(src, weights_only=False)
            bbx_xys = cached["bbx_xys"].float()
            img_wh = cached.get("img_wh", torch.tensor([width, height]))
        else:  # detect + track the subject (bring-your-own-data)
            bbx_xyxy = tracker.get_one_track(str(video)).float()  # (F, 4)
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (F, 3)
            img_wh = torch.tensor([width, height])

        features = extractor.extract_video_features(str(video), bbx_xys)  # (F, D)
        torch.save(
            {"features": features, "bbx_xys": bbx_xys, "img_wh": img_wh, "backbone": backbone, "feat_dim": feat_dim},
            dst,
        )
        written += 1

    Log.info(
        f"[ok]Wrote {written}[/] feature file(s) to [muted]{out}[/]" + (f" · skipped {skipped}" if skipped else "")
    )
    if written and feat_dim:
        console.print(
            f"  → train on these with [gvhmr]network.imgseq_dim={feat_dim}[/] and a dataset "
            f"[gvhmr]imgfeat_subdir[/] pointing at [muted]{out}[/] (see docs/EXTENSIBILITY.md)."
        )
