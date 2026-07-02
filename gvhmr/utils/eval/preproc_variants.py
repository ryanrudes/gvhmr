"""Regenerate the benchmark preprocessing with swapped stages — so `gvhmr eval` can measure them.

The canonical benchmark packs ship *frozen* preprocessing (YOLOv8x boxes, ViTPose keypoints, HMR2
features — computed once upstream), which is what makes the paper numbers reproducible but also makes a
detector/2D-pose swap invisible to `gvhmr eval`. This module rebuilds those artifacts with **your chosen
stages** into a *variant cache* next to the canonical files (never overwriting them):

    <DS>/hmr4d_support/preproc_variants/<slug>/     # slug e.g. "yolo26x-vitpose"
        3DPW:  preproc_test_bbx.pt · preproc_test_kp2d.pt · imgfeats/3dpw_test{,_flip}/<vid>.pt
        EMDB:  emdb_preproc.pt (vid → {bbx_xys, kp2d, features}) · imgfeats/emdb_flip/<vid>.pt

The test dataset loaders accept ``preproc_variant=<slug>`` and read from that cache instead (ground
truth always stays canonical). RICH is not supported: its pack has no per-sequence videos and the raw
dataset is registration-gated.

**Videos are required and not redistributable.** The packs ship an *empty* ``videos/`` dir; the raw
datasets (3DPW: https://virtualhumans.mpi-inf.mpg.de/3DPW/ · EMDB: https://eth-ait.github.io/emdb/)
ship image sequences. :func:`build_videos_from_raw` composes the expected ``videos/<name>.mp4`` (30 fps)
from an official raw download — a one-time step, after which any number of variants can be generated.

**Identity guard (3DPW is multi-person).** A fresh detector track on a two-person video may lock onto
the *other* person than the canonical track did — scoring that would be garbage, not a detector
comparison. Each regenerated track is IoU-checked against the canonical one; on identity mismatch the
canonical boxes are kept for that sequence (recorded and reported), so a variant differs from canonical
only by the stage under test, never by *who* is being tracked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gvhmr.utils.console import console
from gvhmr.utils.pylogger import Log

#: Median-IoU threshold under which a regenerated track is considered a different person.
IDENTITY_IOU_THR = 0.5


def variant_slug(detector: str | None, pose2d: str | None, backbone: str | None = None) -> str:
    """Deterministic cache name for a stage selection (defaults spelled out, e.g. ``yolo26x-vitpose``)."""
    parts = [detector or "yolo", pose2d or "vitpose"]
    if backbone and backbone != "hmr2":
        parts.append(backbone)
    return "-".join(parts)


def xys_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame IoU of two square-box tracks in the ``(x, y, size)`` convention. (F,3),(F,3) → (F,)."""
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 2] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 2] / 2
    ix = (torch.min(ax1 + a[:, 2], bx1 + b[:, 2]) - torch.max(ax1, bx1)).clamp(min=0)
    iy = (torch.min(ay1 + a[:, 2], by1 + b[:, 2]) - torch.max(ay1, by1)).clamp(min=0)
    inter = ix * iy
    union = a[:, 2] ** 2 + b[:, 2] ** 2 - inter
    return inter / union.clamp(min=1e-9)


def same_identity(new_xys: torch.Tensor, canonical_xys: torch.Tensor) -> bool:
    """Whether a regenerated track follows the same person as the canonical one (median IoU)."""
    n = min(len(new_xys), len(canonical_xys))
    if n == 0:
        return False
    return xys_iou(new_xys[:n].float(), canonical_xys[:n].float()).median().item() >= IDENTITY_IOU_THR


# --- video composition from the official raw downloads --------------------------------------------

#: dataset → (raw image-dir pattern relative to the raw root, target video name source)
_RAW_LAYOUTS = {
    "3dpw": "imageFiles/{name}",  # official 3DPW: imageFiles/<seq>/image_00000.jpg
    "emdb": "{p}/{seq}/images",  # official EMDB: P3/28_outdoor_walk_lunges/images/*.jpg
}


def _raw_image_dir(dataset: str, raw_dir: Path, name: str) -> Path:
    if dataset == "3dpw":
        return raw_dir / _RAW_LAYOUTS["3dpw"].format(name=name)
    p, seq = name.split("_", 1)  # P3_28_outdoor_walk_lunges → P3/28_outdoor_walk_lunges
    return raw_dir / _RAW_LAYOUTS["emdb"].format(p=p, seq=seq)


def build_videos_from_raw(dataset: str, raw_dir: Path, names: list[str], videos_dir: Path) -> None:
    """Compose the pack's (empty) ``videos/<name>.mp4`` from an official raw download, 30 fps.

    Skips videos that already exist; raises with the expected layout when the raw dir doesn't match.
    """
    import cv2

    from gvhmr.utils.video_io_utils import get_writer

    videos_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (videos_dir / f"{n}.mp4").exists()]
    if not todo:
        return
    for name in todo:
        img_dir = _raw_image_dir(dataset, Path(raw_dir), name)
        frames = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
        if not frames:
            raise FileNotFoundError(
                f"No frames at {img_dir} — --raw-dir must point at the official {dataset.upper()} download "
                f"(expected layout: {_RAW_LAYOUTS[dataset]})."
            )
        writer = get_writer(videos_dir / f"{name}.mp4", fps=30, crf=17)
        for f in frames:
            writer.write_frame(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
        writer.close()
        Log.info(f"[ok]composed[/] videos/{name}.mp4 ({len(frames)} frames)")


# --- per-sequence stage extraction ------------------------------------------------------------------


@dataclass
class VariantReport:
    """What the generator did — surfaced in `gvhmr eval`'s output so nothing happens silently."""

    slug: str
    generated: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    identity_fallbacks: list[str] = field(default_factory=list)

    def log(self) -> None:
        if self.generated:
            Log.info(f"[ok]{len(self.generated)} sequence(s) regenerated[/] for variant '{self.slug}'")
        if self.cached:
            Log.info(f"{len(self.cached)} sequence(s) already cached for variant '{self.slug}'")
        if self.identity_fallbacks:
            Log.warning(
                f"[warn]identity guard[/]: {len(self.identity_fallbacks)} sequence(s) kept the CANONICAL "
                f"boxes (the new detector locked onto a different person): "
                f"{', '.join(self.identity_fallbacks)}"
            )


def _extract_stages(video_path: Path, canonical_xys: torch.Tensor, stages: dict, vid: str, report: VariantReport):
    """Run detector (identity-guarded) → pose2d → backbone (+ flipped-video backbone) on one video."""
    from gvhmr.utils.geo.flip_utils import flip_bbx_xys
    from gvhmr.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from gvhmr.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer

    length, width, _ = get_video_lwh(video_path)
    if length != len(canonical_xys):
        raise ValueError(
            f"{vid}: video has {length} frames but the canonical preprocessing has {len(canonical_xys)} — "
            f"the composed/supplied video doesn't match the benchmark protocol."
        )

    # 1) boxes — regenerate with the chosen detector, then identity-check against the canonical track.
    if stages["detector"] is not None:
        bbx_xyxy = stages["detector"].get_one_track(str(video_path)).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        if not same_identity(bbx_xys, canonical_xys):
            report.identity_fallbacks.append(vid)
            bbx_xys = canonical_xys.clone().float()
    else:
        bbx_xys = canonical_xys.clone().float()

    # 2) 2D keypoints on the variant boxes.
    kp2d = stages["pose2d"].extract(str(video_path), bbx_xys)

    # 3) features on the variant boxes; the flip-test protocol needs them on the mirrored video too.
    features = stages["backbone"].extract_video_features(str(video_path), bbx_xys)
    flip_video = video_path.with_name(f"_tmp_flip_{video_path.name}")
    try:
        reader = get_video_reader(video_path)
        writer = get_writer(flip_video, fps=30, crf=17)
        for img in reader:
            writer.write_frame(np.ascontiguousarray(img[:, ::-1]))
        writer.close()
        reader.close()
        flip_xys = flip_bbx_xys(bbx_xys, width)
        flip_features = stages["backbone"].extract_video_features(str(flip_video), flip_xys)
    finally:
        flip_video.unlink(missing_ok=True)

    return {
        "bbx_xys": bbx_xys,
        "kp2d": kp2d.float().cpu(),
        "features": features.float().cpu(),
        "flip_bbx_xys": flip_xys,
        "flip_features": flip_features.float().cpu(),
    }


def _resolve_stage(group: str, name: str) -> tuple[str, dict]:
    """Resolve a stage selection through the demo's Hydra config group (the same mechanism the demo's
    ``--detector/--pose2d/--backbone`` use), so every preset (``yolo26x``, ``rtmpose``, …) and its knobs
    mean exactly the same thing here. Returns the implementation name + ctor kwargs."""
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from gvhmr.configs import register_store_gvhmr

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        node = compose(config_name="demo", overrides=[f"{group}={name}", "video_name=_"])[group]
    conf = OmegaConf.to_container(node, resolve=True)
    impl = conf.pop("name")
    return impl, {k: v for k, v in conf.items() if v is not None}


def _make_stages(detector: str | None, pose2d: str | None, backbone: str | None) -> dict:
    """Construct the chosen stage implementations (detector=None ⇒ keep canonical boxes)."""
    from gvhmr.utils.preproc.base import make_backbone, make_detector, make_pose2d

    stages: dict = {"detector": None}
    if detector is not None:
        impl, kwargs = _resolve_stage("detector", detector)
        stages["detector"] = make_detector(impl, **kwargs)
    impl, kwargs = _resolve_stage("pose2d", pose2d or "vitpose")
    stages["pose2d"] = make_pose2d(impl, **kwargs)
    impl, kwargs = _resolve_stage("backbone", backbone or "hmr2")
    stages["backbone"] = make_backbone(impl, **kwargs)
    return stages


# --- dataset-specific generation (canonical schema in, canonical schema out) ------------------------


def generate_3dpw_variant(
    support_dir: Path,
    slug: str,
    detector: str | None,
    pose2d: str | None,
    backbone: str | None = None,
    overwrite: bool = False,
) -> VariantReport:
    """Write ``preproc_variants/<slug>/`` for 3DPW (same file schema the canonical loader reads)."""
    report = VariantReport(slug=slug)
    labels = torch.load(support_dir / "test_3dpw_gt_labels.pt", weights_only=False)
    vid2bbx = torch.load(support_dir / "preproc_test_bbx.pt", weights_only=False)

    out = support_dir / "preproc_variants" / slug
    feat_dir, flip_dir = out / "imgfeats/3dpw_test", out / "imgfeats/3dpw_test_flip"
    feat_dir.mkdir(parents=True, exist_ok=True)
    flip_dir.mkdir(parents=True, exist_ok=True)

    stages = _make_stages(detector, pose2d, backbone)
    new_bbx: dict = {}
    new_kp2d: dict = {}
    bbx_pt, kp2d_pt = out / "preproc_test_bbx.pt", out / "preproc_test_kp2d.pt"
    if bbx_pt.exists() and not overwrite:
        new_bbx = torch.load(bbx_pt, weights_only=False)
        new_kp2d = torch.load(kp2d_pt, weights_only=False)

    for vid in labels:
        if vid in new_bbx and (feat_dir / f"{vid}.pt").exists() and not overwrite:
            report.cached.append(vid)
            continue
        vname = labels[vid]["vname"]
        video_path = support_dir / f"videos/{vname}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(
                f"{video_path} missing — the packs ship no videos (not redistributable). Pass --raw-dir "
                f"pointing at the official 3DPW download to compose them once."
            )
        r = _extract_stages(video_path, vid2bbx[vid]["bbx_xys"], stages, vid, report)
        new_bbx[vid] = {"bbx_xys": r["bbx_xys"]}
        new_kp2d[vid] = r["kp2d"]
        img_wh = labels[vid]["img_wh"]
        torch.save({"features": r["features"], "bbx_xys": r["bbx_xys"], "img_wh": img_wh}, feat_dir / f"{vid}.pt")
        torch.save({"features": r["flip_features"], "bbx_xys": r["flip_bbx_xys"]}, flip_dir / f"{vid}.pt")
        torch.save(new_bbx, bbx_pt)  # checkpoint progress after every sequence (resumable)
        torch.save(new_kp2d, kp2d_pt)
        report.generated.append(vid)
        Log.info(f"[ok]{vid}[/] regenerated ({len(report.generated) + len(report.cached)}/{len(labels)})")
    return report


def generate_emdb_variant(
    support_dir: Path,
    slug: str,
    detector: str | None,
    pose2d: str | None,
    backbone: str | None = None,
    overwrite: bool = False,
) -> VariantReport:
    """Write ``preproc_variants/<slug>/`` for EMDB (overlay file + flip imgfeats; GT stays canonical)."""
    report = VariantReport(slug=slug)
    labels = torch.load(support_dir / "emdb_vit_v4.pt", weights_only=False)

    out = support_dir / "preproc_variants" / slug
    flip_dir = out / "imgfeats/emdb_flip"
    flip_dir.mkdir(parents=True, exist_ok=True)
    overlay_pt = out / "emdb_preproc.pt"
    overlay: dict = {}
    if overlay_pt.exists() and not overwrite:
        overlay = torch.load(overlay_pt, weights_only=False)

    stages = _make_stages(detector, pose2d, backbone)
    for vid in labels:
        if vid in overlay and (flip_dir / f"{vid}.pt").exists() and not overwrite:
            report.cached.append(vid)
            continue
        video_path = support_dir / f"videos/{vid}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(
                f"{video_path} missing — the packs ship no videos (not redistributable). Pass --raw-dir "
                f"pointing at the official EMDB download to compose them once."
            )
        r = _extract_stages(video_path, labels[vid]["bbx_xys"], stages, vid, report)
        overlay[vid] = {"bbx_xys": r["bbx_xys"], "kp2d": r["kp2d"], "features": r["features"]}
        torch.save({"features": r["flip_features"], "bbx_xys": r["flip_bbx_xys"]}, flip_dir / f"{vid}.pt")
        torch.save(overlay, overlay_pt)  # checkpoint progress after every sequence (resumable)
        report.generated.append(vid)
        Log.info(f"[ok]{vid}[/] regenerated ({len(report.generated) + len(report.cached)}/{len(labels)})")
    return report


def dataset_video_names(dataset: str, support_dir: Path) -> list[str]:
    """The video basenames a dataset's ``videos/`` dir must contain (3DPW: per-video, not per-person)."""
    if dataset == "3dpw":
        labels = torch.load(support_dir / "test_3dpw_gt_labels.pt", weights_only=False)
        return sorted({labels[v]["vname"] for v in labels})
    return sorted(torch.load(support_dir / "emdb_vit_v4.pt", weights_only=False))


def variant_complete(dataset: str, support_dir: Path, slug: str) -> bool:
    """Whether a variant cache covers every sequence of the dataset (cheap check, no stage imports)."""
    out = support_dir / "preproc_variants" / slug
    if dataset == "3dpw":
        if not (out / "preproc_test_bbx.pt").exists():
            return False
        labels = torch.load(support_dir / "test_3dpw_gt_labels.pt", weights_only=False)
        done = torch.load(out / "preproc_test_bbx.pt", weights_only=False)
        return all(v in done and (out / f"imgfeats/3dpw_test/{v}.pt").exists() for v in labels)
    if dataset == "emdb":
        if not (out / "emdb_preproc.pt").exists():
            return False
        labels = torch.load(support_dir / "emdb_vit_v4.pt", weights_only=False)
        done = torch.load(out / "emdb_preproc.pt", weights_only=False)
        return all(v in done and (out / f"imgfeats/emdb_flip/{v}.pt").exists() for v in labels)
    return False
