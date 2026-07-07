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


def variant_slug(
    detector: str | None, pose2d: str | None, backbone: str | None = None, stage_overrides: list[str] | None = None
) -> str:
    """Deterministic cache name for a stage selection (defaults spelled out, e.g. ``yolo26x-vitpose``).
    Stage ``--set`` knobs (e.g. ``pose2d.flip_test=false``) fold into the slug so different knob sets
    don't collide in the cache — ``yolo-vitpose__pose2d.flip_test=false``."""
    parts = [detector or "yolo", pose2d or "vitpose"]
    if backbone and backbone != "hmr2":
        parts.append(backbone)
    slug = "-".join(parts)
    if stage_overrides:
        tag = ",".join(sorted(stage_overrides)).replace("/", "").replace(" ", "")
        slug = f"{slug}__{tag}"
    return slug


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

#: Raw datasets served directly by their official hosts (no credentials) → auto-fetchable, like every
#: other asset: dataset → (url, exact size, license url). EMDB is NOT here — its download is
#: credential-gated (institutional email), so it stays a manual --raw-dir step.
RAW_AUTOFETCH: dict[str, tuple[str, int, str]] = {
    "3dpw": (
        "https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip",
        4_610_672_154,
        "https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html",
    ),
}


def fetch_raw_dataset(dataset: str, dest_root: Path) -> Path:
    """Download + extract an auto-fetchable raw dataset (resumable; exact-size verified). Returns the
    directory usable as ``raw_dir`` (e.g. ``<dest_root>`` containing ``imageFiles/``). Idempotent —
    an already-extracted copy is reused; a partial ``.zip.part`` resumes via HTTP Range."""
    import urllib.request
    import zipfile

    from gvhmr.utils.console import track
    from gvhmr.utils.net import ensure_ca_bundle

    url, size, license_url = RAW_AUTOFETCH[dataset]
    dest_root = Path(dest_root)
    marker = dest_root / _RAW_LAYOUTS[dataset].split("/")[0]  # e.g. imageFiles/
    if marker.is_dir() and any(marker.iterdir()):
        return dest_root

    ensure_ca_bundle()
    dest_root.mkdir(parents=True, exist_ok=True)
    part = dest_root / (Path(url).name + ".part")
    Log.info(
        f"[{dataset}] downloading the official raw dataset ({size / 1e9:.1f}GB) from {url} — "
        f"by continuing you accept its research license: {license_url}"
    )
    start = part.stat().st_size if part.exists() else 0
    if start < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(req) as resp, open(part, "ab") as out:
            total_mb = (size - start) / 1e6
            for _ in track(range(int(total_mb) + 1), desc=f"{dataset} raw ({size / 1e9:.1f}GB)"):
                chunk = resp.read(1_000_000)
                if not chunk:
                    break
                out.write(chunk)
    got = part.stat().st_size
    if got != size:
        raise OSError(f"{part} is {got} bytes, expected {size} — re-run to resume the download.")
    with zipfile.ZipFile(part) as z:
        for name in track(z.namelist(), desc=f"extracting {Path(url).name}"):
            z.extract(name, dest_root)
    part.unlink()  # 4.6GB of dead weight once extracted
    Log.info(f"[ok]{dataset} raw dataset ready[/] → [muted]{dest_root}[/]")
    return dest_root


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


# --- shared stage caches ----------------------------------------------------------------------------
# The grid's economics: a combo (detector, pose2d) does NOT need three fresh model passes. Boxes depend
# only on the detector; features depend only on the boxes (the backbone is fixed per sweep); only the
# keypoints are truly per-combo. So each stage caches independently under
# ``preproc_variants/_stages/{boxes,feats,kp2d}/<key>/<vid>.pt`` and combo dirs are assembled from
# them — a full detector×pose2d grid costs O(detectors) heavy passes + O(combos) keypoint passes, not
# O(combos) of everything. The mirrored video (flip-test) is likewise composed ONCE per video, shared
# by every stage (``videos_flip/``). Stage files are written atomically under an flock, so several
# sweep agents on one box can generate concurrently without duplicating or corrupting work.


class _LazyStages:
    """Construct stage implementations on first use — full cache hits load no models at all."""

    def __init__(
        self, detector: str | None, pose2d: str | None, backbone: str | None, overrides: list[str] | None = None
    ):
        self._names = {"detector": detector, "pose2d": pose2d or "vitpose", "backbone": backbone or "hmr2"}
        self._overrides = list(overrides or [])
        self._built: dict = {}

    def get(self, stage: str):
        if stage not in self._built:
            from gvhmr.utils.preproc.base import make_backbone, make_detector, make_pose2d

            impl, kwargs = _resolve_stage(stage, self._names[stage], self._overrides)
            maker = {"detector": make_detector, "pose2d": make_pose2d, "backbone": make_backbone}[stage]
            self._built[stage] = maker(impl, **kwargs)
        return self._built[stage]


def _locked(path: Path):
    """An exclusive advisory lock scoped to ``path`` (context manager) — agents on one box serialize
    per stage-file instead of duplicating a model pass."""
    import fcntl
    from contextlib import contextmanager

    @contextmanager
    def cm():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(path.suffix + ".lock"), "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    return cm()


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _ensure_flip_video(support_dir: Path, video_path: Path) -> Path:
    """The mirrored copy of a pack video (flip-test features run on it) — composed once, shared by
    every detector/combo, kept under ``videos_flip/`` next to the pack's ``videos/``."""
    from gvhmr.utils.video_io_utils import get_video_reader, get_writer

    flip_path = support_dir / "videos_flip" / video_path.name
    if flip_path.exists():
        return flip_path
    with _locked(flip_path):
        if flip_path.exists():
            return flip_path
        tmp = flip_path.with_suffix(".tmp.mp4")
        reader = get_video_reader(video_path)
        writer = get_writer(tmp, fps=30, crf=17)
        for img in reader:
            writer.write_frame(np.ascontiguousarray(img[:, ::-1]))
        writer.close()
        reader.close()
        tmp.replace(flip_path)
    return flip_path


def _stage_boxes(
    support_dir: Path,
    video_path: Path,
    vid: str,
    detector: str | None,
    canonical_xys: torch.Tensor,
    stages: _LazyStages,
    overwrite: bool,
) -> dict:
    """Per-DETECTOR box track (identity-guarded): ``{"bbx_xys", "identity_fallback"}``."""
    from gvhmr.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from gvhmr.utils.video_io_utils import get_video_lwh

    length = get_video_lwh(video_path)[0]
    if length != len(canonical_xys):
        raise ValueError(
            f"{vid}: video has {length} frames but the canonical preprocessing has {len(canonical_xys)} — "
            f"the composed/supplied video doesn't match the benchmark protocol."
        )
    if detector is None:
        return {"bbx_xys": canonical_xys.clone().float(), "identity_fallback": False}
    cache = support_dir / "preproc_variants/_stages/boxes" / detector / f"{vid}.pt"
    if cache.exists() and not overwrite:
        return torch.load(cache, weights_only=False)
    with _locked(cache):
        if cache.exists() and not overwrite:
            return torch.load(cache, weights_only=False)
        bbx_xyxy = stages.get("detector").get_one_track(str(video_path)).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        fallback = not same_identity(bbx_xys, canonical_xys)
        if fallback:
            bbx_xys = canonical_xys.clone().float()
        out = {"bbx_xys": bbx_xys, "identity_fallback": fallback}
        _atomic_save(out, cache)
    return out


def _stage_kp2d(
    support_dir: Path,
    video_path: Path,
    vid: str,
    detector: str | None,
    pose2d: str | None,
    bbx_xys: torch.Tensor,
    stages: _LazyStages,
    overwrite: bool,
) -> torch.Tensor:
    """Per-(detector, pose2d) keypoints — the only genuinely per-combo model pass."""
    key = f"{detector or 'canonical'}-{pose2d or 'vitpose'}"
    cache = support_dir / "preproc_variants/_stages/kp2d" / key / f"{vid}.pt"
    if cache.exists() and not overwrite:
        return torch.load(cache, weights_only=False)
    with _locked(cache):
        if cache.exists() and not overwrite:
            return torch.load(cache, weights_only=False)
        kp2d = stages.get("pose2d").extract(str(video_path), bbx_xys).float().cpu()
        _atomic_save(kp2d, cache)
    return kp2d


def _stage_feats(
    support_dir: Path,
    video_path: Path,
    vid: str,
    detector: str | None,
    backbone: str | None,
    bbx_xys: torch.Tensor,
    stages: _LazyStages,
    overwrite: bool,
) -> dict:
    """Per-(detector, backbone) features incl. the flip-test pass — shared by every pose2d choice:
    ``{"features", "flip_features", "flip_bbx_xys"}``."""
    from gvhmr.utils.geo.flip_utils import flip_bbx_xys
    from gvhmr.utils.video_io_utils import get_video_lwh

    key = f"{detector or 'canonical'}-{backbone or 'hmr2'}"
    cache = support_dir / "preproc_variants/_stages/feats" / key / f"{vid}.pt"
    if cache.exists() and not overwrite:
        return torch.load(cache, weights_only=False)
    with _locked(cache):
        if cache.exists() and not overwrite:
            return torch.load(cache, weights_only=False)
        width = get_video_lwh(video_path)[1]
        features = stages.get("backbone").extract_video_features(str(video_path), bbx_xys)
        flip_video = _ensure_flip_video(support_dir, video_path)
        flip_xys = flip_bbx_xys(bbx_xys, width)
        flip_features = stages.get("backbone").extract_video_features(str(flip_video), flip_xys)
        out = {
            "features": features.float().cpu(),
            "flip_features": flip_features.float().cpu(),
            "flip_bbx_xys": flip_xys,
        }
        _atomic_save(out, cache)
    return out


def _extract_stages(
    support_dir: Path,
    video_path: Path,
    canonical_xys: torch.Tensor,
    combo: tuple[str | None, str | None, str | None],
    stages: _LazyStages,
    vid: str,
    report: VariantReport,
    overwrite: bool,
):
    """Assemble one sequence's artifacts from the shared stage caches (computing what's missing)."""
    detector, pose2d, backbone = combo
    boxes = _stage_boxes(support_dir, video_path, vid, detector, canonical_xys, stages, overwrite)
    if boxes["identity_fallback"]:
        report.identity_fallbacks.append(vid)
    bbx_xys = boxes["bbx_xys"]
    kp2d = _stage_kp2d(support_dir, video_path, vid, detector, pose2d, bbx_xys, stages, overwrite)
    feats = _stage_feats(support_dir, video_path, vid, detector, backbone, bbx_xys, stages, overwrite)
    return {"bbx_xys": bbx_xys, "kp2d": kp2d, **feats}


def _resolve_stage(group: str, name: str, extra_overrides: list[str] | None = None) -> tuple[str, dict]:
    """Resolve a stage selection through the demo's Hydra config group (the same mechanism the demo's
    ``--detector/--pose2d/--backbone`` use), so every preset (``yolo26x``, ``rtmpose``, …) and its knobs
    mean exactly the same thing here. ``extra_overrides`` are stage-scoped ``--set`` knobs (e.g.
    ``pose2d.flip_test=false``); only those addressing THIS ``group`` apply, so the same list is safe to
    pass to every stage. Returns the implementation name + ctor kwargs."""
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from gvhmr.configs import register_store_gvhmr

    scoped = [o for o in (extra_overrides or []) if o.split("=", 1)[0].split(".", 1)[0] == group]
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        node = compose(config_name="demo", overrides=[f"{group}={name}", *scoped, "video_name=_"])[group]
    conf = OmegaConf.to_container(node, resolve=True)
    impl = conf.pop("name")
    return impl, {k: v for k, v in conf.items() if v is not None}


# --- dataset-specific generation (canonical schema in, canonical schema out) ------------------------


def generate_3dpw_variant(
    support_dir: Path,
    slug: str,
    detector: str | None,
    pose2d: str | None,
    backbone: str | None = None,
    overwrite: bool = False,
    stage_overrides: list[str] | None = None,
) -> VariantReport:
    """Write ``preproc_variants/<slug>/`` for 3DPW (same file schema the canonical loader reads)."""
    report = VariantReport(slug=slug)
    labels = torch.load(support_dir / "test_3dpw_gt_labels.pt", weights_only=False)
    vid2bbx = torch.load(support_dir / "preproc_test_bbx.pt", weights_only=False)

    out = support_dir / "preproc_variants" / slug
    feat_dir, flip_dir = out / "imgfeats/3dpw_test", out / "imgfeats/3dpw_test_flip"
    feat_dir.mkdir(parents=True, exist_ok=True)
    flip_dir.mkdir(parents=True, exist_ok=True)

    stages = _LazyStages(detector, pose2d, backbone, stage_overrides)
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
        r = _extract_stages(
            support_dir,
            video_path,
            vid2bbx[vid]["bbx_xys"],
            (detector, pose2d, backbone),
            stages,
            vid,
            report,
            overwrite,
        )
        new_bbx[vid] = {"bbx_xys": r["bbx_xys"]}
        new_kp2d[vid] = r["kp2d"]
        img_wh = labels[vid]["img_wh"]
        torch.save({"features": r["features"], "bbx_xys": r["bbx_xys"], "img_wh": img_wh}, feat_dir / f"{vid}.pt")
        torch.save({"features": r["flip_features"], "bbx_xys": r["flip_bbx_xys"]}, flip_dir / f"{vid}.pt")
        torch.save(new_bbx, bbx_pt)  # checkpoint progress after every sequence (resumable)
        torch.save(new_kp2d, kp2d_pt)
        report.generated.append(vid)
        Log.info(f"[ok]{vid}[/] assembled ({len(report.generated) + len(report.cached)}/{len(labels)})")
    return report


def generate_emdb_variant(
    support_dir: Path,
    slug: str,
    detector: str | None,
    pose2d: str | None,
    backbone: str | None = None,
    overwrite: bool = False,
    stage_overrides: list[str] | None = None,
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

    stages = _LazyStages(detector, pose2d, backbone, stage_overrides)
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
        r = _extract_stages(
            support_dir,
            video_path,
            labels[vid]["bbx_xys"],
            (detector, pose2d, backbone),
            stages,
            vid,
            report,
            overwrite,
        )
        overlay[vid] = {"bbx_xys": r["bbx_xys"], "kp2d": r["kp2d"], "features": r["features"]}
        torch.save({"features": r["flip_features"], "bbx_xys": r["flip_bbx_xys"]}, flip_dir / f"{vid}.pt")
        torch.save(overlay, overlay_pt)  # checkpoint progress after every sequence (resumable)
        report.generated.append(vid)
        Log.info(f"[ok]{vid}[/] assembled ({len(report.generated) + len(report.cached)}/{len(labels)})")
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
