"""Single-video demo pipeline (preprocess → recover motion → render), Rich-flavored.

This is the implementation behind ``gvhmr demo``. It is import-heavy (torch, hydra,
preproc models), so the Typer command in :mod:`gvhmr.cli` lazy-imports it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from rich.panel import Panel
from rich.table import Table

from gvhmr import PROJ_ROOT
from gvhmr.configs import register_store_gvhmr
from gvhmr.model.gvhmr.gvhmr_pl_demo import DemoPL
from gvhmr.utils.console import console, progress_phase, rule, status, track
from gvhmr.utils.device import device_name, get_device, predict_device, to_device
from gvhmr.utils.geo.hmr_cam import (
    convert_f_to_K,
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from gvhmr.utils.geo.rotations import quaternion_to_matrix
from gvhmr.utils.geo_transform import apply_T_on_points, compute_cam_angvel, compute_T_ayfz2ay
from gvhmr.utils.intrinsics import load_intrinsics_file, load_intrinsics_for_undistort
from gvhmr.utils.net_utils import detach_to_cpu
from gvhmr.utils.postproc_world import compose_world_from_dust3r
from gvhmr.utils.preproc.base import make_backbone, make_detector, make_pose2d, missing_requirements
from gvhmr.utils.preproc.box_adapter import BoxAdapter
from gvhmr.utils.pylogger import Log
from gvhmr.utils.smplx_utils import make_smplx
from gvhmr.utils.video_io_utils import (
    get_video_fps,
    get_video_lwh,
    get_video_reader,
    get_video_rotation,
    get_writer,
    merge_videos_horizontal,
    read_video_np,
    save_video,
)
from gvhmr.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from gvhmr.utils.vis.renderer import get_global_cameras_static, get_ground_params_from_points
from gvhmr.utils.vis.renderer_gl import make_renderer
from gvhmr.utils.vis.skeleton import build_skeleton_mesh, resolve_joint_subset

# Body-model assets shipped inside the package (resolved relative to the repo root).
SMPLX2SMPL_PATH = PROJ_ROOT / "gvhmr/utils/body_model/smplx2smpl_sparse.pt"
SMPL_J_REGRESSOR_PATH = PROJ_ROOT / "gvhmr/utils/body_model/smpl_neutral_J_regressor.pt"

CRF = 23  # 17 is lossless, every +6 halves the mp4 size
MODEL_FPS = 30  # GVHMR is trained at 30fps (AMASS/BEDLAM downsampled to 30); inputs are resampled to it


def _sane_focal(value) -> int | None:
    try:
        f = int(round(float(str(value).split()[0])))
    except (ValueError, IndexError):
        return None
    return f if 8 <= f <= 400 else None  # sane lens range


def focal_mm_from_metadata(video_path) -> int | None:
    """Best-effort 35mm-equivalent focal length from the video's metadata.

    Phone videos store the focal in QuickTime metadata that ``ffprobe`` doesn't expose, so we
    prefer **exiftool**'s ``FocalLengthIn35mmFormat`` — and that value already **accounts for the
    lens + zoom** used at capture (e.g. iPhone reports ~15mm on the 0.5× ultrawide, ~24mm at 1×,
    ~48mm at 2×). Falls back to ffprobe tags, then ``None`` → the diagonal-FOV heuristic. A
    user-supplied ``--f_mm`` always overrides this. (A single value assumes constant zoom; the
    demo uses one intrinsics for the clip, so mid-clip zoom changes aren't tracked per-frame.)
    """
    import json
    import shutil
    import subprocess

    # 1) exiftool — the reliable source for phone QuickTime/EXIF focal length.
    if shutil.which("exiftool") is not None:
        try:
            out = subprocess.run(
                ["exiftool", "-s3", "-n", "-FocalLengthIn35mmFormat", str(video_path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if (f := _sane_focal(out.stdout.strip())) is not None:
                return f
        except (OSError, subprocess.SubprocessError):
            pass

    # 2) ffprobe stream/format tags (rare for mp4, but free).
    if shutil.which("ffprobe") is not None:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            meta = json.loads(out.stdout or "{}")
        except (OSError, ValueError, subprocess.SubprocessError):
            meta = {}
        tags = dict(meta.get("format", {}).get("tags", {}))
        for stream in meta.get("streams", []):
            tags.update(stream.get("tags", {}))
        for key, value in tags.items():
            # a 35mm-equivalent focal-length tag
            if "focal" in key.lower() and "35" in key and (f := _sane_focal(value)) is not None:
                return f
    return None


def _find_intrinsics_sidecar(video: Path) -> Path | None:
    """A ``<video-stem>.intrinsics.{json,npz}`` next to the input video, if present (auto-detected)."""
    for ext in (".intrinsics.json", ".intrinsics.npz"):
        cand = video.parent / (video.stem + ext)
        if cand.exists():
            return cand
    return None


def _stage_intrinsics(src, dest_dir: Path) -> Path:
    """Normalize a user-supplied intrinsics source into a file :func:`load_intrinsics_file` can read.

    Accepts a path (used in place), a ``dict`` (→ ``intrinsics.json``), or a 3x3 / (L,3,3) array/tensor
    treated as ``K`` (→ ``intrinsics.npz``). Returns the resolved path."""
    if isinstance(src, (str, Path)):
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"intrinsics file not found: {p}")
        return p
    dest_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(src, dict):

        def _jsonable(v):
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            return v.tolist() if isinstance(v, np.ndarray) else v

        p = dest_dir / "intrinsics.json"
        p.write_text(json.dumps({k: _jsonable(v) for k, v in src.items()}))
        return p
    arr = src.detach().cpu().numpy() if torch.is_tensor(src) else np.asarray(src)
    p = dest_dir / "intrinsics.npz"
    np.savez(p, K=arr)
    return p


def resolve_intrinsics(cfg, length: int, width: int, height: int) -> torch.Tensor:
    """Build the per-frame ``(L, 3, 3)`` full-image ``K`` for the clip.

    Precedence (high→low): an intrinsics sidecar (``cfg.intrinsics_path`` — fx/fy/cx/cy or K, per-frame
    ok) > ``cfg.f_px`` (focal in pixels) > ``cfg.f_mm`` (focal in mm) > the diagonal-FOV heuristic. The
    last two branches reproduce the historical construction byte-for-byte, so the default path stays
    golden-identical (see docs/ACCURACY.md)."""
    path = getattr(cfg, "intrinsics_path", None)
    if path:
        return load_intrinsics_file(path, length=length, width=width, height=height)
    f_px = getattr(cfg, "f_px", None)
    if f_px is not None:
        return convert_f_to_K(float(f_px), width, height).repeat(length, 1, 1)
    if cfg.f_mm is not None:
        return create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    return estimate_K(width, height).repeat(length, 1, 1)


def _apply_undistortion(cfg) -> None:
    """If the intrinsics sidecar declares lens ``distortion``, undistort the staged frames **in place** and
    swap in the corrected pinhole ``K``.

    GVHMR is pinhole-only, so the faithful way to honour a wide-angle/fisheye lens is to rectify the pixels
    the model sees (YOLO/ViTPose/HMR2 all read the staged video), not to pass coefficients the network never
    learned. Idempotent + cache-friendly: a prior run's ``intrinsics_undistorted.json`` short-circuits it.
    See docs/CAMERA_METADATA.md."""
    path = getattr(cfg, "intrinsics_path", None)
    if not path:
        return
    corrected = Path(cfg.preprocess_dir) / "intrinsics_undistorted.json"
    if corrected.exists():  # undistorted on a previous run (staged video already rectified) — reuse
        cfg.intrinsics_path = str(corrected)
        return
    length, width, height = get_video_lwh(cfg.video_path)
    spec = load_intrinsics_for_undistort(path, length=length, width=width, height=height)
    if spec is None:
        return  # no 'distortion' field → the frames are already pinhole, nothing to rectify
    K, dist, meta = spec
    alpha, per_frame = meta["alpha"], meta["per_frame"]

    def _maps(k, d):
        # Corrected pinhole matrix (same output size; alpha 0 crops the invalid border, 1 keeps all source px).
        new_k, _roi = cv2.getOptimalNewCameraMatrix(k, d, (width, height), alpha, (width, height))
        m1, m2 = cv2.initUndistortRectifyMap(k, d, None, new_k, (width, height), cv2.CV_16SC2)
        return new_k, m1, m2

    tmp = Path(cfg.video_path).with_suffix(".undist.mp4")
    reader = get_video_reader(cfg.video_path)
    writer = get_writer(str(tmp), fps=MODEL_FPS, crf=CRF)
    if per_frame:
        # A zoom changes focal AND distortion per frame — rebuild the rectification map each frame and emit a
        # per-frame corrected K (needs zoom-aware calibration; see docs/CAMERA_METADATA.md).
        new_ks = []
        for t, img in enumerate(track(reader, total=length, desc="Undistorting frames (per-frame)")):
            new_k, m1, m2 = _maps(np.ascontiguousarray(K[t]), np.ascontiguousarray(dist[t]))
            writer.write_frame(cv2.remap(img, m1, m2, cv2.INTER_LINEAR))
            new_ks.append(new_k.tolist())
        corrected_K = new_ks  # (L, 3, 3)
    else:
        new_k, m1, m2 = _maps(K, dist)  # one map, reused for every frame (the common, fixed-lens case)
        for img in track(reader, total=length, desc="Undistorting frames"):
            writer.write_frame(cv2.remap(img, m1, m2, cv2.INTER_LINEAR))
        corrected_K = new_k.tolist()  # (3, 3)
    writer.close()
    reader.close()
    os.replace(tmp, cfg.video_path)  # replaces the staged file (or a fast-path symlink) with the rectified one
    corrected.write_text(json.dumps({"K": corrected_K, "width": width, "height": height}))
    cfg.intrinsics_path = str(corrected)
    tag = "per-frame " if per_frame else ""
    Log.info(f"Undistorted staged frames [ok]({length}f)[/] → {tag}corrected pinhole K [muted](alpha={alpha})[/]")


def ensure_deps(cfg) -> None:
    """Fail fast — before any download or compute — when a chosen backend's dependency is missing,
    with the exact install command (instead of a mid-run ImportError traceback)."""
    selections = {
        "detector": cfg.detector.name,
        "pose2d": cfg.pose2d.name,
        "backbone": cfg.backbone.name,
    }
    if not cfg.static_cam:
        selections["camera"] = cfg.camera.name
        # The scene-aware cameras import from clones under third-party/ (see scripts/setup_scene_aware.sh).
        if cfg.camera.name in ("dust3r", "vggt") and not (PROJ_ROOT / "third-party" / cfg.camera.name).is_dir():
            console.print(
                f"[err]camera '{cfg.camera.name}' is not set up[/] — it runs from a clone under "
                f"[muted]{PROJ_ROOT / 'third-party'}[/].\nRun [gvhmr]scripts/setup_scene_aware.sh[/] once "
                f"(clones DUSt3R/VGGT/Depth-Anything + fetches weights), then re-run."
            )
            raise SystemExit(1)
    missing = missing_requirements(selections)
    if missing:
        from gvhmr.utils import localconfig

        lines = "\n".join(f"  • [gvhmr]{module}[/] — needed by {why}" for why, module, _ in missing)
        wanted = sorted({fix.split("--extra ")[-1].split()[0] for _, _, fix in missing if "--extra" in fix})
        scripts = [fix for _, _, fix in dict.fromkeys(missing) if "--extra" not in fix]  # e.g. setup_dpvo.sh
        if localconfig.env_table():  # a recorded env → the no-uv fix: extend the recorded extras, then sync
            needed = list(dict.fromkeys(localconfig.env_extras() + wanted))
            steps = [f"gvhmr config set extras {','.join(needed)}"] if needed != localconfig.env_extras() else []
            steps += ["gvhmr env sync"] if wanted else []
            fixes = "\n".join(f"  {s}" for s in steps + scripts)
            tail = ""
        else:
            fixes = "\n".join(f"  {fix}" for fix in dict.fromkeys(fix for _, _, fix in missing))
            tail = (
                "\n[dim](On Linux+CUDA also keep your torch extra, e.g. `uv sync --extra preproc --extra cu128` "
                "— see docs/INSTALL.md. Tip: `gvhmr config init` records the env so this becomes `gvhmr env sync`.)[/]"
            )
        console.print(f"[err]Missing optional dependencies for this run:[/]\n{lines}\nInstall with:\n{fixes}{tail}")
        raise SystemExit(1)


def ensure_assets(camera: str, want_render: bool = True, repo: str | None = None) -> None:
    """Auto-fetch the checkpoints this run needs (so `gvhmr demo` just works).

    Body models are registration-gated: :func:`assets.ensure_body_models` credential-fetches them from the
    official MPI source when a login is available (``gvhmr auth smpl`` / ``$SMPLX_USER``/``$SMPLX_PW`` for
    SMPL-X, ``$SMPL_USER``/``$SMPL_PW`` for SMPL — separate accounts), otherwise raises the actionable
    sign-up instructions.
    """
    from gvhmr.utils import assets

    need = ["gvhmr", "hmr2", "vitpose", "yolo"] + (["dpvo"] if camera == "dpvo" else [])
    todo = {n: assets.ASSETS[n] for n in need if not assets.is_present(assets.ASSETS[n])}
    if todo:
        sz = sum(a.size for a in todo.values()) / 1e9
        with status(f"Fetching {len(todo)} missing checkpoint(s) [{', '.join(todo)}] ({sz:.1f}GB)"):
            assets.fetch(todo, repo=repo)
        Log.info(f"[ok]Checkpoints ready[/] [muted]({assets.CHECKPOINT_ROOT})[/]")
    # Motion recovery needs SMPL-X; overlay rendering additionally needs SMPL. Resolve now (fail/fetch early).
    assets.ensure_body_models(smpl=want_render)


def _ctor_kwargs(node) -> dict:
    """A pluggable-stage config node → ctor kwargs: drop the ``name`` selector and any ``null``
    (⇒ let the implementation's ctor default stand). Keeps a group whose knobs match today's defaults
    byte-identical to the old direct construction."""
    d = OmegaConf.to_container(node, resolve=True)
    return {k: v for k, v in d.items() if k != "name" and v is not None}


def build_demo_cfg(
    video: Path,
    *,
    output_root: str | None,
    static_cam: bool,
    use_dpvo: bool,
    f_mm: int | None,
    verbose: bool,
    render_scale: float | None,
    f_px: float | None = None,
    intrinsics=None,
    config_overrides: list[str] | None = None,
):
    """Compose the demo ``DictConfig`` from typed CLI args and stage the input video.

    ``f_px`` (focal in pixels) and ``intrinsics`` (a sidecar path, a dict, or a 3x3 / (L,3,3) ``K``
    array/tensor) are the faithful camera-intrinsics inputs — see :func:`resolve_intrinsics` for the
    precedence. ``config_overrides`` are extra raw Hydra overrides (the pluggable-stage selections,
    ``--recipe``, and ``--set`` passthrough) assembled by :func:`run`; they are applied *after* the base
    options so a ``--set`` can override anything, and a name selector (``detector=yolo11x``) a
    ``--recipe``."""
    video = Path(video)
    assert video.exists(), f"Video not found at {video}"
    length, width, height = get_video_lwh(video)
    Log.info(f"Input: [muted]{video}[/]  (L,W,H) = ({length}, {width}, {height})")

    # Resolve the camera-intrinsics source. Precedence (high→low): explicit --intrinsics sidecar >
    # --f_px (pixels) > --f_mm (mm) > an auto-detected <video>.intrinsics.json next to the input >
    # 35mm-equiv focal from video metadata > the diagonal-FOV heuristic (in load_data_dict). Pixels and
    # the sidecar are the faithful routes — the mm path round-trips through a full-frame assumption and
    # can carry neither a real principal point nor per-frame values. See docs/ACCURACY.md.
    intrinsics_src = intrinsics
    if intrinsics is None and f_px is None and f_mm is None:
        auto = _find_intrinsics_sidecar(video)
        if auto is not None:
            intrinsics_src = auto
            Log.info(f"Camera intrinsics from sidecar [ok]{auto.name}[/]")
        else:
            f_mm = focal_mm_from_metadata(video)
            if f_mm is not None:
                Log.info(f"Focal length [ok]{f_mm}mm[/] (35mm-equiv) read from video metadata")

    with initialize_config_module(version_base="1.3", config_module="gvhmr.configs"):
        overrides = [
            f"video_name={video.stem}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
            f"use_dpvo={use_dpvo}",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")
        if f_px is not None:
            overrides.append(f"f_px={f_px}")
        if render_scale is not None:
            overrides.append(f"render_scale={render_scale}")
        if output_root is not None:
            overrides.append(f"output_root={output_root}")
        # Pluggable-stage selections / --recipe / --set (assembled by run()), applied last.
        overrides += list(config_overrides or [])
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Log.info(f"Output dir: [muted]{cfg.output_dir}[/]")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Stage the intrinsics source (a path is used in place; a dict / array is normalized into the
    # preprocess dir). load_data_dict then reads it against the *staged* frame size via resolve_intrinsics.
    if intrinsics_src is not None:
        cfg.intrinsics_path = str(_stage_intrinsics(intrinsics_src, Path(cfg.preprocess_dir)))

    # Stage the input into the output dir: re-encode at the model's 30fps (resampling if the
    # source differs — GVHMR integrates per-frame velocities, so a 60fps clip fed as-is would
    # come out at half speed and out-of-distribution) with the phone-orientation flag baked in.
    if not Path(cfg.video_path).exists():
        # `.exists()` follows symlinks, so a DANGLING link from a prior run (its target since deleted)
        # slips past the guard — clear it first so symlink_to/get_writer below don't hit FileExistsError
        # (→ copyfile-through-a-broken-link) or write through the stale link. unlink acts on the link itself.
        Path(cfg.video_path).unlink(missing_ok=True)
        src_fps = get_video_fps(video)
        resample = abs(src_fps - MODEL_FPS) > 1.5
        if not resample and get_video_rotation(video) == 0:
            # Already ~30fps and upright: skip the whole decode+re-encode — link the original in place.
            # (Higher fidelity too: models read the original pixels instead of a recompressed copy. The
            # rotation guard protects decoders that ignore the phone-orientation flag, e.g. ultralytics'.)
            try:
                Path(cfg.video_path).symlink_to(Path(video).resolve())
            except OSError:  # filesystem without symlinks → plain copy
                import shutil

                shutil.copyfile(video, cfg.video_path)
            Log.info(f"Staging skipped: input already ~{MODEL_FPS}fps and upright [muted](linked in place)[/]")
            _apply_undistortion(cfg)  # rectifies the linked file in place if the sidecar has lens distortion
            return cfg
        keep = None
        if resample:
            n_out = max(1, round(length * MODEL_FPS / src_fps))
            keep = [min(length - 1, round(j * src_fps / MODEL_FPS)) for j in range(n_out)]
            Log.info(f"Resampling [warn]{src_fps:.2f}fps[/] → {MODEL_FPS}fps ({length} → {n_out} frames)")
        reader = get_video_reader(video)
        writer = get_writer(cfg.video_path, fps=MODEL_FPS, crf=CRF)
        if keep is None:
            for img in track(reader, total=length, desc="Staging video"):
                writer.write_frame(img)
        else:
            ptr = 0
            for i, img in enumerate(track(reader, total=length, desc="Staging @30fps")):
                while ptr < len(keep) and keep[ptr] == i:  # write each kept frame (dup if upsampling)
                    writer.write_frame(img)
                    ptr += 1
        writer.close()
        reader.close()
    _apply_undistortion(cfg)  # no-op unless the sidecar declares lens distortion (also handles the cached case)
    return cfg


def flip_feat_path(cfg) -> Path:
    """Path for the HMR2 features re-extracted on the horizontally-flipped video (flip-test)."""
    return Path(cfg.paths.vit_features).with_name("vit_features_flip.pt")


@torch.no_grad()
def run_preprocess(cfg, flip_test: bool = False) -> None:
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # 0) Kick the CPU-only SimpleVO camera off FIRST, in a background thread: it needs only the video
    #    (not boxes/pose), and cv2/pycolmap release the GIL — so it genuinely overlaps the GPU stages
    #    below instead of serializing after them (~free 15-30% on moving-camera runs). GPU camera
    #    backends (dpvo/dust3r/vggt) stay serial — they'd contend for the same device.
    vo_thread = None
    _vo_result: dict = {}
    if (
        not static_cam
        and cfg.camera.name == "simplevo"
        and not Path(paths.slam).exists()
        and os.environ.get("GVHMR_NO_VO_OVERLAP", "").strip().lower() not in ("1", "true", "yes")
    ):
        import threading

        from gvhmr.utils.preproc import SimpleVO

        cam = cfg.camera
        simple_vo = SimpleVO(
            cfg.video_path, scale=cam.scale, step=cam.step, method=cam.method, f_mm=cfg.f_mm, progress=False
        )

        def _run_vo():  # stash result/exception; re-raised (or collected) at stage 4
            try:
                _vo_result["value"] = simple_vo.compute()
            except BaseException as e:  # noqa: BLE001 — surfaced to the main thread at stage 4
                _vo_result["error"] = e

        # daemon=True: if an EARLIER GPU stage raises (no person / OOM) or the user hits Ctrl-C, the
        # process fails fast instead of blocking at exit joining a still-running VO (a compiled SIFT/
        # pycolmap call can't be interrupted). The happy path join()s it at stage 4.
        vo_thread = threading.Thread(target=_run_vo, name="simplevo", daemon=True)
        vo_thread.start()
        Log.info("SimpleVO camera tracking started [muted](overlapped with GPU preprocessing)[/]")

    # 1) bbox tracking (pluggable detector; default YOLO). Config group cfg.detector — swap with --detector.
    with progress_phase("Tracking person"):
        if not Path(paths.bbx).exists():
            tracker = make_detector(cfg.detector.name, **_ctor_kwargs(cfg.detector))
            bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
            # Optional box-distribution adapter (A4): renormalize a swapped detector's framing toward the
            # released distribution the pipeline expects. Default null → identity → skipped (byte-identical).
            adapter = BoxAdapter.from_config(cfg.get("box_adapt", None))
            if not adapter.is_identity:
                bbx_xys = adapter.apply(bbx_xys).float()
                Log.info(f"box adapter: scale {adapter.scale:.3f}, shift ({adapter.dx:+.3f}, {adapter.dy:+.3f})·size")
            torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
            del tracker
        else:
            bbx_xys = torch.load(paths.bbx, weights_only=False)["bbx_xys"]
            Log.info(f"bbox cached: [muted]{paths.bbx}[/]")
        if verbose:
            video = read_video_np(video_path)
            bbx_xyxy = torch.load(paths.bbx, weights_only=False)["bbx_xyxy"]
            save_video(draw_bbx_xyxy_on_image_batch(bbx_xyxy, video), cfg.paths.bbx_xyxy_video_overlay)

    # ViTPose and the HMR2/DINOv2 backbone both build the SAME cropped batch — get_batch(video_path,
    # bbx_xys, img_ds=0.5), a deterministic decode+GaussianBlur+crop_and_resize with no RNG. When BOTH
    # stages need to run and both consume get_batch, decode+crop ONCE and feed the shared tensor to each:
    # byte-identical output (each gets exactly the tensor it would have built), one fewer full decode.
    shared_imgs = shared_bbx = None
    _need_pose, _need_feat = not Path(paths.vitpose).exists(), not Path(paths.vit_features).exists()
    if _need_pose and _need_feat and cfg.pose2d.name == "vitpose" and cfg.backbone.name in ("hmr2", "dinov2"):
        from gvhmr.utils.preproc.vitfeat_extractor import get_batch

        shared_imgs, shared_bbx = get_batch(video_path, bbx_xys, img_ds=0.5)  # (F,3,256,256), (F,3)

    # 2) 2D keypoints (pluggable 2D-pose; default ViTPose → COCO-17). Config group cfg.pose2d — swap with --pose2d.
    with progress_phase("2D pose estimation"):
        if not Path(paths.vitpose).exists():
            vitpose_extractor = make_pose2d(cfg.pose2d.name, **_ctor_kwargs(cfg.pose2d))
            if shared_imgs is not None:  # ViTPose's tensor branch takes the pre-cropped imgs + adjusted bbx
                vitpose = vitpose_extractor.extract(shared_imgs, shared_bbx)
            else:
                vitpose = vitpose_extractor.extract(video_path, bbx_xys)
            torch.save(vitpose, paths.vitpose)
            del vitpose_extractor
        else:
            vitpose = torch.load(paths.vitpose, weights_only=False)
            Log.info(f"vitpose cached: [muted]{paths.vitpose}[/]")
        if verbose:
            video = read_video_np(video_path)
            save_video(draw_coco17_skeleton_batch(video, vitpose, 0.5), paths.vitpose_video_overlay)

    # 3) image features (pluggable backbone; default HMR2 ViT). Swapping it needs a retrain — see
    #    docs/EXTENSIBILITY.md Tier B. Config group cfg.backbone — swap with --backbone.
    with progress_phase("Image features"):
        if not Path(paths.vit_features).exists():
            extractor = make_backbone(cfg.backbone.name, **_ctor_kwargs(cfg.backbone))
            if (
                shared_imgs is not None
            ):  # HMR2/DINOv2 tensor branch takes the pre-cropped imgs (bbx required but ignored)
                vit_features = extractor.extract_video_features(shared_imgs, shared_bbx)
            else:
                vit_features = extractor.extract_video_features(video_path, bbx_xys)
            torch.save(vit_features, paths.vit_features)
            del extractor
        else:
            Log.info(f"vit_features cached: [muted]{paths.vit_features}[/]")
    del shared_imgs, shared_bbx

    # 3b) flip-test: re-extract image features on the horizontally-flipped video (TTA)
    if flip_test:
        with progress_phase("Flip-test features"):
            if not flip_feat_path(cfg).exists():
                import numpy as np

                from gvhmr.utils.geo.flip_utils import flip_bbx_xys

                length, width, _ = get_video_lwh(video_path)
                flip_video = Path(cfg.preprocess_dir) / "flipped_input.mp4"
                if not flip_video.exists():
                    reader, writer = get_video_reader(video_path), get_writer(flip_video, fps=30, crf=CRF)
                    for img in track(reader, total=length, desc="Mirroring video"):
                        writer.write_frame(np.ascontiguousarray(img[:, ::-1]))
                    writer.close()
                    reader.close()
                extractor = make_backbone(cfg.backbone.name, **_ctor_kwargs(cfg.backbone))
                feat_flip = extractor.extract_video_features(str(flip_video), flip_bbx_xys(bbx_xys, width))
                torch.save(feat_flip, flip_feat_path(cfg))
                del extractor

    # 4) camera (SimpleVO / DPVO / DUSt3R-SLAM), unless static. Config group cfg.camera — swap with --camera.
    if not static_cam:
        with progress_phase("Camera motion"):
            if vo_thread is not None:
                # SimpleVO ran concurrently with the GPU stages above — join + collect (re-raise on error).
                vo_thread.join()
                if "error" in _vo_result:
                    raise _vo_result["error"]
                torch.save(_vo_result["value"], paths.slam)  # (L, 4, 4) numpy
            elif not Path(paths.slam).exists():
                cam = cfg.camera
                if cam.name == "dust3r":
                    # scene-aware, metric camera on MPS/CPU/CUDA — recovers translation (DPVO does too but
                    # is CUDA-only; SimpleVO's translation is unreliable). (L, 4, 4) metric T_w2c numpy.
                    from gvhmr.utils.preproc.dust3r_slam import run_dust3r_slam

                    with status("DUSt3R scene-aware camera tracking"):
                        result = run_dust3r_slam(
                            cfg.video_path,
                            max_depth=cam.max_depth,
                            depth_model=cam.get("depth_model", "unidepth"),  # 2x better scale, see ROADMAP A2
                        )
                    Log.info(f"DUSt3R camera: scale {result['scale']:.1f}, reconstruction conf {result['conf']:.2f}")
                    torch.save(result["T_w2c"], paths.slam)
                elif cam.name == "vggt":
                    # VGGT: one feed-forward pass for camera + depth (scale-ambiguous), Depth-Anything fixes
                    # the metric scale — same T_w2c contract as dust3r, faster/no global-alignment optimizer.
                    from gvhmr.utils.preproc.vggt_slam import run_vggt_slam

                    with status("VGGT scene-aware camera tracking"):
                        result = run_vggt_slam(
                            cfg.video_path,
                            max_depth=cam.max_depth,
                            depth_model=cam.get("depth_model", "depth_anything_v2"),
                        )
                    Log.info(f"VGGT camera: scale {result['scale']:.1f}, depth conf {result['conf']:.2f}")
                    torch.save(result["T_w2c"], paths.slam)
                elif cam.name == "dpvo":
                    from gvhmr.utils.preproc.slam import SLAMModel

                    length, width, height = get_video_lwh(cfg.video_path)
                    K_fullimg = estimate_K(width, height)
                    model = SLAMModel(
                        video_path, width, height, convert_K_to_K4(K_fullimg), buffer=cam.buffer, resize=cam.resize
                    )
                    with status("DPVO camera tracking"):
                        while model.track():
                            pass
                    torch.save(model.process(), paths.slam)  # (L, 7) numpy
                else:  # simplevo
                    from gvhmr.utils.preproc import SimpleVO

                    simple_vo = SimpleVO(
                        cfg.video_path, scale=cam.scale, step=cam.step, method=cam.method, f_mm=cfg.f_mm
                    )
                    torch.save(simple_vo.compute(), paths.slam)  # (L, 4, 4) numpy
            else:
                Log.info(f"camera cached: [muted]{paths.slam}[/]")

    Log.info(f"Preprocess done in [ok]{Log.time() - tic:.1f}s[/]")


def load_data_dict(cfg, flip_test: bool = False) -> dict:
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam, weights_only=False)
        if cfg.camera.name == "dpvo":  # (L, 7) quaternion + translation
            R_w2c = quaternion_to_matrix(torch.from_numpy(traj[:, [6, 3, 4, 5]])).mT
        else:  # simplevo / dust3r: (L, 4, 4) T_w2c (dust3r's is metric)
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    K_fullimg = resolve_intrinsics(cfg, length, width, height)
    cam_angvel = compute_cam_angvel(R_w2c)
    bbx_xys = torch.load(paths.bbx, weights_only=False)["bbx_xys"]
    kp2d = torch.load(paths.vitpose, weights_only=False)
    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": kp2d,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_imgseq": torch.load(paths.vit_features, weights_only=False),
    }
    if flip_test:
        from gvhmr.utils.geo.flip_utils import flip_bbx_xys, flip_kp2d_coco17

        data["flip_test"] = {
            "length": data["length"],
            "bbx_xys": flip_bbx_xys(bbx_xys, width),
            "kp2d": flip_kp2d_coco17(kp2d, width),
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": torch.load(flip_feat_path(cfg), weights_only=False),
        }
    return data


@torch.no_grad()
def recover_motion(
    model: DemoPL,
    data: dict,
    *,
    static_cam: bool,
    flip_test_data: dict | None = None,
    world_from_incam: bool = False,
    cam_name: str | None = None,
    slam_path=None,
) -> dict:
    """Run the model's ``predict`` + (for a moving scene-aware camera) the metric-camera world compose.

    The single source of truth for turning a ``load_data_dict`` dict into the final ``pred`` — shared by
    ``gvhmr demo`` and the :mod:`gvhmr.inference` library so the two can never drift. Returns the CPU
    ``pred`` dict (``smpl_params_global`` / ``smpl_params_incam`` / ``K_fullimg`` / ``net_outputs``).
    """
    pred = detach_to_cpu(
        model.predict(
            data,
            static_cam=static_cam,
            flip_test_data=flip_test_data,
            world_from_incam=world_from_incam,
        )
    )
    # Moving + scene-aware: replace the world trajectory with the in-cam carry through the metric camera
    # (captures traversal a following camera induces; the velocity prior misses it).
    if cam_name in ("dust3r", "vggt") and not static_cam and slam_path is not None:
        compose_world_from_dust3r(pred, torch.load(slam_path, weights_only=False))
    return pred


def render_incam(cfg, device, skeleton_overlay: bool = False, joint_indices=None, mesh: str = "smpl") -> None:
    incam_video_path = Path(cfg.paths.incam_video)
    if mesh == "smplx":  # native SMPL-X render → distinct output paths (never shadows the SMPL cache)
        incam_video_path = incam_video_path.with_stem(incam_video_path.stem + "_smplx")
    overlay_path = incam_video_path.with_stem(incam_video_path.stem + "_skeleton")
    want_mesh = not incam_video_path.exists()
    want_overlay = skeleton_overlay and not overlay_path.exists()
    if not want_mesh and not want_overlay:
        Log.info(f"in-cam video cached: [muted]{incam_video_path}[/]")
        return

    pred = torch.load(cfg.paths.hmr4d_results, weights_only=False)
    smplx = make_smplx("supermotion").to(device)
    smplx_out = smplx(**to_device(pred["smpl_params_incam"], device))
    joints_c = None
    if mesh == "smplx":  # full ~10475-vertex SMPL-X surface + SMPL-X faces (predicted params are native SMPL-X)
        faces = smplx.faces
        pred_c_verts = smplx_out.vertices
        if want_overlay:  # SMPL-X joints[:24]: first 22 body joints match SMPL; joints 22/23 differ (see docs)
            joints_c = smplx_out.joints[:, :24].cpu().numpy()
    else:  # default: project SMPL-X verts to the 6890-vertex SMPL topology (byte-identical to before)
        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=False).to(device)
        faces = make_smplx("smpl").faces
        pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
        if want_overlay:  # in-cam joints from the SMPL verts (same regressor the world frame uses)
            J_regressor = torch.load(SMPL_J_REGRESSOR_PATH, weights_only=False).to(device)
            joints_c = einsum(J_regressor, pred_c_verts, "j v, l v i -> l j i").cpu().numpy()

    length, width, height = get_video_lwh(cfg.video_path)
    render_scale = float(cfg.get("render_scale", 1.0))
    rw, rh = max(1, round(width * render_scale)), max(1, round(height * render_scale))
    K = pred["K_fullimg"][0].clone()
    K[:2] *= render_scale

    renderer = make_renderer(rw, rh, device=device, faces=faces, K=K)
    reader = get_video_reader(cfg.video_path)
    mesh_writer = get_writer(incam_video_path, fps=30, crf=CRF) if want_mesh else None
    overlay_writer = get_writer(overlay_path, fps=30, crf=CRF) if want_overlay else None
    for i, img_raw in track(enumerate(reader), total=length, desc="Rendering in-cam"):
        if render_scale != 1.0:
            img_raw = cv2.resize(img_raw, (rw, rh))
        v = pred_c_verts[i].to(device)
        if mesh_writer is not None:
            mesh_writer.write_frame(renderer.render_mesh(v, img_raw, [0.8, 0.8, 0.8]))
        if overlay_writer is not None:
            sv, sf, sc = build_skeleton_mesh(joints_c[i], joint_indices)
            overlay_writer.write_frame(
                renderer.render_mesh(v, img_raw, [0.8, 0.8, 0.8], extra_meshes=[(sv, sf, sc)], extra_on_top=True)
            )
    for w in (mesh_writer, overlay_writer):
        if w is not None:
            w.close()
    reader.close()


def render_global(
    cfg, device, skeleton: bool = False, skeleton_overlay: bool = False, joint_indices=None, mesh: str = "smpl"
) -> None:
    global_video_path = Path(cfg.paths.global_video)
    if mesh == "smplx":  # native SMPL-X render → distinct output paths
        global_video_path = global_video_path.with_stem(global_video_path.stem + "_smplx")
    skel_path = global_video_path.with_stem(global_video_path.stem + "_skeleton")  # skeleton only
    overlay_path = global_video_path.with_stem(global_video_path.stem + "_meshskel")  # mesh + skeleton
    want_mesh = not global_video_path.exists()
    want_skel = skeleton and not skel_path.exists()
    want_overlay = skeleton_overlay and not overlay_path.exists()
    if not (want_mesh or want_skel or want_overlay):
        Log.info(f"world video cached: [muted]{global_video_path}[/]")
        return

    pred = torch.load(cfg.paths.hmr4d_results, weights_only=False)
    smplx = make_smplx("supermotion").to(device)
    smplx_out = smplx(**to_device(pred["smpl_params_global"], device))

    if mesh == "smplx":  # place the native SMPL-X surface using its own joints (SMPL J-regressor can't
        faces = smplx.faces  # apply to 10475 verts) — same XZ-origin / ground / face-Z transform as SMPL
        pred_ay_verts = smplx_out.vertices
        pred_ay_joints = smplx_out.joints[:, :24]

        def move_to_start_point_face_z_smplx(verts, joints):
            verts, joints = verts.clone(), joints.clone()
            offset = joints[0, 0].clone()  # frame-0 pelvis (SMPL-X joint 0)
            offset[1] = verts[:, :, [1]].min()
            verts = verts - offset
            joints = joints - offset
            T_ay2ayfz = compute_T_ayfz2ay(joints[[0]], inverse=True)
            return apply_T_on_points(verts, T_ay2ayfz), apply_T_on_points(joints, T_ay2ayfz)

        verts_glob, joints_glob = move_to_start_point_face_z_smplx(pred_ay_verts, pred_ay_joints)
    else:  # default: SMPL topology (byte-identical to before)
        smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=False).to(device)
        faces = make_smplx("smpl").faces
        J_regressor = torch.load(SMPL_J_REGRESSOR_PATH, weights_only=False).to(device)
        pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

        def move_to_start_point_face_z(verts):
            "XZ to origin, Start from the ground, Face-Z"
            verts = verts.clone()
            offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]
            offset[1] = verts[:, :, [1]].min()
            verts = verts - offset
            T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
            return apply_T_on_points(verts, T_ay2ayfz)

        verts_glob = move_to_start_point_face_z(pred_ay_verts)
        joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")

    joints_glob_np = joints_glob.cpu().numpy()  # for the skeleton mesh builder
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(), beta=2.0, cam_height_degree=20, target_center_height=1.0
    )

    length, width, height = get_video_lwh(cfg.video_path)
    render_scale = float(cfg.get("render_scale", 1.0))
    rw, rh = max(1, round(width * render_scale)), max(1, round(height * render_scale))
    _, _, K = create_camera_sensor(rw, rh, 24)
    renderer = make_renderer(rw, rh, device=device, faces=faces, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().to(device) * 0.8

    mesh_writer = get_writer(global_video_path, fps=30, crf=CRF) if want_mesh else None
    skel_writer = get_writer(skel_path, fps=30, crf=CRF) if want_skel else None
    overlay_writer = get_writer(overlay_path, fps=30, crf=CRF) if want_overlay else None
    for i in track(range(length), desc="Rendering world"):
        renderer.create_camera(global_R[i], global_T[i])
        if mesh_writer is not None:
            mesh_writer.write_frame(renderer.render_with_ground(verts_glob[[i]], color[None]))
        if want_skel or want_overlay:
            sv, sf, sc = build_skeleton_mesh(joints_glob_np[i], joint_indices)
        if skel_writer is not None:  # skeleton only (no body), on the ground
            skel_writer.write_frame(renderer.render_with_ground(draw_body=False, extra_meshes=[(sv, sf, sc)]))
        if overlay_writer is not None:  # mesh with the skeleton on top
            overlay_writer.write_frame(
                renderer.render_with_ground(
                    verts_glob[[i]], color[None], extra_meshes=[(sv, sf, sc)], extra_on_top=True
                )
            )
    for w in (mesh_writer, skel_writer, overlay_writer):
        if w is not None:
            w.close()


def _render(
    cfg, device, skeleton: bool = False, skeleton_overlay: bool = False, joint_indices=None, mesh: str = "smpl"
) -> bool:
    """Render the overlay videos. Returns True if produced, False if skipped (missing deps).

    ``mesh='smplx'`` renders the native ~10475-vertex SMPL-X surface (opt-in) to ``_smplx``-suffixed
    paths, leaving the default SMPL-topology outputs untouched.
    """
    # pytorch3d rasterizes on CPU/CUDA (not MPS); render on CPU when the device is MPS.
    render_device = device if device.type == "cuda" else torch.device("cpu")
    paths = cfg.paths
    suffix = "_smplx" if mesh == "smplx" else ""

    def _sfx(p: str) -> str:  # append the mesh suffix to a video path's stem
        pp = Path(p)
        return str(pp.with_stem(pp.stem + suffix)) if suffix else str(pp)

    try:
        render_incam(cfg, render_device, skeleton_overlay=skeleton_overlay, joint_indices=joint_indices, mesh=mesh)
        render_global(
            cfg,
            render_device,
            skeleton=skeleton,
            skeleton_overlay=skeleton_overlay,
            joint_indices=joint_indices,
            mesh=mesh,
        )
        horiz = _sfx(paths.incam_global_horiz_video)
        if not Path(horiz).exists():
            merge_videos_horizontal([_sfx(paths.incam_video), _sfx(paths.global_video)], horiz)
        return True
    except (ImportError, AssertionError, FileNotFoundError) as e:
        from gvhmr.utils.assets import BODY_MODEL_ROOT

        Log.warning(
            f"[warn]Rendering skipped[/] ([muted]{type(e).__name__}: {e}[/])\n"
            f"  The overlay renderer needs the SMPL body model ([muted]smpl/SMPL_NEUTRAL.pkl[/] under "
            f"[muted]{BODY_MODEL_ROOT}[/], registration-gated — `gvhmr download` prints the layout) and a "
            f"working GL context (headless boxes fall back to EGL, then pytorch3d). Motion predictions are "
            f"saved either way; `gvhmr info` shows what's missing."
        )
        return False


def run(
    video: Path,
    *,
    output_root: str | None = None,
    static_cam: bool = False,
    use_dpvo: bool = False,
    slam: str | None = None,
    camera: str | None = None,
    f_mm: int | None = None,
    f_px: float | None = None,
    intrinsics=None,
    verbose: bool = False,
    render_scale: float | None = None,
    no_render: bool = False,
    flip_test: bool = False,
    incam_world_traj: bool = True,
    fast: bool = False,
    skeleton: bool = False,
    skeleton_overlay: bool = False,
    skeleton_joints: str | None = None,
    smplx: bool = False,
    detector: str | None = None,
    pose2d: str | None = None,
    backbone: str | None = None,
    detector_ckpt: str | None = None,
    pose2d_ckpt: str | None = None,
    recipe: str | None = None,
    set_overrides: list[str] | None = None,
) -> None:
    """Run the full single-video demo with a Rich, staged display."""
    mesh = "smplx" if smplx else "smpl"  # native ~10475-vert SMPL-X surface vs the default SMPL topology
    torch.set_num_threads(os.cpu_count() or 1)  # pytorch3d CPU rasterizer scales with torch threads
    joint_indices = resolve_joint_subset(skeleton_joints)  # validates the spec up front
    # Fall back to the config file's [models] defaults for any stage not chosen on the CLI (precedence:
    # CLI flag > config file > the released demo default). Manage the file with `gvhmr config`.
    from gvhmr.utils import localconfig

    detector = detector or localconfig.model_default("detector")
    pose2d = pose2d or localconfig.model_default("pose2d")
    backbone = backbone or localconfig.model_default("backbone")
    # Resolve the camera backend: --camera is the selector; --slam / --use-dpvo are deprecated aliases.
    cam = camera or slam or ("dpvo" if use_dpvo else None) or localconfig.model_default("camera")

    # Assemble the pluggable-stage config overrides (Tier A). Order matters for precedence: a --recipe
    # first (a committable bundle of choices), then the stage selectors (CLI flag or config-file default),
    # then the raw --set passthrough last — so --set wins over everything.
    config_overrides: list[str] = []
    if fast and recipe is None:
        # --fast IS --recipe fast: a @package _global_ recipe that MERGES flip-off + ByteTrack onto whatever
        # stages are selected. Raw `pose2d.flip_test=`/`detector.tracker=` overrides would ConfigComposition-
        # crash on any non-default --pose2d/--detector that lacks those keys (rtmpose, yolo11x, …).
        recipe = "fast"
    elif fast and recipe is not None:
        console.print("[warn]--fast ignored[/]: an explicit --recipe was given (add flip-off/ByteTrack via --set).")
    if recipe is not None:
        config_overrides.append(f"+recipe={recipe}")
    if detector is not None:
        config_overrides.append(f"detector={detector}")
    if pose2d is not None:
        config_overrides.append(f"pose2d={pose2d}")
    if backbone is not None:
        config_overrides.append(f"backbone={backbone}")
    if cam is not None:
        config_overrides.append(f"camera={cam}")
    if detector_ckpt is not None:
        config_overrides.append(f"detector.ckpt={detector_ckpt}")
    if pose2d_ckpt is not None:
        config_overrides.append(f"pose2d.ckpt_path={pose2d_ckpt}")
    if flip_test:  # the CLI flag forces flip-test on; a recipe/--set can also set cfg.flip_test
        config_overrides.append("flip_test=true")
    config_overrides += list(set_overrides or [])

    console.print(Panel.fit("[gvhmr]GVHMR[/] · world-grounded human motion recovery", border_style="gvhmr"))
    cfg = build_demo_cfg(
        video,
        output_root=output_root,
        static_cam=static_cam,
        use_dpvo=use_dpvo,
        f_mm=f_mm,
        f_px=f_px,
        intrinsics=intrinsics,
        verbose=verbose,
        render_scale=render_scale,
        config_overrides=config_overrides,
    )
    device = get_device()
    Log.info(f"Device: [ok]{device_name(device)}[/] ({device})")
    paths = cfg.paths
    cam_name = cfg.camera.name  # resolved camera backend (simplevo / dpvo / dust3r)
    flip_test = cfg.flip_test  # resolved from the CLI flag / a --recipe / --set

    ensure_deps(cfg)  # fail fast (with the install command) before any download or compute
    # SMPL body model is only needed for the SMPL-topology render; native SMPL-X render needs only SMPL-X.
    ensure_assets(cam_name, want_render=(not no_render and mesh == "smpl"))

    rule("Preprocess")
    run_preprocess(cfg, flip_test=flip_test)
    data = load_data_dict(cfg, flip_test=flip_test)

    rule("Recover motion")
    if not Path(paths.hmr4d_results).exists():
        with status("Loading model + checkpoint"):
            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            # The network runs on CPU by default — it's ~6.7x faster there (launch-bound); preproc + render
            # already ran/​run on the GPU. Override with $GVHMR_PREDICT_DEVICE=cuda.
            model = model.eval().to(predict_device())
        tic = Log.sync_time()
        # On a static camera, derive world translation from the in-cam motion (captures scene
        # traversal the velocity prior misses — gliding/skateboarding). No-op for moving cameras.
        world_from_incam = cfg.static_cam and incam_world_traj
        if world_from_incam:
            Log.info("Static camera: world trajectory from in-cam motion [muted](--no-incam-world-traj to disable)[/]")
        if cam_name in ("dust3r", "vggt") and not cfg.static_cam:
            Log.info(f"Moving camera: world trajectory from the [ok]{cam_name} metric camera[/] (scene-aware)")
        with status("Recovering SMPL motion" + (" (flip-test)" if flip_test else "")):
            pred = recover_motion(
                model,
                data,
                static_cam=cfg.static_cam,
                flip_test_data=data.get("flip_test"),
                world_from_incam=world_from_incam,
                cam_name=cam_name,
                slam_path=paths.slam,
            )
        torch.save(pred, paths.hmr4d_results)
        Log.info(f"Recovered [ok]{data['length'] / 30:.1f}s[/] of motion in [ok]{Log.sync_time() - tic:.2f}s[/]")
    else:
        Log.info(f"motion cached: [muted]{paths.hmr4d_results}[/]")

    rendered = True
    if not no_render:
        rule("Render" + (" · native SMPL-X" if mesh == "smplx" else ""))
        rendered = _render(
            cfg, device, skeleton=skeleton, skeleton_overlay=skeleton_overlay, joint_indices=joint_indices, mesh=mesh
        )

    # Summary
    _sfx = "_smplx" if mesh == "smplx" else ""
    horiz = Path(paths.incam_global_horiz_video)
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="muted")
    summary.add_column()
    summary.add_row("motion", str(paths.hmr4d_results))
    if rendered and not no_render:
        summary.add_row("overlay", str(horiz.with_stem(horiz.stem + _sfx)) if _sfx else str(horiz))
        gv = Path(paths.global_video)
        gv = gv.with_stem(gv.stem + _sfx) if _sfx else gv
        if skeleton:
            summary.add_row("skeleton", str(gv.with_stem(gv.stem + "_skeleton")))
        if skeleton_overlay:
            summary.add_row("mesh+skeleton", str(gv.with_stem(gv.stem + "_meshskel")))
    console.print(Panel(summary, title="[ok]Done[/]", border_style="ok", expand=False))
