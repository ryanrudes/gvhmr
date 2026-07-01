"""Single-video demo pipeline (preprocess → recover motion → render), Rich-flavored.

This is the implementation behind ``gvhmr demo``. It is import-heavy (torch, hydra,
preproc models), so the Typer command in :mod:`gvhmr.cli` lazy-imports it.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import hydra
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from rich.panel import Panel
from rich.table import Table

from gvhmr import PROJ_ROOT
from gvhmr.configs import register_store_gvhmr
from gvhmr.model.gvhmr.gvhmr_pl_demo import DemoPL
from gvhmr.utils.console import console, rule, status, track
from gvhmr.utils.device import device_name, get_device, to_device
from gvhmr.utils.geo.hmr_cam import convert_K_to_K4, create_camera_sensor, estimate_K, get_bbx_xys_from_xyxy
from gvhmr.utils.geo.rotations import quaternion_to_matrix
from gvhmr.utils.geo_transform import apply_T_on_points, compute_cam_angvel, compute_T_ayfz2ay
from gvhmr.utils.net_utils import detach_to_cpu
from gvhmr.utils.postproc_world import compose_world_from_dust3r
from gvhmr.utils.preproc import SimpleVO
from gvhmr.utils.preproc.base import make_backbone, make_detector, make_pose2d
from gvhmr.utils.pylogger import Log
from gvhmr.utils.smplx_utils import make_smplx
from gvhmr.utils.video_io_utils import (
    get_video_fps,
    get_video_lwh,
    get_video_reader,
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


def ensure_assets(camera: str) -> None:
    """Auto-fetch the checkpoints this run needs (so `gvhmr demo` just works); body models are gated."""
    from gvhmr.utils import assets

    need = ["gvhmr", "hmr2", "vitpose", "yolo"] + (["dpvo"] if camera == "dpvo" else [])
    todo = {n: assets.ASSETS[n] for n in need if not assets.is_present(assets.ASSETS[n])}
    if todo:
        sz = sum(a.size for a in todo.values()) / 1e9
        with status(f"Fetching {len(todo)} missing checkpoint(s) [{', '.join(todo)}] ({sz:.1f}GB)"):
            assets.fetch(todo)
        Log.info(f"[ok]Checkpoints ready[/] [muted]({assets.CHECKPOINT_ROOT})[/]")
    if not (assets.BODY_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz").exists():
        raise FileNotFoundError(
            f"SMPL-X body model not found under {assets.BODY_MODEL_ROOT}. These are registration-gated: "
            f"sign up at https://smpl-x.is.tue.mpg.de/ + https://smpl.is.tue.mpg.de/, then place them there "
            f"(or set $GVHMR_BODY_MODELS). `gvhmr download` prints the exact layout."
        )


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
    config_overrides: list[str] | None = None,
):
    """Compose the demo ``DictConfig`` from typed CLI args and stage the input video.

    ``config_overrides`` are extra raw Hydra overrides (the pluggable-stage selections, ``--recipe``,
    and ``--set`` passthrough) assembled by :func:`run`; they are applied *after* the base options so a
    ``--set`` can override anything, and a name selector (``detector=yolo11``) overrides a ``--recipe``."""
    video = Path(video)
    assert video.exists(), f"Video not found at {video}"
    length, width, height = get_video_lwh(video)
    Log.info(f"Input: [muted]{video}[/]  (L,W,H) = ({length}, {width}, {height})")

    # When no focal is given, try to read the true one from the video metadata (improves the
    # world-frame scale); falls back to the diagonal-FOV heuristic in load_data_dict if absent.
    if f_mm is None:
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

    # Stage the input into the output dir: re-encode at the model's 30fps (resampling if the
    # source differs — GVHMR integrates per-frame velocities, so a 60fps clip fed as-is would
    # come out at half speed and out-of-distribution) with the phone-orientation flag baked in.
    if not Path(cfg.video_path).exists():
        src_fps = get_video_fps(video)
        resample = abs(src_fps - MODEL_FPS) > 1.5
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

    # 1) bbox tracking (pluggable detector; default YOLO). Config group cfg.detector — swap with --detector.
    if not Path(paths.bbx).exists():
        tracker = make_detector(cfg.detector.name, **_ctor_kwargs(cfg.detector))
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx, weights_only=False)["bbx_xys"]
        Log.info(f"bbox cached: [muted]{paths.bbx}[/]")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx, weights_only=False)["bbx_xyxy"]
        save_video(draw_bbx_xyxy_on_image_batch(bbx_xyxy, video), cfg.paths.bbx_xyxy_video_overlay)

    # 2) 2D keypoints (pluggable 2D-pose; default ViTPose → COCO-17). Config group cfg.pose2d — swap with --pose2d.
    if not Path(paths.vitpose).exists():
        vitpose_extractor = make_pose2d(cfg.pose2d.name, **_ctor_kwargs(cfg.pose2d))
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
    if not Path(paths.vit_features).exists():
        extractor = make_backbone(cfg.backbone.name, **_ctor_kwargs(cfg.backbone))
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"vit_features cached: [muted]{paths.vit_features}[/]")

    # 3b) flip-test: re-extract image features on the horizontally-flipped video (TTA)
    if flip_test and not flip_feat_path(cfg).exists():
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
        if not Path(paths.slam).exists():
            cam = cfg.camera
            if cam.name == "dust3r":
                # scene-aware, metric camera on MPS/CPU/CUDA — recovers translation (DPVO does too but
                # is CUDA-only; SimpleVO's translation is unreliable). (L, 4, 4) metric T_w2c numpy.
                from gvhmr.utils.preproc.dust3r_slam import run_dust3r_slam

                with status("DUSt3R scene-aware camera tracking"):
                    result = run_dust3r_slam(cfg.video_path, max_depth=cam.max_depth)
                Log.info(f"DUSt3R camera: scale {result['scale']:.1f}, reconstruction conf {result['conf']:.2f}")
                torch.save(result["T_w2c"], paths.slam)
            elif cam.name == "vggt":
                # VGGT: one feed-forward pass for camera + depth (scale-ambiguous), Depth-Anything fixes
                # the metric scale — same T_w2c contract as dust3r, faster/no global-alignment optimizer.
                from gvhmr.utils.preproc.vggt_slam import run_vggt_slam

                with status("VGGT scene-aware camera tracking"):
                    result = run_vggt_slam(cfg.video_path, max_depth=cam.max_depth)
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
                simple_vo = SimpleVO(cfg.video_path, scale=cam.scale, step=cam.step, method=cam.method, f_mm=cfg.f_mm)
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
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
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


def render_incam(cfg, device, skeleton_overlay: bool = False, joint_indices=None) -> None:
    incam_video_path = Path(cfg.paths.incam_video)
    overlay_path = incam_video_path.with_stem(incam_video_path.stem + "_skeleton")
    want_mesh = not incam_video_path.exists()
    want_overlay = skeleton_overlay and not overlay_path.exists()
    if not want_mesh and not want_overlay:
        Log.info(f"in-cam video cached: [muted]{incam_video_path}[/]")
        return

    pred = torch.load(cfg.paths.hmr4d_results, weights_only=False)
    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=False).to(device)
    faces_smpl = make_smplx("smpl").faces

    smplx_out = smplx(**to_device(pred["smpl_params_incam"], device))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
    joints_c = None
    if want_overlay:  # in-cam joints from the SMPL verts (same regressor the world frame uses)
        J_regressor = torch.load(SMPL_J_REGRESSOR_PATH, weights_only=False).to(device)
        joints_c = einsum(J_regressor, pred_c_verts, "j v, l v i -> l j i").cpu().numpy()

    length, width, height = get_video_lwh(cfg.video_path)
    render_scale = float(cfg.get("render_scale", 1.0))
    rw, rh = max(1, round(width * render_scale)), max(1, round(height * render_scale))
    K = pred["K_fullimg"][0].clone()
    K[:2] *= render_scale

    renderer = make_renderer(rw, rh, device=device, faces=faces_smpl, K=K)
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


def render_global(cfg, device, skeleton: bool = False, skeleton_overlay: bool = False, joint_indices=None) -> None:
    global_video_path = Path(cfg.paths.global_video)
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
    smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=False).to(device)
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(SMPL_J_REGRESSOR_PATH, weights_only=False).to(device)

    smplx_out = smplx(**to_device(pred["smpl_params_global"], device))
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
    renderer = make_renderer(rw, rh, device=device, faces=faces_smpl, K=K)
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


def _render(cfg, device, skeleton: bool = False, skeleton_overlay: bool = False, joint_indices=None) -> bool:
    """Render the overlay videos. Returns True if produced, False if skipped (missing deps)."""
    # pytorch3d rasterizes on CPU/CUDA (not MPS); render on CPU when the device is MPS.
    render_device = device if device.type == "cuda" else torch.device("cpu")
    paths = cfg.paths
    try:
        render_incam(cfg, render_device, skeleton_overlay=skeleton_overlay, joint_indices=joint_indices)
        render_global(
            cfg, render_device, skeleton=skeleton, skeleton_overlay=skeleton_overlay, joint_indices=joint_indices
        )
        if not Path(paths.incam_global_horiz_video).exists():
            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
        return True
    except (ImportError, AssertionError, FileNotFoundError) as e:
        Log.warning(
            f"[warn]Rendering skipped[/] — needs pytorch3d (the render extra) + the SMPL body model. "
            f"Predictions are saved. ([muted]{type(e).__name__}: {e}[/])"
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
    verbose: bool = False,
    render_scale: float | None = None,
    no_render: bool = False,
    flip_test: bool = False,
    incam_world_traj: bool = True,
    skeleton: bool = False,
    skeleton_overlay: bool = False,
    skeleton_joints: str | None = None,
    detector: str | None = None,
    pose2d: str | None = None,
    backbone: str | None = None,
    detector_ckpt: str | None = None,
    pose2d_ckpt: str | None = None,
    recipe: str | None = None,
    set_overrides: list[str] | None = None,
) -> None:
    """Run the full single-video demo with a Rich, staged display."""
    torch.set_num_threads(os.cpu_count() or 1)  # pytorch3d CPU rasterizer scales with torch threads
    joint_indices = resolve_joint_subset(skeleton_joints)  # validates the spec up front
    # Resolve the camera backend: --camera is the selector; --slam / --use-dpvo are deprecated aliases.
    cam = camera or slam or ("dpvo" if use_dpvo else None)

    # Assemble the pluggable-stage config overrides (Tier A). Order matters for precedence: a --recipe
    # first (a committable bundle of choices), then explicit name selectors and weight tweaks, then the
    # raw --set passthrough last — so --set wins over everything and --detector wins over a --recipe.
    config_overrides: list[str] = []
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
        verbose=verbose,
        render_scale=render_scale,
        config_overrides=config_overrides,
    )
    device = get_device()
    Log.info(f"Device: [ok]{device_name(device)}[/] ({device})")
    paths = cfg.paths
    cam_name = cfg.camera.name  # resolved camera backend (simplevo / dpvo / dust3r)
    flip_test = cfg.flip_test  # resolved from the CLI flag / a --recipe / --set

    ensure_assets(cam_name)  # auto-fetch missing checkpoints (gated body models raise a clear error)

    rule("Preprocess")
    run_preprocess(cfg, flip_test=flip_test)
    data = load_data_dict(cfg, flip_test=flip_test)

    rule("Recover motion")
    if not Path(paths.hmr4d_results).exists():
        with status("Loading model + checkpoint"):
            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().to(device)
        tic = Log.sync_time()
        # On a static camera, derive world translation from the in-cam motion (captures scene
        # traversal the velocity prior misses — gliding/skateboarding). No-op for moving cameras.
        world_from_incam = cfg.static_cam and incam_world_traj
        if world_from_incam:
            Log.info("Static camera: world trajectory from in-cam motion [muted](--no-incam-world-traj to disable)[/]")
        with status("Recovering SMPL motion" + (" (flip-test)" if flip_test else "")):
            pred = detach_to_cpu(
                model.predict(
                    data,
                    static_cam=cfg.static_cam,
                    flip_test_data=data.get("flip_test"),
                    world_from_incam=world_from_incam,
                )
            )
        # Moving + scene-aware: replace the world trajectory with the in-cam carry through the DUSt3R
        # metric camera (captures traversal a following camera induces; the velocity prior misses it).
        if cam_name in ("dust3r", "vggt") and not cfg.static_cam:
            Log.info(f"Moving camera: world trajectory from the [ok]{cam_name} metric camera[/] (scene-aware)")
            compose_world_from_dust3r(pred, torch.load(paths.slam, weights_only=False))
        torch.save(pred, paths.hmr4d_results)
        Log.info(f"Recovered [ok]{data['length'] / 30:.1f}s[/] of motion in [ok]{Log.sync_time() - tic:.2f}s[/]")
    else:
        Log.info(f"motion cached: [muted]{paths.hmr4d_results}[/]")

    rendered = True
    if not no_render:
        rule("Render")
        rendered = _render(
            cfg, device, skeleton=skeleton, skeleton_overlay=skeleton_overlay, joint_indices=joint_indices
        )

    # Summary
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="muted")
    summary.add_column()
    summary.add_row("motion", str(paths.hmr4d_results))
    if rendered and not no_render:
        summary.add_row("overlay", str(paths.incam_global_horiz_video))
        gv = Path(paths.global_video)
        if skeleton:
            summary.add_row("skeleton", str(gv.with_stem(gv.stem + "_skeleton")))
        if skeleton_overlay:
            summary.add_row("mesh+skeleton", str(gv.with_stem(gv.stem + "_meshskel")))
    console.print(Panel(summary, title="[ok]Done[/]", border_style="ok", expand=False))
