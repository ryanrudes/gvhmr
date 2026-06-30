"""SLOPER4D → :class:`WorldSeqGT` adapter for ``tools/eval/eval_world.py``.

SLOPER4D (Dai et al., CVPR 2023) is real outdoor capture with 200 m–1.3 km world-frame trajectories —
the long-traversal, following-camera regime the velocity prior misses. It is **public, no registration**:
http://www.lidarhumanmotion.net/data-sloper4d/ (see ``scripts/setup_eval_datasets.sh``).

Each sequence ships a ``*_labels.pkl`` with (verbatim key names from the official ``src/data_loader.py``):
  ``RGB_info``    → ``fps``, ``width``, ``height``, ``intrinsics``
  ``RGB_frames``  → ``file_basename`` (image names), ``cam_pose`` (n,4,4) **world→camera**
  ``second_person`` → ``opt_pose`` (n,72 axis-angle = global_orient[:3] + body_pose[3:]),
                      ``opt_trans`` (n,3), ``beta`` (n,10), ``gender``   ← the filmed subject, SMPL

GT body & camera are subsampled to 30 fps (the model's rate) with a shared index map so they stay frame-
aligned with the prediction. The single tracked subject matches SLOPER4D's single ``second_person``.

NOTE: SLOPER4D's on-disk layout (image-folder vs. packed video, exact ``intrinsics`` ordering) has varied
across releases; this adapter handles the documented forms and raises a clear error otherwise — worth a
sanity check on the first real sequence.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch

from gvhmr.utils.eval.world_eval import WorldSeqGT, resample_indices
from gvhmr.utils.pylogger import Log


def _default_root() -> Path:
    base = os.environ.get("GVHMR_DATA", str(Path.home() / "Datasets/GVHMR"))
    return Path(base) / "sloper4d"


def _intrinsics_to_K(intr) -> torch.Tensor:
    a = np.asarray(intr, dtype=np.float32).reshape(-1)
    if a.size == 4:  # [fx, fy, cx, cy]
        fx, fy, cx, cy = a
    elif a.size == 9:  # row-major 3x3
        K = a.reshape(3, 3)
        return torch.from_numpy(K).float()
    else:
        raise ValueError(f"Unrecognised SLOPER4D intrinsics of size {a.size}: expected 4 or 9")
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)


def _find_image_dir(pkl_path: Path, first_basename: str) -> Path | None:
    """Locate the folder holding ``file_basename`` images near the labels pkl."""
    for parent in (pkl_path.parent, *pkl_path.parent.parents[:3]):
        for cand in parent.rglob(first_basename):
            return cand.parent
    return None


def _stage_video(image_dir: Path, basenames: list[str], keep: list[int], out_mp4: Path) -> None:
    import cv2

    from gvhmr.utils.video_io_utils import get_writer

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = get_writer(str(out_mp4), fps=30, crf=23)
    for i in keep:
        img = cv2.imread(str(image_dir / basenames[i]))
        if img is None:
            writer.close()
            raise FileNotFoundError(f"missing SLOPER4D frame {image_dir / basenames[i]}")
        writer.write_frame(img[:, :, ::-1])  # BGR→RGB
    writer.close()


def iter_sequences(data_root: str | None = None, limit: int | None = None):
    root = Path(data_root) if data_root else _default_root()
    assert root.exists(), f"SLOPER4D root not found at {root} (set --data-root or $GVHMR_DATA)"
    pkls = sorted(root.rglob("*_labels.pkl"))
    assert pkls, f"no *_labels.pkl under {root} — run scripts/setup_eval_datasets.sh"
    if limit:
        pkls = pkls[:limit]

    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        vid = pkl_path.stem.replace("_labels", "")
        info, frames, person = d["RGB_info"], d["RGB_frames"], d["second_person"]
        src_fps = float(info.get("fps", 20))

        basenames = list(frames["file_basename"])
        cam_pose = np.asarray(frames["cam_pose"], dtype=np.float32)  # (n,4,4) world→cam
        pose = np.asarray(person["opt_pose"], dtype=np.float32)  # (n,72)
        trans = np.asarray(person["opt_trans"], dtype=np.float32)  # (n,3)
        beta = np.asarray(person["beta"], dtype=np.float32)  # (n,10)
        n = min(len(basenames), len(cam_pose), len(pose), len(trans))
        if not (len(basenames) == len(cam_pose) == len(pose)):
            Log.warning(f"[warn]{vid}[/]: misaligned lengths (imgs={len(basenames)}, cam={len(cam_pose)}, "
                        f"pose={len(pose)}) — truncating to {n}")  # fmt: skip

        keep = resample_indices(n, src_fps)  # native → 30 fps
        image_dir = _find_image_dir(pkl_path, basenames[0])
        if image_dir is None:
            Log.warning(f"[warn]{vid}[/]: image folder not found near {pkl_path}; skipping")
            continue
        out_mp4 = root / "_staged30" / f"{vid}.mp4"
        if not out_mp4.exists():
            _stage_video(image_dir, basenames, keep, out_mp4)

        keep_t = torch.tensor(keep)
        pose_t = torch.from_numpy(pose[:n])[keep_t]
        betas = torch.from_numpy(beta[:n])[keep_t] if beta.ndim == 2 else torch.from_numpy(beta).expand(len(keep), 10)
        smpl_params = {
            "global_orient": pose_t[:, :3],
            "body_pose": pose_t[:, 3:72],
            "transl": torch.from_numpy(trans[:n])[keep_t],
            "betas": betas,
        }
        yield WorldSeqGT(
            vid=f"sloper4d_{vid}",
            frames_mp4=out_mp4,
            smpl_params=smpl_params,
            body_type="smpl",
            gender=str(person.get("gender", "neutral")),
            T_w2c=torch.from_numpy(cam_pose[:n])[keep_t],
            K_fullimg=_intrinsics_to_K(info["intrinsics"]),
            fps=30.0,
            length=len(keep),
            mask=None,
        )
