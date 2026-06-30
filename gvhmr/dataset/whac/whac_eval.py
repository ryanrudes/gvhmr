"""WHAC-A-Mole → :class:`WorldSeqGT` adapter for ``tools/eval/eval_world.py``.

WHAC-A-Mole (Yin et al., 2024) is **synthetic** with *exact* camera + SMPL-X ground truth — the clean
control that isolates whether the composition math is right (the ``gt-cam`` mode), with zero SLAM/depth
error. Download (HuggingFace, ~50 GB images + npz): https://huggingface.co/datasets/waanqii/WHAC-A-Mole
(see ``scripts/setup_eval_datasets.sh``).

Annotations are **HumanData** ``.npz`` (the SMPLCap/mmhuman3d standard): a pickled dict with
``image_path`` (N,), an ``smplx`` sub-dict (``global_orient``, ``body_pose``, ``transl``, ``betas``),
camera intrinsics under ``meta``/``cam_param`` (``focal_length``, ``principal_point``), and — because
WHAC is world-grounded — per-frame world↔camera extrinsics, plus ``track_id`` for the multi-person scenes.

⚠️ SCHEMA-PENDING: the one thing not verifiable without the (large, gated) download is whether ``smplx``
is stored in **world** or **per-frame camera** coordinates, and the exact extrinsics key. Both paths are
implemented; select with ``$WHAC_FRAME`` (``world`` default | ``camera``). Run with ``--probe`` first to
dump the actual keys, then confirm/flip the switch. Procrustes per-chunk alignment forgives a constant
world-frame offset but **not** a camera-frame trajectory left unlifted — hence the explicit switch.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from gvhmr.utils.eval.world_eval import WorldSeqGT, resample_indices
from gvhmr.utils.pylogger import Log

# Common HumanData extrinsics spellings (world→camera), checked in order.
_EXTRINSIC_KEYS = ("cam_pose", "world2cam", "extrinsics", "T_w2c", "cam_param")


def _default_root() -> Path:
    base = os.environ.get("GVHMR_DATA", str(Path.home() / "Datasets/GVHMR"))
    return Path(base) / "whac"


def _load_humandata(npz_path: Path) -> dict:
    raw = np.load(npz_path, allow_pickle=True)
    # HumanData packs everything under a single object array, or as top-level npz members.
    if "__data_len__" in raw or "smplx" in raw:
        return {k: raw[k] for k in raw.files}
    if len(raw.files) == 1:  # some dumps wrap the whole dict in one entry
        return raw[raw.files[0]].item()
    return {k: raw[k] for k in raw.files}


def probe(npz_path: Path) -> None:
    """Print the actual key tree of a HumanData npz so the schema switches can be confirmed."""
    d = _load_humandata(Path(npz_path))
    Log.info(f"[gvhmr]WHAC probe[/] {npz_path}")
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            inner = v.item()
            if isinstance(inner, dict):
                for kk, vv in inner.items():
                    Log.info(f"  {k}.{kk}: {getattr(vv, 'shape', type(vv).__name__)}")
                continue
        Log.info(f"  {k}: {getattr(v, 'shape', type(v).__name__)}")


def _as_dict(v):
    return v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1 else v


def _body_pose_63(bp: np.ndarray, n: int) -> torch.Tensor:
    return torch.from_numpy(bp.reshape(n, -1)[:, :63].astype(np.float32))


def _extrinsics_w2c(d: dict, n: int) -> torch.Tensor | None:
    for key in _EXTRINSIC_KEYS:
        if key in d:
            val = _as_dict(d[key])
            if isinstance(val, dict) and "R" in val and "T" in val:  # cam_param: R (.,3,3) + T (.,3)
                R = np.asarray(val["R"], np.float32).reshape(-1, 3, 3)
                T = np.asarray(val["T"], np.float32).reshape(-1, 3)
                R = np.broadcast_to(R, (n, 3, 3))
                T = np.broadcast_to(T, (n, 3))
                M = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
                M[:, :3, :3], M[:, :3, 3] = R, T
                return torch.from_numpy(M)
            arr = np.asarray(val, np.float32)
            if arr.shape[-2:] == (4, 4):
                return torch.from_numpy(np.broadcast_to(arr.reshape(-1, 4, 4), (n, 4, 4)).copy())
    return None


def _intrinsics_K(meta: dict) -> torch.Tensor:
    f = np.asarray(meta.get("focal_length", [1000.0, 1000.0]), np.float32).reshape(-1)
    pp = np.asarray(meta.get("principal_point", [0.0, 0.0]), np.float32).reshape(-1)
    fx, fy = (f[0], f[1]) if f.size >= 2 else (f[0], f[0])
    cx, cy = (pp[0], pp[1]) if pp.size >= 2 else (0.0, 0.0)
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)


def _lift_to_world(smpl_params: dict, T_w2c: torch.Tensor) -> dict:
    """Camera-frame SMPL-X → world: rotate orientation by R_c2w (translation handled via FK + recenter)."""
    from gvhmr.utils.geo.rotations import axis_angle_to_matrix, matrix_to_axis_angle

    R_c2w = torch.linalg.inv(T_w2c)[:, :3, :3]
    go = axis_angle_to_matrix(smpl_params["global_orient"])
    smpl_params = dict(smpl_params)
    smpl_params["global_orient"] = matrix_to_axis_angle(R_c2w @ go)
    smpl_params["transl"] = torch.einsum("fij,fj->fi", R_c2w, smpl_params["transl"]) + torch.linalg.inv(T_w2c)[:, :3, 3]
    return smpl_params


def iter_sequences(data_root: str | None = None, limit: int | None = None):
    root = Path(data_root) if data_root else _default_root()
    assert root.exists(), f"WHAC-A-Mole root not found at {root} (set --data-root or $GVHMR_DATA)"
    npzs = sorted(root.rglob("*.npz"))
    assert npzs, f"no *.npz under {root} — run scripts/setup_eval_datasets.sh"
    if limit:
        npzs = npzs[:limit]
    frame_mode = os.environ.get("WHAC_FRAME", "world")

    for npz_path in npzs:
        d = _load_humandata(npz_path)
        vid = npz_path.stem
        smplx = _as_dict(d.get("smplx", d.get("smpl", {})))
        meta = _as_dict(d.get("meta", {}))
        image_path = [str(p) for p in np.asarray(d["image_path"]).reshape(-1)]
        n = len(image_path)

        T_w2c = _extrinsics_w2c(d, n)
        if T_w2c is None:
            Log.warning(f"[warn]{vid}[/]: no world→cam extrinsics among {_EXTRINSIC_KEYS}; skipping "
                        f"(run with --probe to inspect)")  # fmt: skip
            continue

        go = np.asarray(smplx["global_orient"], np.float32).reshape(n, 3)
        smpl_params = {
            "global_orient": torch.from_numpy(go),
            "body_pose": _body_pose_63(np.asarray(smplx["body_pose"], np.float32), n),
            "transl": torch.from_numpy(np.asarray(smplx["transl"], np.float32).reshape(n, 3)),
            "betas": torch.from_numpy(np.asarray(smplx["betas"], np.float32).reshape(n, -1)[:, :10]),
        }
        if frame_mode == "camera":
            smpl_params = _lift_to_world(smpl_params, T_w2c)

        keep = resample_indices(n, float(meta.get("fps", 30)))
        keep_t = torch.tensor(keep)
        out_mp4 = _stage(root, vid, image_path, keep)
        if out_mp4 is None:
            continue

        yield WorldSeqGT(
            vid=f"whac_{vid}",
            frames_mp4=out_mp4,
            smpl_params={k: v[keep_t] for k, v in smpl_params.items()},
            body_type="smplx",
            gender="neutral",
            T_w2c=T_w2c[keep_t],
            K_fullimg=_intrinsics_K(meta),
            fps=30.0,
            length=len(keep),
            mask=None,
        )


def _stage(root: Path, vid: str, image_path: list[str], keep: list[int]) -> Path | None:
    import cv2

    from gvhmr.utils.video_io_utils import get_writer

    out_mp4 = root / "_staged30" / f"{vid}.mp4"
    if out_mp4.exists():
        return out_mp4
    # image_path entries are relative to the dataset root (HumanData convention).
    resolved = [(root / p) if not Path(p).is_absolute() else Path(p) for p in image_path]
    if not resolved[0].exists():  # fall back: search by basename
        hit = next(root.rglob(Path(image_path[0]).name), None)
        if hit is None:
            Log.warning(f"[warn]{vid}[/]: images not found (first={image_path[0]}); skipping")
            return None
        base = hit.parent
        resolved = [base / Path(p).name for p in image_path]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = get_writer(str(out_mp4), fps=30, crf=23)
    for i in keep:
        img = cv2.imread(str(resolved[i]))
        if img is None:
            writer.close()
            raise FileNotFoundError(f"missing WHAC frame {resolved[i]}")
        writer.write_frame(img[:, :, ::-1])
    writer.close()
    return out_mp4
