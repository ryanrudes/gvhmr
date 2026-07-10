"""Build GVHMR-format H36M training artifacts from NeuralAnnot SMPL-X + Moon's COCO-format images.

This is the clean replacement for GVHMR's released MoSh-derived H36M cache: NeuralAnnot's SMPL-X pseudo-GT is
frame-aligned to the images *by construction* (same ``[action_idx][subaction_idx][frame_idx]`` indexing), so
there is NO frame-matching problem (see docs/ROADMAP.md A1 / /data/gvhmr/data/H36M/RAW_IMAGES_NOTES.md).

CPU-only. Produces, for every ``mid = S{subj}@{Action}[_1]@{cam_serial}`` (GVHMR's convention; subaction 1 →
``Action``, subaction 2 → ``Action_1``):
  - smplxpose (GVHMR ``smpl_params_glob`` layout: body_pose/global_orient/betas/transl) + cam_Rt + cam_K
  - a frame manifest (ordered image paths) — the input to ``gvhmr extract-features --backbone <name>``
  - bbx_xys (cx, cy, size) from the annotation bbox — the ``--bbx-from`` boxes that hold crops fixed

The only remaining (GPU) step for a backbone swap is running ``extract-features`` over the manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

H36M = Path("/data/gvhmr/data/H36M")
ANN = H36M / "annotations"
NA = H36M / "neuralannot_smplx"
IMAGES = H36M / "images"
OUT = H36M / "neuralannot_prepared"

TRAIN_SUBJECTS = [1, 5, 6, 7, 8]  # GVHMR H36M-train (S9/S11 are H36M's test split; available but unused)
CAM_IDX_TO_SERIAL = {1: "54138969", 2: "55011271", 3: "58860488", 4: "60457274"}


def _mid(subj: int, action_name: str, subaction_idx: int, serial: str) -> str:
    suffix = "" if subaction_idx == 1 else f"_{subaction_idx - 1}"  # sub1→"", sub2→"_1" (GVHMR convention)
    return f"S{subj}@{action_name}{suffix}@{serial}"


def _bbx_xys(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x + w / 2, y + h / 2, float(max(w, h))]  # (cx, cy, size)


def prepare():
    OUT.mkdir(exist_ok=True)
    smplxpose: dict = {}
    manifest: dict = {}
    bbx: dict = {}
    stats = {"mids": 0, "frames": 0, "missing_gt": 0, "missing_img": 0}

    for subj in TRAIN_SUBJECTS:
        data = json.load(open(ANN / f"Human36M_subject{subj}_data.json"))
        cams = json.load(open(ANN / f"Human36M_subject{subj}_camera.json"))
        na = json.load(open(NA / f"Human36M_subject{subj}_SMPLX_NeuralAnnot.json"))
        annos = {a["image_id"]: a["bbox"] for a in data["annotations"]}
        a2name = {im["action_idx"]: im["action_name"] for im in data["images"]}

        # group image rows by (action, subaction, cam), ordered by frame_idx
        groups: dict = {}
        for im in data["images"]:
            groups.setdefault((im["action_idx"], im["subaction_idx"], im["cam_idx"]), []).append(im)
        for rows in groups.values():
            rows.sort(key=lambda im: im["frame_idx"])

        for (act, sub, cam), rows in groups.items():
            serial = CAM_IDX_TO_SERIAL[cam]
            mid = _mid(subj, a2name[act], sub, serial)
            na_seq = na.get(str(act), {}).get(str(sub), {})
            paths, bp, go, be, tr, boxes = [], [], [], [], [], []
            for im in rows:
                fkey = str(im["frame_idx"])
                if fkey not in na_seq:  # NeuralAnnot occasionally drops a frame
                    stats["missing_gt"] += 1
                    continue
                img = IMAGES / im["file_name"]
                if not img.exists():
                    stats["missing_img"] += 1
                    continue
                g = na_seq[fkey]
                paths.append(str(img))
                go.append(g["root_pose"])
                bp.append(g["body_pose"])
                be.append(g["shape"])
                tr.append(g["trans"])
                boxes.append(_bbx_xys(annos[im["id"]]))
            if len(paths) < 2:
                continue
            fx, fy = cams[str(cam)]["f"]
            cx, cy = cams[str(cam)]["c"]
            Kmat = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            R = torch.tensor(cams[str(cam)]["R"])
            t = torch.tensor(cams[str(cam)]["t"]).reshape(3) / 1000.0
            Rt = torch.eye(4)
            Rt[:3, :3] = R
            Rt[:3, 3] = t
            smplxpose[mid] = {
                "smpl_params_glob": {
                    "body_pose": torch.tensor(bp),
                    "global_orient": torch.tensor(go),
                    "betas": torch.tensor(be),
                    "transl": torch.tensor(tr),
                },
                "cam_Rt": Rt,
                "cam_K": Kmat,
            }
            manifest[mid] = paths
            bbx[mid] = torch.tensor(boxes)
            stats["mids"] += 1
            stats["frames"] += len(paths)

    torch.save(smplxpose, OUT / "smplxpose_neuralannot.pt")
    torch.save(bbx, OUT / "bbx_xys_neuralannot.pt")
    json.dump(manifest, open(OUT / "frames_manifest.json", "w"))
    print(f"[prepare_h36m_neuralannot] wrote {stats['mids']} mids, {stats['frames']} frames → {OUT}")
    print(f"  missing GT frames: {stats['missing_gt']}, missing images: {stats['missing_img']}")
    return smplxpose, manifest, bbx, stats


if __name__ == "__main__":
    prepare()
