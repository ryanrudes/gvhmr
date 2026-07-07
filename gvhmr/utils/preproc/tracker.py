from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from gvhmr.utils.assets import YOLO_CKPT
from gvhmr.utils.console import track
from gvhmr.utils.device import get_device
from gvhmr.utils.net_utils import moving_average_smooth
from gvhmr.utils.seq_utils import (
    frame_id_to_mask,
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    rearrange_by_mask,
)
from gvhmr.utils.video_io_utils import get_video_lwh

# The released default; any ultralytics-loadable detector (yolov8x … yolov11/12/26x.pt) drops in.
DEFAULT_YOLO_CKPT = YOLO_CKPT


class Tracker:
    """Person detector/tracker (ultralytics YOLO). Satisfies the ``Detector`` protocol (base.py)."""

    def __init__(self, ckpt=None, conf: float = 0.5) -> None:
        # https://docs.ultralytics.com/modes/predict/
        ckpt = ckpt or DEFAULT_YOLO_CKPT
        # Progress label from the actual checkpoint (e.g. "yolo11x") — don't hard-code a version.
        self.name = Path(str(ckpt)).stem or "yolo"
        self.yolo = YOLO(ckpt)
        self.conf = conf  # default 0.25, wham/gvhmr 0.5
        # ultralytics device convention: 0 for cuda:0, else the type string ("mps"/"cpu").
        device = get_device()
        self.device = 0 if device.type == "cuda" else device.type

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": self.device,
            "conf": self.conf,
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
            "half": self.device == 0,  # fp16 on CUDA (~1.5-2x, negligible detection change at conf=0.5)
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in track(results, total=get_video_lwh(video_path)[0], desc=f"Tracking ({self.name})"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_one_track(self, video_path):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        track_id = id_sorted[0]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

        return bbx_xyxy_one_track
