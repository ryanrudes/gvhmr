"""RTMPose 2D-keypoint backend — a modern, ungated alternative to ViTPose (Tier A).

RTMPose runs via **rtmlib** on **ONNXRuntime** — no mmpose/mmcv, and the ONNX model auto-downloads on
first use. This is the concrete proof the 2D-pose stage is swappable: select ``pose2d=rtmpose`` and it
emits the same **COCO-17** ``(F, 17, 3)`` ``[x, y, conf]`` contract the trained network asserts
(``relative_transformer.py``; see ``base.py``). It's a genuinely different architecture from ViTPose
(SimCC top-down vs. heatmap) yet fits the same slot because the *output format* matches.

Install the optional dep:  ``uv sync --extra rtmpose``  (rtmlib + onnxruntime).

The default model is an RTMPose-m trained on *body7* → **17 keypoints in COCO order** (matches ViTPose).
Point ``onnx_model`` at a larger variant (rtmpose-l/x) or another COCO-17 checkpoint via the config.
"""

from __future__ import annotations

import numpy as np
import torch

from gvhmr.utils.console import track
from gvhmr.utils.device import get_device
from gvhmr.utils.video_io_utils import get_video_lwh, get_video_reader

# RTMPose-m / body7 / 256×192 — 17 COCO keypoints. rtmlib downloads + unpacks the .zip on first use.
DEFAULT_RTMPOSE_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
DEFAULT_INPUT_SIZE = (192, 256)  # (w, h) — matches the "256x192" model


class RTMPoseExtractor:
    """2D-keypoint estimator (RTMPose via rtmlib). Satisfies the ``Pose2D`` protocol; emits COCO-17."""

    def __init__(
        self,
        onnx_model: str = DEFAULT_RTMPOSE_ONNX,
        model_input_size=DEFAULT_INPUT_SIZE,
        device: str | None = None,
        tqdm_leave: bool = True,
    ) -> None:
        try:
            from rtmlib import RTMPose
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "RTMPose needs the 'rtmpose' extra — install it with `uv sync --extra rtmpose` "
                "(rtmlib + onnxruntime). On a CUDA box, `onnxruntime-gpu` enables the GPU."
            ) from e

        # onnxruntime execution provider: CUDA when available, else CPU (portable, incl. Apple Silicon).
        dev = device or ("cuda" if get_device().type == "cuda" else "cpu")
        self.pose = RTMPose(onnx_model, model_input_size=tuple(model_input_size), backend="onnxruntime", device=dev)
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, bbx_xys) -> torch.Tensor:
        """``video_path`` (str) + ``bbx_xys`` (F,3) center+size → COCO-17 ``(F, 17, 3)`` in image pixels."""
        assert isinstance(video_path, str), "RTMPose reads full frames + the per-frame bbox; pass a video path"
        bbx_xys = torch.as_tensor(bbx_xys).float()
        c, s = bbx_xys[:, :2], bbx_xys[:, 2:3]
        xyxy = torch.cat([c - s / 2, c + s / 2], dim=1).numpy()  # (F,4) square boxes, [x1,y1,x2,y2]

        reader = get_video_reader(video_path)
        out = []
        for i, img in enumerate(
            track(reader, total=get_video_lwh(video_path)[0], desc="RTMPose", leave=self.tqdm_leave)
        ):
            img_bgr = np.ascontiguousarray(img[:, :, ::-1])  # rtmlib/ONNX expect BGR (cv2 convention)
            kpts, scores = self.pose(img_bgr, bboxes=xyxy[i : i + 1])  # (1,17,2), (1,17)
            out.append(torch.from_numpy(np.concatenate([kpts[0], scores[0][:, None]], axis=-1)).float())  # (17,3)
        reader.close()
        return torch.stack(out, dim=0)  # (F, 17, 3)
