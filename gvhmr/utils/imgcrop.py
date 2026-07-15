"""Video → normalized person crops for the HMR2 ViT — the single crop implementation.

Lives here (not under ``gvhmr.utils.preproc``) so it imports with **no** heavy/optional deps: the preproc
package ``__init__`` eagerly pulls DPVO (CUDA), which cannot initialize in a forked DataLoader worker — and
joint-training datasets call this in their workers. Only cv2/numpy/torch + the dpvo-free video reader and
the vendored HMR2 crop math are imported. ``vitfeat_extractor`` re-exports ``get_batch`` from here, so the
preproc feature extractor and the crops-serving datasets share one identical crop path.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from gvhmr.network.hmr2.utils.preproc import IMAGE_MEAN, IMAGE_STD, crop_and_resize
from gvhmr.utils.video_io_utils import read_video_np


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video", start_frame=0, end_frame=-1):
    """``start_frame``/``end_frame`` trim the decode to a clip (joint-training reads one clip's crops per
    item, not the whole video); defaults read the whole video, so the preproc extractor is unchanged."""
    if path_type == "video":
        imgs = read_video_np(input_path, start_frame=start_frame, end_frame=end_frame, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256
    return imgs, bbx_xys
