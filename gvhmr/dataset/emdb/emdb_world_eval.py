"""EMDB-2 adapter for the trainer-free world-frame eval (``tools/eval/eval_world.py``).

EMDB-2 is *the* field-standard global benchmark (it's what GVHMR reports), so it is the right arbiter
for "which world-grounding stack is better" — a question ROADMAP A2 currently answers on **one video**.

Everything the eval needs is already in the released pack (``emdb_vit_v4.pt``): world-frame SMPL
params, the metric ``T_w2c``, ``K_fullimg``, gender, and the validity mask. The one missing piece is
the RGB itself — the pack ships an EMPTY ``videos/`` (EMDB's download is credential-gated), and the
official release ships image *sequences*, not mp4. So point ``--data-root`` at the official download
(the dir holding ``P0/ … P9/``) and the videos get composed once, by the same
``build_videos_from_raw`` the benchmark's preproc-variant path uses.

    uv run python tools/eval/eval_world.py --dataset emdb \
        --data-root /data/gvhmr/data/EMDB/raw --modes prior gt-cam dust3r
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import torch

from gvhmr.utils.assets import DATA_ROOT
from gvhmr.utils.eval.preproc_variants import build_videos_from_raw
from gvhmr.utils.eval.world_eval import WorldSeqGT
from gvhmr.utils.pylogger import Log

from .utils import EMDB2_NAMES


def iter_sequences(data_root: str | Path | None = None, limit: int | None = None) -> Iterator[WorldSeqGT]:
    """Yield the 25 EMDB-2 sequences as :class:`WorldSeqGT`.

    ``data_root`` is the **official EMDB download** (the dir containing ``P0/ … P9/``), used only to
    compose the pack's missing ``videos/*.mp4`` on first run. Defaults to ``$GVHMR_EMDB_RAW`` or
    ``<DATA_ROOT>/EMDB/raw``. The ground truth always comes from the released pack.
    """
    support = DATA_ROOT / "EMDB" / "hmr4d_support"
    labels = torch.load(support / "emdb_vit_v4.pt", weights_only=False)

    names = EMDB2_NAMES[:limit] if limit else list(EMDB2_NAMES)

    videos_dir = support / "videos"
    missing = [n for n in names if not (videos_dir / f"{n}.mp4").exists()]
    if missing:
        raw = Path(data_root or os.environ.get("GVHMR_EMDB_RAW") or (DATA_ROOT / "EMDB" / "raw"))
        if not raw.is_dir():
            raise FileNotFoundError(
                f"{len(missing)} EMDB videos are missing and the raw download isn't at {raw}.\n"
                f"  EMDB's pack ships an empty videos/ (the dataset is credential-gated). Download it from\n"
                f"  https://emdb.ait.ethz.ch/ (P0.zip … P9.zip), extract, and pass --data-root <dir with P0/ … P9/>."
            )
        Log.info(f"[EMDB] composing {len(missing)} missing videos from {raw} (once)")
        build_videos_from_raw("emdb", raw, missing, videos_dir)

    for name in names:
        label = labels[name]
        sp = label["smpl_params"]
        T_w2c = label["T_w2c"].float()
        yield WorldSeqGT(
            vid=name,
            frames_mp4=videos_dir / f"{name}.mp4",
            # EMDB's GT is world-frame SMPL (gendered) — exactly what the global protocol scores.
            smpl_params={k: v.float() for k, v in sp.items()},
            body_type="smpl",
            gender=label["gender"],
            T_w2c=T_w2c,
            K_fullimg=label["K_fullimg"].float(),
            fps=30.0,
            length=int(T_w2c.shape[0]),
            mask=label.get("mask"),
        )
