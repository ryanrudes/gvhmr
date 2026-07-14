"""Fetch the BEDLAM **videos** GVHMR trains on, from the official HuggingFace mirror.

Unblocks the ROADMAP A1 backbone track: to re-extract image features with a different backbone you
need the RGB the released features were computed from. GVHMR's BEDLAM pack ships the features, the
labels and an *empty* ``videos/`` — BEDLAM itself is registration-gated (https://bedlam.is.tue.mpg.de/).

Only ~20 GB is needed, not the ~2.2 TB the dataset weighs:
  * GVHMR reads **mp4** (``bedlam.py``: ``read_video_np(root/"videos"/mid2vname(mid), ...)``), so the
    ``png/`` frame tars — 2.17 TB of them — are irrelevant.
  * Only the **30 scenes** the training mids reference are fetched, not all of BEDLAM.

    huggingface-cli login          # needs access to Intelligent-Systems/BEDLAM
    uv run python scripts/fetch_bedlam_videos.py           # ~20 GB -> <DATA>/BEDLAM/hmr4d_support/videos/
    uv run python scripts/fetch_bedlam_videos.py --check   # verify only, download nothing

Idempotent: scenes already extracted are skipped, so a re-run resumes.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import torch

from gvhmr.dataset.bedlam.utils import mid2vname
from gvhmr.utils.assets import DATA_ROOT
from gvhmr.utils.console import console, rule, track
from gvhmr.utils.hf_token import resolve_hf_token

REPO = "Intelligent-Systems/BEDLAM"
SUPPORT = DATA_ROOT / "BEDLAM" / "hmr4d_support"


def needed_videos() -> dict[str, set[str]]:
    """scene -> {seq_000000.mp4, …} that the training mids actually reference."""
    mids: set[str] = set()
    for f in ("mid_to_valid_range_all60.pt", "mid_to_valid_range_maxspan60.pt"):
        mids |= set(torch.load(SUPPORT / f, weights_only=False).keys())
    out: dict[str, set[str]] = {}
    for mid in mids:
        scene, seq = mid2vname(mid).split("/", 1)  # "<scene>/seq_000001.mp4"
        out.setdefault(scene, set()).add(seq)
    return out


def check(want: dict[str, set[str]]) -> int:
    videos = SUPPORT / "videos"
    missing = 0
    for scene, seqs in sorted(want.items()):
        have = {p.name for p in (videos / scene).glob("*.mp4")}
        gap = seqs - have
        missing += len(gap)
        mark = "[ok]✓[/]" if not gap else f"[warn]{len(gap)} missing[/]"
        console.print(f"  {scene:<52} {len(have):>5}/{len(seqs):<5} {mark}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="Verify what's present; download nothing.")
    ap.add_argument("--keep-tars", action="store_true", help="Don't delete each tar after extracting it.")
    args = ap.parse_args()

    rule("[gvhmr]BEDLAM videos[/]")
    want = needed_videos()
    n_vids = sum(len(v) for v in want.values())
    console.print(f"training mids reference [ok]{n_vids}[/] videos across [ok]{len(want)}[/] scenes")

    if args.check:
        missing = check(want)
        console.print(f"\n{'[ok]all present[/]' if not missing else f'[warn]{missing} videos missing[/]'}")
        raise SystemExit(0 if not missing else 1)

    from huggingface_hub import hf_hub_download

    token = resolve_hf_token(None)
    if not token:
        console.print(
            "[warn]No HuggingFace token.[/] BEDLAM is gated — request access at "
            f"https://huggingface.co/datasets/{REPO}, then `huggingface-cli login`."
        )
        raise SystemExit(1)

    videos = SUPPORT / "videos"
    staging = DATA_ROOT / "BEDLAM" / "raw_mp4"
    for scene in track(sorted(want), desc="scenes"):
        dest = videos / scene
        if len(list(dest.glob("*.mp4"))) >= len(want[scene]):
            continue  # already extracted — resume cheaply
        tar_path = hf_hub_download(
            REPO, f"{scene}/mp4/{scene}_mp4.tar", repo_type="dataset", token=token, local_dir=str(staging)
        )
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as t:
            # tar holds "<scene>/mp4/seq_x.mp4"; GVHMR wants "videos/<scene>/seq_x.mp4" — strip "mp4/"
            for m in t.getmembers():
                if not m.name.endswith(".mp4"):
                    continue
                src = t.extractfile(m)
                if src is None:
                    continue
                (dest / Path(m.name).name).write_bytes(src.read())
        if not args.keep_tars:
            Path(tar_path).unlink(missing_ok=True)

    console.print()
    missing = check(want)
    if missing:
        console.print(f"\n[warn]{missing} videos still missing[/] — re-run to resume.")
        raise SystemExit(1)
    console.print(f"\n[ok]BEDLAM videos ready[/] → [muted]{videos}[/]  (A1 backbone re-extraction unblocked)")


if __name__ == "__main__":
    main()
