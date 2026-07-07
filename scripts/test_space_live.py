#!/usr/bin/env python3
"""Smoke-test the live HF Space: upload a tiny clip and wait for a response.

Requires: ``pip install gradio_client`` and a reachable Space (not sleeping).
Exits 0 on success, 1 on failure. Used manually and in CI (optional).

    python scripts/test_space_live.py
    python scripts/test_space_live.py --space ryanrudes/gvhmr-demo --timeout 900
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE = "ryanrudes/gvhmr-demo"
DEFAULT_VIDEO = REPO_ROOT / "docs" / "example_video" / "tennis.mp4"


def _tiny_mp4(path: Path) -> None:
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Live HF Space upload smoke test")
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Clip with a visible person")
    parser.add_argument("--timeout", type=int, default=900, help="Client timeout (seconds)")
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"FAIL: video not found: {args.video}", file=sys.stderr)
        return 2

    try:
        from gradio_client import Client, handle_file
        from huggingface_hub import HfApi
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2

    vars_ = HfApi().get_space_variables(args.space)
    ssr = vars_.get("GRADIO_SSR_MODE")
    print(f"Space variables: GRADIO_SSR_MODE={getattr(ssr, 'value', ssr)}")

    with tempfile.TemporaryDirectory() as td:
        video = args.video
        print(f"Connecting to {args.space} …")
        client = Client(args.space, verbose=False)
        print("Uploading 1 s test clip (static camera) …")
        try:
            job = client.submit(
                handle_file(str(video)),
                True,  # static_camera
                "simplevo",  # camera
                False,  # flip_test
                "ryanrudes/gvhmr",  # hub_repo
                "",  # hub_revision
                "yolo",  # detector
                "vitpose",  # pose2d
                "hmr2",  # backbone
                True,  # native SMPL-X mesh (Space default — needs only the SMPL-X body model, not gated SMPL)
                False,  # render_overlay — off by default (the fast, reliable npz path on cpu-basic)
                api_name="/run",
            )
            result = job.result(timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            if "403" in text or "failed to download file" in text:
                print(
                    "FAIL: Gradio SSR file fetch 403 — set Space variable GRADIO_SSR_MODE=false and restart/republish.",
                    file=sys.stderr,
                )
            else:
                print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if not result or len(result) < 2:
        print(f"FAIL: unexpected result shape: {result!r}", file=sys.stderr)
        return 1
    print(f"OK: overlay={result[0]!r}, summary keys={list((result[2] or {}).keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
