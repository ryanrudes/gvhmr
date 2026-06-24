"""Backward-compat entry point for the single-video demo.

Prefer the CLI:  [gvhmr demo VIDEO -s]. This shim keeps the old flags working and
forwards to :func:`gvhmr.cli.demo.run`.

    python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s
    # equivalently:
    gvhmr demo docs/example_video/tennis.mp4 -s
"""

import argparse
from pathlib import Path

from gvhmr.cli.demo import run


def main() -> None:
    p = argparse.ArgumentParser(description="GVHMR single-video demo (legacy entry; prefer `gvhmr demo`).")
    p.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    p.add_argument("-o", "--output_root", type=str, default=None)
    p.add_argument("-s", "--static_cam", action="store_true", help="Camera is static (skip visual odometry).")
    p.add_argument("--use_dpvo", action="store_true", help="Use DPVO (CUDA-only).")
    p.add_argument("--f_mm", type=int, default=None, help="Full-frame focal length in mm.")
    p.add_argument("--render_scale", type=float, default=None, help="Overlay resolution fraction (0.5 default).")
    p.add_argument("--no_render", action="store_true", help="Skip overlay videos.")
    p.add_argument("--flip_test", action="store_true", help="Flip-test TTA (accuracy; +1 feature pass).")
    p.add_argument("-v", "--verbose", action="store_true", help="Save intermediate-result overlays.")
    a = p.parse_args()
    run(
        Path(a.video),
        output_root=a.output_root,
        static_cam=a.static_cam,
        use_dpvo=a.use_dpvo,
        f_mm=a.f_mm,
        verbose=a.verbose,
        render_scale=a.render_scale,
        no_render=a.no_render,
        flip_test=a.flip_test,
    )


if __name__ == "__main__":
    main()
