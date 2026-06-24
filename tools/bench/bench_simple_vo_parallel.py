"""
Benchmark SimpleVO with different num_workers values.

Usage:
    python tools/bench/bench_simple_vo_parallel.py [--video PATH] [--workers 1 2 4 8] \\
            [--scale 0.5] [--step 8] [--method sift]

Measures wall time of SimpleVO.compute() for each worker count, and reports
the max element-wise diff of the resulting T_w2c stack against the serial
baseline. The diff is non-zero because pycolmap RANSAC is non-deterministic
(random_seed = -1); small per-pair noise accumulates through the 4x4 matmul
chain. This is a sanity check, not a correctness assertion.

Reference timings (host: 192-core CPU, opencv 4.11, pycolmap 4.0.4):

  tennis.mp4, scale=0.5, step=8, sift (39 pairs, 312 interpolated frames):
     workers   time(s)   speedup   max|T - T_serial|
           1      3.66      1.00x   0.00e+00  (baseline)
           2      1.86      1.97x   5.63e-02
           4      1.25      2.93x   2.93e-02
           8      1.02      3.74x   —
          16      0.93      4.10x   —
          32      0.96      3.97x   —

  Notes on the 1/2/4 diff column (from `--check`):
    - workers=1 is bit-identical to itself (it IS the baseline).
    - workers={2,4} diverge by ~3-6e-2 in max element; this is the
      expected pycolmap RANSAC non-determinism accumulating through the
      4x4 T_w2c chain, NOT a correctness bug. Different runs of the same
      worker count will also differ by a similar magnitude.
    - The number tends to shrink with more workers in this particular
      run, but that ordering is noise — don't read a trend into it.

  tennis.mp4, scale=0.5, step=4, sift (78 pairs):
     workers   time(s)   speedup
           1      6.25      1.00x
           2      3.43      1.82x
           4      2.10      2.97x
           8      1.43      4.36x
          16      1.08      5.80x
          32      1.06      5.88x

Speedup saturates around 16 workers despite the host having far more cores;
the bottleneck is the serial portions (`read_video_np` decoding and the
final T_w2c accumulation), not GIL contention. cv2 SIFT/FLANN and pycolmap
release the GIL during their C++ work, so a threaded pool is sufficient.
"""

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path so `import gvhmr...` works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# `gvhmr.utils.preproc.__init__` eagerly imports Tracker / Extractor / VitPose,
# which pull in ultralytics + the HMR2 network. We only need SimpleVO here,
# so stub those siblings before the package-level import runs.
for _name, _attr in [
    ("gvhmr.utils.preproc.tracker", "Tracker"),
    ("gvhmr.utils.preproc.vitfeat_extractor", "Extractor"),
    ("gvhmr.utils.preproc.vitpose", "VitPoseExtractor"),
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        setattr(_m, _attr, type(_attr, (), {}))
        sys.modules[_name] = _m

from gvhmr.utils.preproc.relpose.simple_vo import SimpleVO  # noqa: E402
from gvhmr.utils.preproc.relpose.solver_two_view import (  # noqa: E402
    PycolmapRansacTwoViewGeometrySolver,
)

# Bench-only robustness: pycolmap occasionally fails geometry estimation on
# degenerate frame pairs and returns answer.cam2_from_cam1 == None, which
# crashes the original solver. Fall back to identity so the run can complete.
_orig_solve = PycolmapRansacTwoViewGeometrySolver.solve


def _safe_solve(self, pts0, pts1):
    try:
        return _orig_solve(self, pts0, pts1)
    except AttributeError:
        return np.eye(4)


PycolmapRansacTwoViewGeometrySolver.solve = _safe_solve


def run_once(video, scale, step, method, num_workers):
    vo = SimpleVO(
        video_path=str(video),
        scale=scale,
        step=step,
        method=method,
        num_workers=num_workers,
    )
    t0 = time.perf_counter()
    T_w2c = vo.compute()
    dt = time.perf_counter() - t0
    T_w2c = np.asarray(T_w2c)
    return dt, T_w2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        default="docs/example_video/tennis.mp4",
        help="Path to input video (relative to repo root or absolute).",
    )
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--method", default="sift", choices=["sift", "orb"])
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify parallel outputs match the serial baseline.",
    )
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        # Resolve relative to repo root (this file lives at tools/bench/...)
        video = Path(__file__).resolve().parents[2] / video
    assert video.exists(), f"Video not found: {video}"

    print(f"Video:   {video}")
    print(f"scale={args.scale}  step={args.step}  method={args.method}")
    print(f"Workers: {args.workers}")
    print("-" * 60)

    results = []
    baseline_T = None
    for nw in args.workers:
        # Warm-up could matter for SIFT/FLANN; we just run once and report.
        dt, T = run_once(video, args.scale, args.step, args.method, nw)
        results.append((nw, dt, T.shape[0]))
        if baseline_T is None:
            baseline_T = T
            max_diff = 0.0
        else:
            # Same shape expected; identical sample_idxs path.
            max_diff = float(np.max(np.abs(T - baseline_T)))
        print(f"workers={nw:2d}  time={dt:7.2f}s  frames={T.shape[0]:5d}  max|T - T_baseline|={max_diff:.2e}")
        if args.check and max_diff > 1e-3 and nw != args.workers[0]:
            print(f"  WARNING: output differs from baseline by {max_diff:.3e}")

    print("-" * 60)
    base_t = results[0][1]
    print(f"{'workers':>8}  {'time(s)':>8}  {'speedup':>8}")
    for nw, dt, _ in results:
        print(f"{nw:>8d}  {dt:>8.2f}  {base_t / dt:>8.2f}x")


if __name__ == "__main__":
    main()
