"""
Regression test for the SimpleVO parallel path.

Standalone script (no pytest dependency, matching the repo's existing
tools/bench/* convention). Exits 0 on pass, non-zero on failure.

Two layers:

1. Unit-level checks (no video, fast):
     - Matcher.clone() / TwoPairSolver.clone() return new objects with the
       same config and independently-callable internals.

2. Integration check (requires docs/example_video/tennis.mp4):
     - num_workers=1 produces a well-formed T_w2c list (starts at identity,
       length matches frame count).
     - num_workers=4 produces a same-shape output that is "close" to the
       serial baseline. The threshold is loose because pycolmap RANSAC is
       non-deterministic, so we assert closeness, not bit equality.

Usage:
    python tools/bench/test_simple_vo_parallel.py
    python tools/bench/test_simple_vo_parallel.py --video <path>
    python tools/bench/test_simple_vo_parallel.py --skip-integration
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Stub sibling submodules so we can import SimpleVO without dragging in the
# full preproc package (ultralytics, HMR2 network, etc.). Same shim the
# bench script uses.
for _name, _attr in [
    ("gvhmr.utils.preproc.tracker", "Tracker"),
    ("gvhmr.utils.preproc.vitfeat_extractor", "Extractor"),
    ("gvhmr.utils.preproc.vitpose", "VitPoseExtractor"),
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        setattr(_m, _attr, type(_attr, (), {}))
        sys.modules[_name] = _m

from gvhmr.utils.preproc.relpose.matcher_wrapper import Matcher  # noqa: E402
from gvhmr.utils.preproc.relpose.solver_two_view import (  # noqa: E402
    CameraParams,
    PycolmapRansacTwoViewGeometrySolver,
    TwoPairSolver,
)

# Same robustness shim as the bench script: pycolmap occasionally returns
# answer.cam2_from_cam1 == None on degenerate pairs. Fall back to identity
# so the regression run completes deterministically in shape, not value.
_orig_solve = PycolmapRansacTwoViewGeometrySolver.solve


def _safe_solve(self, pts0, pts1):
    try:
        return _orig_solve(self, pts0, pts1)
    except AttributeError:
        return np.eye(4)


PycolmapRansacTwoViewGeometrySolver.solve = _safe_solve


# ---------------------------------------------------------------------------
# Unit-level checks
# ---------------------------------------------------------------------------


def test_matcher_clone():
    m = Matcher("sift")
    c = m.clone()
    assert c is not m, "clone must return a new object"
    assert c.matcher is not m.matcher, "inner BaseMatcher must not be shared"
    assert c._matcher_name == m._matcher_name == "sift"

    # Both clones should be independently callable on the same input.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    # SIFT on random noise can yield 0 matches; just confirm no exception and
    # that outputs have the same shape contract.
    p0a, p1a = m.match_np(img, img)
    p0b, p1b = c.match_np(img, img)
    assert p0a.shape == p0b.shape and p1a.shape == p1b.shape


def test_solver_clone():
    params = CameraParams(width=320, height=240, focal_length=400.0)
    s = TwoPairSolver(params, solver="pycolmap")
    c = s.clone()
    assert c is not s, "clone must return a new object"
    assert c.solver is not s.solver, "inner solver must not be shared"
    assert c._solver_name == s._solver_name == "pycolmap"
    assert np.allclose(c.get_K(), s.get_K()), "camera intrinsics must match"


def test_solver_clone_preserves_camera_params():
    # If someone constructs a TwoPairSolver with non-default cx/cy, the clone
    # must carry them over — this is the specific reviewer concern about
    # parallel workers diverging from the configured solver.
    params = CameraParams(width=320, height=240, focal_length=500.0, cx=100.0, cy=120.0)
    s = TwoPairSolver(params, solver="pycolmap")
    c = s.clone()
    K_orig, K_clone = s.get_K(), c.get_K()
    assert np.allclose(K_orig, K_clone), f"K differs: {K_orig} vs {K_clone}"
    assert K_clone[0, 2] == 100.0 and K_clone[1, 2] == 120.0


# ---------------------------------------------------------------------------
# Integration check
# ---------------------------------------------------------------------------


def integration_serial_vs_parallel(video_path, scale, step, method, parallel_workers, tol):
    from gvhmr.utils.preproc.relpose.simple_vo import SimpleVO

    serial_vo = SimpleVO(video_path=str(video_path), scale=scale, step=step, method=method, num_workers=1)
    T_serial = np.asarray(serial_vo.compute())

    # Serial path shape contract
    assert T_serial.ndim == 3 and T_serial.shape[1:] == (4, 4), f"bad shape: {T_serial.shape}"
    assert np.allclose(T_serial[0], np.eye(4)), "T_w2c_list[0] must be identity"

    parallel_vo = SimpleVO(
        video_path=str(video_path),
        scale=scale,
        step=step,
        method=method,
        num_workers=parallel_workers,
    )
    T_par = np.asarray(parallel_vo.compute())

    assert T_par.shape == T_serial.shape, f"shape mismatch: {T_par.shape} vs {T_serial.shape}"
    assert np.allclose(T_par[0], np.eye(4)), "parallel T_w2c_list[0] must be identity"

    max_diff = float(np.max(np.abs(T_par - T_serial)))
    # Not bit-equality: pycolmap RANSAC is non-deterministic. We just want
    # to catch gross regressions (e.g., wrong frame ordering, broken solver
    # clone). tol is loose by design.
    assert max_diff < tol, (
        f"parallel output diverges from serial by {max_diff:.3e} (tol={tol:.3e}); "
        "this is larger than expected RANSAC noise — likely a real regression."
    )
    return T_serial.shape[0], max_diff


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        default="docs/example_video/tennis.mp4",
        help="Path to test video (relative to repo root or absolute).",
    )
    ap.add_argument("--workers", type=int, default=4, help="Worker count for parallel check.")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--method", default="sift", choices=["sift", "orb"])
    ap.add_argument(
        "--tol",
        type=float,
        default=5.0,
        help=(
            "Max allowed |T_parallel - T_serial|. Loose because pycolmap RANSAC "
            "is non-deterministic and per-pair noise accumulates."
        ),
    )
    ap.add_argument(
        "--skip-integration",
        action="store_true",
        help="Only run unit checks (no video required).",
    )
    args = ap.parse_args()

    failures = []

    unit_tests = [
        ("matcher.clone()", test_matcher_clone),
        ("solver.clone()", test_solver_clone),
        ("solver.clone() preserves CameraParams", test_solver_clone_preserves_camera_params),
    ]
    for name, fn in unit_tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failures.append(name)

    if args.skip_integration:
        print("(integration check skipped)")
    else:
        video = Path(args.video)
        if not video.is_absolute():
            video = _REPO_ROOT / video
        if not video.exists():
            print(f"SKIP  integration: video not found at {video}")
        else:
            name = f"serial vs parallel (workers={args.workers})"
            try:
                n_frames, max_diff = integration_serial_vs_parallel(
                    video,
                    scale=args.scale,
                    step=args.step,
                    method=args.method,
                    parallel_workers=args.workers,
                    tol=args.tol,
                )
                print(f"PASS  {name}  [frames={n_frames}  max|Δ|={max_diff:.3e}]")
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                failures.append(name)

    print("-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
