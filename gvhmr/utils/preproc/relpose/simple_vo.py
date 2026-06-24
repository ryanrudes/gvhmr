import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gvhmr.utils.console import track
from gvhmr.utils.pylogger import Log
from gvhmr.utils.video_io_utils import read_video_np

from .matcher_wrapper import Matcher
from .solver_two_view import CameraParams, TwoPairSolver, interpolate_missing_frames
from .utils import focal_length_from_mm


def _resolve_delta(delta, last_delta, n_degenerate):
    """Carry forward the last valid relative pose when a pair is degenerate (delta is None).

    A degenerate two-view geometry (fast motion / low texture) used to freeze the camera
    (identity); reusing the previous frame's relative motion is a closer continuation.
    """
    if delta is None:
        return last_delta, n_degenerate + 1
    return delta, n_degenerate


def _log_degenerate(n_degenerate, n_pairs):
    if n_degenerate:
        Log.info(f"[SimpleVO] carried forward {n_degenerate}/{n_pairs} degenerate frame-pairs")


class SimpleVO:
    def __init__(self, video_path, scale=0.5, step=8, method="sift", f_mm=None, num_workers=1):
        self.video_path = video_path
        self.scale = scale
        self.step = step
        self.method = method
        self.f_mm = 24 if f_mm is None else f_mm  # fullframe camera focal length in mm
        self.num_workers = num_workers

    def compute(self):
        # Read video
        frames = read_video_np(self.video_path, scale=self.scale)

        # Downsample frames, and interpolate missing frames
        F_all = frames.shape[0]
        sample_idxs = np.arange(0, F_all, self.step)
        if sample_idxs[-1] != F_all - 1:
            sample_idxs = np.concatenate([sample_idxs, [F_all - 1]])
        frames = frames[sample_idxs]
        F, H, W, C = frames.shape
        Log.info(f"[SimpleVO] Chosen frames shape: {frames.shape}")

        matcher: Matcher = Matcher(self.method)
        camera_params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))
        solver: TwoPairSolver = TwoPairSolver(camera_params, solver="pycolmap")

        # TODO:We should use different pipelines for different methods
        T_w2c_list = self.process_video_T_w2c_list_np(frames, matcher, solver)

        # Interpolate missing frames
        T_w2c_list = interpolate_missing_frames(T_w2c_list, sample_idxs)

        return T_w2c_list

    def process_video_T_w2c_list_np(self, frames, matcher: Matcher, solver: TwoPairSolver):
        if self.num_workers <= 1:
            # Serial path: identical to the pre-parallel implementation, including
            # incremental accumulation (so a mid-loop exception leaves a partial
            # T_w2c_list, matching prior behavior).
            T_w2c_list = [np.eye(4)]
            prev_frame = frames[0]
            last_delta = np.eye(4, dtype=np.float32)
            n_degenerate = 0
            for frame_idx in track(range(1, len(frames))):
                curr_frame = frames[frame_idx]
                pts0, pts1 = matcher.match_np(prev_frame, curr_frame)
                T_delta = solver.solve(pts0, pts1)
                last_delta, n_degenerate = _resolve_delta(T_delta, last_delta, n_degenerate)
                T_w2c_list.append(last_delta @ T_w2c_list[-1])
                prev_frame = curr_frame
            _log_degenerate(n_degenerate, len(frames) - 1)
            return T_w2c_list

        # Parallel path: each thread owns its own Matcher/TwoPairSolver to avoid
        # sharing cv2.SIFT / pycolmap internals across threads. Per-thread
        # instances are cloned from the passed-in objects so configuration
        # (matcher type/args, solver backend, camera params) carries over.
        #
        # Note: this is NOT bit-for-bit equivalent to the serial path. pycolmap
        # RANSAC is non-deterministic, and accumulation is deferred until all
        # deltas are collected, so partial-failure behavior also differs.
        n_pairs = len(frames) - 1
        tls = threading.local()

        def get_worker_state():
            if not hasattr(tls, "matcher"):
                tls.matcher = matcher.clone()
                tls.solver = solver.clone()
            return tls.matcher, tls.solver

        def solve_pair(i):
            m, s = get_worker_state()
            pts0, pts1 = m.match_np(frames[i], frames[i + 1])
            return s.solve(pts0, pts1)

        T_deltas = [None] * n_pairs
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for i, T in track(
                zip(range(n_pairs), ex.map(solve_pair, range(n_pairs))),
                total=n_pairs,
            ):
                T_deltas[i] = T

        T_w2c_list = [np.eye(4)]
        last_delta = np.eye(4, dtype=np.float32)
        n_degenerate = 0
        for T_delta in T_deltas:
            last_delta, n_degenerate = _resolve_delta(T_delta, last_delta, n_degenerate)
            T_w2c_list.append(last_delta @ T_w2c_list[-1])
        _log_degenerate(n_degenerate, len(T_deltas))
        return T_w2c_list
