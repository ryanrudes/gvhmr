"""Shared, opt-in diagnostics for the eval metric callbacks (3DPW / EMDB / RICH).

The three ``MetricMocap`` callbacks each compute rich per-frame, per-sequence arrays and then keep
only the mean. When diagnostics are enabled (``diagnostics=True`` ctor flag OR the
``$GVHMR_EVAL_DIAGNOSTICS`` env var), they call :func:`stash_diagnostics` at epoch end — AFTER the
unchanged ``metrics_avg`` / ``metrics_summary`` logic and BEFORE the aggregator is reset — to preserve
the full distribution (std, percentiles, per-sequence, per-joint, timing) on the LightningModule.

This module writes ONLY to new attributes (``pl_module.metrics_detailed`` / ``metrics_raw``); it never
touches ``metrics_summary`` or the ``val_metric_<DS>/<k>`` logging, so default eval output is byte-identical.
"""

from __future__ import annotations

import os

import numpy as np

from gvhmr.utils.eval.eval_utils import summarize_metric_array, summarize_per_joint, summarize_per_sequence

#: SMPL 24-joint names (EMDB/RICH camcoord use the 24-joint smpl_neutral J-regressor).
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]  # fmt: skip

#: 3DPW's smpl_3dpw14_J_regressor is the "common-14" (LSP/H36M) order — NOT the first 14 SMPL joints.
#: (pelvis_idxs=[2,3] = right_hip/left_hip confirms this order.)
COMMON14_JOINT_NAMES = [
    "right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle", "right_wrist",
    "right_elbow", "right_shoulder", "left_shoulder", "left_elbow", "left_wrist", "neck", "head_top",
]  # fmt: skip


def gather_diagnostics(perjoint_agg: dict | None, timing: dict | None) -> None:
    """All-gather the per-joint + timing aggregators across ranks and merge in place (mirrors the
    callbacks' metric_aggregator gather; no-op on a single device). MUST be called on ALL ranks (it's a
    collective op) before the rank-0-only :func:`stash_diagnostics`."""
    import torch

    from gvhmr.utils.comm.gather import all_gather

    with torch.inference_mode(False):  # allow the in-place update all_gather does
        pj_gathered = all_gather(perjoint_agg) if perjoint_agg else None
        t_gathered = all_gather(timing) if timing else None
    if pj_gathered:
        for metric in list(perjoint_agg.keys()):
            for d in pj_gathered:
                perjoint_agg[metric].update(d.get(metric, {}))
    if t_gathered:
        for d in t_gathered:
            timing.update(d)


def diagnostics_enabled(flag: bool) -> bool:
    """Diagnostics are on if the ctor flag is set OR ``$GVHMR_EVAL_DIAGNOSTICS`` is truthy."""
    if flag:
        return True
    env = os.environ.get("GVHMR_EVAL_DIAGNOSTICS", "").strip().lower()
    return env not in ("", "0", "false", "no")


def build_detailed(
    dataset_id: str,
    metric_aggregator: dict,
    perjoint_agg: dict | None = None,
    timing: dict | None = None,
    joint_labels: list | None = None,
) -> dict:
    """Assemble the detailed diagnostics dict for one dataset (pure; no side effects)."""
    detail: dict = {"n_sequences": 0, "distribution": {}, "per_sequence": {}, "n_frames": {}}
    for metric, vid2arr in metric_aggregator.items():
        if not vid2arr:
            continue
        pooled = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in vid2arr.values()])
        detail["distribution"][metric] = summarize_metric_array(pooled)
        detail["per_sequence"][metric] = summarize_per_sequence(vid2arr)
        detail["n_frames"][metric] = int(pooled.size)
        detail["n_sequences"] = max(detail["n_sequences"], len(vid2arr))
    if perjoint_agg:
        detail["per_joint"] = {
            metric: summarize_per_joint(vid2arr, joint_labels) for metric, vid2arr in perjoint_agg.items() if vid2arr
        }
    if timing:
        vals = np.asarray(list(timing.values()), dtype=np.float64)
        detail["timing"] = {
            "total_s": float(vals.sum()),
            "mean_s": float(vals.mean()) if vals.size else 0.0,
            "max_s": float(vals.max()) if vals.size else 0.0,
            "per_sequence_s": {str(k): float(v) for k, v in timing.items()},
        }
    return detail


def stash_diagnostics(
    pl_module,
    dataset_id: str,
    metric_aggregator: dict,
    perjoint_agg: dict | None = None,
    timing: dict | None = None,
    joint_labels: list | None = None,
) -> None:
    """Stash detailed stats + the raw per-sequence arrays on the LightningModule (opt-in path only).

    ``metrics_detailed[dataset_id]`` holds JSON-safe summaries; ``metrics_raw[dataset_id]`` holds the raw
    numpy arrays (for npz/artifact dumps). Both use shallow copies so the callback's subsequent aggregator
    reset does not free them. If a Lightning logger is present (real-training validation, not ``gvhmr eval``
    whose task sets ``logger: null``), also emit extended scalars + histograms — guarded so it can never
    fire under eval/sweep.
    """
    detail = build_detailed(dataset_id, metric_aggregator, perjoint_agg, timing, joint_labels)

    if not hasattr(pl_module, "metrics_detailed"):
        pl_module.metrics_detailed = {}
    pl_module.metrics_detailed[dataset_id] = detail

    raw: dict = {
        metric: {vid: np.asarray(a) for vid, a in vid2arr.items()} for metric, vid2arr in metric_aggregator.items()
    }
    if perjoint_agg:
        for metric, vid2arr in perjoint_agg.items():
            raw[f"perjoint_{metric}"] = {vid: np.asarray(a) for vid, a in vid2arr.items()}
    if not hasattr(pl_module, "metrics_raw"):
        pl_module.metrics_raw = {}
    pl_module.metrics_raw[dataset_id] = raw

    # Real-training validation only: stream extended scalars + histograms to the live logger.
    logger = getattr(pl_module, "logger", None)
    if logger is not None and hasattr(logger, "experiment"):
        step = getattr(pl_module, "current_epoch", 0)
        scalars = {}
        for metric, dist in detail["distribution"].items():
            for stat in ("std", "median", "p05", "p95", "p99", "max"):
                if stat in dist:
                    scalars[f"val_metric_{dataset_id}/{metric}_{stat}"] = dist[stat]
        try:
            logger.log_metrics(scalars, step=step)
            import wandb  # noqa: PLC0415

            if hasattr(logger.experiment, "log"):
                for metric, vid2arr in metric_aggregator.items():
                    pooled = np.concatenate([np.asarray(a).ravel() for a in vid2arr.values()])
                    logger.experiment.log({f"val_hist_{dataset_id}/{metric}": wandb.Histogram(pooled)}, step=step)
        except Exception:  # noqa: BLE001 — never let diagnostics logging break a training run
            pass
