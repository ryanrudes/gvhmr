"""The GVHMR inference library — a small, HuggingFace-style API over the demo pipeline.

Public surface:

* :func:`~gvhmr.inference.pipeline.pipeline` — ``gvhmr.pipeline("human-motion-recovery")`` factory.
* :func:`~gvhmr.inference.pipeline.recover` — ``gvhmr.recover("video.mp4")`` one-liner.
* :class:`~gvhmr.inference.pipeline.GVHMRPipeline` — the callable end-to-end pipeline.
* :class:`~gvhmr.inference.model.GVHMR` — the core model (``from_pretrained`` / ``predict`` / ``push_to_hub``).
* :class:`~gvhmr.inference.result.MotionResult` — the rich return value (meshes, joints, render, export).

All of these are also re-exported at the top level (``gvhmr.pipeline``, ``gvhmr.recover``, ``gvhmr.GVHMR``,
``gvhmr.GVHMRPipeline``, ``gvhmr.MotionResult``).
"""

from __future__ import annotations

from gvhmr.inference import hub
from gvhmr.inference.model import GVHMR
from gvhmr.inference.pipelines import GVHMRPipeline, pipeline, recover
from gvhmr.inference.result import MotionResult

__all__ = ["GVHMR", "GVHMRPipeline", "MotionResult", "pipeline", "recover", "hub"]
