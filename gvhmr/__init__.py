import os
from pathlib import Path

# On macOS, torch and the preprocessing wheels (opencv / ultralytics) each bundle their
# own libomp, and the OpenMP runtime aborts on the duplicate. This is the standard
# workaround; setdefault so an explicit user setting always wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJ_ROOT = Path(__file__).resolve().parents[1]

__version__ = "1.0.5"

#: The HuggingFace-style inference API, re-exported at the top level. Loaded lazily (via ``__getattr__``)
#: so ``import gvhmr`` stays instant — torch/hydra are only imported when you first touch one of these.
__all__ = ["GVHMR", "GVHMRPipeline", "MotionResult", "pipeline", "recover", "PROJ_ROOT", "__version__"]


def os_chdir_to_proj_root():
    """useful for running notebooks in different directories."""
    os.chdir(PROJ_ROOT)


def __getattr__(name: str):
    """Lazily expose the inference API (PEP 562) without importing torch on ``import gvhmr``."""
    if name in ("pipeline", "recover", "GVHMRPipeline"):
        from gvhmr.inference import pipelines as _pipelines_mod

        return getattr(_pipelines_mod, name)
    if name == "GVHMR":
        from gvhmr.inference.model import GVHMR

        return GVHMR
    if name == "MotionResult":
        from gvhmr.inference.result import MotionResult

        return MotionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
