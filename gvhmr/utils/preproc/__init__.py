"""Per-video preprocessing backends: detection/tracking, 2D pose, image features, camera.

The heavy implementations live behind the ``preproc`` / ``slam`` extras (ultralytics, pycolmap, the
vendored ViTPose/HMR2, DPVO). Their imports are **guarded** so this package — and the lightweight
``base`` registry of pluggable backends — import cleanly on the base install (e.g. macOS/CI without the
extra). A name is simply absent if its dependency isn't installed; the demo path requires the extra.
"""

from contextlib import suppress

with suppress(ImportError):
    from gvhmr.utils.preproc.relpose.simple_vo import SimpleVO
with suppress(ImportError):
    from gvhmr.utils.preproc.tracker import Tracker
with suppress(ImportError):
    from gvhmr.utils.preproc.vitfeat_extractor import Extractor
with suppress(ImportError):
    from gvhmr.utils.preproc.vitpose import VitPoseExtractor
with suppress(ImportError):
    from gvhmr.utils.preproc.slam import SLAMModel
