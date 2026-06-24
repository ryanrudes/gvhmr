import os
from pathlib import Path

# On macOS, torch and the preprocessing wheels (opencv / ultralytics) each bundle their
# own libomp, and the OpenMP runtime aborts on the duplicate. This is the standard
# workaround; setdefault so an explicit user setting always wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJ_ROOT = Path(__file__).resolve().parents[1]


def os_chdir_to_proj_root():
    """useful for running notebooks in different directories."""
    os.chdir(PROJ_ROOT)
