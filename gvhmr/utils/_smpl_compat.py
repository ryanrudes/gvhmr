"""Compatibility shims for loading legacy SMPL ``.pkl`` body models.

The official SMPL ``.pkl`` files store ``chumpy`` arrays, and ``chumpy`` (0.70,
unmaintained) uses a couple of APIs removed in modern Python / NumPy:

- ``inspect.getargspec`` (removed in Python 3.11) and
- the ``np.bool`` / ``np.float`` / ``np.int`` / ... aliases (removed in NumPy 2.0).

:func:`apply` re-adds them (idempotently) so SMPL ``.pkl`` files load on Python 3.13
+ NumPy 2.x. SMPL-X ``.npz`` files don't need this. Call it once before loading a SMPL
body model (``make_smplx`` does).
"""

from __future__ import annotations

import inspect

import numpy as np

_applied = False


def apply() -> None:
    """Idempotently install the legacy chumpy/SMPL compatibility shims."""
    global _applied
    if _applied:
        return
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, target in {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }.items():
        # Check the module dict directly — `hasattr(np, "object")` would trigger NumPy's
        # deprecation __getattr__ and emit a FutureWarning.
        if name not in np.__dict__:
            setattr(np, name, target)
    _applied = True
