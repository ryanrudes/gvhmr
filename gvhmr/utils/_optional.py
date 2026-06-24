"""Helpers for optional, extra-gated dependencies.

GVHMR installs in layers (see ``pyproject.toml`` extras): the base install runs
on CPU / Apple-Silicon MPS, while heavy/platform-specific pieces — pytorch3d mesh
rendering, the preprocessing models, DPVO SLAM — live behind extras. Modules that
need such a dependency should stay *importable* without it and fail loudly only
when the feature is actually used.
"""

from __future__ import annotations


def missing_dependency(feature: str, *, package: str, extra: str | None = None) -> ImportError:
    """Build a clear ``ImportError`` for an unavailable optional dependency.

    Args:
        feature: human-readable name of what the caller tried to use (e.g. "Mesh rendering").
        package: the import/distribution name that is missing (e.g. "pytorch3d").
        extra: the GVHMR extra that provides it, if any (e.g. "render").
    """
    hint = f"`pip install gvhmr[{extra}]`" if extra else f"install `{package}`"
    return ImportError(
        f"{feature} requires the optional dependency `{package}`, which is not installed. "
        f"To enable it, {hint}. See docs/INSTALL.md for platform notes."
    )
