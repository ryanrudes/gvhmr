"""Rotation conversions used throughout GVHMR (the single import point).

Historically the codebase imported these directly from ``pytorch3d.transforms``,
which made every geometry/model/dataset module hard-depend on pytorch3d — a
package whose prebuilt wheels are CUDA/Linux/py3.10-only. These are pure-PyTorch
functions, so GVHMR now ships a frozen, byte-identical copy under
``gvhmr.utils._vendor.pytorch3d`` and re-exports it here. The numerics are exactly
pytorch3d ``v0.7.6``; only the import path changed.

This indirection is the *one* place to swap implementations: point first-party
code at ``gvhmr.utils.geo.rotations`` and the backend can change (vendored copy,
a real pytorch3d install, etc.) without touching call sites.
"""

from __future__ import annotations

from gvhmr.utils._vendor.pytorch3d import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_raw_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    so3_exp_map,
    so3_log_map,
    standardize_quaternion,
)

__all__ = [
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "euler_angles_to_matrix",
    "matrix_to_axis_angle",
    "matrix_to_euler_angles",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "quaternion_apply",
    "quaternion_invert",
    "quaternion_multiply",
    "quaternion_raw_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rotation_6d_to_matrix",
    "so3_exp_map",
    "so3_log_map",
    "standardize_quaternion",
]
