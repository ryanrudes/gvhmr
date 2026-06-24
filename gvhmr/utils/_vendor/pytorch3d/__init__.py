"""Vendored subset of ``pytorch3d.transforms`` (rotation conversions + SO(3)).

These are pure-PyTorch rotation utilities copied verbatim from PyTorch3D so that
GVHMR's inference and geometry paths run on CPU / Apple-Silicon MPS **without** a
pytorch3d install (whose prebuilt wheels are CUDA/Linux/py3.10-only and whose C++
extensions do not build easily off-Linux). Mesh *rendering* still needs the real
pytorch3d — see the ``render`` extra.

Behaviour is identical to upstream by construction (verbatim copies). The only
edits are import-path patches, each tagged ``# [GVHMR vendor patch]``.

See ``README.md`` in this directory for exact provenance.
"""

from .rotation_conversions import (
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
    standardize_quaternion,
)
from .so3 import so3_exp_map, so3_log_map

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
    "standardize_quaternion",
    "so3_exp_map",
    "so3_log_map",
]
