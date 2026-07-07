"""Native SMPL-X mesh option (opt-in) — topology-keyed verts/joints/faces.

Guards that ``topology='smplx'`` yields the full ~10475-vertex SMPL-X surface while the default
``'smpl'`` path stays the 6890-vertex SMPL topology (byte-identical default is the golden test's job).
Skips when the (registration-gated) SMPL-X body model isn't installed.
"""

from __future__ import annotations

import pytest
import torch


def _smplx_available() -> bool:
    try:
        from gvhmr.inference._smpl import _load_assets

        _load_assets("cpu")
        return True
    except Exception:  # noqa: BLE001 — body model missing / not registered
        return False


pytestmark = pytest.mark.skipif(not _smplx_available(), reason="SMPL-X body model not installed")


def _zero_params(length: int = 2) -> dict:
    return {
        "global_orient": torch.zeros(length, 3),
        "body_pose": torch.zeros(length, 63),
        "betas": torch.zeros(length, 10),
        "transl": torch.zeros(length, 3),
    }


def test_topology_shapes_smpl_vs_smplx() -> None:
    from gvhmr.inference._smpl import params_to_verts_joints

    params = _zero_params(2)
    v_smpl, j_smpl, f_smpl = params_to_verts_joints(params, topology="smpl")
    assert v_smpl.shape == (2, 6890, 3)
    assert j_smpl.shape == (2, 24, 3)

    v_x, j_x, f_x = params_to_verts_joints(params, topology="smplx")
    assert v_x.shape[0] == 2 and v_x.shape[1] > 10000 and v_x.shape[2] == 3  # native SMPL-X surface
    assert j_x.shape == (2, 24, 3)
    assert f_x.shape[0] > f_smpl.shape[0]  # SMPL-X has ~20908 faces vs SMPL's 13776


def test_default_topology_is_smpl() -> None:
    from gvhmr.inference._smpl import params_to_verts_joints

    v, _, _ = params_to_verts_joints(_zero_params(1))  # no topology kw → default 'smpl'
    assert v.shape == (1, 6890, 3)


def test_motionresult_smplx_accessors() -> None:
    from gvhmr.inference.result import MotionResult

    params = _zero_params(2)
    r = MotionResult(smpl_params_world=params, smpl_params_camera=params, intrinsics=torch.eye(3).expand(2, 3, 3))
    # default accessors = SMPL topology
    assert r.vertices_world.shape == (2, 6890, 3)
    assert r.faces.shape[1] == 3 and r.faces.shape[0] == 13776
    # native SMPL-X accessors
    assert r.vertices_world_smplx.shape[1] > 10000
    assert r.joints_world_smplx.shape == (2, 24, 3)
    assert r.faces_smplx.shape[0] > r.faces.shape[0]
    # both cached independently (default unchanged after touching smplx)
    assert r.vertices_camera.shape == (2, 6890, 3)
