"""Turn recovered SMPL(-X) parameters into meshes and joints.

A faithful, cached extraction of the vertex/joint math the demo renderer uses
(:func:`gvhmr.cli.demo.render_incam` / ``render_global``): run the SuperMotion SMPL-X body model,
map its vertices to the 6890-vertex **SMPL** topology with the committed ``smplx2smpl`` sparse matrix,
then regress the 24 **SMPL** joints. Shared by :class:`gvhmr.inference.MotionResult` so a mesh you get
from the library matches, vertex-for-vertex, what ``gvhmr demo`` renders.

Requires the (registration-gated) SMPL-X body model — the library fetches it for you; see
:func:`gvhmr.utils.assets.ensure_body_models`.
"""

from __future__ import annotations

import functools

import torch
from einops import einsum

from gvhmr import PROJ_ROOT
from gvhmr.utils.device import to_device
from gvhmr.utils.smplx_utils import make_smplx

# Committed body-model assets (not gated — derived maps shipped inside the package).
SMPLX2SMPL_PATH = PROJ_ROOT / "gvhmr/utils/body_model/smplx2smpl_sparse.pt"
SMPL_J_REGRESSOR_PATH = PROJ_ROOT / "gvhmr/utils/body_model/smpl_neutral_J_regressor.pt"


@functools.lru_cache(maxsize=4)
def _load_assets(device_str: str):
    """Load (and cache per device) the SMPL-X model, the SMPL-X→SMPL map, the J regressor, and SMPL faces."""
    device = torch.device(device_str)
    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load(SMPLX2SMPL_PATH, weights_only=False).to(device)
    j_regressor = torch.load(SMPL_J_REGRESSOR_PATH, weights_only=False).to(device)
    faces = make_smplx("smpl").faces  # (13776, 3) int
    return smplx, smplx2smpl, j_regressor, faces


@torch.no_grad()
def params_to_verts_joints(
    params: dict, device="cpu", *, topology: str = "smpl"
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """SMPL(-X) parameters → ``(vertices, joints, faces)``, all on CPU.

    ``params`` is one of ``MotionResult.smpl_params_world`` / ``smpl_params_camera``
    (``global_orient`` / ``body_pose`` / ``betas`` / ``transl``).

    ``topology='smpl'`` (default): map the SMPL-X surface to the 6890-vertex SMPL topology + 24 SMPL joints
    — byte-identical to the demo renderer. ``topology='smplx'``: the native ~10475-vertex SMPL-X surface,
    SMPL-X faces, and ``smplx_out.joints[:, :24]`` (first 22 match SMPL; joints 22/23 are SMPL-X jaw/eye —
    the native SMPL-X render uses the same choice, so the two stay in lockstep).
    """
    smplx, smplx2smpl, j_regressor, faces = _load_assets(str(torch.device(device)))
    out = smplx(**to_device(dict(params), device))
    if topology == "smplx":
        verts = out.vertices  # (L, 10475, 3)
        joints = out.joints[:, :24]  # (L, 24, 3)
        return verts.cpu(), joints.cpu(), smplx.faces
    verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in out.vertices])  # (L, 6890, 3)
    joints = einsum(j_regressor, verts, "j v, l v i -> l j i")  # (L, 24, 3)
    return verts.cpu(), joints.cpu(), faces


def smpl_faces(device="cpu"):
    """The (13776, 3) SMPL triangle faces (shared across frames)."""
    return _load_assets(str(torch.device(device)))[3]


def smplx_faces(device="cpu"):
    """The (~20908, 3) native SMPL-X triangle faces (shared across frames)."""
    return _load_assets(str(torch.device(device)))[0].faces
