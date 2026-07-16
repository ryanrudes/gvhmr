""":class:`MotionResult` — the friendly return value of the inference API.

Wraps the raw ``pred`` dict that ``DemoPL.predict`` produces (``smpl_params_global`` / ``smpl_params_incam``
/ ``K_fullimg`` / ``net_outputs``) behind HuggingFace-ish names (``world`` / ``camera``) and lazy accessors
for meshes, joints, files, and overlay videos — without changing any numbers.

Frames: **world** = gravity-aligned world frame (the paper's global frame); **camera** = in-camera frame.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class MotionResult:
    """Recovered SMPL motion for one video, in the world and camera frames.

    Attributes:
        smpl_params_world: ``{global_orient (L,3), body_pose (L,63), betas (L,10), transl (L,3)}`` — world frame.
        smpl_params_camera: same keys, in the camera frame.
        intrinsics: ``(L, 3, 3)`` per-frame pinhole ``K`` (full-image pixels).
        betas_per_frame: ``(L, 10)`` the per-frame (frame-varying) shape params as the network predicts
            them, *before* the sequence-averaging that produces the single ``betas`` used for the mesh.
            ``None`` if the producing pipeline didn't surface them. See :attr:`betas_per_frame`.
        bbx_xys: ``(L, 3)`` per-frame crop box ``(cx, cy, size)`` in full-image pixels — the network's
            input crop. Feeds :meth:`depth_reliability`; ``None`` if not produced.
        kp2d: ``(L, J, 3)`` per-frame detected 2D keypoints ``(x, y, conf)`` — the pose observation;
            ``None`` if not produced.
        fps: frames per second of the recovered motion (GVHMR runs at 30).
        camera: the camera backend used (``simplevo`` / ``dpvo`` / ``dust3r`` / ``vggt`` / ``static``).
        video_path: the (staged, 30fps) input video, if produced by the pipeline.
    """

    smpl_params_world: dict[str, torch.Tensor]
    smpl_params_camera: dict[str, torch.Tensor]
    intrinsics: torch.Tensor
    betas_per_frame: torch.Tensor | None = None
    bbx_xys: torch.Tensor | None = None
    kp2d: torch.Tensor | None = None
    fps: float = 30.0
    camera: str = "simplevo"
    video_path: Path | None = None
    output_dir: Path | None = None

    # Internals for save()/render() — the full prediction and the demo config that produced it.
    raw: dict = field(default_factory=dict, repr=False)
    _cfg: object = field(default=None, repr=False)
    _device: object = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- basic geometry --------------------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return int(self.smpl_params_world["transl"].shape[0])

    def __len__(self) -> int:
        return self.num_frames

    def __repr__(self) -> str:
        return f"MotionResult(frames={self.num_frames}, fps={self.fps:g}, camera={self.camera!r})"

    # ---- meshes & joints (lazy; need the SMPL-X body model) ------------------------------------
    def _verts_joints(self, frame: str, topology: str = "smpl"):
        key = f"vj_{frame}_{topology}"
        if key not in self._cache:
            from gvhmr.inference._smpl import params_to_verts_joints

            params = self.smpl_params_world if frame == "world" else self.smpl_params_camera
            device = self._device if self._device is not None else "cpu"
            verts, joints, faces = params_to_verts_joints(params, device=str(device), topology=topology)
            self._cache[key] = (verts, joints)
            self._cache[f"faces_{topology}"] = faces
        return self._cache[key]

    @property
    def vertices_world(self) -> torch.Tensor:
        """SMPL mesh vertices in the world frame, ``(L, 6890, 3)``."""
        return self._verts_joints("world")[0]

    @property
    def vertices_camera(self) -> torch.Tensor:
        """SMPL mesh vertices in the camera frame, ``(L, 6890, 3)``."""
        return self._verts_joints("camera")[0]

    @property
    def joints_world(self) -> torch.Tensor:
        """24 SMPL joints in the world frame, ``(L, 24, 3)``."""
        return self._verts_joints("world")[1]

    @property
    def joints_camera(self) -> torch.Tensor:
        """24 SMPL joints in the camera frame, ``(L, 24, 3)``."""
        return self._verts_joints("camera")[1]

    @property
    def faces(self) -> np.ndarray:
        """SMPL triangle faces, ``(13776, 3)`` — shared across frames."""
        if "faces_smpl" not in self._cache:
            from gvhmr.inference._smpl import smpl_faces

            self._cache["faces_smpl"] = smpl_faces(str(self._device or "cpu"))
        return self._cache["faces_smpl"]

    # ---- native SMPL-X (opt-in): the full ~10475-vertex surface the model predicts natively ------
    @property
    def vertices_world_smplx(self) -> torch.Tensor:
        """Native SMPL-X mesh vertices in the world frame, ``(L, ~10475, 3)``."""
        return self._verts_joints("world", "smplx")[0]

    @property
    def vertices_camera_smplx(self) -> torch.Tensor:
        """Native SMPL-X mesh vertices in the camera frame, ``(L, ~10475, 3)``."""
        return self._verts_joints("camera", "smplx")[0]

    @property
    def joints_world_smplx(self) -> torch.Tensor:
        """SMPL-X joints[:24] in the world frame, ``(L, 24, 3)`` (first 22 match SMPL; 22/23 are jaw/eye)."""
        return self._verts_joints("world", "smplx")[1]

    @property
    def joints_camera_smplx(self) -> torch.Tensor:
        """SMPL-X joints[:24] in the camera frame, ``(L, 24, 3)``."""
        return self._verts_joints("camera", "smplx")[1]

    @property
    def faces_smplx(self) -> np.ndarray:
        """Native SMPL-X triangle faces, ``(~20908, 3)`` — shared across frames."""
        if "faces_smplx" not in self._cache:
            from gvhmr.inference._smpl import smplx_faces

            self._cache["faces_smplx"] = smplx_faces(str(self._device or "cpu"))
        return self._cache["faces_smplx"]

    @property
    def betas_avg(self) -> torch.Tensor:
        """The single sequence-averaged shape ``(10,)`` actually used to build the mesh (all frames share it)."""
        return self.smpl_params_world["betas"][0]

    def depth_reliability(self) -> dict[str, torch.Tensor] | None:
        """Per-frame in-camera-depth diagnostics. **``weight`` does NOT predict depth error — see below.**

        .. warning::
           This shipped in v1.6.0 on the *unmeasured* hypothesis that out-of-distribution crops inflate
           ``betas`` and push ``tz`` far. Measured against EMDB-1 ground truth (2026-07-16,
           ``scripts/depth_bias_probe.py``, 24k frames), **that hypothesis is refuted**:

           - The dominant depth error is the **focal**, not the crops. ``tz = 2·f/(s·b)`` is exactly
             linear in ``f``, and the default ``estimate_K`` heuristic (``f = √(w²+h²)``) runs 1.67× long
             on phone footage → **+71% depth error on 100% of frames**. With GT intrinsics: +6%.
           - The residual bias is **not** crop-size dependent — relative error *grows* with crop size
             (+2.97% → +8.08%), the opposite of the OOD story, and ``corr(betas_mag, |err|) = −0.373``
             (anti-correlated with the inflation hypothesis).
           - ``corr(weight, |depth error|)`` is **−0.028 with correct intrinsics** — i.e. nothing. Its
             apparent −0.650 under the default heuristic is an artifact: ``weight`` is mostly ``bbx_px``,
             a distance proxy, and a wrong focal makes error scale with distance. It tracks how far away
             the person is, not how wrong the depth is.

           **Do not weight multi-view depth fusion by ``weight``.** The dominant error is a per-video
           constant (the focal) that no per-frame proxy can express — pass calibrated ``--f_px`` /
           ``--intrinsics`` instead. The raw fields below remain useful as diagnostics.

        - ``bbx_px`` ``(L,)`` — person pixel-height (``bbx_xys[:,2]``), absolute, comparable across views.
        - ``betas_mag`` ``(L,)`` — ``‖betas_per_frame‖``; shape departure from the ~N(0,1) prior.
        - ``betas_std`` ``(L,)`` — ``‖betas_per_frame − betas_avg‖``; shape instability across the clip.
        - ``weight`` ``(L,)`` in ``(0, 1]`` — **deprecated, not predictive** (see the warning). Retained
          only so v1.6.0 consumers don't break on a missing key.

        Returns ``None`` if ``betas_per_frame`` / ``bbx_xys`` weren't produced. See docs/CAMERA_METADATA.md.
        """
        if self.betas_per_frame is None or self.bbx_xys is None:
            return None
        bbx_px = self.bbx_xys[..., 2].float()  # (L,) absolute person pixel-height
        bpf = self.betas_per_frame.float()  # (L, 10)
        betas_mag = bpf.norm(dim=-1)  # (L,)
        betas_std = (bpf - bpf.mean(dim=0, keepdim=True)).norm(dim=-1)  # (L,)
        med = bbx_px.median().clamp_min(1e-6)
        weight = (bbx_px / med).clamp(max=1.0) / (1.0 + betas_std)  # within-view; monotone, no magic thresh
        return {"bbx_px": bbx_px, "betas_mag": betas_mag, "betas_std": betas_std, "weight": weight}

    # ---- serialization ---------------------------------------------------------------------------
    def to_dict(self) -> dict:
        """A plain dict of the friendly fields (tensors), for programmatic use."""
        out = {
            "smpl_params_world": self.smpl_params_world,
            "smpl_params_camera": self.smpl_params_camera,
            "intrinsics": self.intrinsics,
            "fps": self.fps,
            "camera": self.camera,
            "num_frames": self.num_frames,
        }
        if self.betas_per_frame is not None:
            out["betas_per_frame"] = self.betas_per_frame
        if self.bbx_xys is not None:
            out["bbx_xys"] = self.bbx_xys
        if self.kp2d is not None:
            out["kp2d"] = self.kp2d
        return out

    def save(self, path: str | Path) -> Path:
        """Save the full raw prediction as a ``.pt`` — the exact format ``gvhmr demo`` writes
        (``hmr4d_results.pt``), so it round-trips with the rest of the toolchain."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.raw or self._raw_from_fields(), path)
        return path

    def save_npz(self, path: str | Path) -> Path:
        """Export the SMPL parameters + intrinsics to a portable ``.npz`` (numpy, no torch needed to read)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"fps": np.asarray(self.fps), "camera": np.asarray(self.camera), "intrinsics": _np(self.intrinsics)}
        for frame, params in (("world", self.smpl_params_world), ("camera", self.smpl_params_camera)):
            for k, v in params.items():
                arrays[f"{frame}_{k}"] = _np(v)
        if self.betas_per_frame is not None:
            arrays["betas_per_frame"] = _np(self.betas_per_frame)
        if self.bbx_xys is not None:
            arrays["bbx_xys"] = _np(self.bbx_xys)
        if self.kp2d is not None:
            arrays["kp2d"] = _np(self.kp2d)
        np.savez(path, **arrays)
        return path

    def _raw_from_fields(self) -> dict:
        raw = {
            "smpl_params_global": self.smpl_params_world,
            "smpl_params_incam": self.smpl_params_camera,
            "K_fullimg": self.intrinsics,
        }
        if self.betas_per_frame is not None:
            raw["betas_per_frame"] = self.betas_per_frame
        if self.bbx_xys is not None:
            raw["bbx_xys"] = self.bbx_xys
        if self.kp2d is not None:
            raw["kp2d"] = self.kp2d
        return raw

    # ---- rendering (reuses the demo's moderngl overlay renderer) ---------------------------------
    def render(
        self,
        path: str | Path | None = None,
        *,
        view: str = "both",
        skeleton: bool = False,
        skeleton_overlay: bool = False,
        skeleton_joints: str | None = None,
        render_scale: float | None = None,
        mesh: str = "smpl",
    ) -> Path:
        """Render the overlay video(s) and return the path.

        ``view`` selects which video to return: ``"both"`` (in-cam ∥ world side-by-side, the default),
        ``"world"``, or ``"camera"``. ``mesh='smplx'`` renders the native ~10475-vertex SMPL-X surface
        (needs only the SMPL-X body model) instead of the default SMPL-topology mesh. Needs the
        pipeline-staged artifacts (produced by ``pipeline(video)``). If ``path`` is given the chosen
        video is copied there.
        """
        if self._cfg is None:
            raise RuntimeError(
                "render() needs the artifacts the pipeline stages on disk. Produce this result via "
                "gvhmr.pipeline(...)(video) (optionally with output_dir=), then call render()."
            )
        from omegaconf import OmegaConf

        from gvhmr.cli.demo import _render
        from gvhmr.utils.device import get_device
        from gvhmr.utils.vis.skeleton import resolve_joint_subset

        cfg = self._cfg
        if render_scale is not None:
            OmegaConf.update(cfg, "render_scale", float(render_scale))
        # The renderers read the motion back from disk — make sure it is there.
        if not Path(cfg.paths.hmr4d_results).exists():
            self.save(cfg.paths.hmr4d_results)
        joint_indices = resolve_joint_subset(skeleton_joints)
        device = self._device or get_device()
        ok = _render(
            cfg, device, skeleton=skeleton, skeleton_overlay=skeleton_overlay, joint_indices=joint_indices, mesh=mesh
        )
        if not ok:
            raise RuntimeError(
                "Rendering was skipped (missing SMPL body model or no GL context). "
                "See `gvhmr info`; the motion itself is available via .save()/.save_npz()."
            )
        produced = Path(
            {
                "both": cfg.paths.incam_global_horiz_video,
                "world": cfg.paths.global_video,
                "camera": cfg.paths.incam_video,
            }[view]
        )
        if mesh == "smplx":  # native SMPL-X renders to _smplx-suffixed paths
            produced = produced.with_stem(produced.stem + "_smplx")
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(produced, path)
            return path
        return produced


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
