from __future__ import annotations

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from gvhmr.configs import MainStore, builds
from gvhmr.model.gvhmr.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik
from gvhmr.utils.geo.flip_utils import avg_smplx_aa, flip_smplx_params
from gvhmr.utils.geo.hmr_cam import normalize_kp2d
from gvhmr.utils.geo.rotations import axis_angle_to_matrix
from gvhmr.utils.net_utils import torch_load_mmap
from gvhmr.utils.pylogger import Log


class DemoPL(pl.LightningModule):
    def __init__(self, pipeline) -> None:
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)

    def _batch(self, data: dict) -> dict:
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        return {k: v.to(self.device) for k, v in batch.items()}  # device-agnostic (cuda / mps / cpu)

    @torch.no_grad()
    def predict(
        self,
        data: dict,
        static_cam: bool = False,
        flip_test_data: dict | None = None,
        world_from_incam: bool = False,
    ) -> dict:
        """auto add batch dim
        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (F, 3)
            "bbx_xys": (F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 3)
            "f_imgseq": (F, 3, 256, 256)
        }

        flip_test_data: optional dict with the same keys built from the horizontally-flipped
            video (flipped kp2d/bbx and re-extracted f_imgseq). When given, runs flip-test TTA:
            the model is evaluated on the input and its mirror and the in-cam SMPL is averaged
            (the same averaging the eval/benchmark path uses), reducing left/right bias.

        world_from_incam: for a **static camera**, derive the world root trajectory from the
            in-cam prediction (the in-cam displacement rotated into the world frame) instead of
            the learned velocity prior. The prior infers translation from stepping locomotion and
            misses scene traversal without foot-stepping (skateboarding, scooters, gliding); the
            in-cam signal carries the true motion. Only valid when the camera barely moves.
        """
        do_flip_test = flip_test_data is not None
        # Postpone post-processing (IK / static-joint) until after flip averaging, mirroring
        # the eval path; otherwise the default path runs it inline (byte-identical to before).
        batch = self._batch(data)
        outputs = self.pipeline.forward(batch, train=False, postproc=not do_flip_test, static_cam=static_cam)

        if do_flip_test:
            self._merge_flip_test(outputs, flip_test_data, static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            # Per-frame (pre-avgbeta) shape params, (L, 10). The `betas` inside smpl_params_* is the
            # single sequence-averaged shape the model actually uses; this is the frame-varying signal
            # before that collapse — exposed for callers who want it, never fed back into rendering.
            "betas_per_frame": outputs["pred_betas_per_frame"][0],
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        if world_from_incam:
            self._world_transl_from_incam(pred)
        return pred

    @staticmethod
    def _world_transl_from_incam(pred: dict) -> None:
        """Replace the world root trajectory with the in-cam displacement rotated into the world
        frame (static-camera only). Keeps the model's world orientation + articulation; only the
        translation changes. Frame-consistent: the same camera→world rotation that maps the body
        orientation also maps the displacement, so there is no heading mismatch."""
        ic, gl = pred["smpl_params_incam"], pred["smpl_params_global"]
        # camera→world rotation from the model's own frame-0 orientation pair (constant for a
        # static camera): R_world @ R_cam⁻¹.
        r_c2w = axis_angle_to_matrix(gl["global_orient"][0]) @ axis_angle_to_matrix(ic["global_orient"][0]).mT
        disp_cam = ic["transl"] - ic["transl"][0]  # (L, 3) person displacement in the camera frame
        gl["transl"] = gl["transl"][0] + (r_c2w @ disp_cam.mT).mT

    def _merge_flip_test(self, outputs: dict, flip_test_data: dict, static_cam: bool) -> None:
        """Average in-cam SMPL with the mirrored prediction, then post-process once.

        Faithful port of ``GvhmrPL.validation_step``'s flip-test: average betas / body_pose /
        global_orient (keep the base translation — the mirror flips it), copy pose+shape into
        the global branch, then run static-joint + IK post-processing on the global frame.
        """
        flipped = self.pipeline.forward(self._batch(flip_test_data), train=False, postproc=False)

        base = {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()}
        mirror = flip_smplx_params({k: v[0] for k, v in flipped["pred_smpl_params_incam"].items()})
        avg = base.copy()
        avg["betas"] = (base["betas"] + mirror["betas"]) / 2
        avg["body_pose"] = avg_smplx_aa(base["body_pose"], mirror["body_pose"])
        avg["global_orient"] = avg_smplx_aa(base["global_orient"], mirror["global_orient"])

        outputs["pred_smpl_params_incam"] = {k: v[None] for k, v in avg.items()}
        outputs["pred_smpl_params_global"]["betas"] = avg["betas"][None]
        outputs["pred_smpl_params_global"]["body_pose"] = avg["body_pose"][None]
        # Average the per-frame betas with the mirror's too (betas are mirror-invariant, so the flip
        # leaves them as-is), matching the shared-beta averaging above.
        outputs["pred_betas_per_frame"] = (outputs["pred_betas_per_frame"] + flipped["pred_betas_per_frame"]) / 2

        endecoder = self.pipeline.endecoder
        pp = pp_static_joint_cam if static_cam else pp_static_joint
        outputs["pred_smpl_params_global"]["transl"] = pp(outputs, endecoder)
        body_pose = process_ik(outputs, endecoder)
        outputs["pred_smpl_params_global"]["body_pose"] = body_pose
        outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

    def load_pretrained_model(self, ckpt_path: str) -> None:
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt type: {ckpt_path}")

        # weights_only=False: this is a trusted Lightning checkpoint carrying more than
        # tensors; torch>=2.6 defaults weights_only=True, which would refuse to load it.
        state_dict = torch_load_mmap(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


MainStore.store(name="gvhmr_pl_demo", node=builds(DemoPL, pipeline="${pipeline}"), group="model/gvhmr")
