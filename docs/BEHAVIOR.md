# Behaviour preservation

The prime directive of this fork is: **the model must compute exactly what it did
upstream.** This file lists the traps that can silently change model output, and the
testing approach that guards them. Read it before editing numeric / model code.

## The test net

`tests/` is a **characterization (regression) suite**, not a spec. It pins the current
behaviour of the pure math the model depends on so refactors can't change numerics. It
runs on CPU / Apple-Silicon MPS with **no GPU, checkpoints, or datasets**.

Covered today (≈ 220 tests):

- `gvhmr/utils/geo/quaternion.py` — quaternion algebra, 6D, slerp (wxyz convention)
- `gvhmr/utils/geo/transforms.py` — the QuaterNet-derived rotation helpers
- `gvhmr/network/base_arch/embeddings/rotary_embedding.py` — RoPE math
- `gvhmr/utils/eval/eval_utils.py` — similarity-transform alignment (**pins the SVD V/Vᴴ risk**)
- `gvhmr/utils/geo/hmr_cam.py` — camera intrinsics, bbox, projection
- `gvhmr/utils/seq_utils.py`, `gvhmr/utils/net_utils.py` — sequence/tensor helpers
- `gvhmr/model/gvhmr/utils/stats_compose.py` — the 151-dim normalization layout
- `gvhmr/model/common_utils/scheduler.py` — warmup/step LR
- `gvhmr/utils/preproc/relpose/transformation_np.py` — numpy pose interpolation
- `gvhmr/utils/_vendor/pytorch3d` & `gvhmr/network/hmr2/utils/geometry.py` — rotation conversions
- config registration + `demo`/`train` composition; device + CPU/MPS forward parity

**Workflow:** before changing behaviour-sensitive code, ensure a test pins the current
behaviour; keep it green through the change. When you fix a bug, add a test (or convert
an `xfail`).

## Landmines

1. **The 151-dim latent layout** (`EnDecoder.decode`):
   `[0:126] body_pose_r6d / [126:136] betas / [136:142] global_orient_c /
   [142:148] global_orient_gv / [148:151] local_transl_vel`.
   `stats_compose` mean/std vectors must keep this exact order; `gvhmr_pipeline` masks
   `[..., 142:]` for 3DPW. Reordering silently corrupts decode and loss.

2. **Checkpoint loading is by module-attribute name** (`load_state_dict`). Renaming a
   submodule/buffer (`blocks.N`, `pred_cam_head`, `static_conf_head`, `ROPE.encoding`,
   `pred_cam_mean/std`) or changing a ctor default (`output_dim=151`, `latent_dim=512`,
   `num_layers=12`, `cliffcam_dim=3`, …) breaks loading the released checkpoint. The
   package rename is safe for this (state-dict keys are attribute paths, not import paths).

3. **Numerically load-bearing constants** must stay byte-identical:
   `pred_cam_mean=[1.0606,-0.0027,0.2702]`, `pred_cam_std=[0.1784,0.0956,0.0764]`,
   `clamp_min 0.25`, visible-mask threshold `0.5`, rotary base `10000`, LayerNorm eps
   `1e-6`, `drop_path_rate 0.55`, GELU `approximate='tanh'`; eval foot verts
   `[3216,3387,6617,6787]`, pelvis `[1,2]`, fps `30`, m→mm `1000`; preproc crop
   `[:,:,:,32:224]`, HMR2 crop `[:,:,:,32:-32]`; bbox aspect `[192,256]`, enlarge `1.2`;
   joint ids `[7,10,8,11,20,21]`.

4. **`torch.svd` → `torch.linalg.svd` returns Vᴴ (= Vᵀ), not V.** Transpose back, or the
   recovered rotation inverts. Pinned by `tests/test_eval_utils.py`. (Already converted in
   `eval_utils.py`, `geo_transform.py`, `flip_utils.py`.)

5. **`autocast(enabled=False)` forces fp32** for FK/IK/rotary/global-rollout and must stay
   *disabled*. The modernized form is `torch.amp.autocast("cuda", enabled=False)` (a no-op
   on CPU/MPS, fp32-forcing on CUDA).

6. **Quaternion convention split:** `geo/quaternion.py` and the vendored pytorch3d are
   **wxyz** (real-first); the lower half of `gvhmr/utils/matrix.py` is **xyzw** (IsaacGym).
   `matrix.py` also has duplicate definitions where the *later* one wins — preserve which
   impl is effective per call site if you split it.

7. **Dataset RNG ordering is load-bearing.** Camera/pose augmentors draw `np.random` in a
   branch-specific order; reordering or switching to `Generator` changes the training
   stream. Leave augmentation untouched in mechanical passes.

8. **Config/code mismatch:** `gvhmr/configs/siga24_release.yaml` sets the network `_target_`
   to `NetworkEncoderRoPEV2`, which **does not exist** (only `NetworkEncoderRoPE` does). The
   live demo path composes `demo.yaml` and uses V1. Don't assume `siga24_release.yaml` loads.

9. **`SimpleVO` serial vs parallel are intentionally not bit-identical** (pycolmap RANSAC
   `random_seed=-1`). The demo default is serial — don't "unify" the paths.
