# Performance

Benchmark the core inference path (model forward + decode + postprocess, excluding
preprocessing/rendering):

```bash
uv run gvhmr bench                       # auto device
GVHMR_DEVICE=cpu uv run gvhmr bench       # force CPU
```

The original optimizations (next section) are **behavior-preserving** — guarded by
`tests/test_golden_inference.py` (golden output fingerprint from the released
checkpoint + CPU determinism + CPU/MPS parity). The later **2026-07 speedup pass**
is **accuracy-first, not all byte-identical**: its accuracy-sensitive defaults (bf16)
are `gvhmr eval`-certified, and the one real trade (flip-test off) is opt-in `--fast`.

## Optimizations applied

| Change | Where | Win | Numerics |
|---|---|---|---|
| In-place forward kinematics at inference | `gvhmr/utils/matrix.py::forward_kinematics` | O(N²)→O(N) tensor assembly (the `torch.cat`-per-joint existed only for autograd) | **byte-identical** |
| Vectorized the static-joint suffix loop | `gvhmr/model/gvhmr/utils/postprocess.py::pp_static_joint_cam` | O(L²)→O(L) via `cumsum` | ≤1e-5 (sum reassoc.) |
| Fused attention | `gvhmr/network/base_arch/transformer/encoder_rope.py` | `F.scaled_dot_product_attention` replaces manual einsum/softmax | ≤1e-3 (fused softmax); big win on CUDA/flash |

Combined: **~205 → ~155 ms/call at L=256 on CPU (≈24% faster)**, with the model
inference running **~50× realtime** on CPU (8.5 s of video in ~170 ms).

## The 2026-07 speedup pass (measured on RTX 6000 Ada, min-of-N)

Priority: **accuracy first, then maximum speed under that constraint.** Preproc (two ViT-Huge models)
is ~85-95% of a run; the network `predict` is small. Everything ran fp32 with tensor cores idle. The
accuracy-safe wins are **on by default**; the one real accuracy trade (ViTPose flip-test off) is opt-in
via `--fast` / `--recipe fast`.

| Change | Speedup | Numerics / accuracy |
|---|---|---|
| Network `predict` on **CPU** (`device.predict_device()`, `$GVHMR_PREDICT_DEVICE`) | ~7-9× on predict (launch-bound) | golden-byte-identical |
| **TF32 + cudnn.benchmark** in `get_device()` (`$GVHMR_DISABLE_TF32`) | 2.7× on fp32 ViT matmul | negligible (≤~1e-3), CUDA-only |
| **bf16 autocast** on ViTPose + HMR2 (`$GVHMR_PREPROC_FP32` to disable) | HMR2 254→64ms (4.0×), ViTPose 252→57ms (4.4×) | feature 1.6% rel, kp2d 0.02px; eval-certified |
| **SDPA/Flash attention** in the two vendored ViTs | 1.1-1.3× (bf16) | fp32 max\|Δ\|≤2e-6 |
| **skip-init + mmap** model loading | HMR2 4.9s→0.35s, ViTPose ~5s→0.42s (~13×) | every param bit-identical |
| **Resident-model cache** (`preproc/base.py`, `$GVHMR_NO_MODEL_CACHE`) | −13s per extra clip (folder/library/Space) | identical |
| **SimpleVO ⇄ GPU overlap** (`$GVHMR_NO_VO_OVERLAP`) | hides the CPU camera stage | identical |
| **Staging re-encode skip** (~30fps upright → symlink), x264 veryfast, cached probe | −1 decode+encode | reads original pixels |
| ViTPose **flip-test off** — OPT-IN `--fast` only | 2.2× on 2D-pose | real train/test trade — not a default |

Net: **~3-4× end-to-end on a long clip** (YOLO, still fp32-CNN, becomes the bottleneck), more on short
clips where the ~13× load win dominates. Retrain-gated levers (faster default detector, RTMPose, fused
single-ViT, TensorRT) are opt-in / future work.

## Verified end-to-end on Apple Silicon

The full pipeline (preprocessing + inference) runs on Apple-Silicon MPS. On `tennis.mp4`
(312 frames / 10.4 s, MPS):

| Stage | Time |
|---|---|
| YOLOv8 tracking | ~24 s |
| ViTPose 2D keypoints | ~31 s |
| HMR2 ViT features | ~15 s |
| **Preprocess total** | **~83 s** |
| GVHMR inference (`predict`) | **~2.9 s** |

Preprocessing dominates wall-clock and is where MPS helps (big batched ViT models).
SimpleVO camera estimation (pycolmap) also runs.

**Rendering** (optional overlay videos) runs on the **GPU via a moderngl (OpenGL/Metal)
renderer** — `gvhmr/utils/vis/renderer_gl.py`, selected by `make_renderer`. It builds a
pinhole camera straight from the model's intrinsics `K` (so the overlay lands exactly where
`perspective_projection` puts the joints) and flat-shades via screen-space derivatives (no
per-vertex normals). On an M2 Max it renders the SMPL body at **~60 fps in-cam and ~330 fps
world-view at 540p** — the full 356-frame skateboard demo renders in **~5 s total** (in-cam
4 s + world 1 s).

This replaced the old pytorch3d path, which had **no Apple-GPU backend** and fell back to a
**CPU software rasterizer** — ~0.6–2.5 s/*frame*, i.e. minutes for a clip (the mesh isn't the
problem; a GPU draws 14k triangles in ~1 ms). pytorch3d is kept as a fallback (CUDA boxes, or
a headless machine with no GL context) behind the `render` extra; `make_renderer` prefers the
GPU path and falls back automatically. `moderngl` is a base dependency, so rendering works out
of the box — **no pytorch3d build needed on a Mac**. `--render_scale` still trades resolution
for speed, but at GPU speeds full resolution is cheap. Rendering never blocks the run — the
SMPL motion is saved first.

## Device note (important)

`get_device()` auto-selects **MPS** on Apple Silicon, but the *GVHMR model inference*
(the demo `predict`) is **faster on CPU** here — it is latency-bound on the IK
post-processing's thousands of tiny sequential ops, where MPS kernel-launch overhead
dominates (≈1900 ms on MPS vs ≈155 ms on CPU at L=256). MPS *does* help the heavy,
batched **preprocessing** models (YOLO / ViTPose / HMR2 ViT), which dominate wall-clock
for real videos. Rule of thumb: preprocessing → MPS/CUDA; the GVHMR model `predict` →
CPU is fine (set `GVHMR_DEVICE=cpu` if you only run the model). Mesh rendering needs
CUDA + pytorch3d regardless.

## Remaining opportunities (not yet done)

Profiled hotspots after the above (L=256, CPU): the **CCD IK** (`process_ik`) is now the
largest cost. Candidate behavior-preserving wins, in rough order of payoff/risk:

1. **Chain-restricted FK in the IK.** `CCD_IK` recomputes forward kinematics over the
   full 22-joint skeleton after every joint update, though only the ~5-joint chain
   changes. Restricting FK to the active chain would cut most of the IK cost. (Medium risk.)
2. **Batch the four IK calls** (left/right leg, left/right hand) instead of running them
   sequentially. (Medium risk — different chains/targets.)
3. **Scan the rollout-merge recurrence** (`process_ik` lines ~126-130) and the
   state-dependent loop in `pp_static_joint_cam` with an associative scan. (Higher risk —
   exact float order.)
4. **`torch.compile`** the network forward for CUDA deployments.

Each must keep `tests/test_golden_inference.py` green.
