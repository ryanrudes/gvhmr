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
| **cudnn.benchmark** in `get_device()` | kernel autotuning on the fixed-shape ViT convs | selection only; no convs in the denoiser, benchmarks unaffected |
| ~~**TF32**~~ — **OFF by default since 2026-07-13** (opt in with `$GVHMR_ENABLE_TF32`) | **none measured** (EMDB eval: 76.0s on → 72.8s off) | ✗ **NOT safe: +3.3mm PA-MPJPE and 4× accel error on EMDB** — see below |
| **bf16 autocast** on ViTPose + HMR2 (`$GVHMR_PREPROC_FP32` to disable) | HMR2 254→64ms (4.0×), ViTPose 252→57ms (4.4×) | feature 1.6% rel, kp2d 0.02px; **eval-certified: bf16-vs-fp32 worst \|Δ\|=0.08mm on 3DPW** (PA-MPJPE 43.71 vs 43.63) — noise |
| **SDPA/Flash attention** in the two vendored ViTs | 1.1-1.3× (bf16) | fp32 max\|Δ\|≤2e-6 |
| **skip-init + mmap** model loading | HMR2 4.9s→0.35s, ViTPose ~5s→0.42s (~13×) | every param bit-identical |
| **Resident-model cache** (`preproc/base.py`, `$GVHMR_NO_MODEL_CACHE`) | −13s per extra clip (folder/library/Space) | identical |
| **SimpleVO ⇄ GPU overlap** (`$GVHMR_NO_VO_OVERLAP`) | hides the CPU camera stage | identical |
| **Staging re-encode skip** (~30fps upright → symlink), x264 veryfast, cached probe | −1 decode+encode | reads original pixels |
| ViTPose **flip-test off** + ByteTrack — OPT-IN `--fast` only | 2.2× on 2D-pose | **eval-certified cost: +2.3 PA-MPJPE / +7.6 MPJPE / +8.7 PVE (mm) on 3DPW** — real, so opt-in only |

Net: **~3-4× end-to-end on a long clip** (YOLO, still fp32-CNN, becomes the bottleneck), more on short
clips where the ~13× load win dominates. Retrain-gated levers (faster default detector, RTMPose, fused
single-ViT, TensorRT) are opt-in / future work.

## The TF32 regression (2026-07-06 → 2026-07-13) — read before adding a "free" fast path

The speedup pass above enabled **TF32** globally in `get_device()` as a free win. It was not free, and it
shipped broken for a week. On the **released checkpoint**, TF32 on vs off:

| EMDB-1 | TF32 on | TF32 off | paper |
|---|---|---|---|
| PA-MPJPE (mm) | 46.0 | **42.7** | 42.7 |
| MPJPE (mm) | 75.6 | **72.6** | 72.6 |
| **Accel (m/s²)** | **14.2** | **3.6** | **3.6** |
| EMDB-2 Jitter | 24.9 | **16.1** | 16.5 |

TF32 is now **off by default** (`gvhmr.utils.device.tf32_enabled`, opt in with `$GVHMR_ENABLE_TF32`),
pinned by `tests/test_device.py::test_tf32_is_off_unless_opted_in`. Three lessons worth keeping:

1. **`autocast(enabled=False)` does not gate TF32.** The original rationale claimed the fp32 FK/IK guards
   kept TF32 away from the numerically-sensitive code. They don't: `autocast` suppresses bf16/fp16
   autocasting only, while TF32 is a separate switch (`allow_tf32`) applying to *every* fp32 matmul. The
   rotary/attention path was never protected.
2. **Derivative metrics amplify what pose metrics hide.** Accel scales `fps²` (900×) and jitter `fps³`
   (27,000×), so a perturbation far too small to move PA-MPJPE on 3DPW (+0.2mm) becomes a **4× accel
   error** on EMDB's full-length sequences. *A benchmark that only looks at pose error cannot certify a
   numerics change.* 3DPW and RICH both looked clean — only EMDB caught it.
3. **It bought nothing.** An EMDB eval takes 76.0s with TF32 and 72.8s without. The hot stages are already
   bf16 (the ViTs — bf16 is *faster* than TF32: 1.68ms vs 3.74ms on a ViT-H MLP block), fp16 (YOLO), or on
   CPU (the network `predict`), so TF32 never engaged where the time actually goes. The "2.7× on fp32 ViT
   matmul" in the old table was a microbenchmark of a path the pipeline does not run.

Certify any numerics change against **`gvhmr eval` including EMDB accel/jitter**, not just 3DPW pose.

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

Each must keep `tests/test_golden_inference.py` green.

## Training: `torch.compile` (opt-in, ~1.9×)

`model.compile_denoiser=true` compiles the RoPE denoiser for **training** (`gvhmr train`). Default off.

Why it pays: the denoiser is *small* (40.9M params, 12×512, L=120) and dominated by many tiny sequential
kernels, so a training step reaches only **~7% MFU** on an RTX 6000 Ada (~12 of ~182 bf16 TFLOPS). It is
launch-overhead-bound, not tensor-core-bound — precisely the workload kernel fusion fixes. Measured
**1.9× on fwd+bwd**, numerically faithful (fp32 max |Δ| = 1.4e-7 vs eager — rounding noise).

Two landmines, both guarded by `tests/test_compile.py`:

- **Compile the `forward`, not the module.** `torch.compile(module)` returns an `OptimizedModule` that
  prefixes every state_dict key with `_orig_mod.`. GVHMR loads **strict**, so that silently breaks every
  checkpoint the run saves — you'd only find out when `gvhmr eval` refuses to load it. `net_utils.compile_forward`
  compiles the bound method, leaving the state_dict untouched and checkpoints interchangeable with eager runs.
- **Don't compile one arm of an A/B and not the other.** It's faithful to fp32 rounding, but training is
  chaotic; keep both arms on the same setting.

The same ~7% MFU is why an **H100 is a weaker upgrade than the spec sheet implies** for *training*: it has
~5× the tensor throughput but only ~3.5× the bandwidth, and at 7% MFU you capture the bandwidth, not the
tensor cores — expect ~1.5-2×, not 5×. (Preproc is the opposite: the two ViT-Huge models are real
tensor-core work, and *that* is where a bigger GPU pays.) The larger training levers are fusion (above), a
bigger batch (the GPU is starved at this size), and DDP across GPUs.
