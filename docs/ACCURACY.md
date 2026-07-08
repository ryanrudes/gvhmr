# Accuracy

How to get the most accurate motion out of GVHMR at inference time (no retraining), and
the evidence behind each lever. The released checkpoint is unchanged — these are test-time
techniques and better-conditioned inputs.

## TL;DR

| Lever | How | Effect | Status |
|---|---|---|---|
| **Flip-test TTA** | `gvhmr demo VIDEO --flip-test` | **−0.66 mm PA-MPJPE on 3DPW** (ground-truth, full test set) | **implemented (opt-in)** |
| **Correct camera intrinsics** | `--f_px <px>` / `--intrinsics <file>` / `--f_mm <mm>` / metadata | Fixes world-frame depth/scale (+ true principal point, per-frame) | **implemented** |
| **DPVO camera** (moving cam) | `--use-dpvo` | More accurate camera trajectory than SimpleVO | available |
| **VO carry-forward** | automatic | Degenerate frame-pairs continue the last motion instead of freezing | **implemented** |
| **Static-cam trajectory** | `--incam-world-traj` (default on for `-s`) | Recovers scene traversal (gliding/skateboarding) the velocity prior misses | **implemented** |

## Measuring accuracy without ground truth

The eval datasets (EMDB/3DPW/RICH) aren't needed to iterate locally. We use two
no-ground-truth proxies on a demo clip:

- **Reprojection error (RE)** — predicted COCO-17 SMPL joints (in-cam) projected with the
  camera `K`, vs. the ViTPose 2D keypoints, confidence-weighted (px). Measures the in-cam 2D
  fit. **Caveat:** it's depth-ambiguous and in-cam only, so it *cannot* see focal/world-frame
  changes and *can be gamed* by anything that overfits 2D.
- **Jitter** — mean joint acceleration (temporal smoothness). An independent check: a
  technique that lowers RE while *raising* jitter is usually trading 3D for 2D fit.

Always read the two together. (The harness lives in the research scratchpad; the core metric
is ~15 lines — project `endecoder`'s coco17 joints with `perspective_projection` and compare
to the cached `vitpose.pt`.)

## Empirical study (tennis.mp4, 312 frames)

| technique | reproj (px) ↓ | jitter ↓ | verdict |
|---|---|---|---|
| baseline (1 forward) | 8.265 | 17.10 | — |
| **flip-test + avg** | **8.10** | **16.9** | ✓ both improve — proven (eval uses it) |
| time-reversal + avg | 8.07 | 16.7 | ✓ both improve (free; not yet wired in) |
| flip × reversal (4-way) | 8.14 | 16.9 | ✓ but no better than the singles |
| multi-crop ×5 + avg | 8.16 | 17.5 | ~ marginal RE, slightly worse jitter |
| drop-imgseq + avg | 7.06 | 19.4 | ✗ **games RE** — jitter ↑, hurts 3D depth |
| MC-dropout ×8 + avg | 8.26 | 19.3 | ✗ neutral (dropout 0.1 too weak) |
| focal sweep (re-run/focal) | flat 8.25–8.27 | — | RE **can't see it** (focal↔depth ambiguity) |

Takeaways: the **ensembling** techniques that average a symmetry-transformed pass (flip,
time-reversal) genuinely help and are safe — they reduce a directional/left-right bias.
`drop-imgseq` is the cautionary tale: it slashes RE by ignoring the image features that
constrain depth, but raises jitter — a metric trap the second proxy catches. Focal is a real
*world-frame* lever that the in-cam proxy structurally can't validate.

## Ground-truth validation (3DPW, full test set)

The reprojection proxy is confirmed by the real benchmark. We downloaded GVHMR's preprocessed
3DPW support data and ran the actual in-cam metrics (PA-MPJPE / MPJPE vs. the 3DPW SMPL
annotations) over all 37 test sequences, baseline vs. flip-test:

| metric (mm) | baseline | flip-test | Δ |
|---|---|---|---|
| PA-MPJPE | 37.01 | **36.35** | **−0.66** |
| MPJPE | 56.53 | **55.74** | **−0.79** |

Flip-test improves both — modest but real and consistent (it's the benchmark-time setting).
The eval runs trainer-free on CPU/MPS (`tools/eval/` — replicates `MetricMocap` exactly); the
3DPW/EMDB/RICH test datasets needed a `weights_only=False` fix to load on torch ≥ 2.6.

## Implemented levers

### Flip-test TTA (`--flip-test`)
Runs the model on the video **and its horizontal mirror**, then averages the in-cam SMPL
(betas/body_pose/global_orient; keeps the base translation, since the mirror flips it). This
is exactly what `GvhmrPL.validation_step` does to produce the paper's benchmark numbers —
the demo just never turned it on. Cost: one extra HMR2 feature pass (on the mirrored video) +
a second, cheap transformer forward. The default path is **byte-identical** when the flag is
off (guarded by `tests/test_golden_inference.py`). Implementation:
`DemoPL.predict(..., flip_test_data=...)` in `gvhmr/model/gvhmr/gvhmr_pl_demo.py`.

### Camera intrinsics (`--f_px` / `--intrinsics`, `--f_mm`, + metadata)
`estimate_K` assumes focal = image diagonal (≈53° diagonal FOV) with a centred principal point
— a guess. The true focal pins world-frame depth/scale; a true principal point also nudges the
recovered translation/orientation. The model reads intrinsics **per-frame** in two places — the
CLIFF-cam network input `[(bcx−cx)/f, (bcy−cy)/f, b/f]` and the metric depth `tz = 2f/(s·b)` —
but consumes only `K[0,0]` (fx) as "the" focal (it assumes **square pixels**, so a separate `fy`
is stored faithfully yet unused) and `K[0,2]/K[1,2]` as the principal point.

Ways to supply intrinsics, **highest precedence first**:

- **`--intrinsics <file>`** — a JSON/NPZ sidecar: `fx`/`fy`/`cx`/`cy` (each a scalar **or a
  per-frame list**, for a zoom/lens-switch) or a full `K` (`(3,3)` or `(L,3,3)`). Optional
  `width`/`height` declare the calibration resolution (auto-rescaled to the staged frames). The
  faithful, fully-general path — the only one carrying a real principal point or per-frame
  values. Auto-detected as `<video>.intrinsics.json` next to the input. (Library: `intrinsics=`
  also accepts a dict or a `K` array/tensor.)
- **`--f_px <px>`** — focal length in **pixels**, straight into `K`. Prefer this over `--f_mm`
  when you know your camera's pixel focal (no lossy mm round-trip).
- **`--f_mm <mm>`** — a full-frame (35mm-equiv) focal, mapped to pixels by the diagonal ratio
  `f_px = √(W²+H²)/√(24²+36²)·f_mm`. Convenient for phones (iPhone 1×≈24, 2×≈48).
- **metadata** — when nothing is passed, a best-effort 35mm-equiv read (exiftool/ffprobe).
- else the diagonal-FOV **heuristic**.

Per-frame arrays must have one value per *staged* 30fps frame (resample them if the source fps
differed). GVHMR is a pure pinhole model, so a `distortion` entry in the sidecar makes the demo
**undistort the staged frames** and swap in the corrected pinhole `K` (for wide-angle/fisheye lenses).
(We verified RE cannot validate the focal lever — it's
depth-ambiguous — so prefer measured intrinsics when world-frame accuracy matters; it won't
visibly change the 2D overlay.) **Full sidecar format + conversions: [CAMERA_METADATA.md](CAMERA_METADATA.md).**

### VO carry-forward (automatic)
For non-static cameras, SimpleVO solves a relative pose per adjacent frame-pair. Degenerate
pairs (fast motion / low texture) used to **freeze** the camera (identity); they now **carry
forward the last valid relative motion**, a closer continuation. `solver_two_view.solve()`
returns `None` on degeneracy; `simple_vo` resolves it and logs how many pairs were affected.

## Phone videos (.MOV) — orientation & focal

Phone clips (iPhone `.MOV`) need two bits of metadata the raw decoder ignores:

- **Orientation.** iPhones store the recording orientation as a **display-matrix rotation
  flag**, not baked into the pixels. The PyAV decode backend returns raw frames, so a clip
  shot upside-down/sideways was processed rotated (everything came out wrong).
  `get_video_rotation` (`gvhmr/utils/video_io_utils.py`) reads the flag and the readers apply
  `np.rot90`; because the demo re-encodes ("stages") the input, the correction is baked into
  the staged video, so **every** downstream consumer (YOLO, ViTPose, HMR2, VO, render) stays
  consistent. No-op for normal videos (k=0).
- **Focal length (zoom-aware).** Phones record the **35mm-equivalent focal length** — which
  already accounts for the lens and zoom in use (iPhone: ~15mm on the 0.5× ultrawide, ~24mm at
  1×, ~48mm at 2×) — in QuickTime metadata that `ffprobe` doesn't surface. `focal_mm_from_metadata`
  prefers **`exiftool`** (`FocalLengthIn35mmFormat`) and feeds it in as `--f_mm` automatically.
  Install `exiftool` (`brew install exiftool`) to enable this; otherwise pass `--f_mm`/`--f_px`.
  (This feeds one value for the clip; for a mid-clip zoom, supply **per-frame** focal via
  `--intrinsics` — see *Camera intrinsics* above.)
- **Frame rate.** GVHMR is a **30fps model** — the training motion is downsampled to 30fps and
  the network integrates *per-frame* velocities, so frame spacing (1/30 s) is baked into the
  dynamics. Phones often shoot **60fps**; fed as-is, the motion comes out at half speed and
  out-of-distribution. The demo now reads the true `avg_frame_rate` (not the misleading
  `r_frame_rate` timebase) and **resamples to 30fps** when staging the video, so any frame rate
  works (60fps → drops every other frame; ~30fps is left untouched). Recording at 30fps or 60fps
  is ideal (clean frame-drop); below 30fps gets upsampled by duplication.

## Global trajectory (scene traversal)

GVHMR recovers **world translation from a learned velocity prior** (`rollout_local_transl_vel` =
`cumsum(R·local_transl_vel)`); the camera only sets the gravity-view *orientation*, never the
translation magnitude. That prior is trained on **stepping locomotion** (walking/running), so it
**misses translation without stepping** — skateboarding, scooters, gliding — where the person
moves but the body looks static. It's a *general* weakness, benchmarked: on EMDB-2 the
**`P8_64_outdoor_skateboard` sequence is the single worst in the dataset** (RTE 12.29 vs ~1.5
mean; stairs sequences fail similarly).

The fix signal exists: the model's **in-cam** prediction tracks the person correctly. Carrying it
through the camera cuts the skateboard trajectory error **22×** (16.3 m → 0.75 m, vs GT camera).

### Moving-camera research (validated method, not yet shipped)

Pursued the general (moving-camera) version with the EMDB-2 harness; full findings:

- **Pure-geometric regresses walking** (WA2-MPJPE 275 → 384 mm) — the in-cam carry is depth-noisy.
- **Naive fusion blows up** (WA2 2070) from a heading/frame mismatch (prior is gravity-view, geometric
  is camera-world).
- **Frame-aligned fusion works.** Both frames share gravity, so they differ by one per-sequence heading
  rotation (solved from frame 0). Pairing the **geometric gross path** with the **prior's local motion**
  (low/high-frequency split) gives, across all 25 EMDB-2 sequences: **RTE 1.92 → 1.72, WAA-MPJPE
  110.6 → 108.1 (both beat the prior)**, the skateboard **RTE 12.29 → 1.2 (10×)**; only the per-2s WA2
  remains a (shrinking) gap — the residual monocular-depth limit.
- **The real DPVO camera is good enough geometrically** — with estimated (not GT) SLAM, the geometric
  trajectory is **3.15 m vs the prior's 16.31 m** on the skateboard (5×) once DPVO's world frame is
  similarity-aligned to metric (camera path mean error 0.62 m over 11.6 m).
- **But the no-GT, test-time pipeline hits a fundamental wall.** Building it without GT needs a scale
  solve; `minimize-acceleration` is unstable, and the principled **foot-contact** scale solve (a planted
  foot is world-stationary) plus a **contact-gated** velocity blend (trust the prior under contact, the
  geometric only during sustained no-contact) gives a *safe small* gain across EMDB-2 (RTE 1.916→1.879,
  WAA 110.6→108.9, **no walking regression**) — **but does not fix the skateboard** (12.29, unchanged).
  The reason is decisive: the model reports **foot contact in 93%** of skateboard frames (the feet look
  planted on the board), so the gliding case defeats *all three* signals these fixes rely on — the
  velocity prior, the contact gate, **and** the contact-based scale solve all assume planted-foot =
  world-stationary, which a moving board violates.

**Conclusion:** a deployable, robust moving-camera fix can't be assembled purely from GVHMR's outputs at
test time — the following-camera + gliding ambiguity needs an **independent world-motion signal**, i.e.
SLAM **scene points** (TRAM's core contribution), not the human. What's solid and reusable: the
GT-camera fusion is validated as the right *method*, the EMDB-2 world harness (`tools/eval/`) is built,
and the failure modes are fully mapped. The shipped, clean win remains the **static-camera** fix above.

**Implemented — static-camera trajectory** (`--incam-world-traj`, default on for `-s`).
When the camera barely moves, there's no scale/translation ambiguity: the world trajectory is just
the **in-cam displacement rotated into the world frame** (`DemoPL._world_transl_from_incam`), using
the same camera→world rotation that maps the body orientation — so it's frame-consistent, no jitter
explosion. It keeps the model's world orientation + articulation and changes only the root path.
On the skateboard clip it recovers **1.76 m** of traversal vs the prior's 0.68 m. **Gated on a
static camera**, so moving-camera results (EMDB, the benchmark) are unchanged by construction —
zero regression — and the default `predict` path stays golden-identical (`world_from_incam=False`).

## Not adopted (and why)

- **drop-imgseq / condition ablation** — games RE, hurts 3D (jitter ↑).
- **MC-dropout** — the model's 0.1 dropout is too weak to form a useful ensemble.
- **Multi-scale bbox TTA** — only marginal here; full benefit needs per-scale feature
  re-extraction (expensive) for little gain.
- **SMPLify-style 2D reprojection refinement** — high ceiling but directly optimizes (and
  thus games) RE; needs strong temporal + pose-prior regularization to avoid pulling joints
  off the 3D manifold. Left as future work behind those guards.

## Candidates for more (validated as promising, not yet shipped)

- **Time-reversal TTA** — measured as good as flip-test and free (no re-extraction); the only
  subtlety is reversing `cam_angvel` correctly for moving cameras. Easy to add for static cam.
- **Overlapping-window averaging** for clips > 120 frames (the transformer switches to a local
  sliding-attention window there; overlapping inference averages out seam effects).
- **VO ensembling** (SimpleVO + DPVO geodesic fusion) for world-frame robustness — measure
  with a foot-skate proxy, not RE.
