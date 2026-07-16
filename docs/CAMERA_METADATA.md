# Camera intrinsics & metadata

GVHMR recovers metric human motion, so it needs the camera's **intrinsics** — the focal length and
principal point. By default it *guesses* them (focal = image diagonal, ≈53° FOV, principal point at
the image centre). If you know your camera's real intrinsics, supplying them **fixes world-frame
depth/scale** and the principal point. This is optional everywhere — GVHMR always falls back to the
guess when you provide nothing.

> **What it changes:** correct intrinsics improve the *metric* camera-frame placement (depth/scale)
> and centering — not the 2D mesh-on-video overlay, which is depth-ambiguous. See
> [ACCURACY.md](ACCURACY.md#camera-intrinsics---f_px---intrinsics---f_mm--metadata).

---

## TL;DR — pick one

| You have… | Use | Notes |
|---|---|---|
| Focal length **in pixels** (`fx`) | `--f_px <px>` | The faithful shortcut; no conversion. |
| Full calibration (`fx, fy, cx, cy`, maybe per-frame) | `--intrinsics file.json` | Most general — the only way to pass a real principal point or per-frame values. |
| A phone / a **35mm-equivalent** focal in mm | `--f_mm <mm>` | iPhone 1×≈24, 2×≈48, 0.5×≈15. Mapped to pixels by the sensor-diagonal ratio. |
| Nothing | *(default)* | Reads 35mm-equiv focal from video metadata (`exiftool`/`ffprobe`), else the FOV heuristic. |

**Precedence, highest first:** `--intrinsics` (or an auto-detected sidecar) → `--f_px` → `--f_mm` →
video metadata → FOV heuristic.

---

## The intrinsics sidecar (`--intrinsics`)

A small **JSON** (or NPZ/NPY) file. The simplest form gives focal length in **pixels** and, optionally,
the principal point:

```json
{
  "width": 1920,
  "height": 1080,
  "fx": 1450.0,
  "fy": 1450.0,
  "cx": 964.2,
  "cy": 541.8
}
```

Fields:

| Key | Required | Default | Meaning |
|---|---|---|---|
| `fx` | **yes** (unless `K` given) | — | Horizontal focal length, **pixels**. |
| `fy` | no | `fx` | Vertical focal length, pixels. *Stored but not consumed* — the model assumes square pixels and reads only `fx`. |
| `cx` | no | image centre (`width/2`) | Principal-point x, pixels. **A real off-centre value genuinely helps.** |
| `cy` | no | image centre (`height/2`) | Principal-point y, pixels. |
| `width`, `height` | no | the video's size | The resolution the intrinsics were **calibrated at**. If it differs from the frames GVHMR processes, all values are rescaled to match. |
| `K` | no | — | A full 3×3 matrix instead of `fx/fy/cx/cy` — see below. |
| `distortion` | no | none | Lens distortion → GVHMR **undistorts the frames** (see below). List `[k1,k2,p1,p2,k3]` or dict. Constant `K` only. |
| `undistort_alpha` | no | `0.0` | Undistort free-scaling: `0` crops the invalid border (fills frame), `1` keeps all source pixels (black borders). |

### Auto-detection

Name the file `<video-stem>.intrinsics.json` (e.g. `dance.mp4` → `dance.intrinsics.json`) and place it
next to the video. GVHMR picks it up automatically — no flag needed. `.intrinsics.npz` works too. An
explicit `--f_px`/`--f_mm` on the command line overrides an auto-detected sidecar.

### Per-frame intrinsics (zoom / lens switch)

Any of `fx`/`fy`/`cx`/`cy` may be a **list**, one value per frame, for a clip whose focal changes:

```json
{ "width": 1920, "height": 1080,
  "fx": [1450.0, 1451.2, 1453.0, "…"],
  "cx": 960.0, "cy": 540.0 }
```

The list length must equal the number of **staged** frames. GVHMR resamples every input to **30 fps**
before inference, so if your source isn't 30 fps, resample your per-frame arrays the same way (or feed
a 30 fps clip). A mismatched length is a hard error.

> Per-frame focal is honoured exactly in the depth computation; it is *mildly* out-of-distribution for
> the network input (trained on one focal per clip), so a smooth zoom degrades gracefully. The
> world-frame trajectory doesn't use intrinsics at all.

### Full matrix (`K`)

```json
{ "K": [[1450.0, 0.0, 964.2],
        [0.0, 1450.0, 541.8],
        [0.0, 0.0,   1.0]] }
```

`K` may be `(3, 3)` (constant) or `(L, 3, 3)` (per-frame). With `width`/`height` present it is rescaled
like the scalar fields.

### Lens distortion (wide-angle / fisheye)

GVHMR is a **pinhole-only** model, so it can't consume distortion coefficients directly — a wide lens
that bows straight limbs corrupts the 2D keypoints and the geometry. The fix is to **rectify the pixels
the model sees**: add a `distortion` entry and GVHMR undistorts the staged frames (`cv2.undistort`) and
swaps in the corrected pinhole `K` automatically. You provide OpenCV coefficients — a list in OpenCV
order or a dict:

```json
{ "width": 1920, "height": 1080,
  "fx": 900.0, "fy": 900.0, "cx": 960.0, "cy": 540.0,
  "distortion": [-0.28, 0.10, 0.0, 0.0, 0.0],
  "undistort_alpha": 0.0 }
```

```json
{ "fx": 900, "fy": 900, "cx": 960, "cy": 540,
  "distortion": {"k1": -0.28, "k2": 0.10, "p1": 0.0, "p2": 0.0, "k3": 0.0} }
```

- Coefficients are OpenCV's `[k1, k2, p1, p2, k3, …]` (length 4/5/8/12/14). They're dimensionless, so they
  need **no** resolution rescale (only `fx/fy/cx/cy` do).
- `undistort_alpha` controls `cv2.getOptimalNewCameraMatrix`: **`0.0`** (default) crops the invalid
  border so the rectified frame fills the image cleanly — the corrected `fx` comes out a bit smaller and
  some FOV is lost at the edges; raise toward **`1.0`** to keep all source pixels, at the cost of black
  borders where no pixel maps.
- This only matters for meaningfully-distorted lenses (GoPro/action cams, ultrawide/fisheye). For a
  normal lens the distortion is a few pixels at the edges — skip it.

#### Per-frame distortion (a zoom)

A zoom changes the focal length **and** the distortion profile together (typically barrel at the wide
end → ~zero mid-range → pincushion at the long end). Supply per-frame distortion — a `(L, N)` array (one
coefficient row per staged frame), or a dict whose coefficients are length-`L` lists — alongside per-frame
`fx`/`fy`; GVHMR rebuilds the rectification map each frame and emits a per-frame corrected `K`:

```json
{ "fx": [900, 902, "…L…"], "fy": [900, 902, "…L…"], "cx": 960, "cy": 540,
  "distortion": [[-0.28, 0.10, 0.0, 0.0], "…one row per staged frame…"] }
```

- You need a **zoom-aware calibration** to produce those rows — coefficients measured at several focal
  lengths, interpolated per frame by the frame's focal (from lens metadata / a zoom encoder). Cinema/
  broadcast lens data (Cooke /i, Zeiss eXtended, a hand-built LUT) has this; consumer clips usually don't.
- If you only have per-frame **focal**, use that alone (it captures most of a zoom's geometric error); a
  single distortion set is broadcast to every frame, and a constant `distortion` with per-frame `fx` works.
- Keep the per-frame coefficients **smooth** — a noisy, jittery warp injects temporal noise into the frames
  the model sees (GVHMR integrates per-frame velocities and is jitter-sensitive).
- Rows must match the **staged** frame count (the clip is resampled to 30 fps first), same as per-frame focal.
- Only for `.MOV`/`.mp4` with a **rotation flag**: express `distortion`/`K` in the displayed (post-rotation)
  frame, since GVHMR undistorts the staged (rotation-baked) video.

### NPZ / NPY

- **`.npz`** — an archive with the same keys (`fx`, `fy`, `cx`, `cy`, `width`, `height`) and/or `K`:
  ```python
  import numpy as np
  np.savez("dance.intrinsics.npz", fx=1450.0, fy=1450.0, cx=964.2, cy=541.8, width=1920, height=1080)
  ```
- **`.npy`** — a single array, interpreted as `K` of shape `(3, 3)` or `(L, 3, 3)`.

---

## How to convert what your camera gives you

- **Focal in pixels already (`fx`, `fy`)** → use them directly (`--f_px fx`, or a sidecar). Don't convert
  to mm and back. Because the model reads only `fx`, if `fx` and `fy` differ (rare — non-square pixels),
  pass `fx` (or the geometric mean `√(fx·fy)`); the difference is negligible in practice.
- **Focal in mm + sensor width in mm** → `fx_px = focal_mm / sensor_width_mm · image_width_px`. Put that
  in `fx` (and `fy`).
- **35mm-equivalent focal (phones)** → just use `--f_mm` (or let metadata provide it).
- **Principal point** → if your calibration gives a real `(cx, cy)`, include it; otherwise omit it and the
  image centre is assumed.
- **Distortion coefficients** → put them in `distortion` (see *Lens distortion* above) and GVHMR
  undistorts the frames for you. Worth it only for wide-angle/fisheye lenses; negligible for normal ones.

---

## Command line

```bash
gvhmr demo clip.mp4 --f_px 1450                 # focal in pixels
gvhmr demo clip.mp4 --intrinsics clip.cam.json  # full sidecar
gvhmr demo clip.mp4                             # auto-detect clip.intrinsics.json, else metadata/heuristic
gvhmr demo clip.mp4 --f_mm 24                   # 35mm-equivalent (phone 1×)
```

`gvhmr demo-folder DIR` auto-detects a `<name>.intrinsics.json` next to each clip.

## Python library

```python
import gvhmr

# focal in pixels
gvhmr.recover("clip.mp4", f_px=1450)

# a sidecar path, a dict, or a K array/tensor
gvhmr.recover("clip.mp4", intrinsics="clip.cam.json")
gvhmr.recover("clip.mp4", intrinsics={"fx": 1450, "cx": 964.2, "cy": 541.8})
gvhmr.recover("clip.mp4", intrinsics=my_K)          # (3,3) or (L,3,3) numpy/torch
```

The recovered `MotionResult.intrinsics` is the resolved `(L, 3, 3)` per-frame `K` (also written by
`result.save_npz(...)`). See [LIBRARY.md](LIBRARY.md).

## Hugging Face Space

The demo Space has an optional **“Camera intrinsics (optional)”** upload — drop in a `.json`/`.npz`
sidecar to use real intrinsics; leave it empty to fall back to metadata / the heuristic.

---

## Landmines

- **Intrinsics are resolution-bound.** Pixel values are tied to the resolution they were measured at.
  Declare `width`/`height` so GVHMR can rescale to the frames it actually processes.
- **Rotation flag.** A phone clip with an orientation flag is rotated when GVHMR stages it; express your
  intrinsics in the *displayed* (post-rotation) frame, or a 90° rotation swaps the effective `fx`/`fy`
  and moves the principal point.
- **One focal, square pixels.** The model consumes only `fx` and the principal point `(cx, cy)`; `fy` is
  stored in the returned `K` but not used.
- **30 fps staging.** Per-frame arrays are indexed against the staged 30 fps clip, not the raw source.

---

## In-camera depth (`tz`): it is the focal, measured against ground truth

> **Measured 2026-07-16 on EMDB-1** (released checkpoint, 17 sequences / 24,117 frames, default no-flip
> path) with `scripts/depth_bias_probe.py`. This **replaces an earlier version of this section that
> attributed the far-bias to out-of-distribution crops inflating `betas`. That hypothesis is refuted** —
> it was reasoned from the mechanism and never measured against ground truth. The numbers below are.

The in-camera translation is `tz = 2·f / (s·b)` (`gvhmr/utils/geo/hmr_cam.py::compute_transl_full_cam`):
`f` = `K[0,0]` (focal in px), `s` = the network's crop-relative predicted scale, `b` = bbox size in
full-image pixels. **`tz ∝ f`, exactly** — a wrong focal biases depth by the same ratio while leaving the
*bearing* (lateral placement) almost untouched.

### The default focal heuristic costs +71% depth error

`estimate_K` uses `f = √(w²+h²)` — a 53° diagonal-FOV assumption. On EMDB's phone footage the true focal
is ~1435 px and the heuristic returns 2400 px: **1.67× too long**. Depth follows almost linearly:

| | mean pred depth | mean signed error | relative | frames predicted FAR |
|---|---|---|---|---|
| GT depth | 2.26 m | — | — | — |
| **default (`estimate_K`)** | **3.84 m** | **+1.58 m** | **+70.9%** | **100.0%** |
| **GT intrinsics** | 2.38 m | +0.12 m | **+6.0%** | 77.9% |

The network sees `K` in its `bbox_info` conditioning and **absorbs only ~9%** of the focal error
(`pred_est/pred_gt` = 1.614 against a focal ratio of 1.672); the remaining 1.61× lands straight on depth.

**Consequences.** *Every EMDB depth number this project has ever reported is ~70% long* — including the
paper protocol, since `emdb_motion_test.py` deliberately discards EMDB's GT intrinsics ("We use estimated
K"). This is invisible to every metric we ship: `mpjpe` is pelvis-aligned, `pa_mpjpe` is Procrustes-aligned,
and 2D reprojection cannot see it either (a body placed farther with a larger `s` reprojects to the same
pixels). **If you consume in-camera depth, pass calibrated intrinsics** (`--f_px` / `--intrinsics`); there
is no EXIF auto-detection, so the default path is always the heuristic. This is also why *"bearing good,
range biased, view-specific"* is the signature of a per-camera focal error — check `fx` first.

### With correct intrinsics: a residual +6% far bias, NOT crop-size dependent

A real, systematic far bias survives calibration (+0.12 m, 78% of frames). But its shape is **the opposite
of the OOD-crop hypothesis** — the relative bias *grows* with crop size, and the whole spread is 5 pp:

| crop size (bbx px) | 464–1067 | 1067–1215 | 1215–1372 | 1372–1560 | 1560–4115 |
|---|---|---|---|---|---|
| relative depth error | +2.97% | +6.03% | +6.80% | +5.96% | **+8.08%** |

Small crops are the *least* biased. So "small/OOD crops → betas inflation → far depth" is refuted twice
over: inverted in sign, and weak in magnitude. With GT `K`, `corr(betas_mag, |err|) = −0.373` — the shape
signal is real but *anti*-correlated with error, again opposite to the inflation story.

### `depth_reliability()` does not predict depth error — do not weight by it

Measured correlation of its `weight` against true `|depth error|`:

| | corr(weight, \|err\|) |
|---|---|
| default (`estimate_K`) | −0.650 |
| **GT intrinsics** | **−0.028** |

The −0.650 is an **artifact**: with a wrong focal the error scales with distance, and the weight is mostly
`bbx_px`, which is itself a distance proxy. It tracks *how far away the person is*, not *how wrong the
depth is*. Once intrinsics are correct — the regime any depth consumer should be in — it predicts
**nothing** (−0.028). It shipped in v1.6.0 with this correlation unmeasured; the fields it exposes
(`bbx_px`, `betas_mag`, `betas_std`, per-frame betas) remain useful as raw diagnostics, but **the `weight`
should not be used to weight multi-view depth fusion.** The dominant error is a per-video constant (the
focal), which no per-frame proxy can express. Fix the focal instead.

What is *excellent* and should be trusted: `body_pose`, `betas` shape, in-cam orientation, and the lateral
bearing — the bias is specifically along the optical axis.

Reproduce: `python scripts/depth_bias_probe.py --split 1 --k est|gt`.
