# Evaluation

## The paper benchmarks — `gvhmr eval`

The canonical numbers (3DPW / EMDB-1 / EMDB-2 / RICH, flip-test + test-time postprocessing — exactly
the paper's protocol) are one command:

```bash
gvhmr eval                       # all three datasets; auto-fetches the data packs + checkpoint
gvhmr eval 3dpw                  # one dataset (~1 min on a modern GPU)
gvhmr eval emdb,rich --json out/metrics.json
gvhmr eval all --ckpt outputs/my_run/checkpoints/last.ckpt   # evaluate your own training run
```

It fetches whatever is missing (preprocessed eval packs → `$GVHMR_DATA_ROOT`, the released checkpoint),
checks the registration-gated body models up front (each dataset needs specific gendered files — the
error lists exactly which and where), runs the same Lightning test tasks as the raw
`gvhmr train global/task=gvhmr/test_* …` invocation, and ends with one table of your numbers next to
the **paper's published numbers** (arXiv 2409.06662) with the delta highlighted — so a regression is
visible at a glance. Metrics: PA-MPJPE / MPJPE / PVE (mm), Accel (m/s²) in camera space;
W-MPJPE₁₀₀ / WA-MPJPE₁₀₀ (mm, per 100-frame chunk), RTE (% of path), Jitter, foot sliding (mm) in
world space.

This pipeline is verified end-to-end: re-verified 2026-07-13, the released checkpoint reproduces the
paper's camera-space numbers exactly (3DPW 36.2/55.6/67.2, EMDB-1 42.7/72.6/84.2 with accel 3.6, RICH
within 0.3 mm) and the world metrics within ~1% (EMDB-2 W-MPJPE 272.8 vs 274.9) on an RTX 6000 Ada.

> **This claim is load-bearing — re-run it after any numerics change.** Between 2026-07-06 and
> 2026-07-13 it was silently false: a global TF32 fast path cost EMDB-1 +3.3 mm PA-MPJPE and **4× the
> acceleration error** (3.6 → 14.2), while 3DPW and RICH moved <0.3 mm and looked fine. TF32 is now
> off by default; see [PERFORMANCE.md](PERFORMANCE.md) for the post-mortem. Certify against **EMDB
> accel/jitter**, not 3DPW pose alone — the derivative metrics are what catch per-frame noise.

**Training reproduction (2026-07-13).** A from-scratch 500-epoch retrain on all four datasets
(`exp=gvhmr/mixed/mixed`, single GPU) reproduces the paper: 3DPW 36.4/55.7/67.3, EMDB-1 **42.4**/73.5/85.5
(accel 3.5), EMDB-2 WA-MPJPE 111.9 / RTE 2.0 / jitter **14.7**, RICH 41.0/69.9/79.0. It beats the paper on
EMDB-1 PA-MPJPE and on jitter, and trails ~3-4 mm on RICH pose and EMDB-2 world translation — consistent
with the halved effective batch (`devices=1` vs the recipe's `devices=2` × 128).

### Full-distribution diagnostics — `gvhmr eval --diagnostics`

By default only the per-metric **mean** survives. `--diagnostics` additionally preserves everything the
metric callbacks compute and normally discard — **std / variance, min / max / median, percentiles
(p01…p99), per-sequence stats, per-joint MPJPE, per-sequence timing**, plus a **provenance** block
(git sha, ckpt sha256, torch/GPU, preproc variant, dataset sizes) — written to `diagnostics.json`
next to `--json`. `--dump-raw` also writes the raw per-sequence arrays as `results_diagnostics/<DS>_raw.npz`
(keys `<metric>__<vid>`, `perjoint_mpjpe__<vid>`). It is purely additive: the printed table, the logged
means, and the `--json` metrics are **byte-identical** with or without it (golden-safe). Enable it in a
sub-process by exporting `GVHMR_EVAL_DIAGNOSTICS=1`, or per callback with `--set callbacks.<name>.diagnostics=true`.

```bash
gvhmr eval all --json out/metrics.json --diagnostics --dump-raw   # means + full distribution + raw arrays
```

`gvhmr sweep … --diagnostics` does the same per trial, logging the extended scalars, per-metric
`wandb.Histogram`s, a per-sequence `wandb.Table`, per-joint bar charts, a raw-arrays `wandb.Artifact`,
and the provenance to `run.summary` — alongside the unchanged `<DS>/<metric>` means.

### Measuring preprocessing swaps — `gvhmr eval --detector / --pose2d`

The packs ship **frozen** preprocessing (YOLOv8x boxes, ViTPose keypoints, HMR2 features — computed
once upstream), so by default a detector/2D-pose swap is *invisible* to the benchmark. To measure one,
`gvhmr eval` can regenerate the preprocessing with your chosen stages into a **variant cache**
(`<DS>/hmr4d_support/preproc_variants/<slug>/` — the canonical files are never touched):

```bash
gvhmr eval 3dpw --detector yolo26x                        # first run: fetches raw 3DPW + regenerates
gvhmr eval 3dpw,emdb --detector yolo26x --pose2d rtmpose  # variants are cached + resumable
gvhmr eval 3dpw --set pose2d.flip_test=false --set detector.tracker=bytetrack.yaml  # tune stage knobs
```

**Stage `--set` knobs.** A `--set` whose key targets a stage group (`detector.`/`pose2d.`/`backbone.`,
e.g. `pose2d.flip_test=false`) drives the **preproc regen** — the same knob the demo's `--set` tweaks,
so you can measure exactly what a stage setting costs on the benchmark (this is how the `--fast` recipe's
flip-off + ByteTrack is certified). Each distinct knob set caches under its own slug. Any non-stage
`--set` still tunes the model/test task.

What to know before trusting a variant number:

- **The raw videos are fetched once.** The packs ship an *empty* `videos/` dir (the footage isn't
  redistributable through the mirror). **3DPW auto-downloads** from its official host
  (`imageFiles.zip`, 4.6 GB, resumable — fetching implies accepting its
  [research license](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html)) into
  `$GVHMR_DATA_ROOT/raw/3DPW`, and the pack's `videos/` is composed from it (30 fps) — after that, any
  number of variants can be generated. [EMDB](https://eth-ait.github.io/emdb/)'s download is
  credential-gated (registration), so it needs a manual download + `--raw-dir` (which also lets you
  point 3DPW at an existing copy). **RICH is not supported** (no per-sequence videos anywhere ungated).
- **Identity guard (3DPW is multi-person).** A fresh detector on a two-person video can lock onto the
  *other* person — that would score a tracking-identity failure, not a detector comparison. Regenerated
  tracks are median-IoU-checked against the canonical track; mismatches keep the canonical boxes for
  that sequence and are reported, so a variant differs from canonical only by the stage under test.
- **The paper column stays canonical.** The summary table's Δ then reads as "what this stage swap
  changes relative to the published protocol". Run once *without* stage flags for your box's canonical
  baseline if you want a same-hardware comparison.
- **Backbone swaps need a retrain.** The features are learned conditioning — `--backbone` on the
  released checkpoint produces meaningless numbers (the command warns); pass a retrained `--ckpt`.

### Comparing many combinations — `gvhmr sweep` (W&B)

To compare evals across *many* stage combinations, run a real [W&B sweep](https://docs.wandb.ai/guides/sweeps)
— each trial is one combo, regenerated/cached as above, benchmarked, and logged as `<DATASET>/<metric>`
(plus `…_vs_paper` deltas). **What's swept is exactly `detector × pose2d`** — the command prints this
dimension table up front. The other stages are deliberately fixed, and the tool says so:

- `canonical` is a first-class value for both swept stages (the pack's frozen paper preprocessing), so
  every sweep contains its baseline point;
- **backbone** is fixed at hmr2 — features are learned conditioning, so a swap is only meaningful with
  a retrained checkpoint (run a separate sweep passing `--ckpt`);
- **camera** is not a benchmark dimension *by construction*: the protocol feeds the model ground-truth
  camera rotation (no visual odometry runs), so simplevo/dpvo/dust3r/vggt can't change these numbers.
  A/B camera backends with the world-eval harness below instead.

```bash
gvhmr sweep run 3dpw --detectors canonical,yolov8x,yolo26x   # create + run locally (raw 3DPW auto-fetches)
gvhmr sweep create 3dpw,emdb --detectors all --pose2ds all   # grid over everything → sweep id
gvhmr sweep agent <sweep_id> --raw-dir ~/ds/EMDB      # run trials; launch on several GPU boxes to parallelize
gvhmr sweep report <sweep_id>                         # (re)build the comparison report
```

Every sweep gets an auto-generated **W&B report** (its URL is printed; panels fill in live): parallel
coordinates across *every* metric (absolute + Δ-vs-paper), a bar chart per metric, and PA-MPJPE vs MPJPE
scatter plots — plus the sweep page's own table/parallel-coordinates. Missing prerequisites (videos,
stage deps) are checked **once up front** — an agent refuses to start rather than crashing every
non-canonical trial; fully-cached grids re-sweep on any box without videos.

`gvhmr sweep create --config my_sweep.yaml` accepts a hand-written W&B sweep spec (e.g. `method: bayes`);
`--dry-run` prints the generated config. Needs `wandb login` once (the W&B service schedules trials —
offline mode can't run sweeps) and the `train` extra (wandb + wandb-workspaces).

**Grid economics.** Generation is cached per *stage*, not per combo
(`preproc_variants/_stages/{boxes,feats,kp2d}/…`, flock-guarded so parallel agents never duplicate a
model pass): boxes are per-detector, the heavy HMR2+flip feature pass is per-detector too, and only the
keypoint pass is per-(detector×pose2d) — so a full `--detectors all --pose2ds all` grid on 3DPW costs
O(detectors) heavy passes + O(combos) cheap ones (~15 h on two GPUs, one-time) rather than everything ×
everything. Every stage is cached forever; re-sweeping a generated grid is ~1 min/trial.

## World-frame evaluation on real datasets (`tools/eval/eval_world.py`)

The fork's headline addition — recovering a *metric* world trajectory for a moving/following camera
(`--camera dust3r|vggt`, and the static-camera in-cam carry) — needs **ground-truth world motion** to measure,
which the golden tests can't provide. `tools/eval/eval_world.py` is a trainer-free harness (CPU/MPS, no
Lightning) that scores the EMDB-2 *global* protocol on real-video datasets and, critically, **A/B's how
the human is placed in the world**:

| mode | world trajectory source | what it isolates |
|---|---|---|
| `prior` | stock velocity-prior output | the baseline we're trying to beat |
| `gt-cam` | in-cam human carried through the dataset's **ground-truth** camera | is the *composition math* right? (zero SLAM/depth error) |
| `dust3r` | in-cam human carried through the full **DUSt3R + Depth-Anything-V2** metric camera | does it survive **real** SLAM/depth noise? |

The model runs **once** per sequence (camera *rotation* fed from GT `T_w2c`, exactly as the EMDB loader
does); each mode only re-composes the world translation, so the comparison is controlled. Metrics
(`compute_global_metrics`, the same code the EMDB callback uses): **W-MPJPE** / **WA-MPJPE** (per 100-frame
chunk, first-2-frame vs full-segment aligned), **RTE** (% of path length), **jitter**, **foot-sliding**.

## Datasets

Two, chosen because they test *different* things:

- **SLOPER4D** — real outdoor, 200 m–1.3 km world trajectories; the long-traversal / following-camera
  regime the velocity prior misses. **Public, no registration.** Use with `--modes prior dust3r` to measure
  the real end-to-end pipeline. SMPL ground truth.
- **WHAC-A-Mole** — synthetic, with *exact* camera + SMPL-X ground truth; the clean control. Use with
  `--modes prior gt-cam` to check the composition is correct independent of SLAM/depth quality. SMPL-X GT.

> EMDB-2 (the field-standard benchmark GVHMR reports) is registration-gated (institutional email); once
> access lands, its existing `global/task=gvhmr/test_emdb` Lightning task is the canonical number.

## Run

```bash
uv sync --extra preproc                       # YOLO/ViTPose/HMR2 are needed to featurize the videos
scripts/setup_eval_datasets.sh sloper4d whac  # fetch (SLOPER4D manual, WHAC via huggingface-cli)
scripts/setup_scene_aware.sh                  # only for the `dust3r` mode

uv run python tools/eval/eval_world.py --dataset sloper4d --modes prior dust3r
uv run python tools/eval/eval_world.py --dataset whac     --modes prior gt-cam --limit 3
```

Data lives under `$GVHMR_DATA/{sloper4d,whac}` (default `~/Datasets/GVHMR`), overridable with
`--data-root`. Per-sequence preproc is cached under `outputs/eval_world/<vid>/`.

## Caveats (read before trusting a number)

- **fps.** GVHMR runs at 30 fps; the adapters resample frames **and** ground truth to 30 with a shared
  index map (`resample_indices`) so prediction and GT stay aligned.
- **Frame convention is alignment-robust — except WHAC's storage frame.** `compute_global_metrics` aligns
  each chunk by Procrustes, so the prediction's gravity-view world frame need not match the dataset's. But
  if a dataset stores SMPL-X in *per-frame camera* coordinates, that must be lifted to world first. WHAC's
  HumanData npz is **schema-pending**: run `--probe <file.npz>` to dump its keys, then set
  `WHAC_FRAME=world|camera` (default `world`). SLOPER4D's `second_person` is already world-frame SMPL.
- **The plumbing is unit-tested offline** (`tests/test_eval_world.py`: the composition recovers a
  following-camera traversal and the metric rewards it, on synthetic tensors), but the **dataset adapters
  are not** — they need a one-time smoke check on the first real sequence (SLOPER4D's on-disk layout and
  WHAC's npz frame convention have both varied across releases). Both raise clear errors on schema
  mismatch rather than scoring garbage silently.
