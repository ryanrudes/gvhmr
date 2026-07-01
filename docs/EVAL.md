# World-frame evaluation on real datasets

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
