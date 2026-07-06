# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates

### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> World-Grounded Human Motion Recovery via Gravity-View Coordinates
> [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Yan Xia](https://isshikihugh.github.io/scholar),
[Zhi Cen](https://scholar.google.com/citations?user=Xyy-uFMAAAAJ),
[Sida Peng](https://pengsida.net/)<sup>†</sup>,
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Xiaowei Zhou](https://xzhou.me/)
> SIGGRAPH Asia 2024

<p align="center">
    <img src=docs/example_video/project_teaser.gif alt="animated" />
</p>

Feed it a video, get back **SMPL/SMPL-X human motion in both the camera frame and the world frame** —
plus rendered overlay videos to see the result.

> [!NOTE]
> This is a **modernized fork** of [`zju3dv/GVHMR`](https://github.com/zju3dv/GVHMR) by
> [@ryanrudes](https://github.com/ryanrudes): one-command install (`uv` + a Typer/Rich `gvhmr` CLI),
> **Apple-Silicon (MPS) support** end-to-end (including a real-time moderngl mesh renderer), automatic
> checkpoint fetching, swappable preprocessing models, scene-aware **metric** cameras
> (`--camera dust3r|vggt` — Mac-friendly alternatives to CUDA-only DPVO), skeleton-overlay exports, a
> CPU test suite, and re-training tooling. **The released model's behaviour is preserved** — the default
> inference path is golden-guarded byte-identical to upstream.

## Quick start

```bash
git clone https://github.com/ryanrudes/gvhmr && cd gvhmr
scripts/install.sh                                   # detects your platform/GPU, installs, fetches checkpoints
bin/gvhmr demo docs/example_video/tennis.mp4 -s      # recover motion from the bundled example video
```

The installer detects macOS vs Linux+NVIDIA vs CPU, picks the matching torch build, records the
choices, and hands off to the **wizard**, which walks you through every optional component (RTMPose,
DPVO, the DUSt3R/VGGT scene cameras, 3D visualization, …), asset locations, and the checkpoint fetch.
From then on you interact only with the `gvhmr` CLI (via `bin/gvhmr`, or `gvhmr` after
`source .venv/bin/activate`), never with uv: `bin/gvhmr config init` re-runs the wizard;
`bin/gvhmr env sync` re-applies the recorded environment if it ever drifts. Two notes, and everything
else just works:

- **Body models are registration-gated** (their license forbids redistribution): register at
  [SMPL-X](https://smpl-x.is.tue.mpg.de/) (required for motion recovery) and
  [SMPL](https://smpl.is.tue.mpg.de/) (mesh rendering), then run `bin/gvhmr auth smpl` once — it
  prompts for each MPI account separately and auto-fetches the files. On a second machine, push your
  copy to a **private** HF repo (`gvhmr body-models push …`) and pull from there instead of re-logging
  in — see [docs/INSTALL.md](docs/INSTALL.md#body-models-registration-gated) (CLI) or
  [docs/LIBRARY.md](docs/LIBRARY.md#body-models--gvhmr-auth-smpl) (pip/API).
- On **Linux + NVIDIA**, torch must match your CUDA driver — `scripts/install.sh` picks the right build
  for you; if you sync manually, see [docs/INSTALL.md](docs/INSTALL.md#cuda--gpu-linux). macOS needs
  nothing special.

Run `bin/gvhmr info` anytime for a full diagnostic (device, installed features, checkpoint status, and
the exact fix for anything missing). Full install guide: [docs/INSTALL.md](docs/INSTALL.md).

Prefer not to install? Try the upstream-hosted
[Colab](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing) or
[HuggingFace Space](https://huggingface.co/spaces/LittleFrog/GVHMR).

## Python library

GVHMR ships a HuggingFace-style Python API (in `gvhmr/inference/`) over the exact same pipeline as the
CLI — same code paths, byte-identical results, but a clean object you can loop over videos with.

```bash
pip install "gvhmr[preproc]"    # base + preprocessing (add cu124/cu126/cu128 on Linux+CUDA; macOS = MPS)
gvhmr auth smpl                 # one-time: your MPI login, to fetch the gated SMPL/SMPL-X body models
```

```python
import gvhmr

# one-liner (caches a default pipeline; a loop of videos loads weights once)
result = gvhmr.recover("dance.mp4")

# or load once, reuse across videos
pipe = gvhmr.pipeline("human-motion-recovery", device="cuda")
result = pipe("dance.mp4", static_camera=True, flip_test=True)

result.smpl_params_world      # world-frame SMPL params (global_orient/body_pose/betas/transl)
result.joints_world           # (L, 24, 3) world-frame joints
result.render("overlay.mp4")  # in-cam ∥ world overlay video
result.save_npz("dance.npz")  # portable SMPL params + intrinsics
```

Body models are registration-gated by MPI and **never redistributed**. On a **new machine**, register at
[SMPL-X](https://smpl-x.is.tue.mpg.de/) + [SMPL](https://smpl.is.tue.mpg.de/) (separate accounts), then
`gvhmr auth smpl` — or set `$SMPLX_USER`/`$SMPLX_PW` and optionally `$SMPL_USER`/`$SMPL_PW` for
CI/containers. Inference auto-fetches on first use (mirror first, then MPI). Already set up elsewhere?
`gvhmr body-models push you/repo` once, then `gvhmr body-models pull` (or `$GVHMR_BODY_MODELS_MIRROR` +
`HF_TOKEN`) on every new box — no MPI login in the loop. Missing SMPL still runs motion recovery; only
overlay rendering is skipped. Tensor-level API: `gvhmr.GVHMR.from_pretrained(...).predict(...)`.
Full guide: [docs/LIBRARY.md](docs/LIBRARY.md).

## Usage

The `gvhmr` command (Typer + Rich) is the main entry point — `gvhmr --help` for the full menu. (Below,
`gvhmr` means `bin/gvhmr` from the repo root, or plain `gvhmr` in an activated venv.)

```bash
gvhmr demo VIDEO.mp4 -s                       # single video, static camera (fastest)
gvhmr demo VIDEO.mp4                          # moving camera (SimpleVO, rotation only)
gvhmr demo VIDEO.mp4 --camera vggt            # moving camera with world translation (any device)
gvhmr demo-folder DIR -o outputs/batch -s     # every video in a folder
gvhmr info                                    # environment & asset diagnostic
gvhmr config init                             # wizard: asset locations, models, managed environment
gvhmr env sync                                # re-apply the recorded environment (fixes any drift)
gvhmr bench                                   # inference latency benchmark
```

Each run writes to `outputs/demo/<video>/`: the recovered motion (`hmr4d_results.pt` — SMPL-X
parameters in camera and world frames), the camera-view render (`1_incam.mp4`), the world-view render
(`2_global.mp4`), and a side-by-side of the two. `--no-render` skips the videos (motion only);
`--render-scale 1` renders full-resolution overlays (default 0.5 for speed); `--verbose` saves
intermediate overlays (detection boxes, 2D keypoints).

**Cameras.** `-s/--static-cam` skips visual odometry entirely. For moving cameras the default is
SimpleVO (rotation only); to recover the camera's **translation** too — e.g. a following/tracking
shot — use a scene-aware **metric** camera: `--camera vggt` or `--camera dust3r` (run
`scripts/setup_scene_aware.sh` once; works on Apple Silicon / CPU / CUDA), or classic
`--camera dpvo` (CUDA only; `scripts/setup_dpvo.sh`).

**Accuracy.** Add `--flip-test` (mirror-averaging TTA — the benchmark-time setting) and pass the true
focal length `--f_mm` if you know it (phone metadata is read automatically when present). Evidence and
measurement methodology: [docs/ACCURACY.md](docs/ACCURACY.md).

**Skeleton exports.** `--skeleton` writes a world-frame skeleton-only video, `--skeleton-overlay` draws
the 24-joint skeleton on top of the mesh videos, `--skeleton-joints legs,left_arm` restricts to a
subset (groups or joint names/indices).

**Device.** Auto-selected (CUDA → MPS → CPU); override with `GVHMR_DEVICE=cpu|mps|cuda`. Everything
except CUDA-only DPVO runs on Apple Silicon, including rendering.

### Swapping models

The detector, 2D-pose estimator, feature backbone, and camera are each a **Hydra config group** — pick
implementations by name, bundle choices into a committable *recipe*, or tweak any knob:

```bash
gvhmr demo clip.mp4 --detector yolo26x --pose2d rtmpose --camera dust3r   # pick implementations
gvhmr demo clip.mp4 --recipe accurate                                     # a committable bundle
gvhmr demo clip.mp4 --set detector.conf=0.4                               # tweak a knob
```

Every YOLO family×size is a preset; RTMPose (`gvhmr config set extras preproc,rtmpose && gvhmr env
sync`) is a verified alternative 2D-pose backend; the feature backbone is learned conditioning, so
swapping it needs a retrain (the tooling for that exists — `gvhmr extract-features` + `gvhmr train`).
Machine-local defaults (asset locations + model versions + the managed environment) live in one
readable config file managed by **`gvhmr config init`**. Full guide:
[docs/CONFIGURATION.md](docs/CONFIGURATION.md); roadmap: [docs/EXTENSIBILITY.md](docs/EXTENSIBILITY.md).

## Reproduce the paper

```bash
gvhmr eval                    # 3DPW + EMDB + RICH: auto-fetches the eval packs, runs the paper
                              # protocol, prints your numbers next to the paper's (verified to match)
gvhmr eval 3dpw --json m.json # one dataset; optionally dump metrics for tracking
gvhmr eval all --ckpt outputs/my_run/checkpoints/last.ckpt    # evaluate your own training run
gvhmr eval 3dpw --detector yolo26x        # benchmark a PREPROCESSING swap (auto-fetches raw 3DPW,
                              # regenerates boxes/keypoints/features; see docs/EVAL.md)
gvhmr sweep run 3dpw --detectors all      # W&B sweep comparing every detector preset on the benchmark

# Train (the released ckpt used 2×4090 for 420 epochs)
gvhmr train exp=gvhmr/mixed/mixed
```

`gvhmr eval` wraps the canonical Lightning test tasks (`gvhmr train global/task=gvhmr/test_* …` still
works) — the only manual step is the registration-gated body models, and the command tells you exactly
which files it needs. Details: [docs/EVAL.md](docs/EVAL.md); training: [docs/TRAINING.md](docs/TRAINING.md)
(training doesn't apply the test-time postprocessing, so its global metrics differ from the benchmark).

## Development

```bash
uv sync --extra dev        # test/lint/type tooling  (or scripts/install.sh --dev)
uv run pre-commit install  # once — auto-formats staged files on commit
make check                 # the required CI gates locally: ruff format --check + pytest
make fmt                   # format the whole tree
```

The test suite is a CPU/MPS characterization net — no GPU, checkpoints, or datasets needed. Start with
[`AGENTS.md`](AGENTS.md) for architecture, conventions, and the behaviour-preservation landmines.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | full install guide: CUDA matrix, extras, weights, troubleshooting |
| [docs/LIBRARY.md](docs/LIBRARY.md) | Python API (`recover`, `MotionResult`), body-model setup (MPI + private mirror) |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | the config file + swapping/configuring every pipeline stage |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | code map: data flow, packages, the 151-dim latent |
| [docs/TRAINING.md](docs/TRAINING.md) | training/eval on any device, dataset packs, backbone retrains |
| [docs/ACCURACY.md](docs/ACCURACY.md) | test-time accuracy levers + how they're measured |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | profiling & latency notes (CPU vs MPS, render scale) |
| [docs/EXTENSIBILITY.md](docs/EXTENSIBILITY.md) | the swappable-everything roadmap & rationale |
| [docs/BEHAVIOR.md](docs/BEHAVIOR.md) / [docs/PROVENANCE.md](docs/PROVENANCE.md) | behaviour-preservation contract; what was vendored/renamed vs upstream |

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

# Acknowledgement

We thank the authors of
[WHAM](https://github.com/yohanshin/WHAM),
[4D-Humans](https://github.com/shubham-goel/4D-Humans),
and [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch) for their great works, without which our project/code would not be possible.

Upstream repository: [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR).
