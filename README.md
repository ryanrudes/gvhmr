# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> [!NOTE]
> **Modernized fork** of [`zju3dv/GVHMR`](https://github.com/zju3dv/GVHMR) by
> [@ryanrudes](https://github.com/ryanrudes) — runs on **Apple Silicon (MPS)**, with `uv`/`pyproject`
> packaging, a Typer + Rich `gvhmr` CLI, a GPU mesh renderer, skeleton-overlay exports, and a
> device-agnostic **scene-aware metric cameras** (`gvhmr demo --camera dust3r|vggt`, Mac-friendly DPVO
> alternatives that recover a *metric* camera). The released model's behaviour is preserved
> (golden-guarded). Details: [`AGENTS.md`](AGENTS.md) · upstream: [`zju3dv/GVHMR`](https://github.com/zju3dv/GVHMR).

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

## News 🔥

- [2025-03-08] By default not using DPVO. We implemented a SimpleVO, which is more efficient and compatible with GVHMR.
- [2025-03-08] We added a new option `f_mm` to specify the focal length of the fullframe camera in mm.

> **This is a modernized fork** of [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR): `uv` +
> `pyproject.toml` packaging, Python 3.13 / modern typing, a CPU/MPS test suite,
> **Apple-Silicon (MPS) support**, a **Typer + Rich `gvhmr` CLI**, a GPU mesh renderer
> (moderngl/Metal), skeleton-overlay video exports, device-agnostic **scene-aware metric cameras**
> (`--camera dust3r|vggt` — DUSt3R or VGGT + Depth-Anything-V2, recovering a metric camera on Mac where
> DPVO is CUDA-only; set up with [`scripts/setup_scene_aware.sh`](scripts/setup_scene_aware.sh)), and
> agent tooling — with model behaviour preserved (golden-guarded).
> See [`AGENTS.md`](AGENTS.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
> [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Setup

Please see [installation](docs/INSTALL.md). In short:

```bash
uv sync                 # base install (CPU / Apple-Silicon MPS); add --extra preproc for the demo
uv run gvhmr download   # fetch model checkpoints (→ inputs/checkpoints; set $GVHMR_CHECKPOINTS to relocate)
uv run gvhmr info       # check device, installed features, and checkpoint status
```

`gvhmr demo` also auto-fetches any missing checkpoints on first run. SMPL/SMPL-X **body models are
registration-gated** — `gvhmr download` prints the sign-up + target path. To keep large assets on a
**high-storage volume** and pick your default model versions, run **`gvhmr config init`** — a Rich wizard
that writes one readable `~/.config/gvhmr/config.toml` (asset locations + `[models]` defaults, with the
options listed inline). `$GVHMR_*` env vars still override it for CI / one-offs. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

On **Linux/CUDA**, add the extra matching your GPU (uv can't auto-detect it for `uv sync`) — check
`nvidia-smi` and pick the nearest ≤ your CUDA: `uv sync --extra cu128` (covers 12.8–13.x), `cu126`,
`cu124`, or `cpu`. macOS needs no extra (MPS). See [installation](docs/INSTALL.md#cuda--gpu-linux).

## Quick Start

### [<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab demo for GVHMR](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing)

### [<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo for GVHMR](https://huggingface.co/spaces/LittleFrog/GVHMR)

### CLI

The `gvhmr` command (Typer + Rich) is the main entry point — run `gvhmr --help` for the
full menu, and `gvhmr info` for a device/extras/checkpoint diagnostic.

```shell
uv run gvhmr info                                              # environment & asset status
uv run gvhmr demo docs/example_video/tennis.mp4 -s            # single video (static camera)
uv run gvhmr demo-folder inputs/demo/folder_in -o outputs/demo/folder_out -s
uv run gvhmr bench                                            # inference latency benchmark
```

Use `-s` for a static camera (skip visual odometry); otherwise the camera is estimated by SimpleVO
(rotation only). For world **translation** on a moving/following camera, use `--camera dpvo` (CUDA) or the
scene-aware **metric** cameras `--camera dust3r` / `--camera vggt` (Apple-Silicon/CPU; see
[Setup](docs/INSTALL.md)). The device is auto-selected (CUDA → MPS → CPU);
override with `GVHMR_DEVICE=mps|cpu|cuda`. `--render-scale` trades overlay resolution for
speed, `--no-render` skips overlays. (The old `python tools/demo/demo.py …` scripts still
work as thin shims.)

For **higher accuracy**, add `--flip-test` (averages the prediction with its mirror — the
benchmark-time setting) and pass the true `--f_mm` if you know the camera's focal length. See
[docs/ACCURACY.md](docs/ACCURACY.md) for the techniques, the evidence, and how they're measured.

### Reproduce

> **Data:** the eval/train sets are gated preprocessed packs — fetch them with
> `uv run gvhmr download --data 3dpw,emdb,rich` (add `amass,h36m,bedlam` for training). They extract under
> `inputs/` by default; set `$GVHMR_DATA_ROOT` to keep them elsewhere — both the download and every dataset
> loader honor it. See [docs/TRAINING.md](docs/TRAINING.md).

1. **Test**:
To reproduce the 3DPW, RICH, and EMDB results in a single run, use the following command:
    ```shell
    uv run gvhmr train global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
    ```
    To test individual datasets, change `global/task` to `gvhmr/test_3dpw`, `gvhmr/test_rich`, or `gvhmr/test_emdb`.

2. **Train**:
To train the model, use the following command:
    ```shell
    # The gvhmr_siga24_release.ckpt is trained with 2x4090 for 420 epochs, note that different GPU settings may lead to different results.
    uv run gvhmr train exp=gvhmr/mixed/mixed
    ```
    During training, note that we do not employ post-processing as in the test script, so the global metrics results will differ (but should still be good for comparison with baseline methods).

## Development

```shell
uv sync --extra dev        # install test/lint/type tooling
uv run pre-commit install  # once — auto-formats staged files on commit (or `make hooks`)
make check                 # the required CI gates locally: ruff format --check + pytest (run before pushing)
make fmt                   # format the whole tree;  make lint / typecheck / test  are the rest
```
See [`AGENTS.md`](AGENTS.md) for architecture, conventions, the behaviour-preservation
landmines, and the upstream-sync workflow.

## Extend & retrain

This fork is being turned from a frozen checkpoint into a **re-trainable, swappable** system — see the
roadmap in [`docs/EXTENSIBILITY.md`](docs/EXTENSIBILITY.md). Already landed:

- **Swap any model by name** — the detector, 2D-pose, feature backbone, and camera are each a **Hydra config
  group**, one config system shared with `train`. Pick an implementation, bundle choices into a *recipe*, or
  tweak any knob — full guide in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md):
  ```shell
  gvhmr demo clip.mp4 --detector yolo11 --pose2d rtmpose --camera dust3r   # pick implementations
  gvhmr demo clip.mp4 --recipe accurate                                    # a committable bundle
  gvhmr demo clip.mp4 --set detector.conf=0.4                              # tweak a knob
  ```
  RTMPose (`--extra rtmpose`) and two scene-aware **metric cameras** (`--camera dust3r` / `vggt`, which
  recover world translation on Apple-Silicon/CPU) are real, verified alternatives. Detector (any YOLO) and
  2D-pose (any COCO-17 estimator) swap freely; the feature backbone is *learned conditioning* → needs a retrain.
- **Retrain on a new backbone** — `gvhmr extract-features VIDEOS OUT --backbone dinov2` writes the training
  feature cache; then `gvhmr train … network.imgseq_dim=<D>`. Verified end-to-end plumbing (Tier B).
- **Training runs on any device** — [`docs/TRAINING.md`](docs/TRAINING.md). A smoke `fit` works even on
  CPU: `GVHMR_DEVICE=cpu uv run gvhmr train exp=gvhmr/mixed/smoke_3dpw`. Real runs are multi-GPU CUDA.
- **Relocatable assets** — one env var each moves checkpoints (`$GVHMR_CHECKPOINTS`), body models
  (`$GVHMR_BODY_MODELS`), and training/eval data (`$GVHMR_DATA_ROOT`) anywhere; `gvhmr download [--data …]`
  fetches into them and every loader reads from them.

All of it is golden-guarded: the released model's inference stays byte-identical.

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
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
