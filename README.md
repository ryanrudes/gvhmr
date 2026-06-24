# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> [!NOTE]
> **Modernized fork** of [`zju3dv/GVHMR`](https://github.com/zju3dv/GVHMR) by
> [@ryanrudes](https://github.com/ryanrudes) — runs on **Apple Silicon (MPS)**, with `uv`/`pyproject`
> packaging, a Typer + Rich `gvhmr` CLI, a GPU mesh renderer, skeleton-overlay exports, and a
> device-agnostic **scene-aware SLAM** backend (`gvhmr demo --slam dust3r`, a Mac-friendly DPVO
> alternative that recovers a *metric* camera). The released model's behaviour is preserved
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
> (moderngl/Metal), skeleton-overlay video exports, a device-agnostic **DUSt3R + Depth-Anything-V2
> scene-aware SLAM backend** (`--slam dust3r` — recovers a metric camera on Mac, where DPVO is
> CUDA-only; set it up with [`scripts/setup_scene_aware.sh`](scripts/setup_scene_aware.sh)), and
> agent tooling — with model behaviour preserved (golden-guarded).
> See [`AGENTS.md`](AGENTS.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
> [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Setup

Please see [installation](docs/INSTALL.md). In short:

```bash
uv sync                 # base install (CPU / Apple-Silicon MPS); add --extra preproc for the demo
uv run gvhmr info       # check device, installed features, and checkpoint status
```

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

Use `-s` for a static camera (skip visual odometry); otherwise the camera is estimated by
SimpleVO (or DPVO with `--use-dpvo`). The device is auto-selected (CUDA → MPS → CPU);
override with `GVHMR_DEVICE=mps|cpu|cuda`. `--render-scale` trades overlay resolution for
speed, `--no-render` skips overlays. (The old `python tools/demo/demo.py …` scripts still
work as thin shims.)

For **higher accuracy**, add `--flip-test` (averages the prediction with its mirror — the
benchmark-time setting) and pass the true `--f_mm` if you know the camera's focal length. See
[docs/ACCURACY.md](docs/ACCURACY.md) for the techniques, the evidence, and how they're measured.

### Reproduce
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
uv run pytest              # CPU/MPS characterization suite (no GPU/checkpoints needed)
uv run ruff check gvhmr tools tests
uv run pyright
```
See [`AGENTS.md`](AGENTS.md) for architecture, conventions, the behaviour-preservation
landmines, and the upstream-sync workflow.

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
