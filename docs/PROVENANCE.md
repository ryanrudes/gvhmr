# Provenance & upstream sync

This repo is a modernized fork of **[zju3dv/GVHMR](https://github.com/zju3dv/GVHMR)**.
This document records what came from where, what was changed structurally, and how to
pull future changes from the original repo.

## Relationship to the original

- The import package was renamed **`hmr4d` → `gvhmr`** (the distribution was already
  `gvhmr`). This was a word-boundary rename, so dataset/output names like
  `hmr4d_support` (preprocessed-data dirs) and `hmr4d_results` (output files) were
  intentionally **kept** for data compatibility.
- Packaging moved from `setup.py` + `requirements.txt` to **PEP 621 `pyproject.toml`**
  (hatchling build backend, `uv` workflow, layered extras).
- Hard `pytorch3d` imports on the inference/geometry path were removed by vendoring
  the rotation math (below) so the package installs and runs on CPU / Apple-Silicon MPS.

The machine-readable map from upstream paths to local paths is
[`docs/upstream_sync.yaml`](upstream_sync.yaml). It is consumed by
[`scripts/upstream_sync.py`](../scripts/upstream_sync.py).

## Vendored third-party code (frozen)

These trees are kept close to their upstreams. **Do not refactor them**; re-vendor from
source instead. Any necessary edit is tagged `# [GVHMR vendor patch]` and documented.

| Path | Upstream | Version | License | Policy |
|---|---|---|---|---|
| `gvhmr/utils/_vendor/pytorch3d/` | [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | `v0.7.6` | BSD-3 | byte-identical copies of `rotation_conversions.py`, `so3.py`, `math.py`; **only import-path patches**. Re-exported by the `gvhmr.utils.geo.rotations` facade. |
| `gvhmr/network/hmr2/` | [4D-Humans / HMR2.0](https://github.com/shubham-goel/4D-Humans) (+ MMLab ViT) | — | per upstream | frozen ViT feature extractor (used as `f_imgseq` provider); `vit.py` attention patched to fused SDPA (`F.scaled_dot_product_attention`, same math) |
| `gvhmr/utils/preproc/vitpose_pytorch/` | [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | — | per upstream | frozen 2D-pose backbone; `backbones/vit.py` attention patched to fused SDPA (`F.scaled_dot_product_attention`, same math) |

Each vendored dir carries its own `README`/`LICENSE` where applicable
(e.g. `gvhmr/utils/_vendor/pytorch3d/README.md`).

### Why vendor pytorch3d's rotation conversions?

The original code imported `pytorch3d.transforms` across ~20 modules. The pinned
pytorch3d wheel is CUDA/Linux/py3.10-only and its C++ extensions don't build easily
off-Linux — so the package couldn't even *import* on macOS. But the rotation
functions actually used (`axis_angle_to_matrix`, `matrix_to_axis_angle`,
`rotation_6d_to_matrix`, `quaternion_to_matrix`, `so3_exp/log_map`, …) are pure
PyTorch. Vendoring them verbatim removes pytorch3d from the entire
inference/geometry path (only mesh *rendering* still needs the real pytorch3d) while
keeping numerics **identical to upstream by construction**.

## Pulling changes from the original repo

```bash
git remote add upstream https://github.com/zju3dv/GVHMR    # one time
python scripts/upstream_sync.py                             # report: upstream changes → local paths
python scripts/upstream_sync.py --since <ref> --show-diff   # include the patches
```

The script reads `docs/upstream_sync.yaml`, lists files changed upstream since
`last_synced_commit`, and maps each to where it now lives here (applying the package
rename and import rewrites). After reconciling, bump `last_synced_commit` and append any
new relocations to the manifest so the next sync stays mechanical — **the manifest is the
contract that keeps upstream merges tractable even after heavy restructuring.**
