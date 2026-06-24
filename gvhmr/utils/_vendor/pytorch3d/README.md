# Vendored: `pytorch3d.transforms` (rotation conversions + SO(3))

Pure-PyTorch rotation utilities copied from **PyTorch3D** so GVHMR's inference and
geometry math runs on CPU / Apple-Silicon MPS without a `pytorch3d` install.

| | |
|---|---|
| Upstream | https://github.com/facebookresearch/pytorch3d |
| Tag | `v0.7.6` (matches the originally pinned `py310_cu121_pyt230` wheel) |
| License | BSD-3-Clause (see `LICENSE` in this directory) |

## Files (verbatim copies)

- `rotation_conversions.py` — `pytorch3d/transforms/rotation_conversions.py`
- `so3.py` — `pytorch3d/transforms/so3.py`
- `math.py` — `pytorch3d/transforms/math.py` (provides `acos_linear_extrapolation`, used by `so3.py`)

## Patches applied

The **only** modifications are import-path rewrites so the files resolve against
each other instead of the full `pytorch3d` package. Each is tagged
`# [GVHMR vendor patch]` in source. No numerical logic was changed, so behaviour
is identical to upstream `v0.7.6`.

1. `rotation_conversions.py`: `from ..common.datatypes import Device` →
   inlined `Device = Union[str, torch.device]`.
2. `so3.py`: `from pytorch3d.transforms import rotation_conversions` →
   `from . import rotation_conversions`.
3. `so3.py`: `from ..transforms import acos_linear_extrapolation` →
   `from .math import acos_linear_extrapolation`.

## Re-syncing from upstream

```bash
TAG=v0.7.6  # or a newer tag
base="https://raw.githubusercontent.com/facebookresearch/pytorch3d/$TAG/pytorch3d/transforms"
for f in rotation_conversions so3 math; do curl -fsSL "$base/$f.py" -o "$f.py"; done
# then re-apply the three patches above (search for "[GVHMR vendor patch]" in git history)
```

First-party code does **not** import these modules directly; it imports the facade
`gvhmr.utils.geo.rotations`, which is the single swap point.
