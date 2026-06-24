# Research debugging & visualization

Helpers for poking at GVHMR's tensors and internal representations. None of these
change model behaviour.

## Tensor / latent inspection — `gvhmr.utils.debug`

```python
from gvhmr.utils.debug import describe, decompose_latent, summarize_latent, count_parameters, nan_hooks, remove_hooks

describe(pred_x, name="pred_x")
# pred_x: Tensor(1, 16, 151) float32 [mps] min=-3.2 max=2.9 mean=0.01

# Split the 151-dim latent into named parts (matches EnDecoder's decode contract):
parts = decompose_latent(pred_x)          # {"body_pose_r6d": (...,126), "betas": (...,10), ...}
summarize_latent(pred_x)                   # prints a per-component describe()

count_parameters(model, trainable_only=True)
```

The latent layout is `LATENT_LAYOUT` (`body_pose_r6d[0:126]`, `betas[126:136]`,
`global_orient_c[136:142]`, `global_orient_gv[142:148]`, `local_transl_vel[148:151]`),
kept in sync with `docs/BEHAVIOR.md`.

### Finding where a forward goes non-finite

```python
from gvhmr.utils.debug import nan_hooks, remove_hooks
handles = nan_hooks(model)      # warns on the first module emitting NaN/Inf
out = model(**batch)
remove_hooks(handles)
```

## 3D visualization — `gvhmr.utils.wis3d_utils` (install `--extra vis`)

```python
from gvhmr.utils.wis3d_utils import make_wis3d, add_motion_as_lines
wis3d = make_wis3d(name="debug-motion")
add_motion_as_lines(joints, wis3d, name="pred", skeleton_type="coco17")
```

Open the result with the `wis3d` viewer. The demo also writes in-camera and world
overlay videos (`gvhmr demo VIDEO`, needs pytorch3d).

## Devices

Force a device for repro/debugging with `GVHMR_DEVICE=cpu` (or `mps`/`cuda`), or call
`gvhmr.utils.device.get_device(prefer=...)`. CPU is fully deterministic; MPS/CUDA differ
within floating-point tolerance (see `tests/test_device.py`).

## Hydra config inspection

```bash
uv run gvhmr info                                              # device / extras / checkpoints
uv run gvhmr train exp=gvhmr/mixed/mixed --cfg job             # print the composed config
```
