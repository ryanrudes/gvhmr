"""HuggingFace Hub integration for the GVHMR library.

Two directions:

* **download** — resolve model weights from a Hub repo (default :data:`DEFAULT_REPO`), transparently
  falling back to the community mirror :data:`MIRROR_REPO` when a repo/file isn't found. This is what
  ``GVHMR.from_pretrained`` / ``gvhmr.pipeline`` use so ``pip install gvhmr`` "just works" before you've
  published your own repo.
* **publish** — upload the released checkpoints + a generated model card to your own Hub repo
  (``gvhmr publish-hub``), or ``model.save_pretrained`` / ``model.push_to_hub`` a fine-tune.

Registration-gated SMPL/SMPL-X body models are **never** uploaded here (that would violate the MPI
license); they are fetched per-user from the official source — see
:func:`gvhmr.utils.assets.ensure_body_models`.
"""

from __future__ import annotations

from pathlib import Path

#: Where the library looks for weights by default. Publish yours here with ``gvhmr publish-hub``.
DEFAULT_REPO = "ryanrudes/gvhmr"
#: Community mirror used as a fallback (and by the legacy ``gvhmr download`` path).
MIRROR_REPO = "camenduru/GVHMR"

# HF paths of the released checkpoints inside the repo (mirrors the on-disk layout under CHECKPOINT_ROOT).
CKPT_HF_PATHS = {
    "gvhmr": "gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2": "hmr2/epoch=10-step=25000.ckpt",
    "vitpose": "vitpose/vitpose-h-multi-coco.pth",
    "yolo": "yolo/yolov8x.pt",
}


def download(filename: str, repo_id: str = DEFAULT_REPO, *, local_dir=None, token=None, revision=None, **kw) -> Path:
    """Fetch one file from ``repo_id`` (into ``local_dir``), falling back to :data:`MIRROR_REPO`.

    Returns the local path. The fallback lets ``from_pretrained("ryanrudes/gvhmr")`` keep working from the
    community mirror until that repo is published.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    from gvhmr.utils.net import ensure_ca_bundle

    ensure_ca_bundle()
    common = dict(local_dir=(str(local_dir) if local_dir else None), token=token, revision=revision, **kw)
    try:
        return Path(hf_hub_download(repo_id, filename, **common))
    except (RepositoryNotFoundError, EntryNotFoundError):
        if repo_id == MIRROR_REPO:
            raise
        from gvhmr.utils.pylogger import Log

        Log.info(f"[muted]{repo_id}[/] has no {filename}; falling back to mirror [muted]{MIRROR_REPO}[/]")
        # The mirror carries no token/revision assumptions.
        return Path(hf_hub_download(MIRROR_REPO, filename, local_dir=(str(local_dir) if local_dir else None)))


def ensure_gvhmr_ckpt(repo_id: str = DEFAULT_REPO, *, cache_dir=None, token=None, revision=None) -> Path:
    """Ensure the core GVHMR checkpoint is present locally; return its path.

    If it's already under the configured ``CHECKPOINT_ROOT`` at full size we use it; otherwise we fetch it
    from ``repo_id`` (with mirror fallback) into ``CHECKPOINT_ROOT`` so the on-disk layout stays canonical.
    """
    from gvhmr.utils import assets

    if assets.is_present(assets.ASSETS["gvhmr"]):
        return assets.GVHMR_CKPT
    local_dir = cache_dir or assets.CHECKPOINT_ROOT
    download(CKPT_HF_PATHS["gvhmr"], repo_id, local_dir=local_dir, token=token, revision=revision)
    return assets.GVHMR_CKPT


# ----------------------------------------------------------------------------------------------------
# Publishing
# ----------------------------------------------------------------------------------------------------
def model_card(repo_id: str = DEFAULT_REPO) -> str:
    """Generate the Hub model-card (README.md) for a GVHMR weights repo."""
    return f"""---
license: other
license_name: gvhmr-non-commercial
license_link: https://github.com/zju3dv/GVHMR/blob/main/LICENSE
library_name: gvhmr
pipeline_tag: other
tags:
  - human-motion-recovery
  - smpl
  - smplx
  - pose-estimation
  - pytorch
  - siggraph
---

# GVHMR — World-Grounded Human Motion Recovery

Checkpoints for [**GVHMR**](https://github.com/ryanrudes/gvhmr) (Shen et al., *World-Grounded Human
Motion Recovery via Gravity-View Coordinates*, SIGGRAPH Asia 2024). Given a video, GVHMR recovers
SMPL/SMPL-X human motion in both the **camera** and **world** frames.

## Usage

```bash
pip install gvhmr
gvhmr auth smpl          # one-time: your MPI login, to fetch the gated SMPL/SMPL-X body models
```

```python
import gvhmr

pipe = gvhmr.pipeline("human-motion-recovery", model="{repo_id}", device="cuda")
result = pipe("dance.mp4")

result.smpl_params_world      # world-frame SMPL params (global_orient/body_pose/betas/transl)
result.joints_world           # (L, 24, 3) world-frame joints
result.render("overlay.mp4")  # in-cam ∥ world overlay video
result.save_npz("dance.npz")
```

## Contents

| file | what |
|---|---|
| `gvhmr/gvhmr_siga24_release.ckpt` | the trained GVHMR denoiser (the released SIGGRAPH-Asia'24 model) |
| `hmr2/…`, `vitpose/…`, `yolo/…` | preprocessing backbones (feature extractor, 2D pose, detector) |

**Body models are not included.** SMPL/SMPL-X are registration-gated by the Max Planck Institute and
cannot be redistributed — `gvhmr auth smpl` fetches them from the official source with *your* account.

## License

Model weights follow the original GVHMR license (non-commercial research). SMPL/SMPL-X body models are
governed by their own MPI licenses. See the linked repositories.
"""


def _api(token=None):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def publish_checkpoints(
    repo_id: str = DEFAULT_REPO,
    *,
    names=("gvhmr", "hmr2", "vitpose", "yolo"),
    token=None,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    """Upload the released checkpoints + a generated model card to ``repo_id``.

    Reads each checkpoint from the configured ``CHECKPOINT_ROOT`` (fetch them first with ``gvhmr download``).
    **Never** uploads the gated body models. Returns the repo URL.
    """
    from gvhmr.utils import assets

    files: list[tuple[Path, str]] = []
    for n in names:
        a = assets.ASSETS[n]
        if not a.path.exists():
            raise FileNotFoundError(f"{n} checkpoint not found at {a.path} — run `gvhmr download {n}` first.")
        files.append((a.path, a.hf_path))

    if dry_run:
        listing = "\n".join(f"  {src}  →  {repo_id}/{dst}" for src, dst in files)
        return f"[dry-run] would create {repo_id} and upload:\n{listing}\n  <generated README.md>"

    api = _api(token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    for src, dst in files:
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo_id, repo_type="model")
    api.upload_file(
        path_or_fileobj=model_card(repo_id).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}"


def save_pretrained(model, save_dir: str | Path, *, repo_id: str = DEFAULT_REPO) -> Path:
    """Write a self-contained model folder (checkpoint + config.json + model card) for ``model``."""
    import json
    import shutil

    from gvhmr.utils import assets

    save_dir = Path(save_dir)
    (save_dir / "gvhmr").mkdir(parents=True, exist_ok=True)
    ckpt = getattr(model, "ckpt_path", None) or assets.GVHMR_CKPT
    shutil.copyfile(ckpt, save_dir / assets.ASSETS["gvhmr"].hf_path)
    (save_dir / "config.json").write_text(
        json.dumps(
            {"library_name": "gvhmr", "task": "human-motion-recovery", "architecture": "GVHMR"},
            indent=2,
        )
    )
    (save_dir / "README.md").write_text(model_card(repo_id))
    return save_dir


def push_to_hub(model, repo_id: str, *, token=None, private: bool = False) -> str:
    """Push ``model``'s checkpoint + card to ``repo_id`` (for sharing a fine-tune). Returns the repo URL."""
    import tempfile

    api = _api(token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        save_pretrained(model, td, repo_id=repo_id)
        api.upload_folder(folder_path=td, repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def publish_space(space_id: str, space_dir: str | Path, *, token=None, sdk: str = "docker") -> str:
    """Create/update a HuggingFace **Space** from a local app folder. Returns the Space URL.

    ``sdk`` sets the Space runtime on *first* creation; for an existing Space the folder's
    ``README.md`` front-matter (``sdk: docker``) is authoritative. GVHMR's Space is a Docker Space
    (its ``Dockerfile`` pre-installs ``chumpy`` with build isolation disabled).
    """
    api = _api(token)
    api.create_repo(space_id, repo_type="space", space_sdk=sdk, exist_ok=True)
    api.upload_folder(folder_path=str(space_dir), repo_id=space_id, repo_type="space")
    return f"https://huggingface.co/spaces/{space_id}"
