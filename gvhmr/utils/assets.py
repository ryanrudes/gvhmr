"""Model checkpoints & assets — one configurable root, one manifest, one fetch path.

Every released asset the code needs is mirrored on the HuggingFace repo ``camenduru/GVHMR`` with the exact
on-disk layout, so ``gvhmr download`` drops each file where the code expects it — no scavenger hunt, no
manual placement. **All** checkpoint paths in the codebase resolve through :data:`CHECKPOINT_ROOT`
(env ``$GVHMR_CHECKPOINTS``, default ``<repo>/inputs/checkpoints``), so you can keep large weights outside
the working tree (e.g. a shared ``~/Datasets``) by setting one env var — no symlink hacks.

SMPL/SMPL-X **body models are registration-gated** and can't be auto-fetched; see :data:`GATED`.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

from gvhmr import PROJ_ROOT
from gvhmr.utils.localconfig import resolve

# --- Asset roots — a readable config file OR env vars point these at a high-storage volume ---------
# Every root resolves as: env var > config file ([paths] in ~/.config/gvhmr/config.toml) > default
# (see gvhmr/utils/localconfig.py). Manage with `gvhmr config`; import these instead of hard-coding paths.
CHECKPOINT_ROOT, _src_ckpt = resolve("checkpoints", "GVHMR_CHECKPOINTS", PROJ_ROOT / "inputs/checkpoints")
#: SMPL/SMPL-X body models (a `smplx/` and `smpl/` subfolder); defaults under CHECKPOINT_ROOT.
BODY_MODEL_ROOT, _src_body = resolve("body_models", "GVHMR_BODY_MODELS", CHECKPOINT_ROOT / "body_models")
#: Training/eval data packs (the `<DS>/hmr4d_support/` bundles).
DATA_ROOT, _src_data = resolve("data", "GVHMR_DATA_ROOT", PROJ_ROOT / "inputs")
#: DUSt3R / Depth-Anything scene-camera weights (used by `--camera dust3r|vggt`).
SCENE_ROOT, _src_scene = resolve("scene", "GVHMR_DATA", Path("~/Datasets/GVHMR").expanduser())

#: The configurable roots, for `gvhmr config` / `gvhmr info`: key → (env var, resolved path, source, description).
ROOTS: dict[str, tuple[str, Path, str, str]] = {
    "checkpoints": (
        "GVHMR_CHECKPOINTS",
        CHECKPOINT_ROOT,
        _src_ckpt,
        "model checkpoints (gvhmr, hmr2, vitpose, yolo, dpvo)",
    ),
    "data": ("GVHMR_DATA_ROOT", DATA_ROOT, _src_data, "training/eval data packs (<DS>/hmr4d_support)"),
    "body_models": ("GVHMR_BODY_MODELS", BODY_MODEL_ROOT, _src_body, "SMPL / SMPL-X body models (registration-gated)"),
    "scene": (
        "GVHMR_DATA",
        SCENE_ROOT,
        _src_scene,
        "DUSt3R / Depth-Anything scene-camera weights (--camera dust3r|vggt)",
    ),
}

# Named checkpoint paths — import these instead of hard-coding "inputs/checkpoints/...".
GVHMR_CKPT = CHECKPOINT_ROOT / "gvhmr/gvhmr_siga24_release.ckpt"
HMR2_CKPT = CHECKPOINT_ROOT / "hmr2/epoch=10-step=25000.ckpt"
VITPOSE_CKPT = CHECKPOINT_ROOT / "vitpose/vitpose-h-multi-coco.pth"
YOLO_CKPT = CHECKPOINT_ROOT / "yolo/yolov8x.pt"
DPVO_CKPT = CHECKPOINT_ROOT / "dpvo/dpvo.pth"

HF_REPO = "camenduru/GVHMR"  # community mirror of the GVHMR checkpoints + preprocessed data packs


@dataclass(frozen=True)
class Asset:
    """A fetchable checkpoint: its path in the HF repo, its on-disk destination, size, and feature group."""

    hf_path: str
    path: Path
    size: int
    group: str  # "demo" (core inference), "slam" (DPVO)


#: The fetchable checkpoints, by short name. `size` is the exact byte count (used to skip/validate).
ASSETS: dict[str, Asset] = {
    "gvhmr": Asset("gvhmr/gvhmr_siga24_release.ckpt", GVHMR_CKPT, 163_508_011, "demo"),
    "hmr2": Asset("hmr2/epoch=10-step=25000.ckpt", HMR2_CKPT, 2_709_494_041, "demo"),
    "vitpose": Asset("vitpose/vitpose-h-multi-coco.pth", VITPOSE_CKPT, 2_549_075_546, "demo"),
    "yolo": Asset("yolo/yolov8x.pt", YOLO_CKPT, 136_867_539, "demo"),
    "dpvo": Asset("dpvo/dpvo.pth", DPVO_CKPT, 14_167_743, "slam"),
}

#: Registration-gated assets we can't auto-fetch: name → (description, target dir, sign-up URLs).
GATED: dict[str, tuple[str, Path, tuple[str, ...]]] = {
    "body_models": (
        "SMPL + SMPL-X body models (smplx/SMPLX_{GENDER}.npz, smpl/SMPL_{GENDER}.pkl)",
        BODY_MODEL_ROOT,
        ("https://smpl.is.tue.mpg.de/", "https://smpl-x.is.tue.mpg.de/"),
    ),
}

#: Preprocessed training/eval packs on the same HF repo: name → (hf tar.gz, extracted dir under DATA_ROOT, size).
DATA_PACKS: dict[str, tuple[str, str, int]] = {
    "3dpw": ("preproc_data/3DPW_hmr4d_support.tar.gz", "3DPW", 388_384_435),
    "amass": ("preproc_data/AMASS_hmr4d_support.tar.gz", "AMASS", 1_710_718_630),
    "bedlam": ("preproc_data/BEDLAM_hmr4d_support.tar.gz", "BEDLAM", 21_069_046_288),
    "emdb": ("preproc_data/EMDB_hmr4d_support.tar.gz", "EMDB", 511_561_664),
    "h36m": ("preproc_data/H36M_hmr4d_support.tar.gz", "H36M", 3_022_581_043),
    "rich": ("preproc_data/RICH_hmr4d_support.tar.gz", "RICH", 454_219_419),
}


def is_present(a: Asset) -> bool:
    """True if the asset file exists at full size (a size mismatch ⇒ partial/corrupt ⇒ re-fetch)."""
    try:
        return a.path.exists() and a.path.stat().st_size == a.size
    except OSError:
        return False


def select(group: str | None = None, names: list[str] | None = None) -> dict[str, Asset]:
    """Resolve the asset set from a group ("demo"/"slam"/"all") and/or explicit names."""
    if names:
        unknown = set(names) - set(ASSETS)
        if unknown:
            raise KeyError(f"unknown asset(s) {sorted(unknown)}; known: {sorted(ASSETS)}")
        return {n: ASSETS[n] for n in names}
    if group in (None, "all"):
        return dict(ASSETS)
    return {n: a for n, a in ASSETS.items() if a.group == group}


def missing(group: str | None = None) -> list[str]:
    """Names of assets in the group that aren't fully present on disk."""
    return [n for n, a in select(group).items() if not is_present(a)]


def fetch(assets: dict[str, Asset], force: bool = False, repo: str | None = None) -> list[Path]:
    """Download the given assets into CHECKPOINT_ROOT (skips present ones unless ``force``).

    ``repo`` overrides the source HF repo (default :data:`HF_REPO`); a file missing there falls back to the
    community mirror :data:`HF_REPO`, so ``from_pretrained("your/repo")`` works before every file is mirrored.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    from gvhmr.utils.net import ensure_ca_bundle

    ensure_ca_bundle()  # repair misconfigured TLS cert env (common on HPC) before any HTTPS
    source = repo or HF_REPO
    out = []
    for a in assets.values():
        if is_present(a) and not force:
            continue
        try:
            hf_hub_download(source, a.hf_path, local_dir=str(CHECKPOINT_ROOT))
        except (RepositoryNotFoundError, EntryNotFoundError):
            if source == HF_REPO:
                raise
            hf_hub_download(HF_REPO, a.hf_path, local_dir=str(CHECKPOINT_ROOT))
        out.append(a.path)
    return out


# --- Registration-gated body models — fetched per-user from the official MPI source, never re-hosted ----
def body_models_present(smpl: bool = False) -> bool:
    """True when the SMPL-X body model is present (and, if ``smpl``, the SMPL model for rendering)."""
    ok = (BODY_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz").exists()
    if smpl:
        ok = ok and (BODY_MODEL_ROOT / "smpl/SMPL_NEUTRAL.pkl").exists()
    return ok


def _gated_error() -> FileNotFoundError:
    return FileNotFoundError(
        f"SMPL-X body model not found under {BODY_MODEL_ROOT}. These are registration-gated: run "
        f"`gvhmr auth smpl` (one-time MPI login → auto-fetch), or sign up at "
        f"https://smpl-x.is.tue.mpg.de/ + https://smpl.is.tue.mpg.de/ and place them there "
        f"(or set $GVHMR_BODY_MODELS). `gvhmr download` prints the exact layout."
    )


def ensure_body_models(smpl: bool = False, auto: bool = True) -> None:
    """Ensure the gated body models are present, auto-fetching from MPI when credentials are available.

    Fetches only what's missing. If SMPL-X is still absent afterwards, raises the actionable gated error
    (identical to the manual instructions). A missing SMPL model when ``smpl`` is requested only warns —
    motion recovery proceeds; overlay rendering is what needs it.
    """
    from gvhmr.utils.pylogger import Log

    have_smplx = (BODY_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz").exists()
    have_smpl = (BODY_MODEL_ROOT / "smpl/SMPL_NEUTRAL.pkl").exists()
    if have_smplx and (have_smpl or not smpl):
        return

    if auto:
        from gvhmr.utils import mpi_download

        if mpi_download.credentials() is not None:
            try:
                placed = mpi_download.fetch_body_models(
                    BODY_MODEL_ROOT, smplx=not have_smplx, smpl=(smpl and not have_smpl)
                )
                Log.info(f"[ok]Body models fetched[/] from MPI ({len(placed)} file(s)) → [muted]{BODY_MODEL_ROOT}[/]")
                have_smplx = (BODY_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz").exists()
                have_smpl = (BODY_MODEL_ROOT / "smpl/SMPL_NEUTRAL.pkl").exists()
            except (OSError, PermissionError, FileNotFoundError, ImportError) as e:
                Log.warning(f"[warn]Automatic body-model fetch failed[/] ([muted]{e}[/]) — falling back to manual.")

    if not have_smplx:
        raise _gated_error()
    if smpl and not have_smpl:
        Log.warning(
            f"[warn]smpl/SMPL_NEUTRAL.pkl not found[/] under [muted]{BODY_MODEL_ROOT}[/] — motion will be "
            f"recovered, but overlay videos will be skipped. `gvhmr auth smpl` (or sign up at "
            f"https://smpl.is.tue.mpg.de/) to enable rendering."
        )


def fetch_data_pack(name: str, force: bool = False) -> Path:
    """Download + extract a preprocessed data pack under DATA_ROOT (e.g. inputs/3DPW/hmr4d_support/)."""
    from huggingface_hub import hf_hub_download

    from gvhmr.utils.net import ensure_ca_bundle

    ensure_ca_bundle()  # repair misconfigured TLS cert env (common on HPC) before any HTTPS
    hf_path, ds_dir, _ = DATA_PACKS[name]
    target = DATA_ROOT / ds_dir / "hmr4d_support"
    if target.exists() and not force:
        return target
    tar = hf_hub_download(HF_REPO, hf_path, local_dir=str(DATA_ROOT))
    with tarfile.open(tar) as t:
        t.extractall(DATA_ROOT)  # noqa: S202 — trusted mirror; packs extract to <DS>/hmr4d_support/
    return target
