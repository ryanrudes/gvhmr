"""Credentialed download of the registration-gated SMPL / SMPL-X body models from the official source.

The Max Planck Institute licenses SMPL/SMPL-X for non-commercial research and **prohibits
redistribution**, so we can't mirror the model files. Instead this module fetches them straight from the
MPI download portal using the **end-user's own account** — exactly what the official ``smplx`` /
SMPLify-X install scripts do — so each user accepts the license themselves and nothing is re-hosted.

**SMPL and SMPL-X are two separate MPI registrations** (``smpl.is.tue.mpg.de`` /
``smpl-x.is.tue.mpg.de``), each with its own login and license — we never assume one account works for
both. Credentials are resolved *per dataset*: ``$SMPLX_USER``/``$SMPLX_PW`` and ``$SMPL_USER``/``$SMPL_PW``,
else the matching ``[auth.smplx]`` / ``[auth.smpl]`` section of a 0600 file written by ``gvhmr auth smpl``.
Everything here is best-effort: on any failure the caller falls back to the manual sign-up instructions,
so behaviour never regresses relative to "download it yourself".
"""

from __future__ import annotations

import os
import stat
import tomllib
import zipfile
from pathlib import Path

from gvhmr.utils.localconfig import _xdg_config_home

#: The MPI download portal — one POST with form-encoded credentials authenticates and streams the file.
PORTAL = "https://download.is.tue.mpg.de/download.php"

#: Registration site per dataset (each is a *separate* account + license).
SITES = {"smplx": "smpl-x.is.tue.mpg.de", "smpl": "smpl.is.tue.mpg.de"}

#: Per-dataset env vars: (user var, (password vars, first is canonical)). No shared/generic fallback —
#: the two accounts are independent, so a generic ``$MPI_USER`` would just re-introduce the wrong assumption.
_ENV_VARS = {
    "smplx": ("SMPLX_USER", ("SMPLX_PW", "SMPLX_PASSWORD")),
    "smpl": ("SMPL_USER", ("SMPL_PW", "SMPL_PASSWORD")),
}

#: Which archive on the portal carries each model, and how to recognise the files we need inside it.
#: ``domain``/``sfile`` are the portal's query params; ``members`` maps a filename-suffix match → target.
ARCHIVES = {
    "smplx": {
        "domain": "smplx",
        "sfile": "models_smplx_v1_1.zip",
        # SMPLX_{NEUTRAL,MALE,FEMALE}.npz → body_models/smplx/<name>
        "members": [("SMPLX_NEUTRAL.npz", "smplx/SMPLX_NEUTRAL.npz")],
        "extra": [("SMPLX_MALE.npz", "smplx/SMPLX_MALE.npz"), ("SMPLX_FEMALE.npz", "smplx/SMPLX_FEMALE.npz")],
    },
    "smpl": {
        "domain": "smpl",
        "sfile": "SMPL_python_v.1.1.0.zip",
        # basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl → body_models/smpl/SMPL_NEUTRAL.pkl
        "members": [("neutral_lbs_10_207_0_v1.1.0.pkl", "smpl/SMPL_NEUTRAL.pkl")],
        "extra": [
            ("m_lbs_10_207_0_v1.1.0.pkl", "smpl/SMPL_MALE.pkl"),
            ("f_lbs_10_207_0_v1.1.0.pkl", "smpl/SMPL_FEMALE.pkl"),
        ],
    },
}

CRED_PATH = _xdg_config_home() / "gvhmr" / "smpl_credentials.toml"


def _site(dataset: str) -> str:
    return SITES.get(dataset, f"{dataset}.is.tue.mpg.de")


#: Branded, human-facing dataset name for messages (``.upper()`` would render "SMPLX", not "SMPL-X").
BRAND = {"smplx": "SMPL-X", "smpl": "SMPL"}


def _brand(dataset: str) -> str:
    return BRAND.get(dataset.lower(), dataset.upper())


def _toml_str(s: str) -> str:
    """A TOML basic string (double-quoted, minimal escaping) — safe for arbitrary passwords."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _read_file() -> dict:
    """Parse the credentials TOML file, or ``{}`` if it's missing/unreadable."""
    if not CRED_PATH.is_file():
        return {}
    try:
        with open(CRED_PATH, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _file_sections() -> dict[str, tuple[str, str]]:
    """Stored ``(username, password)`` per dataset from the file.

    New files use nested ``[auth.smplx]`` / ``[auth.smpl]`` sections. A pre-split flat ``[auth]``
    (single-account era) is migrated to the **SMPL-X** slot only — that's all it was ever used for —
    so upgrading never silently reuses one login for the other account.
    """
    auth = _read_file().get("auth", {})
    out: dict[str, tuple[str, str]] = {}
    for ds, v in auth.items():
        if isinstance(v, dict) and v.get("username") and v.get("password"):
            out[ds.lower()] = (str(v["username"]), str(v["password"]))
    if auth.get("username") and auth.get("password") and "smplx" not in out:
        out["smplx"] = (str(auth["username"]), str(auth["password"]))
    return out


def credentials(dataset: str = "smplx") -> tuple[str, str] | None:
    """Resolve the MPI login for one dataset (``"smplx"`` or ``"smpl"``); ``None`` if unset.

    SMPL and SMPL-X are **separate accounts** — this never falls back from one to the other. Order:
    ``$<DS>_USER``/``$<DS>_PW`` env vars, then the matching ``[auth.<ds>]`` file section.
    """
    dataset = dataset.lower()
    if dataset not in ARCHIVES:
        raise ValueError(f"unknown body-model dataset {dataset!r} (expected one of {sorted(ARCHIVES)})")
    user_var, pw_vars = _ENV_VARS[dataset]
    user = os.environ.get(user_var)
    pw = next((os.environ[v] for v in pw_vars if os.environ.get(v)), None)
    if user and pw:
        return user, pw
    return _file_sections().get(dataset)


def save_credentials(username: str, password: str, dataset: str = "smplx") -> Path:
    """Persist one dataset's MPI login to the 0600 file (preserving the other account). Returns the path."""
    dataset = dataset.lower()
    if dataset not in ARCHIVES:
        raise ValueError(f"unknown body-model dataset {dataset!r} (expected one of {sorted(ARCHIVES)})")
    sections = _file_sections()  # migrates any legacy flat [auth] → smplx before we merge
    sections[dataset] = (username, password)
    CRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        f"[auth.{ds}]\nusername = {_toml_str(u)}\npassword = {_toml_str(p)}\n" for ds, (u, p) in sections.items()
    )
    # Create the file 0600 *before* writing, so the plaintext passwords are never briefly world-readable
    # (a plain write_text + chmod leaves a umask-0644 window). O_TRUNC replaces any prior content.
    fd = os.open(CRED_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    CRED_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — also fixes an older file created pre-patch
    return CRED_PATH


def _missing_creds_msg(dataset: str) -> str:
    user_var, pw_vars = _ENV_VARS[dataset]
    return (
        f"No MPI credentials for the {_brand(dataset)} account. SMPL and SMPL-X are separate "
        f"registrations — set ${user_var}/${pw_vars[0]} or run `gvhmr auth smpl` "
        f"(register at https://{_site(dataset)}/)."
    )


def _stream_to_file(response, dest: Path, label: str) -> None:
    """Stream an HTTP response body to ``dest`` with a byte-progress bar (when interactive)."""
    from gvhmr.utils.console import console

    total = int(response.headers.get("Content-Length") or 0)
    if not (total and console.is_terminal and not console.quiet):
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        return

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    columns = (
        SpinnerColumn(style="gvhmr"),
        TextColumn("[stage]{task.description}"),
        BarColumn(bar_width=None),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console, transient=True) as prog:
        task = prog.add_task(label, total=total)
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                prog.advance(task, len(chunk))


def _download_archive(domain: str, sfile: str, dest: Path, user: str, pw: str) -> None:
    """POST credentials to the portal and stream the archive to ``dest`` (raises on HTTP / auth failure)."""
    import requests

    from gvhmr.utils.net import ensure_ca_bundle

    ensure_ca_bundle()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        url = f"{PORTAL}?domain={domain}&sfile={sfile}"
        r = s.post(url, data={"username": user, "password": pw}, stream=True, allow_redirects=True, timeout=60)
        if r.status_code in (401, 403):  # authenticated, but this account isn't authorized for `domain`
            raise PermissionError(
                f"MPI refused the {_brand(domain)} download (HTTP {r.status_code}). SMPL and SMPL-X are "
                f"separate accounts: this login isn't registered for {_brand(domain)} or hasn't accepted "
                f"its license. Register + accept it at https://{_site(domain)}/ with this account, then "
                f"retry (`gvhmr auth smpl`)."
            )
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:  # the portal returns its login page (200) on bad credentials
            raise PermissionError(
                f"MPI login failed for the {_brand(domain)} account — check its credentials "
                f"(`gvhmr auth smpl`; register at https://{_site(domain)}/)."
            )
        _stream_to_file(r, dest, f"Downloading {domain.upper()} body model")


def _extract_members(zip_path: Path, target_root: Path, wanted: list[tuple[str, str]]) -> list[Path]:
    """Extract members whose name ends with a wanted suffix to the mapped target path."""
    placed = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for suffix, target_rel in wanted:
            match = next((n for n in names if n.replace("\\", "/").endswith(suffix)), None)
            if match is None:
                continue
            target = target_root / target_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(match) as src, open(target, "wb") as dst:
                dst.write(src.read())
            placed.append(target)
    return placed


def fetch_body_models(
    body_model_root: Path,
    *,
    smpl: bool = True,
    smplx: bool = True,
    genders: bool = False,
    smplx_credentials: tuple[str, str] | None = None,
    smpl_credentials: tuple[str, str] | None = None,
) -> list[Path]:
    """Download + place the requested body models under ``body_model_root``. Returns the files placed.

    ``smplx`` fetches ``smplx/SMPLX_NEUTRAL.npz`` (needed for motion recovery); ``smpl`` fetches
    ``smpl/SMPL_NEUTRAL.pkl`` (needed for mesh rendering). ``genders=True`` also grabs the MALE/FEMALE
    variants (for gendered eval). Each dataset authenticates with its **own** account: pass
    ``smplx_credentials`` / ``smpl_credentials`` to override, else they resolve via :func:`credentials`.
    Raises if a requested dataset's credentials are missing or its download/extract fails.
    """
    import tempfile

    body_model_root = Path(body_model_root)
    overrides = {"smplx": smplx_credentials, "smpl": smpl_credentials}
    placed: list[Path] = []
    todo = [k for k, on in (("smplx", smplx), ("smpl", smpl)) if on]
    with tempfile.TemporaryDirectory() as td:
        for key in todo:
            creds = overrides.get(key) or credentials(key)
            if creds is None:
                raise PermissionError(_missing_creds_msg(key))
            user, pw = creds
            spec = ARCHIVES[key]
            archive = Path(td) / spec["sfile"]
            _download_archive(spec["domain"], spec["sfile"], archive, user, pw)
            wanted = list(spec["members"]) + (list(spec["extra"]) if genders else [])
            got = _extract_members(archive, body_model_root, wanted)
            if not any(t.name.endswith(spec["members"][0][1].split("/")[-1]) for t in got):
                raise FileNotFoundError(
                    f"Downloaded the {key} archive but couldn't find {spec['members'][0][1]} inside it "
                    f"(members: {spec['sfile']}). The MPI archive layout may have changed."
                )
            placed += got
    return placed


def dataset_files(dataset: str, genders: bool = False) -> list[str]:
    """The on-disk relative paths a dataset places (e.g. ``["smplx/SMPLX_NEUTRAL.npz"]``).

    Derived from :data:`ARCHIVES` so the mirror layout stays in lockstep with the MPI-extract layout.
    """
    spec = ARCHIVES[dataset.lower()]
    files = [target for _suffix, target in spec["members"]]
    if genders:
        files += [target for _suffix, target in spec["extra"]]
    return files


def fetch_from_mirror(
    body_model_root: Path,
    repo: str,
    *,
    smpl: bool = True,
    smplx: bool = True,
    genders: bool = False,
    token: str | None = None,
) -> list[Path]:
    """Fetch body models from a **user-controlled** HF repo (their own private copy) into ``body_model_root``.

    This is the seamless alternative to the gated MPI login: one ``hf_hub_download`` per file, no per-run
    MPI credentials. It is **not** redistribution — GVHMR never hosts these files; ``repo`` is a mirror the
    *user* created and keeps private (see ``gvhmr body-models push``). Files already present are skipped;
    files absent from the repo are skipped too (so a partial mirror still helps and the caller can fall
    back to MPI for the rest). Repo-not-found / auth / network errors propagate to the caller.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    from gvhmr.utils.hf_token import resolve_hf_token
    from gvhmr.utils.net import ensure_ca_bundle

    ensure_ca_bundle()
    token = resolve_hf_token(token)
    body_model_root = Path(body_model_root)
    placed: list[Path] = []
    todo = [k for k, on in (("smplx", smplx), ("smpl", smpl)) if on]
    for key in todo:
        for rel in dataset_files(key, genders):
            if (body_model_root / rel).exists():
                continue
            try:
                got = hf_hub_download(repo, rel, local_dir=str(body_model_root), token=token)
            except EntryNotFoundError:
                continue  # not in this mirror — leave it for the MPI fallback
            except HfHubHTTPError as e:
                if e.response.status_code in (401, 403):
                    raise PermissionError(
                        f"Could not read private mirror '{repo}' (HTTP {e.response.status_code}). "
                        "Authenticate with a token that can read the repo: set $HF_TOKEN or "
                        "$HUGGING_FACE_HUB_TOKEN, run `huggingface-cli login`, or pass `--token`. "
                        "In a notebook use os.environ['HF_TOKEN'] = '...' — shell `export` does not "
                        "work in `!` cells."
                    ) from e
                raise
            placed.append(Path(got))
    return placed
