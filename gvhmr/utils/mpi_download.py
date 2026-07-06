"""Credentialed download of the registration-gated SMPL / SMPL-X body models from the official source.

The Max Planck Institute licenses SMPL/SMPL-X for non-commercial research and **prohibits
redistribution**, so we can't mirror the model files. Instead this module fetches them straight from the
MPI download portal using the **end-user's own account** — exactly what the official ``smplx`` /
SMPLify-X install scripts do — so each user accepts the license themselves and nothing is re-hosted.

Credentials come from ``$SMPLX_USER`` / ``$SMPLX_PW`` or a 0600 file written by ``gvhmr auth smpl``.
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


def credentials() -> tuple[str, str] | None:
    """Resolve MPI credentials: ``$SMPLX_USER``/``$SMPLX_PW`` first, else the ``gvhmr auth`` file, else None."""
    user = os.environ.get("SMPLX_USER") or os.environ.get("MPI_USER")
    pw = os.environ.get("SMPLX_PW") or os.environ.get("SMPLX_PASSWORD") or os.environ.get("MPI_PW")
    if user and pw:
        return user, pw
    if CRED_PATH.is_file():
        try:
            with open(CRED_PATH, "rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            return None
        auth = data.get("auth", {})
        if auth.get("username") and auth.get("password"):
            return str(auth["username"]), str(auth["password"])
    return None


def save_credentials(username: str, password: str) -> Path:
    """Persist MPI credentials to a 0600 file for later auto-fetch. Returns the path written."""
    CRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    # TOML literal strings avoid escaping; MPI passwords are the user's own and this file is chmod 600.
    CRED_PATH.write_text(f"[auth]\nusername = '{username}'\npassword = '{password}'\n")
    CRED_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — owner read/write only
    return CRED_PATH


def _download_archive(domain: str, sfile: str, dest: Path, user: str, pw: str) -> None:
    """POST credentials to the portal and stream the archive to ``dest`` (raises on HTTP / auth failure)."""
    import requests

    from gvhmr.utils.net import ensure_ca_bundle

    ensure_ca_bundle()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        url = f"{PORTAL}?domain={domain}&sfile={sfile}"
        r = s.post(url, data={"username": user, "password": pw}, stream=True, allow_redirects=True, timeout=60)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:  # the portal returns its login page (200) on bad credentials
            raise PermissionError("MPI login failed — check your SMPL-X account credentials (`gvhmr auth smpl`).")
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


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
    credentials_pair: tuple[str, str] | None = None,
) -> list[Path]:
    """Download + place the requested body models under ``body_model_root``. Returns the files placed.

    ``smplx`` fetches ``smplx/SMPLX_NEUTRAL.npz`` (needed for motion recovery); ``smpl`` fetches
    ``smpl/SMPL_NEUTRAL.pkl`` (needed for mesh rendering). ``genders=True`` also grabs the MALE/FEMALE
    variants (for gendered eval). Raises if credentials are missing or the download/extract fails.
    """
    import tempfile

    creds = credentials_pair or credentials()
    if creds is None:
        raise PermissionError(
            "No MPI credentials found. Set $SMPLX_USER / $SMPLX_PW or run `gvhmr auth smpl` "
            "(register once at https://smpl-x.is.tue.mpg.de/ and https://smpl.is.tue.mpg.de/)."
        )
    user, pw = creds
    body_model_root = Path(body_model_root)
    placed: list[Path] = []
    todo = [k for k, on in (("smplx", smplx), ("smpl", smpl)) if on]
    with tempfile.TemporaryDirectory() as td:
        for key in todo:
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
