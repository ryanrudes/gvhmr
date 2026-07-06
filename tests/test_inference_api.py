"""Hermetic tests for the HuggingFace-style inference API (``gvhmr.inference``).

Everything here runs on CPU with **no** checkpoints, body models, or network — it pins the pure logic
around the model: the :class:`MotionResult` container (shapes, serialization, the render guard), the
credentialed body-model fetch's credential handling + archive extraction, task validation, and the
generated model card. The heavy end-to-end path is covered by ``test_golden_inference.py``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch


def _synthetic_result(**overrides):
    from gvhmr.inference.result import MotionResult

    length = 6
    world = {
        "global_orient": torch.zeros(length, 3),
        "body_pose": torch.zeros(length, 63),
        "betas": torch.zeros(length, 10),
        "transl": torch.randn(length, 3),
    }
    camera = {k: v.clone() for k, v in world.items()}
    k_full = torch.eye(3).repeat(length, 1, 1)
    kw = dict(
        smpl_params_world=world,
        smpl_params_camera=camera,
        intrinsics=k_full,
        camera="simplevo",
        raw={"smpl_params_global": world, "smpl_params_incam": camera, "K_fullimg": k_full},
    )
    kw.update(overrides)
    return MotionResult(**kw)


# ---- MotionResult ---------------------------------------------------------------------------------
def test_motionresult_basics():
    r = _synthetic_result()
    assert r.num_frames == 6
    assert len(r) == 6
    assert "MotionResult(" in repr(r)
    d = r.to_dict()
    assert set(d) == {"smpl_params_world", "smpl_params_camera", "intrinsics", "fps", "camera", "num_frames"}
    assert d["fps"] == 30.0


def test_motionresult_save_pt_roundtrips_to_hmr4d_format(tmp_path):
    r = _synthetic_result()
    p = r.save(tmp_path / "m.pt")
    loaded = torch.load(p, weights_only=False)
    assert set(loaded) == {"smpl_params_global", "smpl_params_incam", "K_fullimg"}
    assert torch.equal(loaded["smpl_params_global"]["transl"], r.smpl_params_world["transl"])


def test_motionresult_save_without_raw_uses_fields(tmp_path):
    r = _synthetic_result(raw={})
    loaded = torch.load(r.save(tmp_path / "m.pt"), weights_only=False)
    assert set(loaded) == {"smpl_params_global", "smpl_params_incam", "K_fullimg"}


def test_motionresult_save_npz(tmp_path):
    r = _synthetic_result()
    data = np.load(r.save_npz(tmp_path / "m.npz"))
    for frame in ("world", "camera"):
        for key in ("global_orient", "body_pose", "betas", "transl"):
            assert f"{frame}_{key}" in data
    assert data["intrinsics"].shape == (6, 3, 3)
    assert str(data["camera"]) == "simplevo"
    assert float(data["fps"]) == 30.0


def test_motionresult_render_without_pipeline_raises():
    r = _synthetic_result()  # no _cfg → cannot render
    with pytest.raises(RuntimeError, match="artifacts"):
        r.render()


# ---- credentials / MPI fetch ----------------------------------------------------------------------
def _clear_mpi_env(monkeypatch):
    for var in (
        "SMPLX_USER",
        "SMPLX_PW",
        "SMPLX_PASSWORD",
        "SMPL_USER",
        "SMPL_PW",
        "SMPL_PASSWORD",
        "MPI_USER",
        "MPI_PW",
    ):
        monkeypatch.delenv(var, raising=False)


def test_credentials_none_when_absent(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "nope.toml")
    assert mpi_download.credentials("smplx") is None
    assert mpi_download.credentials("smpl") is None


def test_credentials_from_env_are_per_dataset(tmp_path, monkeypatch):
    """SMPL and SMPL-X are separate accounts — one set of env vars must not satisfy the other."""
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "nope.toml")
    monkeypatch.setenv("SMPLX_USER", "u@example.com")
    monkeypatch.setenv("SMPLX_PW", "hunter2")
    assert mpi_download.credentials("smplx") == ("u@example.com", "hunter2")
    assert mpi_download.credentials() == ("u@example.com", "hunter2")  # default dataset is smplx
    assert mpi_download.credentials("smpl") is None  # no cross-account fallback

    monkeypatch.setenv("SMPL_USER", "other@example.com")
    monkeypatch.setenv("SMPL_PW", "s3cret")
    assert mpi_download.credentials("smpl") == ("other@example.com", "s3cret")
    assert mpi_download.credentials("smplx") == ("u@example.com", "hunter2")  # unchanged


def test_saved_credentials_are_independent_per_account(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "cred.toml")

    path = mpi_download.save_credentials("x@a.com", "smplx-pw", dataset="smplx")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    # Saving the SMPL account must not clobber the SMPL-X one (separate accounts, same file).
    mpi_download.save_credentials("y@b.com", "smpl-pw", dataset="smpl")
    assert mpi_download.credentials("smplx") == ("x@a.com", "smplx-pw")
    assert mpi_download.credentials("smpl") == ("y@b.com", "smpl-pw")


def test_password_with_quotes_roundtrips(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "cred.toml")
    tricky = "p\"a\\ss'w"
    mpi_download.save_credentials("me@x.com", tricky, dataset="smplx")
    assert mpi_download.credentials("smplx") == ("me@x.com", tricky)


def test_legacy_flat_auth_file_migrates_to_smplx_only(tmp_path, monkeypatch):
    """A pre-split flat [auth] file resolves as SMPL-X (its only historical use), never as SMPL."""
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    cred = tmp_path / "cred.toml"
    cred.write_text("[auth]\nusername = 'legacy@x.com'\npassword = 'old'\n")
    monkeypatch.setattr(mpi_download, "CRED_PATH", cred)
    assert mpi_download.credentials("smplx") == ("legacy@x.com", "old")
    assert mpi_download.credentials("smpl") is None


def test_fetch_body_models_without_credentials_raises(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "nope.toml")
    with pytest.raises(PermissionError, match="MPI credentials"):
        mpi_download.fetch_body_models(tmp_path / "bm")


def test_fetch_uses_per_dataset_credentials(tmp_path, monkeypatch):
    """Each dataset must download with its OWN account; a missing account for one doesn't borrow the other's."""
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "nope.toml")
    monkeypatch.setenv("SMPLX_USER", "u@example.com")
    monkeypatch.setenv("SMPLX_PW", "hunter2")

    calls = []

    def _fake_download(domain, sfile, dest, user, pw):
        calls.append((domain, user, pw))
        dest.write_bytes(b"")  # extract step is skipped via the stub below

    monkeypatch.setattr(mpi_download, "_download_archive", _fake_download)
    monkeypatch.setattr(mpi_download, "_extract_members", lambda *a, **k: [tmp_path / "smplx/SMPLX_NEUTRAL.npz"])

    # SMPL-X only: uses the smplx account.
    mpi_download.fetch_body_models(tmp_path / "bm", smplx=True, smpl=False)
    assert calls == [("smplx", "u@example.com", "hunter2")]

    # SMPL requested without an SMPL account → raises (no fallback to the SMPL-X login).
    with pytest.raises(PermissionError, match="SMPL account"):
        mpi_download.fetch_body_models(tmp_path / "bm", smplx=False, smpl=True)


def test_extract_members_maps_by_suffix(tmp_path):
    from gvhmr.utils import mpi_download

    archive = tmp_path / "models_smplx_v1_1.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("models/smplx/SMPLX_NEUTRAL.npz", b"neutral-bytes")
        zf.writestr("models/smplx/SMPLX_MALE.npz", b"male-bytes")
    root = tmp_path / "bm"
    placed = mpi_download._extract_members(archive, root, [("SMPLX_NEUTRAL.npz", "smplx/SMPLX_NEUTRAL.npz")])
    assert len(placed) == 1
    assert (root / "smplx/SMPLX_NEUTRAL.npz").read_bytes() == b"neutral-bytes"
    assert not (root / "smplx/SMPLX_MALE.npz").exists()  # not requested


def test_archives_manifest_shape():
    from gvhmr.utils import mpi_download

    assert set(mpi_download.ARCHIVES) == {"smpl", "smplx"}
    for spec in mpi_download.ARCHIVES.values():
        assert spec["domain"] and spec["sfile"] and spec["members"]


def test_dataset_files_match_archive_layout():
    from gvhmr.utils import mpi_download

    assert mpi_download.dataset_files("smplx") == ["smplx/SMPLX_NEUTRAL.npz"]
    assert mpi_download.dataset_files("smpl") == ["smpl/SMPL_NEUTRAL.pkl"]
    with_genders = mpi_download.dataset_files("smplx", genders=True)
    assert "smplx/SMPLX_MALE.npz" in with_genders and "smplx/SMPLX_FEMALE.npz" in with_genders


def test_fetch_from_mirror_downloads_missing_and_skips_present(tmp_path, monkeypatch):
    """The mirror path pulls only missing files, skips ones absent from the repo, into the canonical layout."""
    from huggingface_hub.utils import EntryNotFoundError

    from gvhmr.utils import mpi_download

    root = tmp_path / "bm"
    # SMPL-X already on disk → should be skipped; SMPL missing → should be fetched from the mirror.
    (root / "smplx").mkdir(parents=True)
    (root / "smplx/SMPLX_NEUTRAL.npz").write_bytes(b"present")

    requested = []

    def _fake_hub_download(repo, rel, *, local_dir, token=None):
        requested.append((repo, rel))
        if rel == "smpl/SMPL_NEUTRAL.pkl":
            dest = Path(local_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"pulled")
            return str(dest)
        raise EntryNotFoundError(f"{rel} not in {repo}")  # anything else isn't in this mirror

    # fetch_from_mirror does `from huggingface_hub import hf_hub_download` at call time.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hub_download)

    placed = mpi_download.fetch_from_mirror(root, "me/bm", smplx=True, smpl=True)
    assert [p.name for p in placed] == ["SMPL_NEUTRAL.pkl"]
    assert (root / "smpl/SMPL_NEUTRAL.pkl").read_bytes() == b"pulled"
    # The already-present SMPL-X file was never re-requested.
    assert ("me/bm", "smplx/SMPLX_NEUTRAL.npz") not in requested


# ---- pipeline / hub -------------------------------------------------------------------------------
def test_pipeline_rejects_unknown_task():
    import gvhmr

    with pytest.raises(ValueError, match="unknown task"):
        gvhmr.pipeline("definitely-not-a-task")


def test_pipeline_tasks_listed():
    from gvhmr.inference import pipelines

    assert "human-motion-recovery" in pipelines.TASKS


def test_toplevel_lazy_exports():
    import gvhmr

    for name in ("pipeline", "recover", "GVHMR", "GVHMRPipeline", "MotionResult"):
        assert name in gvhmr.__all__
        assert getattr(gvhmr, name) is not None


def test_model_card_mentions_repo_and_gating():
    from gvhmr.inference import hub

    card = hub.model_card("someone/gvhmr")
    assert "someone/gvhmr" in card
    assert "library_name: gvhmr" in card
    assert "not included" in card.lower() or "gated" in card.lower()  # body-model caveat present


def test_publish_checkpoints_dry_run_lists_without_network(tmp_path, monkeypatch):
    from gvhmr.inference import hub
    from gvhmr.utils import assets

    # Point the manifest at fake, existing files so the dry-run can list them (no upload, no network).
    fake = {}
    for name in ("gvhmr", "hmr2", "vitpose", "yolo"):
        p = tmp_path / assets.ASSETS[name].hf_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        fake[name] = assets.Asset(assets.ASSETS[name].hf_path, p, 1, "demo")
    monkeypatch.setattr(assets, "ASSETS", fake)
    out = hub.publish_checkpoints("someone/gvhmr", dry_run=True)
    assert "dry-run" in out and "someone/gvhmr" in out
