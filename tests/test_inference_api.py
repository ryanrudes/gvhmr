"""Hermetic tests for the HuggingFace-style inference API (``gvhmr.inference``).

Everything here runs on CPU with **no** checkpoints, body models, or network — it pins the pure logic
around the model: the :class:`MotionResult` container (shapes, serialization, the render guard), the
credentialed body-model fetch's credential handling + archive extraction, task validation, and the
generated model card. The heavy end-to-end path is covered by ``test_golden_inference.py``.
"""

from __future__ import annotations

import zipfile

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
    for var in ("SMPLX_USER", "SMPLX_PW", "SMPLX_PASSWORD", "MPI_USER", "MPI_PW"):
        monkeypatch.delenv(var, raising=False)


def test_credentials_none_when_absent(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "nope.toml")
    assert mpi_download.credentials() is None


def test_credentials_from_env(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "nope.toml")
    monkeypatch.setenv("SMPLX_USER", "u@example.com")
    monkeypatch.setenv("SMPLX_PW", "hunter2")
    assert mpi_download.credentials() == ("u@example.com", "hunter2")


def test_credentials_saved_file_is_0600_and_readback(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "cred.toml")
    path = mpi_download.save_credentials("me@x.com", "s3cret")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert mpi_download.credentials() == ("me@x.com", "s3cret")


def test_fetch_body_models_without_credentials_raises(tmp_path, monkeypatch):
    from gvhmr.utils import mpi_download

    _clear_mpi_env(monkeypatch)
    monkeypatch.setattr(mpi_download, "CRED_PATH", tmp_path / "nope.toml")
    with pytest.raises(PermissionError, match="MPI credentials"):
        mpi_download.fetch_body_models(tmp_path / "bm")


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
