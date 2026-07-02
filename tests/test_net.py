"""``gvhmr.utils.net.ensure_ca_bundle`` — repair broken TLS cert env, respect sane ones.

Regression net for the HPC failure mode: a login shell exports ``SSL_CERT_DIR`` pointing at a
*file* (not a hash dir), so OpenSSL issuer lookup fails and every HF/ultralytics download dies
with ``CERTIFICATE_VERIFY_FAILED``. The fix must kick in there, and stay out of the way otherwise.
"""

from __future__ import annotations

import os

import certifi
import pytest

from gvhmr.utils.net import ensure_ca_bundle


def test_broken_ssl_cert_dir_falls_back_to_certifi(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """SSL_CERT_DIR pointing at a file (the HPC bug) → set SSL_CERT_FILE=certifi, drop the bad dir."""
    bogus = tmp_path / "tls-ca-bundle.pem"
    bogus.write_text("")  # a real *file* where a directory is expected
    monkeypatch.setenv("SSL_CERT_DIR", str(bogus))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    result = ensure_ca_bundle()

    assert result == certifi.where()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    assert "SSL_CERT_DIR" not in os.environ  # a non-directory is never usable → removed


def test_broken_ssl_cert_file_falls_back_to_certifi(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """SSL_CERT_FILE pointing at a nonexistent path → replaced with certifi's bundle."""
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "does-not-exist.pem"))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    result = ensure_ca_bundle()

    assert result == certifi.where()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()


def test_valid_ssl_cert_file_is_respected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """An explicit, valid SSL_CERT_FILE (e.g. corporate roots) is left untouched."""
    bundle = tmp_path / "corp-roots.pem"
    bundle.write_text("")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "whatever.pem"))  # even a bad dir alongside

    result = ensure_ca_bundle()

    assert result is None
    assert os.environ["SSL_CERT_FILE"] == str(bundle)  # unchanged


def test_valid_ssl_cert_dir_is_respected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A real hash directory in SSL_CERT_DIR is sane → no-op, and we don't inject SSL_CERT_FILE."""
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))  # tmp_path is a genuine directory
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    result = ensure_ca_bundle()

    assert result is None
    assert os.environ["SSL_CERT_DIR"] == str(tmp_path)
    assert "SSL_CERT_FILE" not in os.environ


def test_clean_env_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither var set (a normal machine) → no change, no certifi injection."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    result = ensure_ca_bundle()

    assert result is None
    assert "SSL_CERT_FILE" not in os.environ
