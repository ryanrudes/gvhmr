"""`gvhmr auth` must never block on a prompt when there is no TTY.

Regression test for a Slurm setup job that died at 00:04:54 with exit 1 — AFTER the venv build and
the data downloads had succeeded. Passing only the SMPL-X flags still fell through to a
`typer.confirm()` for the (optional, render-only) SMPL account, which aborts without a TTY. Anything
that runs in a batch job, CI, or a setup script must configure exactly what it was given and skip the
rest.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from gvhmr.cli.hubcmd import auth_app


@pytest.fixture
def no_tty(monkeypatch, tmp_path):
    """Stdin is not a TTY — a batch job. Credentials go somewhere disposable."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("gvhmr.utils.mpi_download.CRED_PATH", tmp_path / "creds.toml")
    return tmp_path


def test_smplx_only_does_not_prompt_for_smpl(no_tty):
    """The actual failure: SMPL-X supplied, SMPL not — must NOT prompt, must exit 0."""
    result = CliRunner().invoke(
        auth_app,
        ["smpl", "--smplx-username", "me@example.com", "--smplx-password", "pw", "--no-fetch"],
        input="",  # nothing to feed a prompt with — a prompt here means failure
    )
    assert result.exit_code == 0, result.output
    assert "Configure the SMPL account" not in result.output


def test_no_credentials_without_tty_fails_loudly(no_tty):
    """With no TTY and no flags there is nothing to do — fail with a message, don't hang on a prompt."""
    result = CliRunner().invoke(auth_app, ["smpl", "--no-fetch"], input="")
    assert result.exit_code != 0
    assert "No credentials supplied" in result.output


def test_password_with_shell_metacharacters_roundtrips(no_tty, monkeypatch):
    """Passwords are full of $ ! and quotes; they must survive being saved and read back."""
    from gvhmr.utils import mpi_download

    # credentials() prefers the env vars, so clear them to actually exercise the FILE round-trip
    for var in ("SMPLX_USER", "SMPLX_PW", "SMPLX_PASSWORD"):
        monkeypatch.delenv(var, raising=False)

    pw = "p@ss$w0rd!'\"x"
    result = CliRunner().invoke(
        auth_app,
        ["smpl", "--smplx-username", "me@example.com", "--smplx-password", pw, "--no-fetch"],
        input="",
    )
    assert result.exit_code == 0, result.output
    assert mpi_download.credentials(dataset="smplx") == ("me@example.com", pw)
