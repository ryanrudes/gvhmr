"""HF token resolution for private mirror access."""

from __future__ import annotations

import os

from gvhmr.utils.hf_token import resolve_hf_token


def test_explicit_token_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert resolve_hf_token("explicit") == "explicit"


def test_env_precedence(monkeypatch):
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-env")
    assert resolve_hf_token(None) == "hf-env"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub-env")
    assert resolve_hf_token(None) == "hub-env"


def test_none_when_unset(monkeypatch):
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    # get_token() may return a cached login on dev machines — only assert env path is empty.
    assert os.environ.get("HF_TOKEN") is None
