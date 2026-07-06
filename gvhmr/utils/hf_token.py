"""Resolve a Hugging Face API token from CLI args, env vars, or the cached login."""

from __future__ import annotations

import os


def resolve_hf_token(token: str | None = None) -> str | None:
    """Return an HF token for private-repo access, or ``None`` if unauthenticated.

    Precedence: explicit ``token`` → ``$HF_TOKEN`` → ``$HUGGING_FACE_HUB_TOKEN`` →
    ``$HUGGINGFACE_HUB_TOKEN`` → ``huggingface_hub.get_token()`` (``huggingface-cli login`` cache).
    """
    if token:
        return token
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if v := os.environ.get(key, "").strip():
            return v
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001 — optional dep path / network-less envs
        return None
