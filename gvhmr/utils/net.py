"""Networking helpers — make model/asset downloads robust to misconfigured TLS environments."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_ca_bundle() -> str | None:
    """Point OpenSSL at a valid CA bundle when the environment's TLS config is broken.

    Many HPC/cluster login shells export ``SSL_CERT_DIR`` (or ``SSL_CERT_FILE``) to a path of
    the *wrong kind* — e.g. ``SSL_CERT_DIR`` set to a bundle **file** instead of a hash
    directory. OpenSSL's issuer lookup then fails and ``httpx`` / ``huggingface_hub`` /
    ``ultralytics`` die with ``CERTIFICATE_VERIFY_FAILED: unable to get local issuer
    certificate`` even though the network is perfectly fine (``urllib`` may still work, which
    makes it baffling to debug). ``gvhmr download`` / ``demo`` hit exactly this on such boxes.

    When we detect a broken cert var, fall back to :mod:`certifi`'s bundle via ``SSL_CERT_FILE``
    (and drop a non-directory ``SSL_CERT_DIR`` so OpenSSL doesn't choke on it). A **valid**
    ``SSL_CERT_FILE`` is always respected — corporate/MITM roots keep working — and a sane
    environment (cert vars unset or correct) is left completely untouched.

    Returns the bundle path we set, or ``None`` when no change was needed. Idempotent and cheap
    (only imports :mod:`certifi` on the broken-env branch), so it's safe to call before any
    command.
    """
    cert_file = os.environ.get("SSL_CERT_FILE")
    if cert_file and Path(cert_file).is_file():
        return None  # an explicit, valid bundle — respect it (may carry corporate/MITM roots)

    cert_dir = os.environ.get("SSL_CERT_DIR")
    file_broken = cert_file is not None and not Path(cert_file).is_file()
    dir_broken = cert_dir is not None and not Path(cert_dir).is_dir()
    if not (file_broken or dir_broken):
        return None  # nothing obviously broken — leave the environment's defaults alone

    import certifi

    bundle = certifi.where()
    os.environ["SSL_CERT_FILE"] = bundle
    if dir_broken:
        os.environ.pop("SSL_CERT_DIR", None)  # a non-directory here is never usable by OpenSSL
    return bundle
