#!/usr/bin/env python3
"""Help reconcile changes from the original GVHMR repo (zju3dv/GVHMR) into this fork.

This fork renamed the package (``hmr4d`` -> ``gvhmr``), vendored some third-party
code, and may restructure further. That makes a plain ``git merge`` of upstream
noisy. This tool reads ``docs/upstream_sync.yaml`` and, for each file changed
upstream since the last sync, prints **where that change should land here** and
how to translate it — turning an upstream diff into an actionable, mechanical
checklist (designed to be driven by an LLM agent or a human).

Usage
-----
    # one-time: register the original repo as the `upstream` remote
    git remote add upstream https://github.com/zju3dv/GVHMR

    # show upstream changes since the last reconciled commit, mapped to local paths
    python scripts/upstream_sync.py

    # compare against a specific upstream ref, and print the per-file patches
    python scripts/upstream_sync.py --since v1.1 --show-diff

It does not modify any files; it produces a plan you (or an agent) then apply.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: `uv run python scripts/upstream_sync.py` (or `pip install pyyaml`).")

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "docs" / "upstream_sync.yaml"


@dataclass
class FileChange:
    status: str  # A/M/D/R...
    upstream_path: str
    local_path: str
    note: str


def _run(args: list[str]) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}")
    return yaml.safe_load(MANIFEST.read_text())


def map_path(upstream_path: str, manifest: dict) -> tuple[str, str]:
    """Translate an upstream path to its local path + an explanatory note."""
    # Longest-prefix match in path_map (so specific entries win over the catch-all).
    best: tuple[str, str] | None = None
    for entry in sorted(manifest.get("path_map", []), key=lambda e: -len(e["upstream"])):
        up = entry["upstream"]
        if upstream_path == up or upstream_path.startswith(up):
            local = entry["local"] + upstream_path[len(up) :] if up.endswith("/") else entry["local"]
            best = (local, entry.get("note", ""))
            break
    if best is None:
        best = (upstream_path, "no mapping — review manually and add to docs/upstream_sync.yaml")
    # Apply token renames to the path itself (e.g. nested 'hmr4d' segments already
    # handled by the prefix map, but defensive for odd paths).
    local, note = best
    for ren in manifest.get("symbol_renames", []):
        local = re.sub(ren["pattern"], ren["replacement"], local)
    return local, note


def ensure_upstream(remote: str, url: str) -> None:
    remotes = _run(["git", "remote"]).split()
    if remote not in remotes:
        sys.exit(
            f"No '{remote}' remote. Add it once with:\n    git remote add {remote} {url}\nthen re-run this script."
        )
    subprocess.run(["git", "fetch", remote, "--quiet"], cwd=REPO_ROOT, check=True)


def main() -> int:
    manifest = load_manifest()
    up = manifest["upstream"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=up.get("last_synced_commit"), help="upstream ref to compare from")
    ap.add_argument("--ref", default=f"{up['remote']}/main", help="upstream ref to compare to (default upstream/main)")
    ap.add_argument("--show-diff", action="store_true", help="print each upstream file's patch")
    args = ap.parse_args()

    ensure_upstream(up["remote"], up["url"])

    if not args.since:
        sys.exit("No --since and no last_synced_commit in manifest; pass --since <upstream-ref>.")

    name_status = _run(["git", "diff", "--name-status", f"{args.since}..{args.ref}"]).strip()
    if not name_status:
        print(f"No upstream changes between {args.since} and {args.ref}. Already in sync.")
        return 0

    changes: list[FileChange] = []
    for line in name_status.splitlines():
        parts = line.split("\t")
        status, upstream_path = parts[0], parts[-1]
        local_path, note = map_path(upstream_path, manifest)
        changes.append(FileChange(status, upstream_path, local_path, note))

    print(f"# Upstream changes {args.since}..{args.ref}  ({len(changes)} files)\n")
    print(f"{'STATUS':<7} {'UPSTREAM PATH':<48} -> LOCAL PATH")
    print("-" * 100)
    for c in changes:
        exists = "  " if (REPO_ROOT / c.local_path).exists() or c.status == "A" else "??"
        print(f"{c.status:<7} {c.upstream_path:<48} -> {c.local_path} {exists}")
        if c.note:
            print(f"        ↳ {c.note}")

    print("\nLegend: '??' = expected local file not found (path may have moved — check the manifest).")
    print("Apply the import rewrites in docs/upstream_sync.yaml to any imported pytorch3d symbols.")
    print("After reconciling, bump `upstream.last_synced_commit` in docs/upstream_sync.yaml.")

    if args.show_diff:
        for c in changes:
            print(f"\n{'=' * 100}\n# {c.upstream_path}  ->  {c.local_path}\n{'=' * 100}")
            diff = _run(["git", "diff", f"{args.since}..{args.ref}", "--", c.upstream_path])
            print(diff)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
