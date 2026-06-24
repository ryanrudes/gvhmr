"""Backward-compat entry point for the folder demo. Prefer [gvhmr demo-folder FOLDER].

python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
# equivalently:
gvhmr demo-folder inputs/demo/folder_in -o outputs/demo/folder_out -s
"""

import argparse
from pathlib import Path

from gvhmr.cli.demo_folder import run


def main() -> None:
    p = argparse.ArgumentParser(description="GVHMR folder demo (legacy entry; prefer `gvhmr demo-folder`).")
    p.add_argument("-f", "--folder", type=str, required=True)
    p.add_argument("-d", "--output_root", type=str, default=None)
    p.add_argument("-s", "--static_cam", action="store_true", help="Cameras are static.")
    a = p.parse_args()
    run(Path(a.folder), output_root=a.output_root, static_cam=a.static_cam)


if __name__ == "__main__":
    main()
