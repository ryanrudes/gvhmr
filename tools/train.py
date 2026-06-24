"""Backward-compat entry point for training/testing. Prefer [gvhmr train ...].

python tools/train.py exp=gvhmr/mixed/mixed
# equivalently:
gvhmr train exp=gvhmr/mixed/mixed
"""

import sys

from gvhmr.cli.train import run

if __name__ == "__main__":
    run(sys.argv[1:])
