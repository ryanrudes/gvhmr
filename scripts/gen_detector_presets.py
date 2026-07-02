#!/usr/bin/env python
"""Generate the ultralytics-YOLO detector presets under ``gvhmr/configs/detector/``.

Exhaustive presets — one per YOLO family × size — so you can pick any version by name:
``gvhmr demo VIDEO --detector yolo26x`` or ``detector = 'yolo26x'`` in the config file. ultralytics
auto-downloads the weight on first use.

When a new family ships (e.g. YOLO27), add one line to ``FAMILIES`` and re-run this script. Any weight
also works *without* a preset via ``--detector-ckpt <name>.pt``, so you're never blocked on a preset.

    python scripts/gen_detector_presets.py
"""

from __future__ import annotations

from pathlib import Path

# family -> its ultralytics size letters (weight name = family + size, e.g. "yolo26" + "x" -> yolo26x.pt).
# `yolo.yaml` (the released default = yolov8x) is kept by hand and NOT generated here.
FAMILIES: dict[str, list[str]] = {
    "yolov8": ["n", "s", "m", "l", "x"],
    "yolov9": ["t", "s", "m", "c", "e"],
    "yolov10": ["n", "s", "m", "b", "l", "x"],
    "yolo11": ["n", "s", "m", "l", "x"],
    "yolo12": ["n", "s", "m", "l", "x"],
    "yolo26": ["n", "s", "m", "l", "x"],  # latest (Jan 2026), NMS-free
}

OUT = Path(__file__).resolve().parents[1] / "gvhmr" / "configs" / "detector"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for family, sizes in FAMILIES.items():
        for size in sizes:
            name = f"{family}{size}"
            (OUT / f"{name}.yaml").write_text(
                f"# {name} — ultralytics YOLO detector (auto-downloads {name}.pt). Select: --detector {name}\n"
                f"name: yolo\n"
                f"ckpt: {name}.pt\n"
                f"conf: 0.5\n"
            )
            written += 1
    print(f"wrote {written} detector presets to {OUT}")


if __name__ == "__main__":
    main()
