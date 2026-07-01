"""``gvhmr demo-folder`` — run the demo over every video in a folder (in-process)."""

from __future__ import annotations

from pathlib import Path

from gvhmr.utils.console import rule
from gvhmr.utils.pylogger import Log


def run(
    folder: Path,
    *,
    output_root: str | None = None,
    static_cam: bool = False,
    camera: str | None = None,
    use_dpvo: bool = False,
    f_mm: int | None = None,
    render_scale: float | None = None,
    no_render: bool = False,
) -> None:
    from gvhmr.cli.demo import run as run_demo

    folder = Path(folder)
    videos = sorted([*folder.glob("*.mp4"), *folder.glob("*.MP4")])
    Log.info(f"Found [gvhmr]{len(videos)}[/] videos in [muted]{folder}[/]")
    for k, video in enumerate(videos, 1):
        rule(f"[{k}/{len(videos)}] {video.name}")
        run_demo(
            video,
            output_root=output_root,
            static_cam=static_cam,
            camera=camera,
            use_dpvo=use_dpvo,
            f_mm=f_mm,
            render_scale=render_scale,
            no_render=no_render,
        )
