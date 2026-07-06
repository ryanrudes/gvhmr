"""The single Rich console all GVHMR output flows through.

Everything user-facing — logging, progress bars, live status, prints — shares this one
:class:`rich.console.Console` so they coordinate (Rich suspends the live display while a
log line is written, etc.). Import ``console`` for printing, ``track``/``progress`` for
progress bars, and ``status`` for an indeterminate spinner.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from typing import TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

T = TypeVar("T")

# A shared theme so GVHMR's accent colours are consistent across logs, bars, and tables.
THEME = Theme(
    {
        "gvhmr": "bold cyan",
        "stage": "bold magenta",
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "muted": "dim",
    }
)

console = Console(theme=THEME, highlight=False)

# Optional hook for UIs (e.g. Gradio ``gr.Progress``): maps Rich ``track()`` advances to a global 0–1 fraction.
_progress_hook: ContextVar[Callable[[float, str], None] | None] = ContextVar("_progress_hook", default=None)
_progress_phase: ContextVar[tuple[float, float, str] | None] = ContextVar("_progress_phase", default=None)


def _notify_progress(local_frac: float, desc: str) -> None:
    hook = _progress_hook.get()
    if hook is None:
        return
    phase = _progress_phase.get()
    if phase is not None:
        start, end, phase_desc = phase
        frac = start + max(0.0, min(1.0, local_frac)) * (end - start)
        hook(frac, desc or phase_desc)
    else:
        hook(max(0.0, min(1.0, local_frac)), desc)


@contextlib.contextmanager
def progress_hook(callback: Callable[[float, str], None] | None):
    """While active, forward ``track()`` progress to ``callback(fraction, description)``."""
    token = _progress_hook.set(callback)
    try:
        yield
    finally:
        _progress_hook.reset(token)


@contextlib.contextmanager
def progress_phase(start: float, end: float, desc: str = ""):
    """Map nested ``track()`` progress into the global ``[start, end]`` range for the active hook."""
    token = _progress_phase.set((start, end, desc))
    try:
        if _progress_hook.get() is not None:
            _notify_progress(0.0, desc)
        yield
    finally:
        _progress_phase.reset(token)


def progress_columns() -> tuple[ProgressColumn, ...]:
    """The standard GVHMR progress layout: spinner · description · bar · count · times."""
    return (
        SpinnerColumn(style="gvhmr"),
        TextColumn("[stage]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def progress(*, transient: bool = False) -> Progress:
    """A :class:`rich.progress.Progress` bound to the shared console.

    Use directly for multi-task / nested displays::

        with progress() as p:
            t = p.add_task("Rendering", total=n)
            for ...:
                p.advance(t)
    """
    return Progress(*progress_columns(), console=console, transient=transient)


def track(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    description: str | None = None,
    leave: bool = True,
    **_: object,
) -> Iterator[T]:
    """Rich progress-tracked iteration — a drop-in for ``tqdm`` with the GVHMR columns.

    Accepts ``desc``/``description`` (tqdm or rich spelling) and ``leave`` (kept bar vs
    transient). Unknown kwargs are ignored so existing ``tqdm(...)`` call sites translate
    cleanly.
    """
    label = description or desc or "Working"
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    with progress(transient=not leave) as prog:
        task_id = prog.add_task(label, total=total)
        if _progress_hook.get() is not None:
            _notify_progress(0.0, label)
        for item in iterable:
            yield item
            prog.advance(task_id)
            if total:
                completed = prog.tasks[task_id].completed
                _notify_progress(completed / total, label)
            elif _progress_hook.get() is not None:
                _notify_progress(0.0, label)


def status(message: str):
    """An indeterminate spinner for a stage whose length is unknown (context manager)."""
    return console.status(f"[stage]{message}", spinner="dots")


def rule(title: str = "") -> None:
    """A horizontal rule (section divider)."""
    console.rule(f"[gvhmr]{title}" if title else "")


def print(*args: object, **kwargs: object) -> None:  # noqa: A001 - intentional rich print shim
    """Print through the shared Rich console (markup enabled)."""
    console.print(*args, **kwargs)
