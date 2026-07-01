"""``gvhmr download`` — fetch model checkpoints (and optional data packs) into the right place.

Reads the manifest in :mod:`gvhmr.utils.assets` and pulls each file from the HuggingFace mirror into
``CHECKPOINT_ROOT`` with the exact layout the code expects. Skips what's already present; the gated body
models can't be auto-fetched, so it prints the sign-up + the precise target path.
"""

from __future__ import annotations

from gvhmr.utils import assets
from gvhmr.utils.console import console, rule
from gvhmr.utils.pylogger import Log


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n / 1024:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _body_models_present() -> bool:
    return (assets.BODY_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz").exists()


def _report_gated() -> None:
    if _body_models_present():
        console.print("body models        [ok]✓[/] present")
        return
    desc, target, urls = assets.GATED["body_models"]
    console.print(
        f"[warn]body models[/]        registration-gated — can't auto-download.\n"
        f"  Sign up at {' , '.join(urls)}\n"
        f"  then place: [muted]{desc}[/]\n"
        f"  under:      [muted]{target}[/]  (override with $GVHMR_BODY_MODELS)"
    )


def run(what: str = "demo", *, force: bool = False, data: str | None = None) -> None:
    rule("[gvhmr]gvhmr download[/]")
    console.print(f"checkpoint root: [muted]{assets.CHECKPOINT_ROOT}[/]  [dim](override: $GVHMR_CHECKPOINTS)[/]")

    if what in ("demo", "slam", "all"):
        sel = assets.select(group=None if what == "all" else what)
    else:
        sel = assets.select(names=[s.strip() for s in what.split(",") if s.strip()])

    todo = {n: a for n, a in sel.items() if force or not assets.is_present(a)}
    have = [n for n in sel if n not in todo]
    if have:
        console.print(f"already present [ok]✓[/]: {', '.join(have)}")
    if todo:
        console.print(f"fetching {len(todo)} ([ok]{_fmt(sum(a.size for a in todo.values()))}[/]): {', '.join(todo)}")
        assets.fetch(todo, force=force)
        Log.info("[ok]checkpoints fetched[/]")
    elif not have:
        Log.info("nothing to fetch for this selection")

    _report_gated()

    if data:
        rule("data packs")
        console.print(f"data root: [muted]{assets.DATA_ROOT}[/]  [dim](override: $GVHMR_DATA_ROOT)[/]")
        for name in [s.strip() for s in data.split(",") if s.strip()]:
            if name not in assets.DATA_PACKS:
                Log.warning(f"[warn]unknown data pack[/] {name!r}; known: {sorted(assets.DATA_PACKS)}")
                continue
            _, _, size = assets.DATA_PACKS[name]
            with console.status(f"fetching {name} pack ([ok]{_fmt(size)}[/]) + extracting…"):
                target = assets.fetch_data_pack(name, force=force)
            Log.info(f"[ok]{name}[/] → [muted]{target}[/]")

    console.print("\n[dim]Run [gvhmr]gvhmr info[/] to verify. Demo needs: gvhmr, hmr2, vitpose, yolo + body models.[/]")
