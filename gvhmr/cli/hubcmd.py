"""``gvhmr auth`` / ``gvhmr publish-hub`` / ``gvhmr publish-space`` — Hub credentials & publishing.

* ``gvhmr auth smpl`` captures the user's **own** MPI login so the gated SMPL/SMPL-X body models can be
  auto-fetched from the official source (never re-hosted — see :mod:`gvhmr.utils.mpi_download`).
* ``gvhmr publish-hub`` uploads the released checkpoints + a generated model card to a HuggingFace model repo.
* ``gvhmr publish-space`` pushes the bundled gradio demo to a HuggingFace Space.

Bodies lazy-import their implementations (Typer commands stay import-cheap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gvhmr import PROJ_ROOT

auth_app = typer.Typer(
    name="auth",
    help="Store credentials for auto-fetching gated assets (e.g. the SMPL/SMPL-X body models).",
    no_args_is_help=True,
)


@auth_app.command("smpl")
def auth_smpl(
    username: Annotated[str | None, typer.Option("--username", "-u", help="Your MPI account email.")] = None,
    password: Annotated[str | None, typer.Option("--password", "-p", help="Your MPI account password.")] = None,
    fetch: Annotated[bool, typer.Option("--fetch/--no-fetch", help="Download the body models now to verify.")] = True,
) -> None:
    """Save your [bold]Max Planck Institute[/] login so GVHMR can auto-fetch the gated SMPL/SMPL-X models.

    The models are licensed for non-commercial research and [bold]can't be redistributed[/], so GVHMR
    downloads them from the official source with [italic]your own[/] account (register once at
    https://smpl-x.is.tue.mpg.de/ and https://smpl.is.tue.mpg.de/). Credentials are stored 0600, locally.
    """
    from gvhmr.utils import assets, mpi_download
    from gvhmr.utils.console import console, rule

    rule("[gvhmr]gvhmr auth smpl[/]")
    if not username:
        username = typer.prompt("MPI account email")
    if not password:
        password = typer.prompt("MPI account password", hide_input=True)

    path = mpi_download.save_credentials(username, password)
    console.print(f"credentials saved [ok]✓[/] → [muted]{path}[/] [dim](0600; also honors $SMPLX_USER/$SMPLX_PW)[/]")

    if fetch:
        try:
            with console.status("Fetching SMPL-X + SMPL body models from MPI…"):
                placed = mpi_download.fetch_body_models(assets.BODY_MODEL_ROOT, smpl=True, smplx=True)
            console.print(f"[ok]body models fetched[/] ({len(placed)} files) → [muted]{assets.BODY_MODEL_ROOT}[/]")
        except Exception as e:  # noqa: BLE001 — surface any failure with the manual fallback
            console.print(
                f"[warn]auto-fetch failed[/] ([muted]{type(e).__name__}: {e}[/]).\n"
                f"  Credentials are saved; GVHMR will retry on first use. If it keeps failing, download "
                f"manually — `gvhmr download` prints the layout."
            )
    else:
        console.print("[dim]Body models will be fetched automatically on first use.[/]")


def publish_hub(
    repo_id: Annotated[
        str | None, typer.Argument(help="Target HF model repo [dim](default: ryanrudes/gvhmr)[/].")
    ] = None,
    private: Annotated[bool, typer.Option("--private", help="Create the repo as private.")] = False,
    token: Annotated[str | None, typer.Option("--token", help="HF token [dim](else your cached login)[/].")] = None,
    names: Annotated[
        str | None, typer.Option("--names", help="Comma list of checkpoints to upload [dim](default: all four)[/].")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print what would be uploaded, upload nothing.")] = False,
) -> None:
    """Upload the released checkpoints + a model card to a HuggingFace [bold]model repo[/].

    [bold]Never[/] uploads the gated body models. Fetch the checkpoints first with [gvhmr]gvhmr download[/].
    """
    from gvhmr.inference import hub
    from gvhmr.utils.console import console, rule

    rule("[gvhmr]gvhmr publish-hub[/]")
    repo_id = repo_id or hub.DEFAULT_REPO
    name_list = tuple(s.strip() for s in names.split(",")) if names else ("gvhmr", "hmr2", "vitpose", "yolo")
    try:
        result = hub.publish_checkpoints(repo_id, names=name_list, token=token, private=private, dry_run=dry_run)
    except FileNotFoundError as e:
        console.print(f"[err]{e}[/]")
        raise typer.Exit(1) from e
    console.print(result if dry_run else f"[ok]published[/] → [muted]{result}[/]")


def publish_space(
    space_id: Annotated[
        str | None, typer.Argument(help="Target HF Space id [dim](default: ryanrudes/gvhmr-demo)[/].")
    ] = None,
    space_dir: Annotated[
        Path | None, typer.Option("--space-dir", help="Local Space folder [dim](default: <repo>/space)[/].")
    ] = None,
    token: Annotated[str | None, typer.Option("--token", help="HF token [dim](else your cached login)[/].")] = None,
) -> None:
    """Push the bundled gradio demo to a HuggingFace [bold]Space[/] (upload a video → get the overlay)."""
    from gvhmr.inference import hub
    from gvhmr.utils.console import console, rule

    rule("[gvhmr]gvhmr publish-space[/]")
    space_id = space_id or "ryanrudes/gvhmr-demo"
    space_dir = space_dir or (PROJ_ROOT / "space")
    if not space_dir.is_dir():
        console.print(f"[err]Space folder not found:[/] {space_dir}")
        raise typer.Exit(1)
    url = hub.publish_space(space_id, space_dir, token=token)
    console.print(f"[ok]Space published[/] → [muted]{url}[/]")
