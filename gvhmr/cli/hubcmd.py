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
    smplx_username: Annotated[
        str | None, typer.Option("--smplx-username", help="Email for your SMPL-X account (smpl-x.is.tue.mpg.de).")
    ] = None,
    smplx_password: Annotated[str | None, typer.Option("--smplx-password", help="SMPL-X account password.")] = None,
    smpl_username: Annotated[
        str | None, typer.Option("--smpl-username", help="Email for your SMPL account (smpl.is.tue.mpg.de).")
    ] = None,
    smpl_password: Annotated[str | None, typer.Option("--smpl-password", help="SMPL account password.")] = None,
    fetch: Annotated[bool, typer.Option("--fetch/--no-fetch", help="Download the body models now to verify.")] = True,
) -> None:
    """Save your [bold]Max Planck Institute[/] logins so GVHMR can auto-fetch the gated body models.

    [bold]SMPL and SMPL-X are two separate registrations[/] (each its own login + license), so this stores
    them independently — never assuming one account works for both. SMPL-X powers motion recovery; SMPL is
    only needed for mesh-overlay rendering. Register at https://smpl-x.is.tue.mpg.de/ and (for rendering)
    https://smpl.is.tue.mpg.de/. Credentials are stored 0600, locally.
    """
    from gvhmr.utils import assets, mpi_download
    from gvhmr.utils.console import console, rule

    rule("[gvhmr]gvhmr auth smpl[/]")
    console.print("[dim]SMPL and SMPL-X are separate MPI accounts — configure each with its own login.[/]")

    saved: list[str] = []
    # SMPL-X (smpl-x.is.tue.mpg.de) — required for motion recovery.
    if (
        smplx_username
        or smplx_password
        or typer.confirm("Configure the SMPL-X account (motion recovery)?", default=True)
    ):
        smplx_username = smplx_username or typer.prompt("  SMPL-X email [smpl-x.is.tue.mpg.de]")
        smplx_password = smplx_password or typer.prompt("  SMPL-X password", hide_input=True)
        mpi_download.save_credentials(smplx_username, smplx_password, dataset="smplx")
        saved.append("smplx")

    # SMPL (smpl.is.tue.mpg.de) — only needed for mesh-overlay rendering.
    if smpl_username or smpl_password or typer.confirm("Configure the SMPL account (mesh rendering)?", default=True):
        smpl_username = smpl_username or typer.prompt("  SMPL email [smpl.is.tue.mpg.de]")
        smpl_password = smpl_password or typer.prompt("  SMPL password", hide_input=True)
        mpi_download.save_credentials(smpl_username, smpl_password, dataset="smpl")
        saved.append("smpl")

    if not saved:
        console.print("[warn]No credentials entered.[/] Nothing saved.")
        return
    console.print(
        f"credentials saved [ok]✓[/] → [muted]{mpi_download.CRED_PATH}[/] "
        f"[dim](0600; for {', '.join(saved)}; also honors $SMPLX_USER/$SMPLX_PW, $SMPL_USER/$SMPL_PW)[/]"
    )

    if not fetch:
        console.print("[dim]Body models will be fetched automatically on first use.[/]")
        return

    # Fetch each configured dataset independently — one account failing must not mask the other's success.
    for ds in saved:
        try:
            with console.status(f"Fetching {ds.upper()} body model from MPI…"):
                placed = mpi_download.fetch_body_models(
                    assets.BODY_MODEL_ROOT, smplx=(ds == "smplx"), smpl=(ds == "smpl")
                )
            console.print(f"[ok]{ds.upper()} fetched[/] ({len(placed)} files) → [muted]{assets.BODY_MODEL_ROOT}[/]")
        except Exception as e:  # noqa: BLE001 — surface any failure with the manual fallback
            console.print(
                f"[warn]auto-fetch of {ds.upper()} failed[/] ([muted]{type(e).__name__}: {e}[/]).\n"
                f"  Credentials are saved; GVHMR will retry on first use. If it keeps failing, download "
                f"manually — `gvhmr download` prints the layout."
            )


@auth_app.command("wandb")
def auth_wandb(
    key: Annotated[
        str | None,
        typer.Option("--key", help="W&B API key (https://wandb.ai/authorize). Prompted for if omitted."),
    ] = None,
) -> None:
    """Save your Weights & Biases API key to ``~/.netrc`` (0600) so training just logs, everywhere.

    This is the right way to hold the key on a **cluster**: ``$HOME`` is shared, so every compute node
    reads it — the key never goes in an environment variable, a command line, or the Slurm job record
    (``sbatch --export`` writes those into the job record, where ``scontrol show job -d`` and the
    accounting DB can surface them). Same file and format ``wandb login`` writes, so either works.
    """
    import os
    import stat
    import tempfile

    from gvhmr.utils.console import console

    key = key or typer.prompt("W&B API key (https://wandb.ai/authorize)", hide_input=True)
    key = key.strip()
    if not key:
        console.print("[warn]No key entered.[/] Nothing saved.")
        raise typer.Exit(1)

    netrc = Path.home() / ".netrc"
    machine = "api.wandb.ai"

    # Drop any existing api.wandb.ai stanza, keep every other machine's entry intact.
    kept: list[str] = []
    if netrc.exists():
        skipping = False
        for line in netrc.read_text().splitlines():
            head = line.strip().split()
            if head and head[0] == "machine":
                skipping = len(head) > 1 and head[1] == machine
            if not skipping:
                kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()

    body = "\n".join([*kept, f"machine {machine}", "  login user", f"  password {key}", ""])

    # Write via a 0600 temp file in the same dir, then rename: the key is never briefly world-readable.
    fd, tmp = tempfile.mkstemp(dir=str(netrc.parent), prefix=".netrc.")
    try:
        os.write(fd, body.encode())
        os.close(fd)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        os.replace(tmp, netrc)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    console.print(
        f"W&B key saved [ok]✓[/] → [muted]{netrc}[/] [dim](0600, machine {machine})[/]\n"
        "[dim]On a cluster $HOME is shared, so every compute node picks it up — no env var needed.[/]"
    )


body_models_app = typer.Typer(
    name="body-models",
    help="Manage a private HF mirror of [bold]your own[/] SMPL/SMPL-X body models (fast, seamless setup).",
    no_args_is_help=True,
)


@body_models_app.command("push")
def body_models_push(
    repo: Annotated[str, typer.Argument(help="Target HF repo id you control [dim](e.g. you/gvhmr-body-models)[/].")],
    private: Annotated[
        bool, typer.Option("--private/--public", help="Create the repo private [dim](strongly recommended)[/].")
    ] = True,
    genders: Annotated[bool, typer.Option("--genders", help="Also upload the MALE/FEMALE variants.")] = False,
    set_default: Annotated[
        bool, typer.Option("--set-default/--no-set-default", help="Record this repo as your body-models mirror.")
    ] = True,
    token: Annotated[
        str | None, typer.Option("--token", help="HF token [dim](else cached login / $HF_TOKEN)[/].")
    ] = None,
) -> None:
    """Upload the body models on [bold]this machine[/] to a private HF repo [bold]you own[/], for fast reuse.

    This copies [italic]your[/] licensed copy to [italic]your own[/] private storage — it is [bold]not[/]
    redistribution, and GVHMR never hosts these files. The MPI license forbids sharing them, so keep the
    repo private and don't grant others access. Afterwards, setup on any of your machines is a single fast
    pull ([gvhmr]gvhmr body-models pull[/]) with no MPI login in the loop.
    """
    from huggingface_hub import HfApi

    from gvhmr.utils import assets, mpi_download
    from gvhmr.utils.console import console, rule
    from gvhmr.utils.hf_token import resolve_hf_token
    from gvhmr.utils.net import ensure_ca_bundle

    rule("[gvhmr]gvhmr body-models push[/]")
    console.print(
        "[warn]License:[/] SMPL/SMPL-X are MPI-licensed and [bold]must not be redistributed[/]. Push only to a "
        "repo [bold]you[/] control and keep it [bold]private[/] — this is your own copy for your own use."
    )
    if not private and not typer.confirm(
        "A PUBLIC repo would redistribute MPI-licensed files (a license violation). Continue anyway?", default=False
    ):
        raise typer.Exit(1)

    root = assets.BODY_MODEL_ROOT
    uploads = [(root / rel, rel) for key in ("smplx", "smpl") for rel in mpi_download.dataset_files(key, genders)]
    uploads = [(p, rel) for p, rel in uploads if p.exists()]
    if not uploads:
        console.print(f"[err]No body models under {root}[/] — fetch them first (`gvhmr auth smpl`), then push.")
        raise typer.Exit(1)

    ensure_ca_bundle()
    api = HfApi(token=resolve_hf_token(token))
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    # create_repo(exist_ok=True) does NOT change the visibility of a repo that already exists — so a
    # pre-existing PUBLIC repo would silently receive the MPI-licensed files despite the --private default.
    # Verify (and, when private was requested, flip it) before uploading anything.
    if private:
        try:
            is_public = api.repo_info(repo, repo_type="model").private is False
        except Exception:  # noqa: BLE001 — if we can't confirm, treat as unsafe
            is_public = True
        if is_public:
            try:
                api.update_repo_settings(repo_id=repo, repo_type="model", private=True)
                console.print(f"[warn]made existing repo private[/] [muted]{repo}[/]")
            except Exception as e:  # noqa: BLE001
                console.print(
                    f"[err]{repo} already exists as PUBLIC and could not be made private[/] ([muted]{e}[/]).\n"
                    "Uploading SMPL/SMPL-X to a public repo would violate the MPI license — aborting. "
                    "Make the repo private in its HF settings, or push to a new private repo id."
                )
                raise typer.Exit(1) from e
    for path, rel in uploads:
        with console.status(f"Uploading {rel}…"):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=rel, repo_id=repo, repo_type="model")
        console.print(f"  [ok]✓[/] {rel}")
    console.print(
        f"[ok]pushed {len(uploads)} file(s)[/] → [muted]{repo}[/] ({'private' if private else '[err]PUBLIC[/]'})"
    )

    if set_default:
        from gvhmr.cli.config import write_settings

        write_settings(body_models={"mirror": repo})
        console.print(f"[ok]recorded[/] [muted]{repo}[/] as your body-models mirror (tried before MPI).")
    else:
        console.print(f"[dim]Set it as your mirror anytime: gvhmr config set body_models_mirror {repo}[/]")


@body_models_app.command("pull")
def body_models_pull(
    repo: Annotated[
        str | None, typer.Argument(help="Mirror repo to pull from [dim](default: the configured one)[/].")
    ] = None,
    genders: Annotated[bool, typer.Option("--genders", help="Also pull the MALE/FEMALE variants.")] = False,
    token: Annotated[
        str | None, typer.Option("--token", help="HF token [dim](else cached login / $HF_TOKEN)[/].")
    ] = None,
) -> None:
    """Fetch the body models from your private HF mirror into the configured body-models root."""
    from gvhmr.utils import assets, localconfig, mpi_download
    from gvhmr.utils.console import console, rule

    rule("[gvhmr]gvhmr body-models pull[/]")
    repo = repo or localconfig.body_models_mirror()
    if not repo:
        console.print(
            "[err]No mirror configured[/] — pass a repo, or set one with `gvhmr body-models push <repo>` / "
            "`gvhmr config set body_models_mirror <repo>`."
        )
        raise typer.Exit(1)
    try:
        placed = mpi_download.fetch_from_mirror(
            assets.BODY_MODEL_ROOT, repo, smplx=True, smpl=True, genders=genders, token=token
        )
    except PermissionError as e:
        console.print(f"[err]{e}[/]")
        raise typer.Exit(1) from e
    if not placed:
        console.print(f"[warn]Nothing pulled[/] — files already present, or [muted]{repo}[/] has none of them.")
        return
    console.print(f"[ok]pulled {len(placed)} file(s)[/] from [muted]{repo}[/] → [muted]{assets.BODY_MODEL_ROOT}[/]")


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
    # SSR-off fallback. The authoritative fix is app.py's launch(ssr_mode=False) (its launch() runs on
    # HF); this env var is belt-and-suspenders. Both stop the browser uploading inputs to the HF Xet CDN,
    # which gradio then can't re-fetch server-side (403). The file push above auto-triggers a rebuild +
    # restart, so the new app + var take effect without an explicit restart_space (which needs a broader
    # token scope than the cached login usually has).
    try:
        from huggingface_hub import HfApi

        from gvhmr.utils.hf_token import resolve_hf_token

        HfApi(token=resolve_hf_token(token)).add_space_variable(space_id, "GRADIO_SSR_MODE", "false")
        console.print("[ok]Space variable[/] GRADIO_SSR_MODE=false (fallback; app forces ssr_mode=False)")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[warn]Could not set GRADIO_SSR_MODE[/]: {exc}")
    console.print(f"[ok]Space published[/] → [muted]{url}[/] [dim](auto-rebuilding)[/]")
