"""Progress hook + phased mapping for Gradio / other UIs."""

from gvhmr.utils.console import progress_hook, progress_phase, track


def test_track_forwards_to_hook_without_rich() -> None:
    seen: list[tuple[float, str]] = []

    def hook(frac: float, desc: str) -> None:
        seen.append((frac, desc))

    with progress_hook(hook):
        with progress_phase(0.2, 0.4, "Stage"):
            list(track(range(4), desc="Batch"))

    assert seen[0] == (0.2, "Stage")
    assert seen[-1] == (0.4, "Stage")
    assert any(d == "Batch" for _, d in seen[1:-1])
