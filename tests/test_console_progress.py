"""Progress hook + per-stage bars for Gradio / other UIs."""

from gvhmr.utils.console import progress_hook, progress_phase, track


def test_stage_bar_runs_0_to_1_under_stage_label() -> None:
    """Each stage is a self-contained 0→1 bar labelled by the stage, not the low-level track() desc."""
    seen: list[tuple[float, str]] = []

    def hook(frac: float, desc: str) -> None:
        seen.append((frac, desc))

    with progress_hook(hook), progress_phase("Tracking person"):
        list(track(range(4), desc="YoloV8 Tracking"))

    assert seen[0] == (0.0, "Tracking person")  # reset on enter
    assert seen[-1] == (1.0, "Tracking person")  # snapped complete on exit
    # The clean stage label always wins over the caller's low-level desc.
    assert all(desc == "Tracking person" for _, desc in seen)
    # Fractions are a real 0→1 progression that reaches 100%.
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs)
    assert max(fracs) == 1.0 and min(fracs) == 0.0


def test_stage_without_track_loop_still_completes() -> None:
    """A one-shot stage (no track() loop) still reports 0 then 1.0, so its bar reaches 100%."""
    seen: list[tuple[float, str]] = []

    with progress_hook(lambda f, d: seen.append((f, d))), progress_phase("Loading motion data"):
        pass

    assert seen == [(0.0, "Loading motion data"), (1.0, "Loading motion data")]


def test_no_hook_is_a_noop() -> None:
    # Outside a progress_hook, track() must not raise and yields every item.
    with progress_phase("Stage"):
        assert list(track(range(3), desc="x")) == [0, 1, 2]
