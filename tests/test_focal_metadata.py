"""The focal-from-metadata path, and the loudness of its fallback.

Why this file exists: `estimate_K` assumes a ~43mm-equivalent lens (it returns the image diagonal), while
phone footage is ~26mm (main) or ~14mm (ultrawide). Since `tz = 2f/(s*b)` is exactly linear in the focal,
picking the heuristic over real metadata costs ~65%-209% on in-camera DEPTH — silently, because no metric
we ship can see it (mpjpe is pelvis-aligned, pa_mpjpe Procrustes-aligned, reprojection depth-ambiguous).
That silence is the defect these tests pin.
"""

from __future__ import annotations

import pytest

from gvhmr.cli.demo import _sane_focal, _warn_focal_fallback, focal_mm_from_metadata
from gvhmr.utils.geo.hmr_cam import create_camera_sensor, estimate_K


class TestSaneFocal:
    @pytest.mark.parametrize("raw, want", [("26", 26), ("26.0", 26), (" 14 mm ", 14), (48, 48), ("13.5", 14)])
    def test_parses(self, raw, want):
        assert _sane_focal(raw) == want

    @pytest.mark.parametrize("raw", ["", "-", "undef", None, "0", "7", "401", "1e9"])
    def test_rejects_junk_and_out_of_range(self, raw):
        assert _sane_focal(raw) is None


class TestMetadataProbe:
    def test_returns_none_when_no_tools_available(self, monkeypatch, tmp_path):
        """No exiftool and no ffprobe -> None (caller falls back), never an exception."""
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert focal_mm_from_metadata(tmp_path / "nope.mp4") is None

    def test_survives_a_broken_exiftool(self, monkeypatch, tmp_path):
        """A tool that exists but explodes must not take the demo down with it."""
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/exiftool" if name == "exiftool" else None)

        def boom(*a, **k):
            raise OSError("exiftool died")

        monkeypatch.setattr("subprocess.run", boom)
        assert focal_mm_from_metadata(tmp_path / "nope.mp4") is None


class TestFallbackIsLoud:
    """The v1.6.0 bug: no focal metadata -> heuristic, with nothing said to the user."""

    def _warn(self, monkeypatch, caplog, exiftool: bool):
        monkeypatch.setattr(
            "shutil.which", lambda name: "/usr/bin/exiftool" if (exiftool and name == "exiftool") else None
        )
        with caplog.at_level("WARNING", logger="gvhmr"):
            _warn_focal_fallback("/tmp/clip.mov")
        return caplog.text

    def test_warns_and_names_the_consequence(self, monkeypatch, caplog):
        text = self._warn(monkeypatch, caplog, exiftool=True)
        assert "depth" in text.lower()  # must say WHAT is wrong, not just that something is

    def test_points_at_exiftool_when_it_is_missing(self, monkeypatch, caplog):
        assert "exiftool" in self._warn(monkeypatch, caplog, exiftool=False).lower()

    def test_suggests_the_manual_route_when_exiftool_is_present(self, monkeypatch, caplog):
        # exiftool installed but the video has no tag: installing exiftool is not the fix — flags are.
        text = self._warn(monkeypatch, caplog, exiftool=True)
        assert "--f_px" in text or "--intrinsics" in text


class TestHeuristicIsA43mmLens:
    """Pins the arithmetic behind the docs' lens table — the reason the fallback matters."""

    def test_estimate_K_equals_the_image_diagonal(self):
        w, h = 1920, 1080
        assert float(estimate_K(w, h)[0, 0]) == pytest.approx((w**2 + h**2) ** 0.5)

    @pytest.mark.parametrize("f_mm, ratio", [(14, 3.09), (26, 1.66), (48, 0.90)])
    def test_heuristic_over_true_focal_by_lens(self, f_mm, ratio):
        """tz is linear in f, so this ratio IS the in-cam depth error factor."""
        w, h = 1920, 1080
        f_true = float(create_camera_sensor(w, h, f_mm)[2][0, 0])
        assert float(estimate_K(w, h)[0, 0]) / f_true == pytest.approx(ratio, abs=0.02)

    def test_a_43mm_lens_is_the_fixed_point(self):
        """The heuristic is exactly right for ~43mm-equiv and wrong everywhere else."""
        w, h = 1920, 1080
        f_43 = float(create_camera_sensor(w, h, 43)[2][0, 0])
        assert float(estimate_K(w, h)[0, 0]) / f_43 == pytest.approx(1.0, abs=0.01)
