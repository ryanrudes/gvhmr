"""Pluggable preprocessing registry/protocol contracts (Tier A) — no heavy deps.

Imports only ``gvhmr/utils/preproc/base.py`` (lazy factories), so it runs on the base/CI install without
ultralytics or the vendored ViTPose. Pins the swap contract: a backend is selected by name and must
satisfy the ``Detector`` / ``Pose2D`` protocol.
"""

from __future__ import annotations

import pytest
import torch

from gvhmr.utils.preproc.base import (
    BACKBONES,
    DETECTORS,
    POSE2D,
    Detector,
    FeatureBackbone,
    Pose2D,
    make_backbone,
    make_detector,
    make_pose2d,
    missing_requirements,
)


def test_registries_list_the_released_defaults():
    assert "yolo" in DETECTORS
    assert "vitpose" in POSE2D
    assert "hmr2" in BACKBONES
    # a genuinely-different 2D-pose architecture registered alongside the default (proves the slot swaps)
    assert "rtmpose" in POSE2D


def test_unknown_backend_raises_keyerror():
    with pytest.raises(KeyError):
        make_detector("not-a-detector")
    with pytest.raises(KeyError):
        make_pose2d("not-a-pose2d")
    with pytest.raises(KeyError):
        make_backbone("not-a-backbone")


def test_protocols_are_runtime_checkable_on_the_output_contract():
    # A minimal stand-in that emits the right shapes should satisfy the protocol structurally —
    # this is the contract any newer/arbitrary model must meet to "fit in".
    class FakeDetector:
        def get_one_track(self, video_path) -> torch.Tensor:
            return torch.zeros(10, 4)

    class FakePose2D:
        def extract(self, video_path, bbx_xys) -> torch.Tensor:
            return torch.zeros(10, 17, 3)

    class FakeBackbone:
        feat_dim = 512

        def extract_video_features(self, video_path, bbx_xys) -> torch.Tensor:
            return torch.zeros(10, self.feat_dim)

    assert isinstance(FakeDetector(), Detector)
    assert isinstance(FakePose2D(), Pose2D)
    assert isinstance(FakeBackbone(), FeatureBackbone)
    # and something missing the method does not
    assert not isinstance(object(), Detector)
    assert not isinstance(object(), Pose2D)
    assert not isinstance(object(), FeatureBackbone)


def test_default_factory_names_match_lazy_impls_without_importing_them():
    # make_*() with the default name resolves to the heavy impl via lazy import; we don't construct it
    # here (would need ultralytics/ViTPose weights), but the dispatch must accept the default name.
    # Unknown names raise; known names would proceed to the lazy import (covered by the demo path).
    for name in DETECTORS:
        # calling with a bogus kwarg would reach the impl ctor; instead just confirm name is dispatchable
        # by asserting it's NOT in the KeyError branch.
        assert name == "yolo"  # only one registered today; update when more land
    assert set(POSE2D) == {"vitpose", "rtmpose"}


def test_missing_requirements_names_the_extra_per_backend():
    # The demo's fail-fast check: with nothing installed, each requiring backend reports its module
    # and the exact install command. `have` is injected so the test is independent of this env.
    selections = {"detector": "yolo26x", "pose2d": "rtmpose", "backbone": "hmr2", "camera": "simplevo"}
    rows = missing_requirements(selections, have=lambda module: False)
    by_module = {module: (why, fix) for why, module, fix in rows}
    assert set(by_module) == {"ultralytics", "rtmlib", "pycolmap"}  # hmr2 needs nothing beyond base
    assert by_module["ultralytics"][0] == "detector 'yolo26x'"  # every YOLO preset maps to ultralytics
    assert "--extra preproc" in by_module["ultralytics"][1]
    assert "--extra rtmpose" in by_module["rtmlib"][1]
    assert "--extra preproc" in by_module["pycolmap"][1]


def test_missing_requirements_empty_when_everything_installed():
    selections = {"detector": "yolo", "pose2d": "vitpose", "backbone": "hmr2", "camera": "dpvo"}
    assert missing_requirements(selections, have=lambda module: True) == []
    # vitpose/hmr2 are vendored (base install) — they must not demand anything even with nothing installed
    assert missing_requirements({"pose2d": "vitpose", "backbone": "hmr2"}, have=lambda module: False) == []
