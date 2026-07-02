"""``gvhmr eval`` — the benchmark-command plumbing (no GPU/datasets needed).

Pins the pure pieces: dataset parsing, the gated body-model preflight, and — most importantly — that
the command's registries stay truthful (every task it can launch exists on disk, every data pack it
can fetch exists in the asset manifest, and the paper-reference table matches the metric keys the
callbacks actually aggregate, so the summary table can never silently misalign).
"""

from __future__ import annotations

import pytest

from gvhmr import PROJ_ROOT
from gvhmr.cli.evalcmd import (
    BODY_MODEL_FILES,
    COMBINED_TASK,
    DATASETS,
    METRIC_LABELS,
    PAPER_REFERENCE,
    missing_body_models,
    parse_datasets,
)


def test_parse_datasets():
    assert parse_datasets("all") == ["3dpw", "emdb", "rich"]
    assert parse_datasets("3dpw,rich") == ["3dpw", "rich"]
    assert parse_datasets("rich 3dpw rich") == ["rich", "3dpw"]  # order kept, deduped
    with pytest.raises(KeyError):
        parse_datasets("h36m")


def test_missing_body_models(tmp_path):
    (tmp_path / "smplx").mkdir()
    (tmp_path / "smplx/SMPLX_NEUTRAL.npz").touch()
    gap = missing_body_models(["3dpw"], tmp_path)
    assert gap == ["smpl/SMPL_MALE.pkl", "smpl/SMPL_FEMALE.pkl"]
    assert missing_body_models(["rich"], tmp_path) != []  # rich needs gendered SMPL-X too


def test_registries_match_disk_and_manifest():
    from gvhmr.utils import assets

    task_dir = PROJ_ROOT / "gvhmr/configs/global/task"
    for key, (task, pack, ids) in DATASETS.items():
        assert (task_dir / f"{task}.yaml").exists(), f"{key}: task config {task}.yaml missing"
        assert pack in assets.DATA_PACKS, f"{key}: pack {pack!r} not in the download manifest"
        assert key in BODY_MODEL_FILES
        for ds_id in ids:
            assert ds_id in PAPER_REFERENCE, f"{ds_id} has no paper reference row"
    assert (task_dir / f"{COMBINED_TASK}.yaml").exists()


def test_paper_reference_metrics_have_labels():
    # Every reference metric must have a display label (units) — the table renders all of them.
    for ds, ref in PAPER_REFERENCE.items():
        for k in ref:
            assert k in METRIC_LABELS, f"{ds}.{k} missing a display label"
