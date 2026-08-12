"""Second BMW live Demo changes only EfficientAD and bright-streak v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "pipeline/bmw_lab_prepare_efficientad_bright_v2_demo.py"
CONFIG = REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_efficientad_bright_v2_v1.json"
BASE_CONFIG = REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_v1.json"
VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def _module():
    spec = importlib.util.spec_from_file_location("prepare_second_demo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_second_demo_links_candidate_efficientad_only(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    for view in VIEWS:
        template = baseline / "template" / view / "model.json"
        checkpoint = candidate / "efficientad" / view / "model.ckpt"
        template.parent.mkdir(parents=True)
        checkpoint.parent.mkdir(parents=True)
        template.write_text("{}", encoding="utf-8")
        checkpoint.write_bytes(view.encode())
    yolo = baseline / "yolo/train/weights/best.pt"
    yolo.parent.mkdir(parents=True)
    yolo.write_bytes(b"yolo")

    report = _module().prepare_efficientad_candidate_run(candidate, baseline, output)

    assert (output / "template").resolve() == (baseline / "template").resolve()
    assert (output / "efficientad").resolve() == (candidate / "efficientad").resolve()
    assert (output / "yolo").resolve() == (baseline / "yolo").resolve()
    assert report["changed_branches"] == ["efficientad", "bright_streak"]
    with pytest.raises(FileExistsError):
        _module().prepare_efficientad_candidate_run(candidate, baseline, output)


def test_second_demo_config_preserves_baseline_except_two_candidate_branches() -> None:
    baseline = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    candidate = json.loads(CONFIG.read_text(encoding="utf-8"))

    for field in ("capture_config", "roi_config", "prepared_manifest", "yolo"):
        assert candidate[field] == baseline[field]
    assert candidate["training_run"].endswith("bmw_right_batch_20260810_21_efficientad_bright_v2_demo_v1")
    assert candidate["efficientad"]["threshold_artifact"].endswith(
        "bmw_right_batch_20260810_21_efficientad_v1/efficientad/score_analysis/part_thresholds.json"
    )
    threshold = (CONFIG.parent / candidate["efficientad"]["threshold_artifact"]).resolve()
    assert candidate["efficientad"]["threshold_artifact_sha256"] == hashlib.sha256(threshold.read_bytes()).hexdigest()
    assert candidate["bright_streak"]["engine"] == "raw_profile_v2"
    bright = (CONFIG.parent / candidate["bright_streak"]["config"]).resolve()
    assert candidate["bright_streak"]["config_sha256"] == hashlib.sha256(bright.read_bytes()).hexdigest()
