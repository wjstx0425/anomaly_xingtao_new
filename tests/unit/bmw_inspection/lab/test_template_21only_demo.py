"""Contracts for the isolated BMW 21:00 Template-only Demo assets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "pipeline/bmw_lab_prepare_template_21only_demo.py"
BASE_CONFIG = REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_v1.json"
CANDIDATE_CONFIG = (
    REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_template_21only_v1.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("bmw_lab_prepare_template_21only_demo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Template-only Demo preparation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assets(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate/template"
    baseline = tmp_path / "baseline"
    for view in VIEW_ORDER:
        model = candidate / view / "model.json"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text('{"threshold": 0.1}\n', encoding="utf-8")
        checkpoint = baseline / "efficientad" / view / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(view.encode())
    yolo = baseline / "yolo/train/weights/best.pt"
    yolo.parent.mkdir(parents=True)
    yolo.write_bytes(b"yolo")
    return candidate, baseline


def test_prepare_template_only_run_links_only_template_to_the_candidate(tmp_path: Path) -> None:
    module = _load_module()
    candidate, baseline = _assets(tmp_path)
    output = tmp_path / "composite"

    report = module.prepare_template_only_run(candidate, baseline, output)

    assert report["status"] == "complete"
    assert report["changed_branch"] == "template"
    assert (output / "template").is_symlink()
    assert (output / "template").resolve() == candidate.resolve()
    assert (output / "efficientad").resolve() == (baseline / "efficientad").resolve()
    assert (output / "yolo").resolve() == (baseline / "yolo").resolve()
    for view in VIEW_ORDER:
        assert (output / "template" / view / "model.json").is_file()
        assert (output / "efficientad" / view / "model.ckpt").is_file()
    assert (output / "yolo/train/weights/best.pt").is_file()

    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        module.prepare_template_only_run(candidate, baseline, output)


def test_candidate_config_changes_only_demo_identity_run_and_result_root() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))

    assert candidate["training_run"] == (
        "../../../results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_demo_v1"
    )
    assert candidate["result_root"] == "../../../results/bmw_eight_view_demo_template_21only_v1"
    assert candidate["demo_id"] == "bmw-eight-view-right-template-21only-demo-v1"
    for field in (
        "schema_version",
        "capture_config",
        "roi_config",
        "prepared_manifest",
        "bright_streak",
        "efficientad",
        "yolo",
    ):
        assert candidate[field] == base[field]
