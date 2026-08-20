"""Contracts for the independent BMW eight-view laboratory Demo."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    fuse_demo_status,
    load_capture_directory,
    load_demo_config,
    load_manifest_sample,
)


def _branch_result(status: BranchStatus) -> DemoBranchResult:
    return DemoBranchResult(
        branch=DemoBranch.TEMPLATE,
        view_id="front",
        status=status,
        score=0.1,
        threshold=0.2,
        elapsed_ms=1.0,
        reason="test",
        overlay=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def test_fusion_runs_as_all_pass_ng_or_error() -> None:
    assert fuse_demo_status((_branch_result(BranchStatus.PASS),)) is DemoFinalStatus.OK
    assert fuse_demo_status((_branch_result(BranchStatus.NG),)) is DemoFinalStatus.NG
    assert fuse_demo_status((_branch_result(BranchStatus.ERROR),)) is DemoFinalStatus.ERROR


def test_capture_directory_requires_exactly_eight_named_images(tmp_path: Path) -> None:
    for view in VIEW_ORDER:
        assert cv2.imwrite(str(tmp_path / f"{view}.png"), np.full((12, 16, 3), 20, dtype=np.uint8))

    images = load_capture_directory(tmp_path)

    assert tuple(images) == VIEW_ORDER
    assert all(image.shape == (12, 16, 3) for image in images.values())


def test_manifest_sample_loads_all_views_in_canonical_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = []
    for index, view in enumerate(reversed(VIEW_ORDER)):
        image_path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(image_path), np.full((8, 12, 3), index, dtype=np.uint8))
        rows.append({"sample_id": "sample-1", "view_id": view, "source_path": str(image_path)})
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "view_id", "source_path"))
        writer.writeheader()
        writer.writerows(rows)

    images = load_manifest_sample(manifest, "sample-1")

    assert tuple(images) == VIEW_ORDER
    assert all(image.shape == (8, 12, 3) for image in images.values())


def test_demo_config_resolves_current_model_assets(tmp_path: Path) -> None:
    roi = tmp_path / "roi.json"
    capture = tmp_path / "capture.json"
    run = tmp_path / "run"
    roi.write_text("{}", encoding="utf-8")
    capture.write_text("{}", encoding="utf-8")
    for view in VIEW_ORDER:
        model = run / "template" / view / "model.json"
        checkpoint = run / "efficientad" / view / "model.ckpt"
        model.parent.mkdir(parents=True)
        checkpoint.parent.mkdir(parents=True)
        model.write_text("{}", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
    bright = run / "bright_streak/calibrated_config.json"
    yolo = run / "yolo/train/weights/best.pt"
    bright.parent.mkdir(parents=True)
    yolo.parent.mkdir(parents=True)
    bright.write_text("{}", encoding="utf-8")
    yolo.write_bytes(b"checkpoint")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "demo_id": "bmw-eight-view-demo-test",
                "capture_config": str(capture),
                "roi_config": str(roi),
                "prepared_manifest": str(tmp_path / "manifest.csv"),
                "training_run": str(run),
                "result_root": str(tmp_path / "results"),
                "yolo": {"candidate_conf": 0.1, "final_threshold": 0.25, "imgsz": 640},
            }
        ),
        encoding="utf-8",
    )

    config = load_demo_config(config_path)

    assert config.views == VIEW_ORDER
    assert config.yolo_checkpoint == yolo.resolve()
    assert config.bright_streak_config == bright.resolve()
    assert tuple(config.template_models) == VIEW_ORDER
    assert tuple(config.efficientad_checkpoints) == VIEW_ORDER


def test_demo_config_rejects_candidate_threshold_above_final(tmp_path: Path) -> None:
    path = tmp_path / "demo.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "demo_id": "bad",
                "capture_config": "missing.json",
                "roi_config": "missing.json",
                "prepared_manifest": "missing.csv",
                "training_run": "missing",
                "result_root": "results",
                "yolo": {"candidate_conf": 0.5, "final_threshold": 0.25, "imgsz": 640},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate_conf"):
        load_demo_config(path)
