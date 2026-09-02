"""Tests for training Template models from surviving review copies."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.template_review_training import (
    calibration_summary,
    write_reviewed_demo_config,
    write_reviewed_template_models,
)


def _review_package(tmp_path: Path, *, count: int = 4) -> tuple[Path, Path]:
    root = tmp_path / "review"
    root.mkdir()
    rows = []
    for view_index, view in enumerate(VIEW_ORDER):
        view_root = root / "right" / view
        view_root.mkdir(parents=True)
        for index in range(count):
            image = np.full((50, 60, 3), 30 + view_index * 5 + index, dtype=np.uint8)
            cv2.line(image, (5 + index, 5), (45, 40), (200, 120, 80), 2)
            relative = Path("right") / view / f"candidate_{index + 1:02d}.png"
            assert cv2.imwrite(str(root / relative), image)
            rows.append(
                {
                    "hand": "right",
                    "view_id": view,
                    "candidate_index": index + 1,
                    "selected_rank": index + 1,
                    "session_id": "session-a",
                    "sample_id": f"{view}-sample-{index}",
                    "physical_part_id": f"part-{index}",
                    "source_path": f"/source/{view}-{index}.png",
                    "review_image_path": relative.as_posix(),
                    "source_class": "normal",
                    "business_label": "OK",
                    "split": "train",
                }
            )
    with (root / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    roi = tmp_path / "roi.json"
    roi.write_text(
        json.dumps(
            {
                "image_width": 80,
                "image_height": 60,
                "part_rois": {view: [10, 5, 70, 55] for view in VIEW_ORDER},
            }
        ),
        encoding="utf-8",
    )
    return root, roi


def test_writer_uses_every_surviving_copy_without_refill_or_sha(tmp_path: Path) -> None:
    review, roi = _review_package(tmp_path)
    (review / "right" / "front" / "candidate_02.png").unlink()

    models = write_reviewed_template_models(review, "right", roi, tmp_path / "models")

    assert tuple(models) == VIEW_ORDER
    for view in VIEW_ORDER:
        payload = json.loads(models[view].read_text(encoding="utf-8"))
        expected = 3 if view == "front" else 4
        assert len(payload["templates"]) == expected
        assert len(list((models[view].parent / "templates").glob("template_*.png"))) == expected
        assert "sha256" not in json.dumps(payload)
        assert all(image["session_id"] == "session-a" for image in payload["templates"])


def test_writer_requires_three_surviving_copies_per_view(tmp_path: Path) -> None:
    review, roi = _review_package(tmp_path, count=3)
    (review / "right" / "front" / "candidate_01.png").unlink()

    with pytest.raises(ValueError, match="right/front.*at least 3"):
        write_reviewed_template_models(review, "right", roi, tmp_path / "models")


def test_calibration_threshold_uses_only_calibration_scores() -> None:
    first = calibration_summary([0.01, 0.02], [0.04])
    second = calibration_summary([0.01, 0.02], [0.40])

    assert first["threshold"] == pytest.approx(0.022)
    assert second["threshold"] == first["threshold"]
    assert first["final_test_false_reject_count"] == 1


def test_candidate_config_preserves_base_and_writes_new_models_and_thresholds(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    base_payload = {
        "demo_id": "base",
        "prepared_manifest": "/old/manifest.csv",
        "result_root": "/old/results",
        "template": {
            "models": {view: f"/old/{view}/model.json" for view in VIEW_ORDER},
            "thresholds": {view: 0.1 for view in VIEW_ORDER},
            "weighted_regions": {"enabled": True, "weight": 3.0, "thresholds": {view: 0.2 for view in VIEW_ORDER}},
        },
    }
    base.write_text(json.dumps(base_payload, indent=2) + "\n", encoding="utf-8")
    original = base.read_bytes()
    models = {view: tmp_path / "models" / view / "model.json" for view in VIEW_ORDER}
    report = {
        "template_thresholds": {view: 0.01 + index / 1000 for index, view in enumerate(VIEW_ORDER)},
        "weighted_thresholds": {view: 0.02 + index / 1000 for index, view in enumerate(VIEW_ORDER)},
    }
    output = tmp_path / "candidate.json"

    write_reviewed_demo_config(
        base,
        output,
        model_paths=models,
        calibration_report=report,
        prepared_manifest=tmp_path / "manifest.csv",
        result_root=tmp_path / "results",
    )

    assert base.read_bytes() == original
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["template"]["models"] == {view: str(models[view].resolve()) for view in VIEW_ORDER}
    assert payload["template"]["thresholds"] == report["template_thresholds"]
    assert payload["template"]["weighted_regions"]["thresholds"] == report["weighted_thresholds"]
    assert payload["prepared_manifest"] == str((tmp_path / "manifest.csv").resolve())

