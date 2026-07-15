# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for precision-first YOLO auxiliary threshold calibration."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest
from capture_data.zs32_yolo_auxiliary_calibration import (
    AuxiliaryRow,
    run_yolo_auxiliary_calibration,
    select_high_precision_threshold,
)

if TYPE_CHECKING:
    from pathlib import Path

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
FIELDS = (
    "part_id",
    "hand",
    "view",
    "branch",
    "raw_score",
    "gt_label",
    "split",
    "model_version",
    "roi_version",
)


def _row(score: float, label: int) -> AuxiliaryRow:
    """Build one direct selector row."""
    return AuxiliaryRow(
        part_id="part::front",
        physical_part_id="part",
        hand="right",
        view="front",
        score=score,
        label=label,
        split="calibration",
        model_version="yolo-v1",
        roi_version="roi-v1",
    )


def _write_input(path: Path, *, test_front_score: float) -> None:
    """Write complete eight-view fit/test rows with one physical defect and normal part."""
    rows = []
    for view in VIEWS:
        for split, defect_score in (
            ("calibration", 0.9),
            ("test", test_front_score if view == "front" else 0.0),
        ):
            rows.extend(
                (
                    {
                        "part_id": f"right:defect:deform:{split}::{view}",
                        "hand": "right",
                        "view": view,
                        "branch": "yolo",
                        "raw_score": defect_score,
                        "gt_label": 1,
                        "split": split,
                        "model_version": "yolo-v1",
                        "roi_version": "roi-v1",
                    },
                    {
                        "part_id": f"right:normal:{split}::{view}",
                        "hand": "right",
                        "view": view,
                        "branch": "yolo",
                        "raw_score": 0.1,
                        "gt_label": 0,
                        "split": split,
                        "model_version": "yolo-v1",
                        "roi_version": "roi-v1",
                    },
                ),
            )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_selector_prioritizes_recall_subject_to_precision() -> None:
    """The selected threshold must maximize recall only among precision-feasible candidates."""
    rows = (_row(0.9, 1), _row(0.4, 1), _row(0.5, 0), _row(0.1, 0))

    threshold, metrics, status = select_high_precision_threshold(rows, 0.9)

    assert status == "ok"
    assert threshold == pytest.approx(0.9)
    assert metrics is not None
    assert metrics["image_presence_precision"] == pytest.approx(1.0)
    assert metrics["image_presence_recall"] == pytest.approx(0.5)


def test_zero_scores_cannot_become_auxiliary_threshold() -> None:
    """No-box zero scores must remain CLEAR instead of becoming strong evidence."""
    threshold, metrics, status = select_high_precision_threshold((_row(0.0, 1), _row(0.0, 0)), 0.9)

    assert threshold is None
    assert metrics is None
    assert status == "no_feasible_threshold"


def test_test_scores_do_not_change_selected_thresholds_and_or_metrics_are_reported(tmp_path: Path) -> None:
    """Held-out test data may change evaluation but never val-selected thresholds."""
    input_a = tmp_path / "a.csv"
    input_b = tmp_path / "b.csv"
    _write_input(input_a, test_front_score=0.95)
    _write_input(input_b, test_front_score=0.0)

    summary_a = run_yolo_auxiliary_calibration(input_a, tmp_path / "out-a", minimum_fit_image_precision=0.9)
    summary_b = run_yolo_auxiliary_calibration(input_b, tmp_path / "out-b", minimum_fit_image_precision=0.9)
    thresholds_a = json.loads((tmp_path / "out-a" / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    thresholds_b = json.loads((tmp_path / "out-b" / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]

    assert [record["threshold"] for record in thresholds_a] == [record["threshold"] for record in thresholds_b]
    assert summary_a["six_view_or_part_metrics"]["test"]["recall"] == pytest.approx(1.0)
    assert summary_b["six_view_or_part_metrics"]["test"]["recall"] == pytest.approx(0.0)
    assert summary_a["threshold_selection_used_test"] is False


def test_explicit_test_leakage_can_fill_missing_calibration_candidate(tmp_path: Path) -> None:
    """The escape hatch may use test labels only when the artifact declares leakage."""
    input_csv = tmp_path / "leaked.csv"
    _write_input(input_csv, test_front_score=0.95)
    rows = list(csv.DictReader(input_csv.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["view"] == "front" and row["split"] == "calibration" and row["gt_label"] == "1":
            row["raw_score"] = "0.0"
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = run_yolo_auxiliary_calibration(
        input_csv,
        tmp_path / "out",
        minimum_fit_image_precision=0.9,
        use_test_for_selection=True,
    )
    thresholds = json.loads((tmp_path / "out" / "thresholds.json").read_text(encoding="utf-8"))

    assert thresholds["all_views_have_candidate"] is True
    assert thresholds["fit_split"] == "calibration+test"
    assert thresholds["evaluation_split"] == "test_reused_for_selection"
    assert thresholds["test_used_for_selection"] is True
    assert thresholds["data_leakage"] is True
    assert "TEST DATA" in thresholds["leakage_notice"]
    assert summary["threshold_selection_used_test"] is True
    assert summary["data_leakage"] is True


def test_physical_part_cannot_cross_splits(tmp_path: Path) -> None:
    """A physical part identity must never contribute to both fit and evaluation."""
    input_csv = tmp_path / "cross-split.csv"
    _write_input(input_csv, test_front_score=0.95)
    rows = list(csv.DictReader(input_csv.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["split"] == "test":
            row["part_id"] = row["part_id"].replace(":test::", ":calibration::")
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="crosses calibration/test splits"):
        run_yolo_auxiliary_calibration(input_csv, tmp_path / "out", minimum_fit_image_precision=0.9)


def test_physical_part_requires_all_eight_views(tmp_path: Path) -> None:
    """Part-level OR metrics must not silently accept incomplete view groups."""
    input_csv = tmp_path / "missing-view.csv"
    _write_input(input_csv, test_front_score=0.95)
    rows = list(csv.DictReader(input_csv.open(encoding="utf-8", newline="")))
    rows = [
        row for row in rows if not (row["part_id"].startswith("right:normal:calibration") and row["view"] == "back")
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="exactly 8 canonical views"):
        run_yolo_auxiliary_calibration(input_csv, tmp_path / "out", minimum_fit_image_precision=0.9)
