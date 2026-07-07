# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for C789 YOLO dataset export helpers."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_prepare_yolo_dataset_module() -> ModuleType:
    """Load the YOLO dataset export script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "prepare_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("capture_data_prepare_yolo_dataset", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load YOLO dataset export script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, width: int = 100, height: int = 50, value: int = 128) -> None:
    """Write a deterministic RGB test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal part-crop manifest."""
    fieldnames = [
        "source_path",
        "source_label",
        "processed_path",
        "view",
        "hand",
        "position",
        "label",
        "gt_label",
        "split",
        "sample_id",
        "frame_id",
        "slot",
        "slot_row",
        "slot_col",
        "defect_type",
        "crop_width",
        "crop_height",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    """Write bbox annotations in pixel xyxy CSV form."""
    fieldnames = ["image_path", "class_name", "x_min", "y_min", "x_max", "y_max"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_export_yolo_dataset_writes_labels_yaml_manifest_and_previews(tmp_path: Path) -> None:
    """C789 manifest rows and bbox CSV annotations should export to Ultralytics detect format."""
    yolo = load_prepare_yolo_dataset_module()
    defect_image = tmp_path / "parts" / "left" / "top" / "defect" / "part001_slot01" / "images" / "defect.png"
    normal_image = tmp_path / "parts" / "left" / "top" / "normal" / "part001_slot02" / "images" / "normal.png"
    normal_test_image = (
        tmp_path / "parts" / "left" / "top" / "normal_test" / "part001_slot03" / "images" / "normal_test.png"
    )
    _write_image(defect_image)
    _write_image(normal_image)
    _write_image(normal_test_image)
    manifest_path = tmp_path / "part_crop_manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "source_path": str(tmp_path / "raw" / "surface_1_1.png"),
                "source_label": "defect",
                "processed_path": str(defect_image),
                "view": "left_top",
                "hand": "left",
                "position": "top",
                "label": "defect",
                "gt_label": "1",
                "split": "test",
                "sample_id": "part001_slot01",
                "frame_id": "surface_1_1",
                "slot": "slot01",
                "slot_row": "1",
                "slot_col": "1",
                "defect_type": "surface",
                "crop_width": "100",
                "crop_height": "50",
            },
            {
                "source_path": str(tmp_path / "raw" / "normal_a.png"),
                "source_label": "normal",
                "processed_path": str(normal_image),
                "view": "left_top",
                "hand": "left",
                "position": "top",
                "label": "normal",
                "gt_label": "0",
                "split": "train",
                "sample_id": "part001_slot02",
                "frame_id": "normal_a",
                "slot": "slot02",
                "slot_row": "1",
                "slot_col": "2",
                "defect_type": "",
                "crop_width": "100",
                "crop_height": "50",
            },
            {
                "source_path": str(tmp_path / "raw" / "normal_test_a.png"),
                "source_label": "normal",
                "processed_path": str(normal_test_image),
                "view": "left_top",
                "hand": "left",
                "position": "top",
                "label": "normal_test",
                "gt_label": "0",
                "split": "test",
                "sample_id": "part001_slot03",
                "frame_id": "normal_test_a",
                "slot": "slot03",
                "slot_row": "2",
                "slot_col": "1",
                "defect_type": "",
                "crop_width": "100",
                "crop_height": "50",
            },
        ],
    )
    annotations_path = tmp_path / "bbox_annotations.csv"
    _write_annotations(
        annotations_path,
        [
            {
                "image_path": str(defect_image),
                "class_name": "defect",
                "x_min": "10",
                "y_min": "5",
                "x_max": "30",
                "y_max": "25",
            },
        ],
    )

    summary = yolo.export_yolo_dataset(
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        output_root=tmp_path / "yolo",
        positive_val_ratio=0.0,
        positive_test_ratio=0.0,
        seed=0,
        preview_dir=tmp_path / "previews",
    )

    assert summary == {"train": 2, "val": 1, "test": 0}
    label_text = (tmp_path / "yolo" / "labels" / "train" / "defect.txt").read_text(encoding="utf-8").strip()
    assert label_text == "0 0.200000 0.300000 0.200000 0.400000"
    assert (tmp_path / "yolo" / "labels" / "train" / "normal.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "yolo" / "labels" / "val" / "normal_test.txt").read_text(encoding="utf-8") == ""
    data_yaml = (tmp_path / "yolo" / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in data_yaml
    assert "0: defect" in data_yaml
    with (tmp_path / "yolo" / "export_manifest.csv").open(newline="", encoding="utf-8") as file:
        exported = list(csv.DictReader(file))
    assert [row["yolo_split"] for row in exported] == ["train", "train", "val"]
    assert exported[0]["slot"] == "slot01"
    assert exported[0]["defect_type"] == "surface"
    assert exported[0]["annotation_count"] == "1"
    assert (tmp_path / "previews" / "train" / "defect.png").is_file()


def test_export_yolo_dataset_rejects_unannotated_defects_by_default(tmp_path: Path) -> None:
    """Defect crops should not silently become negative YOLO examples when bbox labels are missing."""
    yolo = load_prepare_yolo_dataset_module()
    defect_image = tmp_path / "parts" / "left" / "top" / "defect" / "part001_slot01" / "images" / "defect.png"
    _write_image(defect_image)
    manifest_path = tmp_path / "part_crop_manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "source_path": str(tmp_path / "raw" / "crack_1_1.png"),
                "source_label": "defect",
                "processed_path": str(defect_image),
                "view": "left_top",
                "hand": "left",
                "position": "top",
                "label": "defect",
                "gt_label": "1",
                "split": "test",
                "sample_id": "part001_slot01",
                "frame_id": "crack_1_1",
                "slot": "slot01",
                "slot_row": "1",
                "slot_col": "1",
                "defect_type": "crack",
                "crop_width": "100",
                "crop_height": "50",
            },
        ],
    )
    annotations_path = tmp_path / "bbox_annotations.csv"
    _write_annotations(annotations_path, [])

    try:
        yolo.export_yolo_dataset(
            manifest_path=manifest_path,
            annotations_path=annotations_path,
            output_root=tmp_path / "yolo",
        )
    except ValueError as error:
        assert "missing bbox annotations" in str(error)
    else:
        raise AssertionError("Expected unannotated defect crop to be rejected")


def test_export_yolo_dataset_does_not_match_bbox_by_raw_source_path(tmp_path: Path) -> None:
    """Raw image source paths should not be used as crop-level bbox identifiers."""
    yolo = load_prepare_yolo_dataset_module()
    raw_image = tmp_path / "raw" / "shared_source.png"
    first_crop = tmp_path / "parts" / "defect" / "part001_slot01" / "images" / "slot01.png"
    second_crop = tmp_path / "parts" / "defect" / "part001_slot02" / "images" / "slot02.png"
    _write_image(first_crop)
    _write_image(second_crop)
    manifest_path = tmp_path / "part_crop_manifest.csv"
    base_row = {
        "source_path": str(raw_image),
        "source_label": "defect",
        "view": "left_top",
        "hand": "left",
        "position": "top",
        "label": "defect",
        "gt_label": "1",
        "split": "test",
        "slot_row": "1",
        "defect_type": "crack",
        "crop_width": "100",
        "crop_height": "50",
    }
    _write_manifest(
        manifest_path,
        [
            {
                **base_row,
                "processed_path": str(first_crop),
                "sample_id": "part001_slot01",
                "frame_id": "shared_source",
                "slot": "slot01",
                "slot_col": "1",
            },
            {
                **base_row,
                "processed_path": str(second_crop),
                "sample_id": "part001_slot02",
                "frame_id": "shared_source",
                "slot": "slot02",
                "slot_col": "2",
            },
        ],
    )
    annotations_path = tmp_path / "bbox_annotations.csv"
    with annotations_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_path", "class_name", "x_min", "y_min", "x_max", "y_max"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_path": str(raw_image),
                "class_name": "defect",
                "x_min": "10",
                "y_min": "5",
                "x_max": "30",
                "y_max": "25",
            },
        )

    try:
        yolo.export_yolo_dataset(
            manifest_path=manifest_path,
            annotations_path=annotations_path,
            output_root=tmp_path / "yolo",
    )
    except ValueError as error:
        assert "image identifier" in str(error) or "missing bbox annotations" in str(error)
    else:
        raise AssertionError("Expected raw source_path bbox annotations to be rejected")


def test_export_yolo_dataset_rejects_non_empty_output_without_overwrite(tmp_path: Path) -> None:
    """Re-exporting into a stale YOLO directory should require an explicit overwrite."""
    yolo = load_prepare_yolo_dataset_module()
    defect_image = tmp_path / "parts" / "defect" / "part001_slot01" / "images" / "defect.png"
    _write_image(defect_image)
    manifest_path = tmp_path / "part_crop_manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "source_path": str(tmp_path / "raw" / "crack_1_1.png"),
                "source_label": "defect",
                "processed_path": str(defect_image),
                "view": "left_top",
                "hand": "left",
                "position": "top",
                "label": "defect",
                "gt_label": "1",
                "split": "test",
                "sample_id": "part001_slot01",
                "frame_id": "crack_1_1",
                "slot": "slot01",
                "slot_row": "1",
                "slot_col": "1",
                "defect_type": "crack",
                "crop_width": "100",
                "crop_height": "50",
            },
        ],
    )
    annotations_path = tmp_path / "bbox_annotations.csv"
    _write_annotations(
        annotations_path,
        [
            {
                "image_path": str(defect_image),
                "class_name": "defect",
                "x_min": "10",
                "y_min": "5",
                "x_max": "30",
                "y_max": "25",
            },
        ],
    )
    output_root = tmp_path / "yolo"
    stale_file = output_root / "images" / "train" / "stale.png"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale", encoding="utf-8")

    try:
        yolo.export_yolo_dataset(
            manifest_path=manifest_path,
            annotations_path=annotations_path,
            output_root=output_root,
        )
    except ValueError as error:
        assert "non-empty" in str(error)
        assert "--overwrite" in str(error)
    else:
        raise AssertionError("Expected non-empty output_root to be rejected")
