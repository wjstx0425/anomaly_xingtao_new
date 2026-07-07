# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ROI-level C789 YOLO dataset export helpers."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_roi_yolo_dataset_module() -> ModuleType:
    """Load the ROI YOLO dataset export script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "roi_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("capture_data_roi_yolo_dataset", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load ROI YOLO dataset export script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, width: int = 800, height: int = 800, value: int = 128) -> None:
    """Write a deterministic RGB test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    grid_y = np.tile(np.arange(height, dtype=np.uint8).reshape(height, 1), (1, width))
    image = np.stack((grid_x, grid_y, np.full((height, width), value, dtype=np.uint8)), axis=-1)
    assert cv2.imwrite(str(path), image)


def _write_yolo_label(
    path: Path,
    boxes_xyxy: list[tuple[float, float, float, float]],
    image_width: int,
    image_height: int,
) -> None:
    """Write YOLO labels from pixel-space xyxy boxes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x_center = ((x1 + x2) / 2.0) / image_width
        y_center = ((y1 + y2) / 2.0) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _parse_yolo_boxes(path: Path, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    """Convert YOLO normalized labels to pixel xyxy boxes."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes = []
    for line in text.splitlines():
        class_id, x_center, y_center, width, height = line.split()
        assert class_id == "0"
        x_center_px = float(x_center) * image_width
        y_center_px = float(y_center) * image_height
        width_px = float(width) * image_width
        height_px = float(height) * image_height
        boxes.append(
            (
                x_center_px - (width_px / 2.0),
                y_center_px - (height_px / 2.0),
                x_center_px + (width_px / 2.0),
                y_center_px + (height_px / 2.0),
            ),
        )
    return boxes


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def test_build_gt_center_dataset_writes_expected_outputs_and_valid_labels(tmp_path: Path) -> None:
    """GT-center mode should crop in-bounds ROIs, preserve bbox geometry, and add empty normal labels."""
    roi = load_roi_yolo_dataset_module()
    input_root = tmp_path / "input_yolo"
    normal_root = tmp_path / "normal_yolo"

    train_image = input_root / "images" / "c789_g001_slot01_defect.png"
    val_image = input_root / "images" / "c789_g002_slot01_defect.png"
    _write_image(train_image)
    _write_image(val_image)
    _write_yolo_label(
        input_root / "labels" / "c789_g001_slot01_defect.txt",
        [(80.0, 100.0, 160.0, 180.0), (190.0, 200.0, 240.0, 250.0)],
        800,
        800,
    )
    _write_yolo_label(
        input_root / "labels" / "c789_g002_slot01_defect.txt",
        [(620.0, 600.0, 700.0, 680.0)],
        800,
        800,
    )

    for name in ("normal_a", "normal_b"):
        _write_image(normal_root / "images" / f"c789_g001_slot01_{name}.png", value=110)
    _write_image(normal_root / "images" / "c789_g002_slot01_normal_c.png", value=120)

    summary = roi.build_roi_yolo_dataset(
        input_root=input_root,
        output_root=tmp_path / "roi_gt_center",
        mode="gt-center",
        roi_size=512,
        stride=256,
        val_groups=("g002",),
        normal_source_root=normal_root,
        train_normal_ratio=2,
        val_normal_limit=1,
        preview_limit=2,
        seed=7,
        overwrite=True,
    )

    assert summary["mode"] == "gt-center"
    assert summary["counts_by_split_kind"]["train:defect_roi"] == 2
    assert summary["counts_by_split_kind"]["train:normal_roi"] == 4
    assert summary["counts_by_split_kind"]["val:defect_roi"] == 1
    assert summary["counts_by_split_kind"]["val:normal_roi"] == 1

    output_root = tmp_path / "roi_gt_center"
    assert (output_root / "data.yaml").is_file()
    assert (output_root / "roi_manifest.csv").is_file()
    assert (output_root / "dataset_summary.csv").is_file()
    assert (output_root / "leakage_report.csv").is_file()
    assert any((output_root / "previews").rglob("*.png"))

    manifest_rows = _read_csv(output_root / "roi_manifest.csv")
    fieldnames = set(manifest_rows[0])
    assert {
        "roi_image_path",
        "roi_label_path",
        "slot_processed_path",
        "slot_label_path",
        "split",
        "group_id",
        "slot_id",
        "roi_x1_in_slot",
        "roi_y1_in_slot",
        "roi_x2_in_slot",
        "roi_y2_in_slot",
        "source_bbox_index",
        "included_box_count",
        "kind",
    }.issubset(fieldnames)

    train_defect_rows = [row for row in manifest_rows if row["split"] == "train" and row["kind"] == "defect_roi"]
    assert len(train_defect_rows) == 2
    expected_boxes = [(80.0, 100.0, 160.0, 180.0), (190.0, 200.0, 240.0, 250.0)]
    for train_defect in train_defect_rows:
        assert train_defect["group_id"] == "g001"
        assert train_defect["slot_id"] == "slot01"
        assert train_defect["included_box_count"] == "2"

        roi_x1 = int(train_defect["roi_x1_in_slot"])
        roi_y1 = int(train_defect["roi_y1_in_slot"])
        roi_x2 = int(train_defect["roi_x2_in_slot"])
        roi_y2 = int(train_defect["roi_y2_in_slot"])
        assert 0 <= roi_x1 < roi_x2 <= 800
        assert 0 <= roi_y1 < roi_y2 <= 800
        assert (roi_x2 - roi_x1, roi_y2 - roi_y1) == (512, 512)

        round_trip_boxes = _parse_yolo_boxes(Path(train_defect["roi_label_path"]), 512, 512)
        projected_boxes = [(x1 + roi_x1, y1 + roi_y1, x2 + roi_x1, y2 + roi_y1) for x1, y1, x2, y2 in round_trip_boxes]
        assert len(projected_boxes) == len(expected_boxes)
        for recovered, expected in zip(projected_boxes, expected_boxes, strict=True):
            assert _iou(recovered, expected) >= 0.98

    for row in manifest_rows:
        label_path = Path(row["roi_label_path"])
        text = label_path.read_text(encoding="utf-8").strip()
        if row["kind"] == "normal_roi":
            assert text == ""
            continue
        for line in text.splitlines():
            _, x_center, y_center, width, height = line.split()
            for value in (x_center, y_center, width, height):
                assert 0.0 <= float(value) <= 1.0


def test_build_tile_dataset_emits_empty_tiles_and_group_split(tmp_path: Path) -> None:
    """Tile mode should cover the slot crop with empty labels allowed and split by group ids."""
    roi = load_roi_yolo_dataset_module()
    input_root = tmp_path / "input_yolo"

    _write_image(input_root / "images" / "c789_g001_slot02_train.png", width=600, height=600)
    _write_image(input_root / "images" / "c789_g008_slot02_val.png", width=600, height=600, value=150)
    _write_yolo_label(input_root / "labels" / "c789_g001_slot02_train.txt", [(30.0, 40.0, 90.0, 100.0)], 600, 600)
    _write_yolo_label(input_root / "labels" / "c789_g008_slot02_val.txt", [(520.0, 500.0, 580.0, 560.0)], 600, 600)

    summary = roi.build_roi_yolo_dataset(
        input_root=input_root,
        output_root=tmp_path / "roi_tile",
        mode="tile",
        roi_size=512,
        stride=256,
        val_groups=("g008",),
        normal_source_root=None,
        preview_limit=1,
        seed=3,
        overwrite=True,
    )

    assert summary["mode"] == "tile"
    assert summary["counts_by_split_kind"]["train:defect_roi"] >= 1
    assert summary["counts_by_split_kind"]["val:defect_roi"] >= 1

    manifest_rows = _read_csv(tmp_path / "roi_tile" / "roi_manifest.csv")
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    val_rows = [row for row in manifest_rows if row["split"] == "val"]
    assert train_rows
    assert val_rows
    assert all(row["kind"] == "defect_roi" for row in manifest_rows)
    assert any(Path(row["roi_label_path"]).read_text(encoding="utf-8") == "" for row in manifest_rows)
    assert all(row["group_id"] == "g001" for row in train_rows)
    assert all(row["group_id"] == "g008" for row in val_rows)
    assert all(int(row["roi_x2_in_slot"]) - int(row["roi_x1_in_slot"]) == 512 for row in manifest_rows)
    assert all(int(row["roi_y2_in_slot"]) - int(row["roi_y1_in_slot"]) == 512 for row in manifest_rows)


def test_gt_center_skips_source_boxes_that_do_not_fit_roi(tmp_path: Path) -> None:
    """GT-center mode should not emit clipped labels for boxes larger than the ROI."""
    roi = load_roi_yolo_dataset_module()
    input_root = tmp_path / "input_yolo"

    _write_image(input_root / "images" / "c789_g001_slot05_defect.png", width=900, height=700)
    _write_yolo_label(
        input_root / "labels" / "c789_g001_slot05_defect.txt",
        [(100.0, 250.0, 780.0, 360.0), (320.0, 420.0, 380.0, 480.0)],
        900,
        700,
    )

    summary = roi.build_roi_yolo_dataset(
        input_root=input_root,
        output_root=tmp_path / "roi_gt_center",
        mode="gt-center",
        roi_size=512,
        val_groups=("g002",),
        overwrite=True,
    )

    output_root = tmp_path / "roi_gt_center"
    review_rows = _read_csv(output_root / "roi_review_report.csv")

    assert summary["counts_by_split_kind"]["train:defect_roi"] == 1
    assert summary["roi_review_rows"] == 1
    assert review_rows[0]["slot_id"] == "slot05"
    assert review_rows[0]["reason"] == "source_bbox_not_fully_contained_in_gt_center_roi"
    assert len(list((output_root / "images" / "train").glob("*.png"))) == 1
    assert (output_root / "roi_manifest.csv").is_file()


def test_build_arg_parser_exposes_expected_defaults() -> None:
    """CLI parser should expose the ROI dataset builder defaults."""
    roi = load_roi_yolo_dataset_module()

    parser = roi.build_arg_parser()
    args = parser.parse_args(["--input-root", "in", "--output-root", "out", "--mode", "tile"])

    assert args.input_root == Path("in")
    assert args.output_root == Path("out")
    assert args.mode == "tile"
    assert args.roi_size == 512
    assert args.stride == 256
    assert tuple(args.val_groups) == ("g002", "g008")
    assert args.normal_source_root is None
    assert args.train_normal_ratio == 2
    assert args.val_normal_limit == 150
    assert args.preview_limit == 24
    assert args.seed == 0
    assert args.overwrite is False
