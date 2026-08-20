# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for split-preserving ZS32 tiled YOLO dataset helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from capture_data.tile_zs32_yolo_dataset import (
    TileDetection,
    build_tiled_zs32_yolo_dataset,
    remap_and_merge_detections,
    tile_starts,
)


def _write_image(path: Path, *, width: int, height: int, value: int) -> None:
    """Write one deterministic source image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_yolo_box(
    path: Path,
    *,
    image_width: int,
    image_height: int,
    xyxy: tuple[float, float, float, float] | None,
) -> None:
    """Write one normalized YOLO box or an explicit empty label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if xyxy is None:
        path.write_text("", encoding="utf-8")
        return
    x1, y1, x2, y2 = xyxy
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    path.write_text(f"0 {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.fixture
def zs32_source_dataset(tmp_path: Path) -> Path:
    """Create a minimal split-preserving ZS32 YOLO dataset and mapping."""
    root = tmp_path / "zs32_source"
    specs = (
        ("train", "defect_train", "defect_positive", 2600, 2100, (400.0, 500.0, 600.0, 700.0)),
        ("val", "normal_val", "trusted_normal", 1400, 1300, None),
        ("test", "normal_test", "trusted_normal", 1400, 1300, None),
    )
    rows: list[dict[str, str]] = []
    for index, (split, stem, source_kind, width, height, box) in enumerate(specs, start=1):
        image_path = root / "images" / split / f"{stem}.png"
        label_path = root / "labels" / split / f"{stem}.txt"
        _write_image(image_path, width=width, height=height, value=50 * index)
        _write_yolo_box(
            label_path,
            image_width=width,
            image_height=height,
            xyxy=box,
        )
        rows.append(
            {
                "source_image_path": str(image_path),
                "new_image_path": str(image_path),
                "new_label_path": str(label_path),
                "sample_id": f"right/{'defect/deform' if box else 'normal/normal'}/session-{split}/group{index:03d}",
                "view": ("front", "front_left", "front_right")[index - 1],
                "hand": "right",
                "defect_type": "deform" if box else "normal",
                "session_id": f"session-{split}",
                "group_id": f"group{index:03d}",
                "split": split,
                "source_kind": source_kind,
                "annotation_status": "label_studio_positive" if box else "trusted_empty",
                "pixel_sha256": str(index) * 64,
            },
        )

    with (root / "mapping.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    return root


def test_tile_starts_cover_1280_tiles_with_960_stride_and_flush_last_edge() -> None:
    """The 1280/960 grid should cover both edges without a partial tile."""
    assert list(tile_starts(3000, tile_size=1280, stride=960)) == [0, 960, 1720]
    assert list(tile_starts(2100, tile_size=1280, stride=960)) == [0, 820]
    assert list(tile_starts(1280, tile_size=1280, stride=960)) == [0]


def test_builder_inherits_splits_transforms_boxes_and_samples_one_normal_tile(
    zs32_source_dataset: Path,
    tmp_path: Path,
) -> None:
    """Tiles should keep source splits, remap boxes, and retain one deterministic tile per normal image."""
    first_output = tmp_path / "tiled_first"
    second_output = tmp_path / "tiled_second"
    for output_root in (first_output, second_output):
        build_tiled_zs32_yolo_dataset(
            input_root=zs32_source_dataset,
            output_root=output_root,
            tile_size=1280,
            stride=960,
            normal_tiles_per_image=1,
            seed=42,
        )

    first_rows = _read_csv(first_output / "tile_manifest.csv")
    second_rows = _read_csv(second_output / "tile_manifest.csv")
    assert first_rows
    assert {row["split"] for row in first_rows} == {"train", "val", "test"}

    splits_by_source: dict[str, set[str]] = {}
    for row in first_rows:
        splits_by_source.setdefault(row["source_image_path"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_source.values())

    positive = next(
        row
        for row in first_rows
        if row["source_kind"] == "defect_positive" and row["tile_x1"] == "0" and row["tile_y1"] == "0"
    )
    label_values = [float(value) for value in Path(positive["tile_label_path"]).read_text(encoding="utf-8").split()]
    assert label_values[0] == 0
    assert label_values[1:] == pytest.approx(
        [500.0 / 1280.0, 600.0 / 1280.0, 200.0 / 1280.0, 200.0 / 1280.0],
        abs=1e-6,
    )

    first_normals = [row for row in first_rows if row["source_kind"] == "trusted_normal"]
    second_normals = [row for row in second_rows if row["source_kind"] == "trusted_normal"]
    assert len(first_normals) == len(second_normals) == 2
    assert all(Path(row["tile_label_path"]).read_text(encoding="utf-8") == "" for row in first_normals)
    first_origins = {row["source_image_path"]: (row["tile_x1"], row["tile_y1"]) for row in first_normals}
    second_origins = {row["source_image_path"]: (row["tile_x1"], row["tile_y1"]) for row in second_normals}
    assert first_origins == second_origins

    assert (first_output / "data.yaml").is_file()
    assert (first_output / "images/train").is_dir()
    assert (first_output / "images/val").is_dir()
    assert (first_output / "images/test").is_dir()


def test_builder_refuses_to_replace_a_nonempty_output_by_default(
    zs32_source_dataset: Path,
    tmp_path: Path,
) -> None:
    """A pre-existing output must not be deleted without explicit overwrite authorization."""
    output_root = tmp_path / "existing"
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"non-empty|overwrite"):
        build_tiled_zs32_yolo_dataset(
            input_root=zs32_source_dataset,
            output_root=output_root,
            tile_size=1280,
            stride=960,
            normal_tiles_per_image=1,
            seed=42,
        )

    assert marker.read_text(encoding="utf-8") == "do not replace\n"


def test_tile_detections_map_to_source_coordinates_before_global_nms() -> None:
    """Overlapping tile predictions should map back and collapse to the strongest global detection."""
    first = TileDetection(
        class_id=0,
        confidence=0.90,
        x1=850.0,
        y1=100.0,
        x2=950.0,
        y2=200.0,
        tile_x1=100,
        tile_y1=0,
    )
    second = TileDetection(
        class_id=0,
        confidence=0.75,
        x1=0.0,
        y1=100.0,
        x2=100.0,
        y2=200.0,
        tile_x1=960,
        tile_y1=0,
    )

    merged = remap_and_merge_detections([second, first], iou_threshold=0.5)

    assert len(merged) == 1
    assert merged[0].class_id == 0
    assert merged[0].confidence == pytest.approx(0.90)
    assert (merged[0].x1, merged[0].y1, merged[0].x2, merged[0].y2) == pytest.approx(
        (950.0, 100.0, 1050.0, 200.0),
    )
