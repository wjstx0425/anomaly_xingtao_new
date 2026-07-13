# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fixed six-view ZS32 ROI dataset converter."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import capture_data.zs32_view_roi_dataset as roi_module
import cv2
import numpy as np
import pytest
from capture_data.zs32_view_roi_dataset import (
    VIEWS,
    crop_zs32_yolo_dataset,
    select_view_rois,
    transform_yolo_labels,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def test_stage29_parser_exposes_select_and_convert_commands() -> None:
    """The numbered entrypoint offers the two steps required by the operator."""
    script = Path(__file__).resolve().parents[3] / "pipeline" / "29_zs32_fixed_roi.py"
    spec = importlib.util.spec_from_file_location("pipeline29_zs32_fixed_roi", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    select_args = module.build_parser().parse_args(["select"])
    convert_args = module.build_parser().parse_args(["convert"])

    assert select_args.command == "select"
    assert convert_args.command == "convert"
    assert convert_args.output_root.name == "zs32_six_view_roi_yolo"


def test_transform_yolo_labels_rejects_box_crossing_roi() -> None:
    """A selected ROI must never silently clip a defect box."""
    with pytest.raises(ValueError, match="not fully inside ROI"):
        transform_yolo_labels(
            "0 0.15 0.5 0.2 0.2\n",
            image_width=100,
            image_height=80,
            roi=(10, 10, 90, 70),
            source="crossing.txt",
        )


def test_transform_yolo_labels_can_clip_and_report_partial_boxes() -> None:
    """Clip mode keeps intersections, drops outside boxes, and reports both."""
    statistics: dict[str, int] = {}

    transformed = transform_yolo_labels(
        "0 0.15 0.5 0.2 0.2\n0 0.05 0.5 0.1 0.2\n",
        image_width=100,
        image_height=80,
        roi=(20, 10, 80, 70),
        source="partial.txt",
        partial_box_policy="clip",
        statistics=statistics,
    )

    assert len(transformed.splitlines()) == 1
    assert statistics == {"clipped_boxes": 1, "dropped_boxes": 1}


@pytest.mark.parametrize(
    "label",
    ["0 nan 0.5 0.2 0.2\n", "0 0.5 0.5 -0.2 0.2\n", "1 0.5 0.5 0.2 0.2\n"],
)
def test_transform_yolo_labels_rejects_invalid_values(label: str) -> None:
    """Invalid geometry and non-defect classes must not reach an output dataset."""
    with pytest.raises(ValueError, match="Invalid YOLO label"):
        transform_yolo_labels(
            label,
            image_width=100,
            image_height=80,
            roi=(0, 0, 100, 80),
            source="invalid.txt",
        )


def test_crop_dataset_rejects_unsafe_output_before_reading_manifest(tmp_path: Path) -> None:
    """Overwrite can only target an independent child under repo/dataset."""
    repo_root = tmp_path / "repo"
    input_root = repo_root / "dataset" / "input"
    input_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="safe child directory"):
        crop_zs32_yolo_dataset(
            repo_root=repo_root,
            input_root=input_root,
            output_root=repo_root / "dataset",
            roi_config=repo_root / "missing.json",
            overwrite=True,
        )


def test_selector_shows_one_clean_original_image_per_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each view must call the ROI selector once with an unmodified source image."""
    repo_root = tmp_path / "repo"
    input_root = repo_root / "dataset" / "input"
    image_dir = input_root / "images" / "train"
    label_dir = input_root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rows = []
    expected_paths = []
    for view in VIEWS:
        image_path = image_dir / f"{view}.png"
        label_path = label_dir / f"{view}.txt"
        assert cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
        label_path.write_text("", encoding="utf-8")
        expected_paths.append(image_path)
        rows.append(
            {
                "source_path": str(image_path.relative_to(repo_root)),
                "output_image": str(image_path.relative_to(repo_root)),
                "output_label": str(label_path.relative_to(repo_root)),
                "kind": "normal_real",
                "split": "train",
                "sample_id": view,
                "hand": "right",
                "view": view,
                "source_view": view,
                "defect_type": "",
                "annotation_count": "0",
            },
        )
    with (input_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected_paths = []

    def fake_select(image_path: Path, *_args: object) -> tuple[int, int, int, int]:
        selected_paths.append(image_path)
        return 1, 2, 29, 19

    monkeypatch.setattr(roi_module, "select_roi", fake_select)
    monkeypatch.setattr(roi_module, "save_overlay", lambda *_args: None)

    select_view_rois(
        repo_root=repo_root,
        input_root=input_root,
        config_path=repo_root / "dataset" / "rois.json",
        preview_dir=repo_root / "dataset" / "previews",
    )

    assert selected_paths == expected_paths


def test_crop_dataset_preserves_split_and_empty_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crops and labels retain manifest split/view semantics and empty negatives."""
    repo_root = tmp_path / "repo"
    input_root = repo_root / "dataset" / "input"
    output_root = repo_root / "dataset" / "output"
    progress_calls: list[tuple[str, int]] = []

    def fake_track(sequence: Iterable[object], *, description: str, total: int) -> Iterable[object]:
        progress_calls.append((description, total))
        return sequence

    monkeypatch.setattr(roi_module, "track", fake_track)
    (input_root / "images" / "train").mkdir(parents=True)
    (input_root / "labels" / "train").mkdir(parents=True)

    rows = []
    for name, label, view, kind, source_view in (
        ("defect.png", "0 0.5 0.5 0.2 0.25\n", "front", "defect", "front"),
        ("normal.png", "", "back", "normal_real", "back"),
        ("mirror.png", "", "front_right", "normal_mirror", "front_left"),
    ):
        image_path = input_root / "images" / "train" / name
        label_path = input_root / "labels" / "train" / f"{Path(name).stem}.txt"
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        if kind == "normal_mirror":
            image[:, :, 0] = np.arange(100, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        label_path.write_text(label, encoding="utf-8")
        rows.append(
            {
                "source_path": str(image_path.relative_to(repo_root)),
                "output_image": str(image_path.relative_to(repo_root)),
                "output_label": str(label_path.relative_to(repo_root)),
                "kind": kind,
                "split": "train",
                "sample_id": name,
                "hand": "left" if kind == "normal_mirror" else "right",
                "view": view,
                "source_view": source_view,
                "defect_type": "less" if label else "",
                "annotation_count": "1" if label else "0",
            },
        )

    with (input_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config_path = repo_root / "dataset" / "rois.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_system": "pixel_xyxy_half_open",
                "image_size": {"width": 100, "height": 80},
                "views": {
                    **{view: {"roi": [10, 10, 90, 70]} for view in VIEWS},
                    "front_left": {"roi": [10, 10, 60, 70]},
                    "front_right": {"roi": [0, 0, 20, 20]},
                },
            },
        ),
        encoding="utf-8",
    )

    summary = crop_zs32_yolo_dataset(
        repo_root=repo_root,
        input_root=input_root,
        output_root=output_root,
        roi_config=config_path,
    )

    assert summary == {
        "images": 3,
        "labels": 3,
        "boxes": 1,
        "empty_labels": 2,
        "clipped_boxes": 0,
        "dropped_boxes": 0,
    }
    cropped = cv2.imread(str(output_root / "images" / "train" / "defect.png"))
    assert cropped.shape[:2] == (60, 80)
    mirrored_crop = cv2.imread(str(output_root / "images" / "train" / "mirror.png"))
    mirrored_source = cv2.imread(str(input_root / "images" / "train" / "mirror.png"))
    assert mirrored_crop.shape[:2] == (60, 50)
    assert np.array_equal(mirrored_crop, mirrored_source[10:70, 40:90])
    assert (output_root / "labels" / "train" / "normal.txt").read_text(encoding="utf-8") == ""
    values = (output_root / "labels" / "train" / "defect.txt").read_text(encoding="utf-8").split()
    assert values[0] == "0"
    assert [float(value) for value in values[1:]] == pytest.approx([0.5, 0.5, 0.25, 1 / 3])

    with (output_root / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
        output_rows = list(csv.DictReader(file))
    assert [(row["split"], row["view"]) for row in output_rows] == [
        ("train", "front"),
        ("train", "back"),
        ("train", "front_right"),
    ]
    assert output_rows[0]["output_image"] == "dataset/output/images/train/defect.png"
    assert [output_rows[2][key] for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")] == ["40", "10", "90", "70"]
    assert progress_calls == [("Preflight", 3), ("Cropping", 3)]
