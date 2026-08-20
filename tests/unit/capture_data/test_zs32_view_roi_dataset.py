# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fixed eight-view ZS32 ROI dataset converter."""

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
    load_roi_config,
    select_view_rois,
    transform_yolo_labels,
)
from zs32_inspection.domain.views import CANONICAL_VIEWS

LEGACY_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")

if TYPE_CHECKING:
    from collections.abc import Iterable


EXPECTED_VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def test_roi_dataset_keeps_historical_six_view_default() -> None:
    """The legacy converter API must retain the historical six-view default."""
    assert VIEWS == LEGACY_VIEWS


def test_stage29_parser_exposes_select_and_convert_commands() -> None:
    """The numbered entrypoint offers the two steps required by the operator."""
    script = Path(__file__).resolve().parents[3] / "pipeline" / "29_zs32_fixed_roi.py"
    spec = importlib.util.spec_from_file_location("pipeline29_zs32_fixed_roi", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    select_args = module.build_parser().parse_args(["select", "--repo-root", "/tmp/data-repo"])
    convert_args = module.build_parser().parse_args(["convert", "--repo-root", "/tmp/data-repo"])

    assert select_args.command == "select"
    assert select_args.repo_root == Path("/tmp/data-repo")
    assert convert_args.command == "convert"
    assert convert_args.repo_root == Path("/tmp/data-repo")
    assert convert_args.output_root.name == "zs32_eight_view_roi_yolo"


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


def test_secondary_mirror_uses_horizontally_flipped_source_roi() -> None:
    """A secondary mirrored normal must flip its own asymmetric ROI."""
    rois = {view: (0, 0, 100, 80) for view in VIEWS}
    rois["front_secondary"] = (10, 12, 60, 72)

    roi = roi_module._roi_for_row(  # noqa: SLF001
        {
            "kind": "normal_mirror",
            "view": "front_secondary",
            "source_view": "front_secondary",
        },
        rois,
        image_width=100,
    )

    assert roi == (40, 12, 90, 72)


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
def test_legacy_six_view_yolo_roi_config_replays_by_default(tmp_path: Path) -> None:
    path = tmp_path / "legacy-six.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_system": "pixel_xyxy_half_open",
                "image_size": {"width": 100, "height": 80},
                "views": {view: {"roi": [0, 0, 100, 80]} for view in LEGACY_VIEWS},
            },
        ),
        encoding="utf-8",
    )

    _, _, rois, _ = load_roi_config(path)

    assert VIEWS == LEGACY_VIEWS
    assert tuple(rois) == LEGACY_VIEWS


def test_yolo_strict_eight_mode_requires_exact_canonical_set(tmp_path: Path) -> None:
    path = tmp_path / "strict-eight.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_system": "pixel_xyxy_half_open",
                "image_size": {"width": 100, "height": 80},
                "views": {view: {"roi": [0, 0, 100, 80]} for view in CANONICAL_VIEWS},
            },
        ),
        encoding="utf-8",
    )

    _, _, rois, _ = load_roi_config(path, expected_views=CANONICAL_VIEWS)

    assert tuple(rois) == CANONICAL_VIEWS


@pytest.mark.parametrize("bad_size", [True, 100.5, "100"])
@pytest.mark.parametrize("dimension", ["width", "height"])
def test_yolo_roi_config_rejects_non_integer_image_size(tmp_path: Path, bad_size: object, dimension: str) -> None:
    image_size: dict[str, object] = {"width": 100, "height": 80}
    image_size[dimension] = bad_size
    path = tmp_path / "bad-size.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_system": "pixel_xyxy_half_open",
                "image_size": image_size,
                "views": {view: {"roi": [0, 0, 100, 80]} for view in LEGACY_VIEWS},
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive integer"):
        load_roi_config(path)


@pytest.mark.parametrize("bad_coordinate", [True, 1.5, "1"])
def test_yolo_roi_config_rejects_non_integer_coordinate(tmp_path: Path, bad_coordinate: object) -> None:
    views = {view: {"roi": [0, 0, 100, 80]} for view in LEGACY_VIEWS}
    views["front"] = {"roi": [bad_coordinate, 0, 100, 80]}
    path = tmp_path / "bad-coordinate.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_system": "pixel_xyxy_half_open",
                "image_size": {"width": 100, "height": 80},
                "views": views,
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="four non-bool integers"):
        load_roi_config(path)
