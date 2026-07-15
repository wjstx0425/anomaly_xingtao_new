# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Select eight fixed ZS32 view ROIs and crop an existing YOLO dataset."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Literal

import cv2
from capture_data.select_roi import save_overlay, select_roi
from rich.progress import track
from zs32_inspection.domain.views import CANONICAL_VIEWS

VIEWS = CANONICAL_VIEWS
ROI = tuple[int, int, int, int]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        msg = f"Missing split manifest: {path}"
        raise FileNotFoundError(msg)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        msg = f"Empty split manifest: {path}"
        raise ValueError(msg)
    return rows


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _parse_label(line: str, image_width: int, image_height: int, source: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        msg = f"Invalid YOLO label in {source}: {line!r}"
        raise ValueError(msg)
    try:
        class_id = int(parts[0])
        cx, cy, width, height = (float(value) for value in parts[1:])
    except ValueError as error:
        msg = f"Invalid YOLO label in {source}: {line!r}"
        raise ValueError(msg) from error
    values = (cx, cy, width, height)
    normalized_box = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    if class_id != 0 or not all(math.isfinite(value) for value in values):
        msg = f"Invalid YOLO label in {source}: {line!r}"
        raise ValueError(msg)
    if width <= 0 or height <= 0 or any(value < 0 or value > 1 for value in normalized_box):
        msg = f"Invalid YOLO label in {source}: {line!r}"
        raise ValueError(msg)
    x1 = (cx - width / 2) * image_width
    y1 = (cy - height / 2) * image_height
    x2 = (cx + width / 2) * image_width
    y2 = (cy + height / 2) * image_height
    return class_id, x1, y1, x2, y2


def transform_yolo_labels(
    label_text: str,
    *,
    image_width: int,
    image_height: int,
    roi: ROI,
    source: str,
    partial_box_policy: Literal["error", "clip"] = "error",
    statistics: dict[str, int] | None = None,
) -> str:
    """Transform normalized full-image labels into a fixed ROI coordinate system."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
    if roi_width <= 0 or roi_height <= 0:
        msg = f"Invalid ROI for {source}: {roi}"
        raise ValueError(msg)

    if partial_box_policy not in {"error", "clip"}:
        msg = f"Unknown partial box policy: {partial_box_policy}"
        raise ValueError(msg)

    output = []
    clipped_boxes = 0
    dropped_boxes = 0
    for line in label_text.splitlines():
        if not line.strip():
            continue
        class_id, x1, y1, x2, y2 = _parse_label(line, image_width, image_height, source)
        tolerance = 1e-4
        crosses_roi = (
            x1 < roi_x1 - tolerance
            or y1 < roi_y1 - tolerance
            or x2 > roi_x2 + tolerance
            or y2 > roi_y2 + tolerance
        )
        if crosses_roi:
            if partial_box_policy == "error":
                msg = f"Box is not fully inside ROI: source={source}, box={(x1, y1, x2, y2)}, roi={roi}"
                raise ValueError(msg)
            x1, y1, x2, y2 = max(x1, roi_x1), max(y1, roi_y1), min(x2, roi_x2), min(y2, roi_y2)
            if x2 <= x1 or y2 <= y1:
                dropped_boxes += 1
                continue
            clipped_boxes += 1
        cx = ((x1 + x2) / 2 - roi_x1) / roi_width
        cy = ((y1 + y2) / 2 - roi_y1) / roi_height
        width = (x2 - x1) / roi_width
        height = (y2 - y1) / roi_height
        output.append(f"{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
    if statistics is not None:
        statistics.update({"clipped_boxes": clipped_boxes, "dropped_boxes": dropped_boxes})
    return "\n".join(output) + ("\n" if output else "")


def load_roi_config(path: Path) -> tuple[int, int, dict[str, ROI], dict[str, Any]]:
    """Load and validate an eight-view ROI JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("coordinate_system") != "pixel_xyxy_half_open":
        msg = "ROI config coordinate_system must be pixel_xyxy_half_open."
        raise ValueError(msg)
    image_size = payload.get("image_size", {})
    width, height = int(image_size.get("width", 0)), int(image_size.get("height", 0))
    views = payload.get("views")
    actual_views = set(views) if isinstance(views, dict) else set()
    if width <= 0 or height <= 0 or actual_views != set(VIEWS):
        missing = sorted(set(VIEWS) - actual_views)
        extra = sorted(actual_views - set(VIEWS))
        msg = f"ROI config must contain exactly the eight views: missing={missing}, extra={extra}"
        raise ValueError(msg)
    assert isinstance(views, dict)
    rois: dict[str, ROI] = {}
    for view in VIEWS:
        values = views[view].get("roi", [])
        if len(values) != 4:
            msg = f"Invalid ROI for view {view}: {values}"
            raise ValueError(msg)
        roi = tuple(int(value) for value in values)
        x1, y1, x2, y2 = roi
        invalid_origin = x1 < 0 or y1 < 0
        invalid_extent = x2 <= x1 or y2 <= y1
        outside_image = x2 > width or y2 > height
        if invalid_origin or invalid_extent or outside_image:
            msg = f"ROI is outside configured image for view {view}: {roi}"
            raise ValueError(msg)
        rois[view] = roi
    return width, height, rois, payload


def _roi_for_row(row: dict[str, str], rois: dict[str, ROI], image_width: int) -> ROI:
    """Resolve the effective ROI, mirroring the source-view ROI for mirrored images."""
    view = row.get("view", "")
    if view not in rois:
        msg = f"Unknown view in manifest: {view!r}"
        raise ValueError(msg)
    if row.get("kind") != "normal_mirror":
        return rois[view]

    source_view = row.get("source_view", "")
    if source_view not in rois:
        msg = f"Unknown source_view for mirrored image: {source_view!r}"
        raise ValueError(msg)
    x1, y1, x2, y2 = rois[source_view]
    return image_width - x2, y1, image_width - x1, y2


def _preflight(
    rows: list[dict[str, str]], repo_root: Path, image_width: int, image_height: int, rois: dict[str, ROI],
) -> list[tuple[dict[str, str], Path, str, int, int, ROI]]:
    prepared = []
    for row in track(rows, description="Preflight", total=len(rows)):
        image_path = _resolve_repo_path(row["output_image"], repo_root)
        label_path = _resolve_repo_path(row["output_label"], repo_root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Failed to read image: {image_path}"
            raise FileNotFoundError(msg)
        height, width = image.shape[:2]
        if (width, height) != (image_width, image_height):
            msg = f"Image size mismatch: {image_path} is {width}x{height}, expected {image_width}x{image_height}"
            raise ValueError(msg)
        if not label_path.is_file():
            msg = f"Missing label: {label_path}"
            raise FileNotFoundError(msg)
        roi = _roi_for_row(row, rois, width)
        statistics: dict[str, int] = {}
        transformed = transform_yolo_labels(
            label_path.read_text(encoding="utf-8"),
            image_width=width,
            image_height=height,
            roi=roi,
            source=str(label_path),
            partial_box_policy="clip",
            statistics=statistics,
        )
        prepared.append(
            (row, image_path, transformed, statistics["clipped_boxes"], statistics["dropped_boxes"], roi),
        )
    return prepared


def crop_zs32_yolo_dataset(
    *,
    repo_root: Path,
    input_root: Path,
    output_root: Path,
    roi_config: Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Crop all images and transform labels while preserving the existing split."""
    repo_root, input_root, output_root = repo_root.resolve(), input_root.resolve(), output_root.resolve()
    dataset_root = (repo_root / "dataset").resolve()
    try:
        relative_output = output_root.relative_to(dataset_root)
    except ValueError as error:
        msg = f"Output must be a safe child directory under {dataset_root}: {output_root}"
        raise ValueError(msg) from error
    unsafe_output = not relative_output.parts or output_root == input_root or output_root in input_root.parents
    if unsafe_output:
        msg = f"Output must be a safe child directory separate from input: {output_root}"
        raise ValueError(msg)
    rows = _read_manifest(input_root / "split_manifest.csv")
    image_width, image_height, rois, config_payload = load_roi_config(roi_config)
    prepared = _preflight(rows, repo_root, image_width, image_height, rois)

    if output_root.exists():
        if not overwrite:
            msg = f"Output already exists; pass --overwrite to replace it: {output_root}"
            raise FileExistsError(msg)
        shutil.rmtree(output_root)

    output_rows = []
    boxes = 0
    empty_labels = 0
    clipped_boxes = 0
    dropped_boxes = 0
    for row, image_path, transformed, row_clipped, row_dropped, roi in track(
        prepared,
        description="Cropping",
        total=len(prepared),
    ):
        split = row["split"]
        output_image = output_root / "images" / split / image_path.name
        output_label = output_root / "labels" / split / f"{image_path.stem}.txt"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = roi
        if image is None or not cv2.imwrite(str(output_image), image[y1:y2, x1:x2], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            msg = f"Failed to write cropped image: {output_image}"
            raise RuntimeError(msg)
        output_label.write_text(transformed, encoding="utf-8")
        box_count = len(transformed.splitlines())
        boxes += box_count
        empty_labels += int(box_count == 0)
        clipped_boxes += row_clipped
        dropped_boxes += row_dropped
        updated = dict(row)
        updated.update(
            {
                "output_image": str(output_image.relative_to(repo_root)),
                "output_label": str(output_label.relative_to(repo_root)),
                "annotation_count": str(box_count),
                "clipped_annotation_count": str(row_clipped),
                "dropped_annotation_count": str(row_dropped),
                "roi_x1": str(x1),
                "roi_y1": str(y1),
                "roi_x2": str(x2),
                "roi_y2": str(y2),
            },
        )
        output_rows.append(updated)

    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (output_root / "roi_config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    (output_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    return {
        "images": len(prepared),
        "labels": len(prepared),
        "boxes": boxes,
        "empty_labels": empty_labels,
        "clipped_boxes": clipped_boxes,
        "dropped_boxes": dropped_boxes,
    }


def _selection_reference(
    rows: list[dict[str, str]], view: str, repo_root: Path,
) -> tuple[Path, int, int]:
    view_rows = [row for row in rows if row.get("view") == view]
    if not view_rows:
        msg = f"No manifest rows for view: {view}"
        raise ValueError(msg)
    reference_row = next(
        (row for row in view_rows if row.get("kind") == "normal_real" and row.get("hand") == "right"),
        view_rows[0],
    )
    reference_path = _resolve_repo_path(reference_row["output_image"], repo_root)
    image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Failed to read reference image: {reference_path}"
        raise FileNotFoundError(msg)
    height, width = image.shape[:2]
    return reference_path, width, height


def select_view_rois(
    *,
    repo_root: Path,
    input_root: Path,
    config_path: Path,
    preview_dir: Path,
    max_window_width: int = 1600,
    max_window_height: int = 1000,
) -> dict[str, Any]:
    """Interactively select one fixed ROI for each canonical view."""
    rows = _read_manifest(input_root / "split_manifest.csv")
    existing: dict[str, ROI] = {}
    if config_path.is_file():
        _width, _height, existing, _payload = load_roi_config(config_path)
    selected: dict[str, dict[str, Any]] = {}
    expected_size: tuple[int, int] | None = None
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(VIEWS, start=1):
        reference_path, width, height = _selection_reference(rows, view, repo_root)
        if expected_size is None:
            expected_size = (width, height)
        elif expected_size != (width, height):
            msg = f"Reference image size mismatch for {view}: {width}x{height}"
            raise ValueError(msg)
        print(f"\n[{index}/{len(VIEWS)}] View: {view}")
        print(f"Reference image: {reference_path}")
        roi = select_roi(reference_path, existing.get(view), max_window_width, max_window_height)
        reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        save_overlay(preview_dir / f"{view}_roi.png", reference_image, roi)
        print(f"Selected ROI for {view}: {roi}")
        selected[view] = {"roi": list(roi), "reference_image": str(reference_path.relative_to(repo_root))}

    assert expected_size is not None
    payload = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": expected_size[0], "height": expected_size[1]},
        "views": selected,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(config_path)
    return payload
