# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build a simple delete-to-reject review package for BMW Template images."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


_MANIFEST_FIELDS = (
    "sample_id",
    "physical_part_id",
    "session_id",
    "view_id",
    "source_path",
    "source_class",
    "business_label",
    "split",
)
_OUTPUT_FIELDS = (
    "hand",
    "view_id",
    "candidate_index",
    "selected_rank",
    "session_id",
    "sample_id",
    "physical_part_id",
    "source_path",
    "review_image_path",
    "source_class",
    "business_label",
    "split",
)


@dataclass(frozen=True, slots=True)
class ReviewSource:
    """One hand-specific prepared manifest and its public ROI configuration."""

    hand: str
    manifest: Path
    roi_config: Path

    def __post_init__(self) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        manifest = Path(self.manifest).expanduser().resolve()
        roi_config = Path(self.roi_config).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"prepared manifest does not exist: {manifest}")
        if not roi_config.is_file():
            raise FileNotFoundError(f"ROI config does not exist: {roi_config}")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "roi_config", roi_config)


@dataclass(frozen=True, slots=True)
class _ManifestRow:
    sample_id: str
    physical_part_id: str
    session_id: str
    view_id: str
    source_path: Path
    source_class: str
    business_label: str
    split: str


def _required_text(row: dict[str, str], field: str, row_number: int) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"manifest row {row_number} has empty {field}")
    return value


def _load_rows(path: Path) -> dict[str, list[_ManifestRow]]:
    by_view: dict[str, list[_ManifestRow]] = {view: [] for view in VIEW_ORDER}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(_MANIFEST_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"prepared manifest is missing required fields: {path}")
        for row_number, raw in enumerate(reader, start=2):
            split = _required_text(raw, "split", row_number)
            source_class = _required_text(raw, "source_class", row_number)
            business_label = _required_text(raw, "business_label", row_number)
            if (split, source_class, business_label) != ("train", "normal", "OK"):
                continue
            view_id = _required_text(raw, "view_id", row_number)
            if view_id not in by_view:
                continue
            source_path = Path(_required_text(raw, "source_path", row_number)).expanduser()
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            source_path = source_path.resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"candidate source image does not exist: {source_path}")
            by_view[view_id].append(
                _ManifestRow(
                    sample_id=_required_text(raw, "sample_id", row_number),
                    physical_part_id=_required_text(raw, "physical_part_id", row_number),
                    session_id=_required_text(raw, "session_id", row_number),
                    view_id=view_id,
                    source_path=source_path,
                    source_class=source_class,
                    business_label=business_label,
                    split=split,
                )
            )
    return by_view


def _load_rois(path: Path) -> tuple[tuple[int, int], dict[str, tuple[int, int, int, int]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not parse ROI config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("ROI config must contain a JSON object")
    image_width = payload.get("image_width")
    image_height = payload.get("image_height")
    raw_rois = payload.get("part_rois")
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or image_width <= 0
        or isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_height <= 0
        or not isinstance(raw_rois, dict)
    ):
        raise ValueError("ROI config has invalid image dimensions or part_rois")
    rois: dict[str, tuple[int, int, int, int]] = {}
    for view in VIEW_ORDER:
        value = raw_rois.get(view)
        if (
            not isinstance(value, list)
            or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f"ROI config has invalid {view} ROI")
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
            raise ValueError(f"ROI config has out-of-bounds {view} ROI")
        rois[view] = (x1, y1, x2, y2)
    return (image_width, image_height), rois


def _read_crop(
    row: _ManifestRow,
    image_size: tuple[int, int],
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    image = cv2.imread(str(row.source_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"could not read candidate source image: {row.source_path}")
    expected_width, expected_height = image_size
    if image.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"candidate image size mismatch for {row.sample_id}: "
            f"{image.shape[1]}x{image.shape[0]} != {expected_width}x{expected_height}"
        )
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2].copy()


def _feature(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)
    resized -= resized.mean()
    resized /= resized.std() + 1e-6
    return resized


def _select_indices(rows: Sequence[_ManifestRow], features: np.ndarray, count: int) -> tuple[int, ...]:
    distances = np.mean((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
    selected = [int(np.argmin(distances.sum(axis=1)))]
    selected_parts = {rows[selected[0]].physical_part_id}
    minimum_distance = distances[selected[0]].copy()
    while len(selected) < count:
        remaining = [index for index in range(len(rows)) if index not in selected]
        new_parts = [index for index in remaining if rows[index].physical_part_id not in selected_parts]
        eligible = new_parts or remaining
        next_index = max(eligible, key=lambda index: (float(minimum_distance[index]), -index))
        selected.append(next_index)
        selected_parts.add(rows[next_index].physical_part_id)
        minimum_distance = np.minimum(minimum_distance, distances[next_index])
    return tuple(selected)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    columns, rows = 5, 8
    tile_width, image_height, label_height = 300, 190, 30
    canvas = np.full((rows * (image_height + label_height), columns * tile_width, 3), 245, dtype=np.uint8)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read written candidate image: {path}")
        scale = min(tile_width / image.shape[1], image_height / image.shape[0])
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        column = index % columns
        row = index // columns
        x = column * tile_width + (tile_width - width) // 2
        y = row * (image_height + label_height) + (image_height - height) // 2
        canvas[y : y + height, x : x + width] = resized
        cv2.putText(
            canvas,
            f"candidate_{index + 1:02d}",
            (column * tile_width + 8, row * (image_height + label_height) + image_height + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"could not write contact sheet: {output}")


def build_template_review_package(
    sources: Sequence[ReviewSource],
    output_root: Path,
    *,
    candidate_count: int = 40,
) -> Path:
    """Build candidate crops and contact sheets without training or refilling."""
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 3:
        raise ValueError("candidate_count must be an integer of at least 3")
    if not sources:
        raise ValueError("at least one review source is required")
    if len({source.hand for source in sources}) != len(sources):
        raise ValueError("review source hands must be unique")
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"review output already exists: {output_root}")

    prepared: list[
        tuple[ReviewSource, str, tuple[int, int], tuple[int, int, int, int], list[_ManifestRow], tuple[int, ...]]
    ] = []
    for source in sources:
        by_view = _load_rows(source.manifest)
        image_size, rois = _load_rois(source.roi_config)
        for view in VIEW_ORDER:
            rows = sorted(
                by_view[view],
                key=lambda item: (item.session_id, item.sample_id, item.source_path.as_posix()),
            )
            if len(rows) < candidate_count:
                raise ValueError(
                    f"{source.hand}/{view} needs {candidate_count} eligible candidates, found {len(rows)}"
                )
            features = np.stack([_feature(_read_crop(row, image_size, rois[view])) for row in rows])
            selected = _select_indices(rows, features, candidate_count)
            prepared.append((source, view, image_size, rois[view], rows, selected))

    output_root.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    for source, view, image_size, roi, rows, selected in prepared:
        view_root = output_root / source.hand / view
        view_root.mkdir(parents=True)
        candidate_paths: list[Path] = []
        for rank, row_index in enumerate(selected, start=1):
            row = rows[row_index]
            filename = (
                f"candidate_{rank:02d}__{_safe_name(row.session_id)}__{_safe_name(row.sample_id)}.png"
            )
            candidate_path = view_root / filename
            crop = _read_crop(row, image_size, roi)
            if not cv2.imwrite(str(candidate_path), crop):
                raise OSError(f"could not write candidate image: {candidate_path}")
            candidate_paths.append(candidate_path)
            manifest_rows.append(
                {
                    "hand": source.hand,
                    "view_id": view,
                    "candidate_index": rank,
                    "selected_rank": rank,
                    "session_id": row.session_id,
                    "sample_id": row.sample_id,
                    "physical_part_id": row.physical_part_id,
                    "source_path": str(row.source_path),
                    "review_image_path": candidate_path.relative_to(output_root).as_posix(),
                    "source_class": row.source_class,
                    "business_label": row.business_label,
                    "split": row.split,
                }
            )
        _write_contact_sheet(candidate_paths, view_root / "contact_sheet.jpg")

    with (output_root / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_root / "review_instructions.txt").write_text(
        "BMW Template 候选人工审核说明\n"
        "1. 逐视角查看 contact_sheet.jpg 和 candidate_*.png。\n"
        "2. 直接删除不适合作为正常模板的 candidate_*.png 副本。\n"
        "3. 不要删除 candidate_manifest.csv，也不要修改原始数据。\n"
        "4. 删除后不会自动补齐，候选编号允许有空缺。\n"
        "5. 每个视角至少保留 3 张；审核完成后再运行训练。\n",
        encoding="utf-8",
    )
    return output_root


__all__ = ["ReviewSource", "build_template_review_package"]
