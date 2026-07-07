# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build ROI-level YOLO datasets from flat C789 slot-level YOLO exports."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
GROUP_PATTERN = re.compile(r"(g\d{3})", re.IGNORECASE)
SLOT_PATTERN = re.compile(r"(slot\d+)", re.IGNORECASE)
CLASS_NAMES = ("defect",)


@dataclass(frozen=True)
class XYXYBox:
    """One pixel-space xyxy bounding box."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def x_center(self) -> float:
        """Return bbox center x."""
        return (self.x1 + self.x2) / 2.0

    @property
    def y_center(self) -> float:
        """Return bbox center y."""
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        """Return bbox width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return bbox height."""
        return self.y2 - self.y1


@dataclass(frozen=True)
class SlotSample:
    """One slot crop image plus parsed metadata and bbox labels."""

    image_path: Path
    label_path: Path | None
    group_id: str
    slot_id: str
    split: str
    width: int
    height: int
    boxes: tuple[XYXYBox, ...]


@dataclass(frozen=True)
class RoiTemplate:
    """One ROI rectangle in slot coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    source_bbox_index: int


def _clean_text(value: object) -> str | None:
    """Return a stripped string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    """Create or clean the output root."""
    if output_root.exists() and not output_root.is_dir():
        msg = f"Output root exists but is not a directory: {output_root}"
        raise ValueError(msg)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            msg = f"Output root is non-empty: {output_root}. Pass --overwrite to replace it."
            raise ValueError(msg)
        output_root = output_root.resolve(strict=False)
        forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if output_root in forbidden:
            msg = f"Refusing to overwrite unsafe output root: {output_root}"
            raise ValueError(msg)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _parse_group_id(path: Path) -> str:
    """Parse ``gNNN`` group id from filename."""
    match = GROUP_PATTERN.search(path.stem)
    if match is None:
        msg = f"Could not parse group_id gNNN from filename: {path.name}"
        raise ValueError(msg)
    return match.group(1).lower()


def _parse_slot_id(path: Path) -> str:
    """Parse ``slotNN`` id from filename."""
    match = SLOT_PATTERN.search(path.stem)
    if match is None:
        msg = f"Could not parse slot_id slotNN from filename: {path.name}"
        raise ValueError(msg)
    return match.group(1).lower()


def _yolo_label_path(root: Path, image_path: Path) -> Path:
    """Return the YOLO label path matching one image."""
    images_root = root / "images"
    try:
        relative = image_path.relative_to(images_root)
    except ValueError:
        return root / "labels" / f"{image_path.stem}.txt"
    return root / "labels" / relative.with_suffix(".txt")


def _split_for_image_path(root: Path, image_path: Path, group_id: str, val_groups: set[str]) -> str:
    """Return an explicit split directory or derive one from the group id."""
    try:
        relative = image_path.relative_to(root / "images")
    except ValueError:
        return "val" if group_id in val_groups else "train"
    if relative.parts and relative.parts[0] in {"train", "val", "test"}:
        return relative.parts[0]
    return "val" if group_id in val_groups else "train"


def _load_image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height."""
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise ValueError(msg)
    return image.shape[1], image.shape[0]


def _parse_label_line(line: str, width: int, height: int) -> XYXYBox:
    """Convert one YOLO normalized label line to pixel xyxy."""
    parts = line.split()
    if len(parts) != 5:
        msg = f"Expected YOLO label line with 5 columns, got: {line!r}"
        raise ValueError(msg)
    class_id, x_center, y_center, box_width, box_height = parts
    x_center_px = float(x_center) * width
    y_center_px = float(y_center) * height
    box_width_px = float(box_width) * width
    box_height_px = float(box_height) * height
    x1 = x_center_px - (box_width_px / 2.0)
    y1 = y_center_px - (box_height_px / 2.0)
    x2 = x_center_px + (box_width_px / 2.0)
    y2 = y_center_px + (box_height_px / 2.0)
    return XYXYBox(class_id=int(class_id), x1=x1, y1=y1, x2=x2, y2=y2)


def _load_yolo_boxes(label_path: Path | None, width: int, height: int) -> tuple[XYXYBox, ...]:
    """Load YOLO labels from one ``.txt`` file."""
    if label_path is None or not label_path.is_file():
        return ()
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    return tuple(_parse_label_line(line, width, height) for line in text.splitlines())


def _iter_image_paths(root: Path) -> Iterable[Path]:
    """Yield image paths from ``root/images`` in stable order."""
    image_root = root / "images"
    if not image_root.is_dir():
        msg = f"Missing images directory: {image_root}"
        raise ValueError(msg)
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _load_slot_samples(root: Path, *, val_groups: set[str], expect_labels: bool) -> list[SlotSample]:
    """Load slot samples from a flat YOLO root."""
    samples: list[SlotSample] = []
    for image_path in _iter_image_paths(root):
        width, height = _load_image_size(image_path)
        label_path = _yolo_label_path(root, image_path)
        if expect_labels and not label_path.is_file():
            msg = f"Missing label file for input image: {label_path}"
            raise ValueError(msg)
        group_id = _parse_group_id(image_path)
        slot_id = _parse_slot_id(image_path)
        split = _split_for_image_path(root, image_path, group_id, val_groups)
        if split == "test":
            continue
        samples.append(
            SlotSample(
                image_path=image_path.resolve(),
                label_path=label_path.resolve() if label_path.is_file() else None,
                group_id=group_id,
                slot_id=slot_id,
                split=split,
                width=width,
                height=height,
                boxes=_load_yolo_boxes(label_path if label_path.is_file() else None, width, height),
            ),
        )
    return samples


def _clamp_roi_centered(center_x: float, center_y: float, roi_size: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Return an in-bounds square ROI centered on a bbox center when possible."""
    if roi_size <= 0:
        msg = f"roi_size must be positive, got {roi_size}"
        raise ValueError(msg)
    if width < roi_size or height < roi_size:
        msg = f"Slot crop {width}x{height} is smaller than roi_size={roi_size}"
        raise ValueError(msg)
    half = roi_size / 2.0
    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))
    x1 = min(max(0, x1), width - roi_size)
    y1 = min(max(0, y1), height - roi_size)
    return x1, y1, x1 + roi_size, y1 + roi_size


def _tile_starts(length: int, roi_size: int, stride: int) -> list[int]:
    """Return tile start positions that cover one axis."""
    if stride <= 0:
        msg = f"stride must be positive, got {stride}"
        raise ValueError(msg)
    if length < roi_size:
        msg = f"Slot dimension {length} is smaller than roi_size={roi_size}"
        raise ValueError(msg)
    starts = list(range(0, max(length - roi_size, 0) + 1, stride))
    last = length - roi_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _box_center_in_roi(box: XYXYBox, roi_box: tuple[int, int, int, int]) -> bool:
    """Return whether the box center falls inside one ROI rectangle."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
    return roi_x1 <= box.x_center < roi_x2 and roi_y1 <= box.y_center < roi_y2


def _box_fully_in_roi(box: XYXYBox, roi_box: tuple[int, int, int, int]) -> bool:
    """Return whether the full box is contained in one ROI rectangle."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
    return roi_x1 <= box.x1 and box.x2 <= roi_x2 and roi_y1 <= box.y1 and box.y2 <= roi_y2


def _boxes_in_roi(boxes: Sequence[XYXYBox], roi_box: tuple[int, int, int, int]) -> list[XYXYBox]:
    """Return boxes that are fully contained in one ROI rectangle."""
    return [
        box
        for box in boxes
        if _box_center_in_roi(box, roi_box) and _box_fully_in_roi(box, roi_box)
    ]


def _normalize_box_to_roi(box: XYXYBox, roi_box: tuple[int, int, int, int]) -> str:
    """Return one YOLO normalized label line in ROI coordinates."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
    x1 = max(0.0, min(box.x1 - roi_x1, float(roi_width)))
    y1 = max(0.0, min(box.y1 - roi_y1, float(roi_height)))
    x2 = max(0.0, min(box.x2 - roi_x1, float(roi_width)))
    y2 = max(0.0, min(box.y2 - roi_y1, float(roi_height)))
    if x2 <= x1 or y2 <= y1:
        msg = f"Clipped ROI bbox has non-positive area: {box}"
        raise ValueError(msg)
    x_center = ((x1 + x2) / 2.0) / roi_width
    y_center = ((y1 + y2) / 2.0) / roi_height
    width = (x2 - x1) / roi_width
    height = (y2 - y1) / roi_height
    return f"{box.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _write_data_yaml(output_root: Path) -> None:
    """Write a minimal Ultralytics dataset yaml."""
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: defect",
    ]
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preview(image_path: Path, roi_box: tuple[int, int, int, int], boxes: Sequence[XYXYBox], output_path: Path) -> None:
    """Write a preview ROI image with bbox overlays."""
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image for preview: {image_path}"
        raise ValueError(msg)
    x1, y1, x2, y2 = roi_box
    roi_image = image[y1:y2, x1:x2].copy()
    for box in boxes:
        top_left = (int(round(box.x1 - x1)), int(round(box.y1 - y1)))
        bottom_right = (int(round(box.x2 - x1)), int(round(box.y2 - y1)))
        cv2.rectangle(roi_image, top_left, bottom_right, (0, 0, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), roi_image):
        msg = f"Could not write preview: {output_path}"
        raise RuntimeError(msg)


def _write_roi_crop(source_image: Path, roi_box: tuple[int, int, int, int], output_path: Path) -> None:
    """Crop and write one ROI image."""
    image = cv2.imread(str(source_image))
    if image is None:
        msg = f"Could not read source image: {source_image}"
        raise ValueError(msg)
    x1, y1, x2, y2 = roi_box
    crop = image[y1:y2, x1:x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop):
        msg = f"Could not write ROI image: {output_path}"
        raise RuntimeError(msg)


def _roi_name(sample: SlotSample, mode: str, kind: str, index: int) -> str:
    """Return a stable ROI output filename."""
    return f"{sample.image_path.stem}_{mode}_{kind}_{index:04d}.png"


def _record_row(
    *,
    roi_image_path: Path,
    roi_label_path: Path,
    sample: SlotSample,
    split: str,
    roi_box: tuple[int, int, int, int],
    source_bbox_index: int,
    included_boxes: Sequence[XYXYBox],
    kind: str,
) -> dict[str, str]:
    """Build one ROI manifest row."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
    return {
        "roi_image_path": str(roi_image_path.resolve()),
        "roi_label_path": str(roi_label_path.resolve()),
        "slot_processed_path": str(sample.image_path.resolve()),
        "slot_label_path": str(sample.label_path.resolve()) if sample.label_path is not None else "",
        "split": split,
        "group_id": sample.group_id,
        "slot_id": sample.slot_id,
        "roi_x1_in_slot": str(roi_x1),
        "roi_y1_in_slot": str(roi_y1),
        "roi_x2_in_slot": str(roi_x2),
        "roi_y2_in_slot": str(roi_y2),
        "source_bbox_index": str(source_bbox_index),
        "included_box_count": str(len(included_boxes)),
        "kind": kind,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write CSV rows with headers."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_defect_templates_gt_center(
    sample: SlotSample,
    roi_size: int,
) -> tuple[list[tuple[RoiTemplate, list[XYXYBox]]], list[dict[str, str]]]:
    """Build one ROI per bbox in gt-center mode."""
    templates: list[tuple[RoiTemplate, list[XYXYBox]]] = []
    review_rows: list[dict[str, str]] = []
    for index, box in enumerate(sample.boxes):
        roi_box = _clamp_roi_centered(box.x_center, box.y_center, roi_size, sample.width, sample.height)
        if not _box_fully_in_roi(box, roi_box):
            review_rows.append(
                {
                    "slot_processed_path": str(sample.image_path.resolve()),
                    "slot_label_path": str(sample.label_path.resolve()) if sample.label_path is not None else "",
                    "split": sample.split,
                    "group_id": sample.group_id,
                    "slot_id": sample.slot_id,
                    "source_bbox_index": str(index),
                    "bbox_width": f"{box.width:.6f}",
                    "bbox_height": f"{box.height:.6f}",
                    "roi_size": str(roi_size),
                    "reason": "source_bbox_not_fully_contained_in_gt_center_roi",
                },
            )
            continue
        included = _boxes_in_roi(sample.boxes, roi_box)
        templates.append(
            (
                RoiTemplate(
                    x1=roi_box[0],
                    y1=roi_box[1],
                    x2=roi_box[2],
                    y2=roi_box[3],
                    source_bbox_index=index,
                ),
                included,
            ),
        )
    return templates, review_rows


def _build_defect_templates_tile(sample: SlotSample, roi_size: int, stride: int) -> list[tuple[RoiTemplate, list[XYXYBox]]]:
    """Build tiled ROIs that cover one slot crop."""
    templates: list[tuple[RoiTemplate, list[XYXYBox]]] = []
    tile_index = 0
    for y1 in _tile_starts(sample.height, roi_size, stride):
        for x1 in _tile_starts(sample.width, roi_size, stride):
            roi_box = (x1, y1, x1 + roi_size, y1 + roi_size)
            templates.append(
                (
                    RoiTemplate(x1=x1, y1=y1, x2=x1 + roi_size, y2=y1 + roi_size, source_bbox_index=tile_index),
                    _boxes_in_roi(sample.boxes, roi_box),
                ),
            )
            tile_index += 1
    return templates


def _write_roi_example(
    *,
    output_root: Path,
    sample: SlotSample,
    split: str,
    kind: str,
    mode: str,
    roi_index: int,
    roi_box: tuple[int, int, int, int],
    included_boxes: Sequence[XYXYBox],
    source_bbox_index: int,
    preview_budget: dict[tuple[str, str], int],
) -> dict[str, str]:
    """Write one ROI image, label file, preview, and manifest row."""
    output_name = _roi_name(sample, mode, kind, roi_index)
    roi_image_path = output_root / "images" / split / output_name
    roi_label_path = output_root / "labels" / split / f"{Path(output_name).stem}.txt"
    _write_roi_crop(sample.image_path, roi_box, roi_image_path)
    label_lines = [_normalize_box_to_roi(box, roi_box) for box in included_boxes]
    roi_label_path.parent.mkdir(parents=True, exist_ok=True)
    roi_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    budget_key = (split, kind)
    if preview_budget.get(budget_key, 0) > 0:
        _write_preview(sample.image_path, roi_box, included_boxes, output_root / "previews" / split / kind / output_name)
        preview_budget[budget_key] -= 1

    return _record_row(
        roi_image_path=roi_image_path,
        roi_label_path=roi_label_path,
        sample=sample,
        split=split,
        roi_box=roi_box,
        source_bbox_index=source_bbox_index,
        included_boxes=included_boxes,
        kind=kind,
    )


def build_roi_yolo_dataset(
    *,
    input_root: Path,
    output_root: Path,
    mode: str,
    roi_size: int = 512,
    stride: int = 256,
    val_groups: Sequence[str] = ("g002", "g008"),
    normal_source_root: Path | None = None,
    train_normal_ratio: int = 2,
    val_normal_limit: int = 150,
    preview_limit: int = 24,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build an ROI-level YOLO dataset for ``gt-center`` or ``tile`` mode."""
    if mode not in {"gt-center", "tile"}:
        msg = "mode must be one of: gt-center, tile"
        raise ValueError(msg)
    if train_normal_ratio < 0:
        msg = "train_normal_ratio must be non-negative"
        raise ValueError(msg)
    if val_normal_limit < 0:
        msg = "val_normal_limit must be non-negative"
        raise ValueError(msg)
    if preview_limit < 0:
        msg = "preview_limit must be non-negative"
        raise ValueError(msg)

    normalized_val_groups = {group.lower() for group in val_groups}
    rng = random.Random(seed)
    defect_samples = _load_slot_samples(input_root, val_groups=normalized_val_groups, expect_labels=True)
    normal_samples = (
        _load_slot_samples(normal_source_root, val_groups=normalized_val_groups, expect_labels=False)
        if normal_source_root is not None and mode == "gt-center"
        else []
    )

    _prepare_output_root(output_root, overwrite=overwrite)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    roi_review_rows: list[dict[str, str]] = []
    preview_budget = {
        (split, kind): preview_limit for split in ("train", "val") for kind in ("defect_roi", "normal_roi")
    }
    counts = Counter()
    templates_by_split_slot: dict[tuple[str, str], list[RoiTemplate]] = defaultdict(list)

    for sample in defect_samples:
        if mode == "gt-center":
            templates, review_rows = _build_defect_templates_gt_center(sample, roi_size)
            roi_review_rows.extend(review_rows)
        else:
            templates = _build_defect_templates_tile(sample, roi_size, stride)
        for roi_index, (template, included_boxes) in enumerate(templates):
            row = _write_roi_example(
                output_root=output_root,
                sample=sample,
                split=sample.split,
                kind="defect_roi",
                mode=mode,
                roi_index=roi_index,
                roi_box=(template.x1, template.y1, template.x2, template.y2),
                included_boxes=included_boxes,
                source_bbox_index=template.source_bbox_index,
                preview_budget=preview_budget,
            )
            manifest_rows.append(row)
            counts[f"{sample.split}:defect_roi"] += 1
            templates_by_split_slot[(sample.split, sample.slot_id)].append(template)

    if mode == "gt-center" and normal_samples:
        train_samples_by_slot: dict[str, list[SlotSample]] = defaultdict(list)
        val_samples_by_slot: dict[str, list[SlotSample]] = defaultdict(list)
        for sample in normal_samples:
            if sample.split == "train":
                train_samples_by_slot[sample.slot_id].append(sample)
            else:
                val_samples_by_slot[sample.slot_id].append(sample)

        for slot_id, templates in sorted((key[1], value) for key, value in templates_by_split_slot.items() if key[0] == "train"):
            slot_samples = train_samples_by_slot.get(slot_id, [])
            if not slot_samples:
                continue
            target_count = len(templates) * train_normal_ratio
            for roi_index in range(target_count):
                sample = slot_samples[roi_index % len(slot_samples)]
                template = templates[roi_index % len(templates)]
                row = _write_roi_example(
                    output_root=output_root,
                    sample=sample,
                    split="train",
                    kind="normal_roi",
                    mode=mode,
                    roi_index=roi_index,
                    roi_box=(template.x1, template.y1, template.x2, template.y2),
                    included_boxes=(),
                    source_bbox_index=-1,
                    preview_budget=preview_budget,
                )
                manifest_rows.append(row)
                counts["train:normal_roi"] += 1

        remaining_val_normals = val_normal_limit
        for slot_id, templates in sorted((key[1], value) for key, value in templates_by_split_slot.items() if key[0] == "val"):
            if remaining_val_normals <= 0:
                break
            slot_samples = val_samples_by_slot.get(slot_id, [])
            if not slot_samples:
                continue
            target_count = min(remaining_val_normals, len(slot_samples))
            sample_order = slot_samples[:]
            rng.shuffle(sample_order)
            for roi_index in range(target_count):
                sample = sample_order[roi_index]
                template = templates[roi_index % len(templates)]
                row = _write_roi_example(
                    output_root=output_root,
                    sample=sample,
                    split="val",
                    kind="normal_roi",
                    mode=mode,
                    roi_index=roi_index,
                    roi_box=(template.x1, template.y1, template.x2, template.y2),
                    included_boxes=(),
                    source_bbox_index=-1,
                    preview_budget=preview_budget,
                )
                manifest_rows.append(row)
                counts["val:normal_roi"] += 1
                remaining_val_normals -= 1

    if not manifest_rows:
        msg = "No ROI samples were generated."
        raise RuntimeError(msg)

    _write_csv(output_root / "roi_manifest.csv", manifest_rows)
    summary_rows = [
        {"split": split, "kind": kind, "count": str(counts.get(f"{split}:{kind}", 0))}
        for split in ("train", "val")
        for kind in ("defect_roi", "normal_roi")
    ]
    _write_csv(output_root / "dataset_summary.csv", summary_rows)

    leakage_rows = []
    groups_by_source: dict[str, set[str]] = defaultdict(set)
    splits_by_source: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        source_key = row["slot_processed_path"]
        groups_by_source[source_key].add(row["group_id"])
        splits_by_source[source_key].add(row["split"])
    for source_key in sorted(groups_by_source):
        leakage_rows.append(
            {
                "slot_processed_path": source_key,
                "group_count": str(len(groups_by_source[source_key])),
                "split_count": str(len(splits_by_source[source_key])),
                "status": "ok" if len(splits_by_source[source_key]) == 1 else "review",
            },
        )
    _write_csv(output_root / "leakage_report.csv", leakage_rows)
    _write_csv(output_root / "roi_review_report.csv", roi_review_rows)
    _write_data_yaml(output_root)

    return {
        "mode": mode,
        "counts_by_split_kind": dict(counts),
        "roi_review_rows": len(roi_review_rows),
        "output_root": str(output_root.resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the ROI YOLO dataset CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="Flat YOLO dataset root with images/ and labels/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output ROI YOLO dataset root.")
    parser.add_argument("--mode", choices=("gt-center", "tile"), required=True, help="ROI generation mode.")
    parser.add_argument("--roi-size", type=int, default=512, help="Square ROI size in pixels.")
    parser.add_argument("--stride", type=int, default=256, help="Tile stride in pixels for tile mode.")
    parser.add_argument(
        "--val-groups",
        nargs="+",
        default=["g002", "g008"],
        help="Group ids reserved for validation.",
    )
    parser.add_argument("--normal-source-root", type=Path, help="Optional flat YOLO-like root used to sample normal ROIs.")
    parser.add_argument("--train-normal-ratio", type=int, default=2, help="Train normal ROI count per train defect ROI.")
    parser.add_argument("--val-normal-limit", type=int, default=150, help="Maximum validation normal ROI count.")
    parser.add_argument("--preview-limit", type=int, default=24, help="Preview image budget per split/kind.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output root.")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    return build_arg_parser()


def main() -> None:
    """Run the ROI YOLO dataset builder."""
    args = build_arg_parser().parse_args()
    summary = build_roi_yolo_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        mode=args.mode,
        roi_size=args.roi_size,
        stride=args.stride,
        val_groups=args.val_groups,
        normal_source_root=args.normal_source_root,
        train_normal_ratio=args.train_normal_ratio,
        val_normal_limit=args.val_normal_limit,
        preview_limit=args.preview_limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"roi dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
