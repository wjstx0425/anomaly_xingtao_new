# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Prepare single-part crops from fixed multi-part capture images.

This script turns full fixture images into a nested anomalib Folder-style
dataset where each output image contains one part. It can also mask the two
non-target holes in each crop so left/right-hand hole differences do not
dominate anomaly scores.

Example:
    source .venv/bin/activate

    python capture_data/prepare_part_crops.py \
        --data-root dataset/c789 \
        --output-root dataset/c789_parts \
        --hand left \
        --position top \
        --roi 460,30,3480,2600 \
        --preset c789_left_top_3x2
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")
LABELS = ("normal", "normal_test", "defect")


@dataclass(frozen=True)
class Box:
    """Axis-aligned crop box."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Return box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Return box height."""
        return self.y2 - self.y1

    def as_text(self) -> str:
        """Return x1,y1,x2,y2 text."""
        return f"{self.x1},{self.y1},{self.x2},{self.y2}"


@dataclass(frozen=True)
class SlotSpec:
    """One fixed part slot within the global ROI crop."""

    name: str
    row: int
    col: int
    box: Box


@dataclass(frozen=True)
class EllipseMask:
    """Crop-relative ellipse used to hide one hole."""

    cx: int
    cy: int
    rx: int
    ry: int
    angle: int = 0

    def as_text(self) -> str:
        """Return cx,cy,rx,ry,angle text."""
        return f"{self.cx},{self.cy},{self.rx},{self.ry},{self.angle}"


@dataclass(frozen=True)
class CropPreset:
    """Fixed crop and mask preset."""

    roi: Box
    slots: tuple[SlotSpec, ...]
    slot_hole_masks: dict[str, tuple[EllipseMask, ...]]


C789_LEFT_TOP_3X2 = CropPreset(
    roi=Box(460, 30, 3480, 2600),
    slots=(
        SlotSpec("slot01", 1, 1, Box(0, 0, 1510, 857)),
        SlotSpec("slot02", 1, 2, Box(1510, 0, 3020, 857)),
        SlotSpec("slot03", 2, 1, Box(0, 857, 1510, 1713)),
        SlotSpec("slot04", 2, 2, Box(1510, 857, 3020, 1713)),
        SlotSpec("slot05", 3, 1, Box(0, 1713, 1510, 2570)),
        SlotSpec("slot06", 3, 2, Box(1510, 1713, 3020, 2570)),
    ),
    slot_hole_masks={
        "slot01": (EllipseMask(482, 520, 55, 55), EllipseMask(1043, 519, 75, 55)),
        "slot02": (EllipseMask(521, 511, 55, 55), EllipseMask(1080, 513, 75, 55)),
        "slot03": (EllipseMask(479, 580, 55, 55), EllipseMask(1022, 580, 75, 55)),
        "slot04": (EllipseMask(528, 572, 55, 55), EllipseMask(1091, 573, 75, 55)),
        "slot05": (EllipseMask(484, 649, 55, 55), EllipseMask(1049, 650, 75, 55)),
        "slot06": (EllipseMask(536, 639, 55, 55), EllipseMask(1100, 636, 75, 55)),
    },
)
C789_LEFT_BOTTOM_3X2 = CropPreset(
    roi=Box(350, 320, 3600, 3030),
    slots=(
        SlotSpec("slot01", 1, 1, Box(0, 0, 1650, 780)),
        SlotSpec("slot02", 1, 2, Box(1600, 0, 3250, 780)),
        SlotSpec("slot03", 2, 1, Box(0, 780, 1650, 1730)),
        SlotSpec("slot04", 2, 2, Box(1600, 780, 3250, 1730)),
        SlotSpec("slot05", 3, 1, Box(0, 1730, 1650, 2710)),
        SlotSpec("slot06", 3, 2, Box(1600, 1730, 3250, 2710)),
    ),
    slot_hole_masks={
        "slot01": (EllipseMask(592, 232, 70, 65), EllipseMask(1148, 225, 85, 65)),
        "slot02": (EllipseMask(539, 228, 70, 65), EllipseMask(1097, 222, 85, 65)),
        "slot03": (EllipseMask(590, 368, 70, 65), EllipseMask(1134, 367, 85, 65)),
        "slot04": (EllipseMask(539, 362, 70, 65), EllipseMask(1108, 359, 85, 65)),
        "slot05": (EllipseMask(594, 344, 70, 65), EllipseMask(1160, 339, 85, 65)),
        "slot06": (EllipseMask(552, 338, 70, 65), EllipseMask(1114, 328, 85, 65)),
    },
)
FX11_NO_HAND_TOP_6X1 = CropPreset(
    roi=Box(0, 0, 4024, 3036),
    slots=(
        SlotSpec("slot01", 1, 1, Box(830, 40, 3230, 450)),
        SlotSpec("slot02", 2, 1, Box(830, 440, 3230, 860)),
        SlotSpec("slot03", 3, 1, Box(830, 890, 3230, 1300)),
        SlotSpec("slot04", 4, 1, Box(830, 1290, 3230, 1700)),
        SlotSpec("slot05", 5, 1, Box(830, 1700, 3230, 2110)),
        SlotSpec("slot06", 6, 1, Box(830, 2130, 3230, 2590)),
    ),
    slot_hole_masks={},
)
FX11_NO_HAND_BOTTOM_6X1 = CropPreset(
    roi=Box(0, 0, 4024, 3036),
    slots=(
        SlotSpec("slot01", 1, 1, Box(830, 10, 3230, 420)),
        SlotSpec("slot02", 2, 1, Box(830, 420, 3230, 840)),
        SlotSpec("slot03", 3, 1, Box(830, 870, 3230, 1280)),
        SlotSpec("slot04", 4, 1, Box(830, 1280, 3230, 1700)),
        SlotSpec("slot05", 5, 1, Box(830, 1700, 3230, 2110)),
        SlotSpec("slot06", 6, 1, Box(830, 2130, 3230, 2590)),
    ),
    slot_hole_masks={},
)
MANUAL_EMPTY = CropPreset(
    roi=Box(0, 0, 1, 1),
    slots=(),
    slot_hole_masks={},
)
PRESETS = {
    "c789_left_bottom_3x2": C789_LEFT_BOTTOM_3X2,
    "c789_left_top_3x2": C789_LEFT_TOP_3X2,
    "fx11_no_hand_bottom_6x1": FX11_NO_HAND_BOTTOM_6X1,
    "fx11_no_hand_top_6x1": FX11_NO_HAND_TOP_6X1,
    "manual": MANUAL_EMPTY,
}


def _parse_ints(value: str, expected: int, name: str) -> tuple[int, ...]:
    """Parse comma-separated integers."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != expected:
        msg = f"{name} must have {expected} comma-separated integers."
        raise argparse.ArgumentTypeError(msg)
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as error:
        msg = f"{name} values must be integers."
        raise argparse.ArgumentTypeError(msg) from error
    return values


def parse_box(value: str) -> Box:
    """Parse an x1,y1,x2,y2 box."""
    x1, y1, x2, y2 = _parse_ints(value, 4, "box")
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        msg = "box must satisfy x1 >= 0, y1 >= 0, x2 > x1, y2 > y1."
        raise argparse.ArgumentTypeError(msg)
    return Box(x1, y1, x2, y2)


def parse_ellipse(value: str) -> EllipseMask:
    """Parse a cx,cy,rx,ry[,angle] ellipse mask."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {4, 5}:
        msg = "ellipse must have 4 or 5 comma-separated integers."
        raise argparse.ArgumentTypeError(msg)
    try:
        cx, cy, rx, ry = (int(part) for part in parts[:4])
        angle = int(parts[4]) if len(parts) == 5 else 0
    except ValueError as error:
        msg = "ellipse values must be integers."
        raise argparse.ArgumentTypeError(msg) from error
    if cx < 0 or cy < 0 or rx <= 0 or ry <= 0:
        msg = "ellipse must satisfy cx >= 0, cy >= 0, rx > 0, ry > 0."
        raise argparse.ArgumentTypeError(msg)
    return EllipseMask(cx, cy, rx, ry, angle)


def parse_slot_ellipse(value: str) -> tuple[str, EllipseMask]:
    """Parse a slot-specific ellipse mask as slot:cx,cy,rx,ry[,angle]."""
    if ":" not in value:
        msg = "slot hole mask must be provided as slot:cx,cy,rx,ry[,angle]."
        raise argparse.ArgumentTypeError(msg)
    slot_name, ellipse = value.split(":", maxsplit=1)
    slot_name = slot_name.strip()
    if not slot_name:
        msg = "slot hole mask must include a slot name."
        raise argparse.ArgumentTypeError(msg)
    return slot_name, parse_ellipse(ellipse)


def parse_slot(value: str) -> SlotSpec:
    """Parse a slot specification as name:row,col,x1,y1,x2,y2."""
    if ":" not in value:
        msg = "slot must be provided as name:row,col,x1,y1,x2,y2."
        raise argparse.ArgumentTypeError(msg)
    name, rest = value.split(":", maxsplit=1)
    row, col, x1, y1, x2, y2 = _parse_ints(rest, 6, "slot")
    if row <= 0 or col <= 0:
        msg = "slot row and column must be positive."
        raise argparse.ArgumentTypeError(msg)
    return SlotSpec(name.strip(), row, col, parse_box(f"{x1},{y1},{x2},{y2}"))


def _iter_label_images(label_dir: Path) -> Iterable[Path]:
    """Yield flat or nested images below one label directory."""
    nested_images = sorted(path for path in label_dir.glob("**/images/*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    flat_images = sorted(
        path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    yield from nested_images
    yield from flat_images


def iter_source_images(data_root: Path, hand: str, position: str, labels: Sequence[str]) -> Iterable[tuple[str, Path]]:
    """Yield source label and image paths."""
    view_root = data_root / hand / position
    for label in labels:
        label_dir = view_root / label
        if not label_dir.is_dir():
            continue
        for image_path in _iter_label_images(label_dir):
            yield label, image_path


def view_name(hand: str, position: str) -> str:
    """Return the workflow view name for a hand/position pair."""
    return f"{hand}_{position}"


def sample_id_for_output(source_path: Path, slot: SlotSpec) -> str:
    """Return a stable output sample id for one source image and slot."""
    if source_path.parent.name == "images":
        source_sample_id = source_path.parent.parent.name
    else:
        source_sample_id = source_path.stem
    return f"{source_sample_id}_{slot.name}"


def frame_id_from_name(image_path: Path) -> str:
    """Return a frame id for manifest/debugging."""
    match = re.search(r"_(\d{6})$", image_path.stem)
    return match.group(1) if match else image_path.stem


def input_layout_for_image(image_path: Path, label_dir: Path) -> str:
    """Return a compact description of the source image layout."""
    if image_path.parent.name != "images":
        return "flat"
    relative = image_path.relative_to(label_dir)
    if len(relative.parts) == 2:
        return "label_images"
    if len(relative.parts) == 3:
        return "nested"
    return "deep_nested"


def defect_type_for_image(image_path: Path, label_dir: Path) -> str:
    """Infer defect type from nested path or flat filename prefix."""
    relative = image_path.relative_to(label_dir)
    if image_path.parent.name == "images" and len(relative.parts) >= 4:
        return relative.parts[0]
    stem = re.sub(r"_[1-9]\d*_[1-9]\d*(?:_\d+)?$", "", image_path.stem)
    return stem.split("_", maxsplit=1)[0] if stem else ""


def row_col_to_slot(slots: Sequence[SlotSpec], row: int, col: int) -> SlotSpec | None:
    """Return the slot matching a row/column pair."""
    return next((slot for slot in slots if slot.row == row and slot.col == col), None)


def defect_slots_from_name(image_path: Path, slots: Sequence[SlotSpec]) -> set[str]:
    """Infer the defective slot from a filename suffix.

    Multi-column presets use row/column suffixes like ``surface_3_2_2``.
    One-column presets use the first trailing number as the top-to-bottom slot,
    so ``deform_6_2`` maps to row 6.
    """
    if len({slot.col for slot in slots}) == 1:
        match = re.search(r"_(?P<row>[1-9]\d*)(?:_\d+)*$", image_path.stem)
        if not match:
            return set()
        slot = row_col_to_slot(slots, int(match.group("row")), slots[0].col)
        return {slot.name} if slot else set()

    match = re.search(r"_(?P<row>[1-9]\d*)_(?P<col>[1-9]\d*)(?:_\d+)?$", image_path.stem)
    if not match:
        return set()
    slot = row_col_to_slot(slots, int(match.group("row")), int(match.group("col")))
    return {slot.name} if slot else set()


def load_defect_slot_map(path: Path | None) -> dict[str, set[str]]:
    """Load optional filename-to-slot mapping CSV."""
    if path is None:
        return {}
    mapping: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.get("filename") or row.get("image") or row.get("source_path")
            slots = row.get("slots") or row.get("defect_slots")
            if not key or not slots:
                continue
            mapping[Path(key).name] = {slot.strip() for slot in re.split(r"[;,\s]+", slots) if slot.strip()}
    return mapping


def defect_slots_for_image(
    image_path: Path,
    slots: Sequence[SlotSpec],
    mode: str,
    mapping: dict[str, set[str]],
) -> set[str]:
    """Return slot names that should be labelled defect for one defect source image."""
    mapped = mapping.get(image_path.name) or mapping.get(str(image_path))
    if mapped is not None:
        return mapped
    if mode == "all":
        return {slot.name for slot in slots}
    if mode == "from-name":
        return defect_slots_from_name(image_path, slots)
    if mode == "skip-unmapped":
        return set()
    msg = f"Unsupported defect slot mode: {mode}"
    raise ValueError(msg)


def _assert_box_inside(box: Box, width: int, height: int, context: str) -> None:
    """Validate a box fits inside an image."""
    if box.x2 > width or box.y2 > height:
        msg = f"{context} {box.as_text()} is outside image size {width}x{height}."
        raise ValueError(msg)


def crop_slot(image: np.ndarray, roi: Box, slot: SlotSpec) -> np.ndarray:
    """Crop one slot image from a full source image."""
    height, width = image.shape[:2]
    _assert_box_inside(roi, width, height, "ROI")
    roi_image = image[roi.y1 : roi.y2, roi.x1 : roi.x2]
    _assert_box_inside(slot.box, roi.width, roi.height, f"slot {slot.name}")
    return roi_image[slot.box.y1 : slot.box.y2, slot.box.x1 : slot.box.x2].copy()


def hole_mask_for_crop(crop: np.ndarray, hole_masks: Sequence[EllipseMask]) -> np.ndarray:
    """Create a binary hole mask for one slot crop."""
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for ellipse in hole_masks:
        cv2.ellipse(mask, (ellipse.cx, ellipse.cy), (ellipse.rx, ellipse.ry), ellipse.angle, 0, 360, 255, -1)
    return mask


def mask_holes(crop: np.ndarray, hole_masks: Sequence[EllipseMask], method: str, inpaint_radius: int) -> np.ndarray:
    """Hide crop holes using inpainting or a stable median fill."""
    if not hole_masks:
        return crop
    mask = hole_mask_for_crop(crop, hole_masks)
    if method == "none":
        return crop
    if method == "inpaint":
        return cv2.inpaint(crop, mask, inpaint_radius, cv2.INPAINT_TELEA)
    if method == "median":
        output = crop.copy()
        background = crop[mask == 0]
        fill = (
            np.median(background, axis=0).astype(np.uint8)
            if len(background)
            else np.array([127, 127, 127], np.uint8)
        )
        output[mask > 0] = fill
        return output
    msg = f"Unsupported hole mask method: {method}"
    raise ValueError(msg)


def output_label_for_slot(
    source_label: str,
    slot: SlotSpec,
    defect_slot_names: set[str],
    nondefect_slots_from_defect_images: str,
) -> str | None:
    """Return the output label for one slot, or ``None`` to skip it."""
    if source_label in {"normal", "normal_test"}:
        return source_label
    if source_label != "defect":
        return None
    if slot.name in defect_slot_names:
        return "defect"
    if nondefect_slots_from_defect_images == "normal":
        return "normal"
    return None


def assign_normal_output_labels(
    images: Sequence[tuple[str, Path]],
    mode: str,
    normal_test_ratio: float,
    seed: int,
) -> dict[Path, str]:
    """Return output labels for source normal images that should be split.

    The split is made at full-image level so all slot crops from one source
    image stay in the same output split.
    """
    if mode == "never" or normal_test_ratio <= 0:
        return {}
    if not 0 <= normal_test_ratio < 1:
        msg = "--normal-test-ratio must be >= 0 and < 1."
        raise ValueError(msg)
    if mode not in {"auto", "always"}:
        msg = f"Unsupported normal split mode: {mode}"
        raise ValueError(msg)
    if mode == "auto" and any(label == "normal_test" for label, _ in images):
        return {}

    normal_paths = sorted(path for label, path in images if label == "normal")
    if len(normal_paths) < 2:
        return {}

    test_count = int(len(normal_paths) * normal_test_ratio + 0.5)
    test_count = min(max(1, test_count), len(normal_paths) - 1)
    shuffled = list(normal_paths)
    random.Random(seed).shuffle(shuffled)
    normal_test_paths = set(shuffled[:test_count])
    return {path: ("normal_test" if path in normal_test_paths else "normal") for path in normal_paths}


def unique_output_path(output_dir: Path, filename: str, overwrite: bool) -> Path:
    """Return a non-conflicting output path unless overwriting is enabled."""
    output_path = output_dir / filename
    if overwrite or not output_path.exists():
        return output_path

    stem = output_path.stem
    suffix = output_path.suffix
    index = 1
    while True:
        candidate = output_dir / f"{stem}_dup{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def hole_masks_for_slots(
    preset: CropPreset,
    slots: Sequence[SlotSpec],
    global_override: Sequence[EllipseMask] | None,
    slot_overrides: Sequence[tuple[str, EllipseMask]] | None,
) -> dict[str, tuple[EllipseMask, ...]]:
    """Return hole masks keyed by slot name."""
    if global_override:
        masks = {slot.name: tuple(global_override) for slot in slots}
    else:
        masks = {slot.name: tuple(preset.slot_hole_masks.get(slot.name, ())) for slot in slots}

    for slot_name, ellipse in slot_overrides or ():
        masks.setdefault(slot_name, ())
        masks[slot_name] = (*masks[slot_name], ellipse)
    return masks


def write_preview_overlay(image_path: Path, output_path: Path, roi: Box, slots: Sequence[SlotSpec]) -> None:
    """Write a full-image overlay showing ROI and slot boxes."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise RuntimeError(msg)

    overlay = image.copy()
    cv2.rectangle(overlay, (roi.x1, roi.y1), (roi.x2, roi.y2), (0, 255, 0), 4)
    for slot in slots:
        x1 = roi.x1 + slot.box.x1
        y1 = roi.y1 + slot.box.y1
        x2 = roi.x1 + slot.box.x2
        y2 = roi.y1 + slot.box.y2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 3)
        cv2.putText(overlay, slot.name, (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 200, 0), 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        msg = f"Could not write preview overlay: {output_path}"
        raise RuntimeError(msg)


def prepare_part_crops(args: argparse.Namespace) -> Path:
    """Crop a full-image dataset into a single-part dataset."""
    preset = PRESETS[args.preset]
    roi = args.roi or preset.roi
    slots = tuple(args.slot) if args.slot else preset.slots
    hole_masks_by_slot = hole_masks_for_slots(preset, slots, args.hole_mask, args.slot_hole_mask)
    output_root = args.output_root.resolve()
    output_hand = args.output_hand or args.hand
    output_position = args.output_position or args.position
    output_view_root = output_root / output_hand / output_position
    defect_slot_map = load_defect_slot_map(args.defect_slot_map)
    manifest_rows: list[dict[str, Any]] = []

    images = list(iter_source_images(args.data_root.resolve(), args.hand, args.position, args.labels))
    if not images:
        msg = f"No source images found under {args.data_root / args.hand / args.position}"
        raise RuntimeError(msg)

    if args.preview_overlay:
        write_preview_overlay(images[0][1], args.preview_overlay, roi, slots)

    normal_output_labels = assign_normal_output_labels(
        images,
        args.normal_split_mode,
        args.normal_test_ratio,
        args.normal_split_seed,
    )

    for source_label, source_path in images:
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Could not read image: {source_path}"
            raise RuntimeError(msg)
        source_height, source_width = image.shape[:2]
        label_dir = args.data_root.resolve() / args.hand / args.position / source_label

        defect_slot_names = defect_slots_for_image(source_path, slots, args.defect_slot_mode, defect_slot_map)
        effective_source_label = normal_output_labels.get(source_path, source_label)
        for slot in slots:
            output_label = output_label_for_slot(
                effective_source_label,
                slot,
                defect_slot_names,
                args.nondefect_slots_from_defect_images,
            )
            if output_label is None:
                continue

            crop = crop_slot(image, roi, slot)
            hole_masks = hole_masks_by_slot.get(slot.name, ())
            crop = mask_holes(crop, hole_masks, args.hole_mask_method, args.inpaint_radius)

            sample_id = sample_id_for_output(source_path, slot)
            output_dir = output_view_root / output_label / sample_id / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{source_path.stem}_{slot.name}{source_path.suffix.lower()}"
            output_path = unique_output_path(output_dir, output_name, args.overwrite)
            if not cv2.imwrite(str(output_path), crop):
                msg = f"Could not write crop: {output_path}"
                raise RuntimeError(msg)

            manifest_rows.append(
                {
                    "source_path": str(source_path.resolve()),
                    "source_label": source_label,
                    "processed_path": str(output_path.resolve()),
                    "view": view_name(output_hand, output_position),
                    "hand": output_hand,
                    "position": output_position,
                    "label": output_label,
                    "gt_label": 1 if output_label == "defect" else 0,
                    "split": "train" if output_label == "normal" else "test",
                    "sample_id": sample_id,
                    "frame_id": frame_id_from_name(source_path),
                    "slot": slot.name,
                    "slot_row": slot.row,
                    "slot_col": slot.col,
                    "defect_type": defect_type_for_image(source_path, label_dir) if source_label == "defect" else "",
                    "roi": roi.as_text(),
                    "slot_box": slot.box.as_text(),
                    "source_width": source_width,
                    "source_height": source_height,
                    "crop_width": crop.shape[1],
                    "crop_height": crop.shape[0],
                    "input_layout": input_layout_for_image(source_path, label_dir),
                    "hole_masks": ";".join(mask.as_text() for mask in hole_masks),
                    "hole_mask_method": args.hole_mask_method,
                    "defect_slot_names": ";".join(sorted(defect_slot_names)),
                },
            )

    if not manifest_rows:
        msg = "No crops were written. Check defect-slot mapping and selected labels."
        raise RuntimeError(msg)

    manifest_path = output_root / "part_crop_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts_path = output_root / "part_crop_counts.csv"
    count_rows: dict[tuple[str, str], int] = {}
    for row in manifest_rows:
        key = (str(row["view"]), str(row["label"]))
        count_rows[key] = count_rows.get(key, 0) + 1
    with counts_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["view", "label", "count"])
        writer.writeheader()
        for (view, label), count in sorted(count_rows.items()):
            writer.writerow({"view": view, "label": label, "count": count})
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Raw full-image dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output single-part dataset root.")
    parser.add_argument("--hand", default="left", help="Input/output hand directory, e.g. left or right.")
    parser.add_argument("--position", default="top", help="Input/output position directory, e.g. top.")
    parser.add_argument("--output-hand", help="Optional output hand directory. Defaults to --hand.")
    parser.add_argument("--output-position", help="Optional output position directory. Defaults to --position.")
    parser.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS), help="Labels to crop.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="c789_left_top_3x2", help="Crop/mask preset.")
    parser.add_argument("--roi", type=parse_box, help="Override preset ROI as x1,y1,x2,y2 in full-image coordinates.")
    parser.add_argument(
        "--slot",
        type=parse_slot,
        action="append",
        help="Override preset slots. Repeat as name:row,col,x1,y1,x2,y2 in ROI coordinates.",
    )
    parser.add_argument(
        "--hole-mask",
        type=parse_ellipse,
        action="append",
        help="Override preset hole masks for all slots. Repeat as cx,cy,rx,ry[,angle] in crop coordinates.",
    )
    parser.add_argument(
        "--slot-hole-mask",
        type=parse_slot_ellipse,
        action="append",
        help="Add a slot-specific hole mask. Repeat as slot:cx,cy,rx,ry[,angle].",
    )
    parser.add_argument(
        "--hole-mask-method",
        choices=("inpaint", "median", "none"),
        default="inpaint",
        help="How to hide hole regions.",
    )
    parser.add_argument("--inpaint-radius", type=int, default=9, help="OpenCV inpaint radius for hole masks.")
    parser.add_argument(
        "--defect-slot-mode",
        choices=("from-name", "all", "skip-unmapped"),
        default="from-name",
        help="How to label slots from defect full images when no map CSV is provided.",
    )
    parser.add_argument(
        "--defect-slot-map",
        type=Path,
        help="Optional CSV with filename/image/source_path and slots/defect_slots columns.",
    )
    parser.add_argument(
        "--nondefect-slots-from-defect-images",
        choices=("skip", "normal"),
        default="skip",
        help="What to do with non-defective slots in defect-labelled full images.",
    )
    parser.add_argument(
        "--normal-split-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "How to split source normal images into normal/normal_test. "
            "auto splits only when no source normal_test images exist."
        ),
    )
    parser.add_argument(
        "--normal-test-ratio",
        type=float,
        default=0.25,
        help="Fraction of source normal images written to normal_test when splitting. 0.25 gives a 3:1 split.",
    )
    parser.add_argument("--normal-split-seed", type=int, default=0, help="Random seed for the normal split.")
    parser.add_argument("--preview-overlay", type=Path, help="Optional output image showing ROI and slot boxes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop filenames.")
    return parser


def main() -> None:
    """Run part crop preparation."""
    args = build_parser().parse_args()
    manifest_path = prepare_part_crops(args)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
