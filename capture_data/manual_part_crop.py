# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Manually select fixed part crop boxes and build a part-crop dataset.

Example:
    source .venv/bin/activate

    .venv/bin/python capture_data/manual_part_crop.py \
        --data-root dataset/fx11_2 \
        --output-root dataset/fx11_2_parts_manual \
        --hand no_hand \
        --position top \
        --slot-count 6 \
        --order vertical \
        --save-overlay results/fx11_2/manual_part_crop_preview.png \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.prepare_part_crops import (
    IMAGE_EXTENSIONS,
    LABELS,
    Box,
    SlotSpec,
    iter_source_images,
    parse_slot,
    prepare_part_crops,
    write_preview_overlay,
)


WINDOW_NAME = "Select part boxes"


def _read_image_size(image_path: Path) -> tuple[int, int]:
    """Return image width and height."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise RuntimeError(msg)
    height, width = image.shape[:2]
    return width, height


def _reference_images(data_root: Path, hand: str, position: str, label: str) -> list[Path]:
    """Return candidate reference images from one label directory."""
    return [
        path
        for source_label, path in iter_source_images(data_root, hand, position, [label])
        if source_label == label
    ]


def resolve_reference_image(args: argparse.Namespace) -> Path:
    """Resolve the image used for manual box selection."""
    if args.image is not None:
        return args.image.resolve()

    images = _reference_images(args.data_root.resolve(), args.hand, args.position, args.reference_label)
    if not images:
        msg = f"No {args.reference_label} images found under {args.data_root / args.hand / args.position}."
        raise RuntimeError(msg)
    if args.reference_index >= len(images):
        msg = f"--reference-index {args.reference_index} is outside {len(images)} available images."
        raise RuntimeError(msg)
    return images[args.reference_index].resolve()


def _display_image(image, max_width: int, max_height: int):
    """Return an image scaled for display and the display scale."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        size = (round(width * scale), round(height * scale))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale
    return image.copy(), 1.0


def select_boxes(
    image_path: Path,
    slot_count: int,
    max_window_width: int,
    max_window_height: int,
) -> list[Box]:
    """Open an OpenCV window and let the user select crop boxes."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {image_path}"
        raise RuntimeError(msg)
    height, width = image.shape[:2]
    display, scale = _display_image(image, max_window_width, max_window_height)

    print(f"Reference image: {image_path}")
    print(f"Original size: width={width}, height={height}")
    if scale < 1.0:
        print(f"Display scale: {scale:.4f}")
    print(f"Draw {slot_count} boxes around the parts.")
    print("Press Enter/Space after all boxes are selected. Press c to clear, Esc to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    selections = cv2.selectROIs(WINDOW_NAME, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW_NAME)

    boxes = []
    for x, y, box_width, box_height in selections:
        if box_width <= 0 or box_height <= 0:
            continue
        x1 = round(int(x) / scale)
        y1 = round(int(y) / scale)
        x2 = round((int(x) + int(box_width)) / scale)
        y2 = round((int(y) + int(box_height)) / scale)
        boxes.append(Box(max(0, x1), max(0, y1), min(width, x2), min(height, y2)))

    if len(boxes) != slot_count:
        msg = f"Expected {slot_count} boxes, but got {len(boxes)}."
        raise RuntimeError(msg)
    return boxes


def order_boxes(boxes: list[Box], order: str) -> list[Box]:
    """Order selected boxes into slot order."""
    if order == "selection":
        return boxes
    if order == "vertical":
        return sorted(boxes, key=lambda box: ((box.y1 + box.y2) / 2, (box.x1 + box.x2) / 2))
    if order == "horizontal":
        return sorted(boxes, key=lambda box: ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2))
    if order == "row-major":
        return sorted(boxes, key=lambda box: ((box.y1 + box.y2) / 2, (box.x1 + box.x2) / 2))
    msg = f"Unsupported order: {order}"
    raise ValueError(msg)


def build_slots(boxes: list[Box], order: str, columns: int, prefix: str) -> list[SlotSpec]:
    """Create SlotSpec objects from selected boxes."""
    ordered_boxes = order_boxes(boxes, order)
    slots = []
    for index, box in enumerate(ordered_boxes, start=1):
        if order == "vertical":
            row, col = index, 1
        elif order == "horizontal":
            row, col = 1, index
        else:
            row = (index - 1) // columns + 1
            col = (index - 1) % columns + 1
        slots.append(SlotSpec(f"{prefix}{index:02d}", row, col, box))
    return slots


def save_slots_csv(path: Path, slots: list[SlotSpec]) -> None:
    """Save selected slots for later inspection or reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["name", "row", "col", "x1", "y1", "x2", "y2", "slot_arg"])
        writer.writeheader()
        for slot in slots:
            slot_arg = (
                f"{slot.name}:{slot.row},{slot.col},"
                f"{slot.box.x1},{slot.box.y1},{slot.box.x2},{slot.box.y2}"
            )
            writer.writerow(
                {
                    "name": slot.name,
                    "row": slot.row,
                    "col": slot.col,
                    "x1": slot.box.x1,
                    "y1": slot.box.y1,
                    "x2": slot.box.x2,
                    "y2": slot.box.y2,
                    "slot_arg": slot_arg,
                },
            )


def _print_slot_args(slots: list[SlotSpec]) -> None:
    """Print reusable --slot arguments."""
    print()
    print("Reusable slot arguments:")
    for slot in slots:
        print(
            "  --slot "
            f"{slot.name}:{slot.row},{slot.col},{slot.box.x1},{slot.box.y1},{slot.box.x2},{slot.box.y2}",
        )


def run(args: argparse.Namespace) -> Path | None:
    """Run manual selection and optionally crop the dataset."""
    reference_image = resolve_reference_image(args)
    width, height = _read_image_size(reference_image)
    if args.slot:
        slots = list(args.slot)
        print(f"Using {len(slots)} slot boxes from --slot arguments.")
    else:
        boxes = select_boxes(reference_image, args.slot_count, args.max_window_width, args.max_window_height)
        slots = build_slots(boxes, args.order, args.columns, args.slot_prefix)
    roi = Box(0, 0, width, height)

    slots_csv = args.slots_csv or args.output_root / "manual_part_slots.csv"
    save_slots_csv(slots_csv, slots)
    print(f"Slots saved: {slots_csv}")
    _print_slot_args(slots)

    if args.save_overlay:
        write_preview_overlay(reference_image, args.save_overlay, roi, slots)
        print(f"Overlay saved: {args.save_overlay}")

    if args.no_crop:
        return None

    crop_args = argparse.Namespace(
        data_root=args.data_root,
        output_root=args.output_root,
        hand=args.hand,
        position=args.position,
        output_hand=None,
        output_position=None,
        labels=args.labels,
        preset="manual",
        roi=roi,
        slot=slots,
        hole_mask=[],
        slot_hole_mask=[],
        hole_mask_method=args.hole_mask_method,
        inpaint_radius=args.inpaint_radius,
        defect_slot_mode=args.defect_slot_mode,
        defect_slot_map=args.defect_slot_map,
        nondefect_slots_from_defect_images=args.nondefect_slots_from_defect_images,
        normal_split_mode=args.normal_split_mode,
        normal_test_ratio=args.normal_test_ratio,
        normal_split_seed=args.normal_split_seed,
        preview_overlay=None,
        overwrite=args.overwrite,
    )
    manifest_path = prepare_part_crops(crop_args)
    print(f"Manifest: {manifest_path}")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Raw full-image dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output single-part dataset root.")
    parser.add_argument("--hand", default="no_hand", help="Input/output hand directory.")
    parser.add_argument("--position", default="top", help="Input/output position directory.")
    parser.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS), help="Labels to crop.")
    parser.add_argument("--image", type=Path, help="Optional reference image. Defaults to a dataset image.")
    parser.add_argument(
        "--reference-label",
        choices=LABELS,
        default="normal",
        help="Label used to choose reference image.",
    )
    parser.add_argument("--reference-index", type=int, default=0, help="Reference image index within the chosen label.")
    parser.add_argument("--slot-count", type=int, default=6, help="Number of part boxes to select.")
    parser.add_argument(
        "--order",
        choices=("vertical", "horizontal", "row-major", "selection"),
        default="vertical",
        help="How selected boxes are converted to slot order.",
    )
    parser.add_argument("--columns", type=int, default=1, help="Column count for row-major order.")
    parser.add_argument("--slot-prefix", default="slot", help="Slot name prefix.")
    parser.add_argument(
        "--slot",
        type=parse_slot,
        action="append",
        help="Use a fixed slot box and skip GUI selection. Repeat as name:row,col,x1,y1,x2,y2 in full-image coordinates.",
    )
    parser.add_argument("--max-window-width", type=int, default=1600, help="Maximum display window width.")
    parser.add_argument("--max-window-height", type=int, default=1000, help="Maximum display window height.")
    parser.add_argument("--save-overlay", type=Path, help="Optional full-resolution overlay of selected boxes.")
    parser.add_argument("--slots-csv", type=Path, help="Optional CSV path for selected slot coordinates.")
    parser.add_argument(
        "--hole-mask-method",
        choices=("inpaint", "median", "none"),
        default="none",
        help="Hole masking.",
    )
    parser.add_argument("--inpaint-radius", type=int, default=9, help="OpenCV inpaint radius for hole masks.")
    parser.add_argument(
        "--defect-slot-mode",
        choices=("from-name", "all", "skip-unmapped"),
        default="from-name",
        help="How to label slots from defect full images.",
    )
    parser.add_argument("--defect-slot-map", type=Path, help="Optional filename-to-slot map CSV.")
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
    parser.add_argument("--no-crop", action="store_true", help="Only save selected boxes; do not crop the dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop filenames.")
    return parser


def main() -> None:
    """Run manual part crop selection."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
