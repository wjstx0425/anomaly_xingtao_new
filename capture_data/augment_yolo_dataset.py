# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Create bbox-aware offline augmentations for a YOLO detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    """One normalized YOLO detection box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Augmentation:
    """One deterministic image and bbox augmentation."""

    name: str
    apply: Callable[[np.ndarray, list[YoloBox]], tuple[np.ndarray, list[YoloBox]]]


def _read_labels(path: Path) -> list[YoloBox]:
    """Read a YOLO label file."""
    boxes: list[YoloBox] = []
    if not path.is_file():
        return boxes
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            msg = f"Expected 5 fields in {path}:{line_number}, got {text!r}"
            raise ValueError(msg)
        class_id, x_center, y_center, width, height = parts
        box = YoloBox(int(float(class_id)), float(x_center), float(y_center), float(width), float(height))
        values = (box.x_center, box.y_center, box.width, box.height)
        if any(value < 0.0 or value > 1.0 for value in values):
            msg = f"YOLO box values must be in [0, 1] in {path}:{line_number}: {text!r}"
            raise ValueError(msg)
        boxes.append(box)
    return boxes


def _write_labels(path: Path, boxes: Sequence[YoloBox]) -> None:
    """Write YOLO labels."""
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _box_to_xyxy(box: YoloBox, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert one normalized YOLO box to pixel xyxy."""
    x_center = box.x_center * width
    y_center = box.y_center * height
    box_width = box.width * width
    box_height = box.height * height
    return (
        x_center - box_width / 2.0,
        y_center - box_height / 2.0,
        x_center + box_width / 2.0,
        y_center + box_height / 2.0,
    )


def _xyxy_to_box(class_id: int, xyxy: tuple[float, float, float, float], width: int, height: int) -> YoloBox | None:
    """Convert clipped pixel xyxy to one normalized YOLO box."""
    x_min, y_min, x_max, y_max = xyxy
    x_min = max(0.0, min(float(width), x_min))
    x_max = max(0.0, min(float(width), x_max))
    y_min = max(0.0, min(float(height), y_min))
    y_max = max(0.0, min(float(height), y_max))
    if x_max - x_min < 2.0 or y_max - y_min < 2.0:
        return None
    return YoloBox(
        class_id=class_id,
        x_center=((x_min + x_max) / 2.0) / width,
        y_center=((y_min + y_max) / 2.0) / height,
        width=(x_max - x_min) / width,
        height=(y_max - y_min) / height,
    )


def _transform_boxes(boxes: Sequence[YoloBox], matrix: np.ndarray, width: int, height: int) -> list[YoloBox]:
    """Transform YOLO boxes by an affine matrix and return enclosing boxes."""
    transformed: list[YoloBox] = []
    for box in boxes:
        x_min, y_min, x_max, y_max = _box_to_xyxy(box, width, height)
        corners = np.array(
            [[[x_min, y_min]], [[x_max, y_min]], [[x_max, y_max]], [[x_min, y_max]]],
            dtype=np.float32,
        )
        points = cv2.transform(corners, matrix).reshape(4, 2)
        new_x_min = float(points[:, 0].min())
        new_y_min = float(points[:, 1].min())
        new_x_max = float(points[:, 0].max())
        new_y_max = float(points[:, 1].max())
        new_box = _xyxy_to_box(box.class_id, (new_x_min, new_y_min, new_x_max, new_y_max), width, height)
        if new_box is not None:
            transformed.append(new_box)
    return transformed


def _affine(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    matrix: np.ndarray,
) -> tuple[np.ndarray, list[YoloBox]]:
    """Apply an affine image transform and synchronize boxes."""
    height, width = image.shape[:2]
    output = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return output, _transform_boxes(boxes, matrix, width, height)


def _hflip(image: np.ndarray, boxes: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
    """Horizontally mirror an image and its boxes."""
    height, width = image.shape[:2]
    matrix = np.array([[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.flip(image, 1), _transform_boxes(boxes, matrix, width, height)


def _rotate(angle: float) -> Callable[[np.ndarray, list[YoloBox]], tuple[np.ndarray, list[YoloBox]]]:
    """Build a small rotation augmentation."""

    def apply(image: np.ndarray, boxes: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0).astype(np.float32)
        return _affine(image, boxes, matrix)

    return apply


def _scale_translate(image: np.ndarray, boxes: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
    """Apply a light scale and translation transform."""
    height, width = image.shape[:2]
    scale = 1.05
    tx = 0.03 * width
    ty = -0.03 * height
    matrix = np.array(
        [[scale, 0.0, (1.0 - scale) * width / 2.0 + tx], [0.0, scale, (1.0 - scale) * height / 2.0 + ty]],
        dtype=np.float32,
    )
    return _affine(image, boxes, matrix)


def _brightness_gamma(image: np.ndarray, boxes: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
    """Adjust brightness, contrast, and gamma without changing boxes."""
    adjusted = cv2.convertScaleAbs(image, alpha=1.12, beta=8)
    gamma = 0.9
    table = np.array([((index / 255.0) ** gamma) * 255 for index in range(256)], dtype=np.uint8)
    return cv2.LUT(adjusted, table), list(boxes)


def _blur(image: np.ndarray, boxes: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
    """Apply a light blur without changing boxes."""
    return cv2.GaussianBlur(image, (3, 3), 0), list(boxes)


def _noise(rng: np.random.Generator) -> Callable[[np.ndarray, list[YoloBox]], tuple[np.ndarray, list[YoloBox]]]:
    """Build a light Gaussian-noise augmentation."""

    def apply(image: np.ndarray, boxes: list[YoloBox]) -> tuple[np.ndarray, list[YoloBox]]:
        noise = rng.normal(0.0, 5.0, image.shape).astype(np.float32)
        output = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return output, list(boxes)

    return apply


def build_augmentations(seed: int) -> list[Augmentation]:
    """Return the C789 YOLO offline augmentation set."""
    rng = np.random.default_rng(seed)
    return [
        Augmentation("orig", lambda image, boxes: (image.copy(), list(boxes))),
        Augmentation("hflip", _hflip),
        Augmentation("rot_m5", _rotate(-5.0)),
        Augmentation("rot_p5", _rotate(5.0)),
        Augmentation("bright_gamma", _brightness_gamma),
        Augmentation("blur", _blur),
        Augmentation("noise", _noise(rng)),
        Augmentation("scale_translate", _scale_translate),
    ]


def _iter_images(input_root: Path) -> list[Path]:
    """Return image files from a YOLO image directory."""
    images_dir = input_root / "images"
    if not images_dir.is_dir():
        msg = f"Input root must contain an images directory: {input_root}"
        raise ValueError(msg)
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _draw_preview(image: np.ndarray, boxes: Sequence[YoloBox], path: Path) -> None:
    """Write a bbox preview image."""
    height, width = image.shape[:2]
    preview = image.copy()
    for box in boxes:
        x_min, y_min, x_max, y_max = _box_to_xyxy(box, width, height)
        cv2.rectangle(
            preview,
            (int(round(x_min)), int(round(y_min))),
            (int(round(x_max)), int(round(y_max))),
            (0, 0, 255),
            2,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def _write_data_yaml(output_root: Path) -> None:
    """Write a train-only YOLO data.yaml for smoke training."""
    content = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/train",
            "names:",
            "  0: defect",
            "",
        ],
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def augment_yolo_dataset(
    input_root: Path,
    output_root: Path,
    *,
    seed: int = 0,
    preview_limit: int = 80,
    overwrite: bool = False,
) -> dict[str, int]:
    """Create an augmented train split from a flat YOLO image/label directory."""
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            msg = f"Output root is non-empty: {output_root}. Pass --overwrite to replace it."
            raise ValueError(msg)
        shutil.rmtree(output_root)
    images = _iter_images(input_root)
    if not images:
        msg = f"No images found under {input_root / 'images'}"
        raise ValueError(msg)
    labels_dir = input_root / "labels"
    output_images = output_root / "images" / "train"
    output_labels = output_root / "labels" / "train"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    preview_dir = output_root / "previews"
    manifest_rows: list[dict[str, Any]] = []
    augmentations = build_augmentations(seed)
    preview_count = 0

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = _read_labels(label_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Could not read image: {image_path}"
            raise ValueError(msg)
        for augmentation in augmentations:
            augmented_image, augmented_boxes = augmentation.apply(image, boxes)
            if boxes and not augmented_boxes:
                continue
            output_name = f"{image_path.stem}__{augmentation.name}{image_path.suffix.lower()}"
            output_image = output_images / output_name
            output_label = output_labels / f"{Path(output_name).stem}.txt"
            if not cv2.imwrite(str(output_image), augmented_image):
                msg = f"Could not write image: {output_image}"
                raise RuntimeError(msg)
            _write_labels(output_label, augmented_boxes)
            if preview_count < preview_limit:
                _draw_preview(augmented_image, augmented_boxes, preview_dir / output_name)
                preview_count += 1
            manifest_rows.append(
                {
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "output_image": str(output_image),
                    "output_label": str(output_label),
                    "variant": augmentation.name,
                    "input_boxes": len(boxes),
                    "output_boxes": len(augmented_boxes),
                },
            )

    with (output_root / "augmentation_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(manifest_rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    classes_path = input_root / "classes.txt"
    if classes_path.is_file():
        shutil.copy2(classes_path, output_root / "classes.txt")
    notes_path = input_root / "notes.json"
    if notes_path.is_file():
        shutil.copy2(notes_path, output_root / "notes.json")
    else:
        (output_root / "notes.json").write_text(
            json.dumps({"categories": [{"id": 0, "name": "defect"}]}, indent=2) + "\n",
            encoding="utf-8",
        )
    _write_data_yaml(output_root)
    return {
        "source_images": len(images),
        "augmentations_per_image": len(augmentations),
        "output_images": len(manifest_rows),
        "previews": preview_count,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the augmentation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="YOLO root with flat images/ and labels/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output YOLO root.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic random seed for noise.")
    parser.add_argument("--preview-limit", type=int, default=80, help="Maximum bbox preview images to write.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output root.")
    return parser


def main() -> None:
    """Run offline YOLO augmentation."""
    args = build_parser().parse_args()
    summary = augment_yolo_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        seed=args.seed,
        preview_limit=args.preview_limit,
        overwrite=args.overwrite,
    )
    print(f"augmented YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
