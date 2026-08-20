# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN401, C901, EM101, EM102, TRY003

"""Build and run a minimal tiled YOLO workflow for ZS32 ROI images."""

# ruff: noqa: ANN401, C901, EM101, EM102, TRY003

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class PixelBox:
    """One class-aware box in source-image pixel coordinates."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        """Box area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        """Box center in pixels."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


@dataclass(frozen=True, slots=True)
class TileWindow:
    """One fixed-grid tile before right/bottom padding."""

    x1: int
    y1: int
    x2: int
    y2: int
    tile_size: int

    @property
    def valid_width(self) -> int:
        """Non-padding width."""
        return self.x2 - self.x1

    @property
    def valid_height(self) -> int:
        """Non-padding height."""
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class TileDetection:
    """One detector result in tile-local coordinates."""

    tile_x1: int
    tile_y1: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    valid_width: int = 1280
    valid_height: int = 1280


@dataclass(frozen=True, slots=True)
class MergedDetection:
    """One detector result mapped to original ROI coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class _PredictModel(Protocol):
    """Minimal Ultralytics-compatible prediction protocol."""

    def predict(self, source: Any, **kwargs: Any) -> Sequence[Any]:
        """Run object detection."""


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return deterministic starts that cover an axis and anchor its final tile."""
    if length <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("length, tile_size, and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride cannot exceed tile_size")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _tile_windows(width: int, height: int, tile_size: int, stride: int) -> list[TileWindow]:
    return [
        TileWindow(x, y, min(x + tile_size, width), min(y + tile_size, height), tile_size)
        for y in tile_starts(height, tile_size, stride)
        for x in tile_starts(width, tile_size, stride)
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_boxes(label_path: Path, width: int, height: int) -> list[PixelBox]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        class_id_text, x_center_text, y_center_text, box_width_text, box_height_text = parts
        class_id = int(class_id_text)
        x_center, y_center, box_width, box_height = map(
            float,
            (x_center_text, y_center_text, box_width_text, box_height_text),
        )
        values = (x_center, y_center, box_width, box_height)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Non-finite YOLO label at {label_path}:{line_number}")
        if class_id != 0 or not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            raise ValueError(f"Out-of-contract YOLO label at {label_path}:{line_number}")
        if box_width <= 0 or box_height <= 0 or box_width > 1 or box_height > 1:
            raise ValueError(f"Invalid YOLO box size at {label_path}:{line_number}")
        x1 = (x_center - box_width / 2.0) * width
        y1 = (y_center - box_height / 2.0) * height
        x2 = (x_center + box_width / 2.0) * width
        y2 = (y_center + box_height / 2.0) * height
        tolerance = 1e-3
        if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
            raise ValueError(f"YOLO box exceeds image bounds at {label_path}:{line_number}")
        boxes.append(PixelBox(class_id, max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)))
    return boxes


def _intersection(box: PixelBox, window: TileWindow) -> PixelBox | None:
    x1 = max(box.x1, float(window.x1))
    y1 = max(box.y1, float(window.y1))
    x2 = min(box.x2, float(window.x2))
    y2 = min(box.y2, float(window.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return PixelBox(box.class_id, x1, y1, x2, y2)


def _visible_ratio(box: PixelBox, window: TileWindow) -> float:
    intersection = _intersection(box, window)
    return 0.0 if intersection is None or box.area <= 0 else intersection.area / box.area


def _box_center_in_window(box: PixelBox, window: TileWindow) -> bool:
    center_x, center_y = box.center
    return window.x1 <= center_x <= window.x2 and window.y1 <= center_y <= window.y2


def _best_window(box: PixelBox, windows: Sequence[TileWindow]) -> TileWindow:
    candidates = [window for window in windows if _box_center_in_window(box, window)]
    if not candidates:
        raise ValueError(f"No tile contains box center: {box}")
    return max(
        candidates,
        key=lambda window: (
            _visible_ratio(box, window),
            min(
                box.center[0] - window.x1,
                window.x2 - box.center[0],
                box.center[1] - window.y1,
                window.y2 - box.center[1],
            ),
            -window.y1,
            -window.x1,
        ),
    )


def _normalize_box(box: PixelBox, window: TileWindow) -> str:
    intersection = _intersection(box, window)
    if intersection is None:
        raise ValueError("Cannot normalize a box outside its tile")
    x1 = intersection.x1 - window.x1
    y1 = intersection.y1 - window.y1
    x2 = intersection.x2 - window.x1
    y2 = intersection.y2 - window.y1
    x_center = ((x1 + x2) / 2.0) / window.tile_size
    y_center = ((y1 + y2) / 2.0) / window.tile_size
    width = (x2 - x1) / window.tile_size
    height = (y2 - y1) / window.tile_size
    return f"{box.class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


def _padded_tile(image: np.ndarray, window: TileWindow, padding_value: int) -> np.ndarray:
    tile = np.full((window.tile_size, window.tile_size, 3), padding_value, dtype=np.uint8)
    crop = image[window.y1 : window.y2, window.x1 : window.x2]
    tile[: crop.shape[0], : crop.shape[1]] = crop
    return tile


def _normal_windows(
    windows: Sequence[TileWindow],
    *,
    count: int,
    seed: int,
    identity: str,
) -> list[TileWindow]:
    if count <= 0:
        return []
    ranked = sorted(
        windows,
        key=lambda window: hashlib.sha256(
            f"{seed}:{identity}:{window.x1}:{window.y1}".encode(),
        ).digest(),
    )
    return ranked[: min(count, len(ranked))]


def build_tiled_zs32_yolo_dataset(
    *,
    input_root: Path,
    output_root: Path,
    tile_size: int = 1280,
    stride: int = 960,
    min_visible_ratio: float = 0.8,
    normal_tiles_per_image: int = 1,
    padding_value: int = 114,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a compact tiled dataset while preserving the existing grouped split."""
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"Refusing to overwrite existing tiled dataset: {output_root}")
    if not 0 < min_visible_ratio <= 1:
        raise ValueError("min_visible_ratio must be in (0, 1]")
    if normal_tiles_per_image < 0:
        raise ValueError("normal_tiles_per_image cannot be negative")
    if not 0 <= padding_value <= 255:
        raise ValueError("padding_value must be in [0, 255]")

    mapping_path = input_root / "mapping.csv"
    rows = _read_csv(mapping_path)
    if not rows:
        raise ValueError(f"No source rows found in {mapping_path}")
    required = {
        "new_image_path",
        "new_label_path",
        "sample_id",
        "view",
        "hand",
        "defect_type",
        "session_id",
        "group_id",
        "split",
        "source_kind",
        "pixel_sha256",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Input mapping is missing columns: {sorted(missing)}")

    manifest_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    source_split: dict[str, str] = {}
    pixel_split: dict[str, str] = {}
    minimum_visible = 1.0
    for row in rows:
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Invalid split in mapping: {split}")
        for key, registry in ((row["sample_id"], source_split), (row["pixel_sha256"], pixel_split)):
            previous = registry.setdefault(key, split)
            if previous != split:
                raise ValueError(f"Input leakage across splits: {key}")
        image_path = Path(row["new_image_path"])
        label_path = Path(row["new_label_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode source image: {image_path}")
        height, width = image.shape[:2]
        boxes = _load_boxes(label_path, width, height)
        windows = _tile_windows(width, height, tile_size, stride)
        if row["source_kind"] == "trusted_normal":
            if boxes:
                raise ValueError(f"Trusted normal label must be empty: {label_path}")
            selected_windows = _normal_windows(
                windows,
                count=normal_tiles_per_image,
                seed=seed,
                identity=f"{row['pixel_sha256']}:{row['view']}:{split}",
            )
            assigned_by_window: dict[TileWindow, list[PixelBox]] = {window: [] for window in selected_windows}
            tile_kind = "trusted_normal"
        else:
            if not boxes:
                raise ValueError(f"Defect-positive label must be non-empty: {label_path}")
            assigned_by_window = defaultdict(list)
            for box in boxes:
                window = _best_window(box, windows)
                visible_ratio = _visible_ratio(box, window)
                minimum_visible = min(minimum_visible, visible_ratio)
                if visible_ratio < min_visible_ratio:
                    raise ValueError(
                        f"Best tile visibility {visible_ratio:.6f} is below {min_visible_ratio:.6f}: {image_path}",
                    )
                assigned_by_window[window].append(box)
            selected_windows = sorted(assigned_by_window, key=lambda window: (window.y1, window.x1))
            tile_kind = "defect_positive"

        for tile_index, window in enumerate(selected_windows):
            tile_name = f"{image_path.stem}__tile_x{window.x1:04d}_y{window.y1:04d}.png"
            output_image = output_root / "images" / split / tile_name
            output_label = output_root / "labels" / split / f"{Path(tile_name).stem}.txt"
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_image), _padded_tile(image, window, padding_value)):
                raise OSError(f"Could not write tile image: {output_image}")
            tile_boxes = assigned_by_window[window]
            label_lines = [_normalize_box(box, window) for box in tile_boxes]
            output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            counts[f"{split}:images"] += 1
            counts[f"{split}:{tile_kind}"] += 1
            counts[f"{split}:boxes"] += len(tile_boxes)
            manifest_rows.append(
                {
                    "source_image_path": str(image_path),
                    "source_label_path": str(label_path),
                    "tile_image_path": str(output_image),
                    "tile_label_path": str(output_label),
                    "sample_id": row["sample_id"],
                    "view": row["view"],
                    "hand": row["hand"],
                    "defect_type": row["defect_type"],
                    "session_id": row["session_id"],
                    "group_id": row["group_id"],
                    "split": split,
                    "source_kind": row["source_kind"],
                    "pixel_sha256": row["pixel_sha256"],
                    "tile_index": tile_index,
                    "tile_x1": window.x1,
                    "tile_y1": window.y1,
                    "tile_x2": window.x2,
                    "tile_y2": window.y2,
                    "valid_width": window.valid_width,
                    "valid_height": window.valid_height,
                    "padding_right": tile_size - window.valid_width,
                    "padding_bottom": tile_size - window.valid_height,
                    "box_count": len(tile_boxes),
                },
            )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    manifest_columns = tuple(manifest_rows[0])
    _write_csv(output_root / "tile_manifest.csv", manifest_rows, manifest_columns)
    split_summary = {
        split: {
            "images": counts[f"{split}:images"],
            "positive_tiles": counts[f"{split}:defect_positive"],
            "trusted_normal_tiles": counts[f"{split}:trusted_normal"],
            "boxes": counts[f"{split}:boxes"],
        }
        for split in SPLITS
    }
    summary: dict[str, Any] = {
        "images": len(manifest_rows),
        "positive_tiles": sum(value["positive_tiles"] for value in split_summary.values()),
        "trusted_normal_tiles": sum(value["trusted_normal_tiles"] for value in split_summary.values()),
        "boxes": sum(value["boxes"] for value in split_summary.values()),
        "source_images": len(rows),
        "tile_size": tile_size,
        "stride": stride,
        "overlap": tile_size - stride,
        "min_visible_ratio": min_visible_ratio,
        "minimum_achieved_visible_ratio": minimum_visible,
        "normal_tiles_per_image": normal_tiles_per_image,
        "padding_value": padding_value,
        "seed": seed,
        "split_inherited": True,
        "splits": split_summary,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def remap_and_merge_detections(
    detections: Sequence[TileDetection],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    iou_threshold: float = 0.5,
) -> list[MergedDetection]:
    """Map tile detections to source coordinates and apply global NMS.

    Image dimensions are optional for callers that only need offset remapping.
    When supplied, they clip detections to the source-image boundary.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if image_width is not None and image_width <= 0:
        raise ValueError("image_width must be positive when provided")
    if image_height is not None and image_height <= 0:
        raise ValueError("image_height must be positive when provided")
    mapped = []
    for detection in detections:
        center_x = (detection.x1 + detection.x2) / 2.0
        center_y = (detection.y1 + detection.y2) / 2.0
        if not 0 <= center_x < detection.valid_width or not 0 <= center_y < detection.valid_height:
            continue
        x1 = max(0.0, detection.x1 + detection.tile_x1)
        y1 = max(0.0, detection.y1 + detection.tile_y1)
        x2 = max(0.0, detection.x2 + detection.tile_x1)
        y2 = max(0.0, detection.y2 + detection.tile_y1)
        if image_width is not None:
            x1 = min(x1, float(image_width))
            x2 = min(x2, float(image_width))
        if image_height is not None:
            y1 = min(y1, float(image_height))
            y2 = min(y2, float(image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        mapped.append(MergedDetection(x1, y1, x2, y2, detection.confidence, detection.class_id))
    if not mapped:
        return []
    boxes_xywh = [[item.x1, item.y1, item.x2 - item.x1, item.y2 - item.y1] for item in mapped]
    scores = [item.confidence for item in mapped]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    kept = [mapped[int(index)] for index in np.asarray(indices).reshape(-1)] if len(indices) else []
    return sorted(kept, key=lambda item: item.confidence, reverse=True)


def _discover_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported source image: {source}")
        return [source.resolve()]
    if not source.is_dir():
        raise FileNotFoundError(f"Inference source does not exist: {source}")
    images = sorted(path.resolve() for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found below inference source: {source}")
    return images


def run_tiled_inference(
    *,
    model: _PredictModel,
    source: Path,
    output_dir: Path,
    tile_size: int = 1280,
    stride: int = 960,
    confidence: float = 0.01,
    tile_iou: float = 0.7,
    merge_iou: float = 0.5,
    batch: int = 16,
    device: str = "0",
    padding_value: int = 114,
) -> dict[str, Any]:
    """Run grid inference and publish source-coordinate merged predictions."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite existing tiled inference output: {output_dir}")
    images = _discover_images(source)
    prediction_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode inference image: {image_path}")
        height, width = image.shape[:2]
        windows = _tile_windows(width, height, tile_size, stride)
        tiles = [_padded_tile(image, window, padding_value) for window in windows]
        results = model.predict(
            source=tiles,
            imgsz=tile_size,
            conf=confidence,
            iou=tile_iou,
            batch=batch,
            device=device,
            verbose=False,
        )
        if len(results) != len(windows):
            raise ValueError(f"Model returned {len(results)} results for {len(windows)} tiles")
        tile_detections = []
        for window, result in zip(windows, results, strict=True):
            boxes = result.boxes
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            for coordinates, score, class_id in zip(xyxy, scores, classes, strict=True):
                tile_detections.append(
                    TileDetection(
                        tile_x1=window.x1,
                        tile_y1=window.y1,
                        valid_width=window.valid_width,
                        valid_height=window.valid_height,
                        x1=float(coordinates[0]),
                        y1=float(coordinates[1]),
                        x2=float(coordinates[2]),
                        y2=float(coordinates[3]),
                        confidence=float(score),
                        class_id=int(class_id),
                    ),
                )
        merged = remap_and_merge_detections(
            tile_detections,
            image_width=width,
            image_height=height,
            iou_threshold=merge_iou,
        )
        image_rows.append(
            {
                "source_image_path": str(image_path),
                "width": width,
                "height": height,
                "tile_count": len(windows),
                "raw_detection_count": len(tile_detections),
                "merged_detection_count": len(merged),
            },
        )
        for detection_index, detection in enumerate(merged):
            prediction_rows.append(
                {
                    "source_image_path": str(image_path),
                    "detection_index": detection_index,
                    **asdict(detection),
                },
            )

    output_dir.mkdir(parents=True)
    prediction_columns = (
        "source_image_path",
        "detection_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "class_id",
    )
    _write_csv(output_dir / "predictions.csv", prediction_rows, prediction_columns)
    _write_csv(output_dir / "image_summary.csv", image_rows, tuple(image_rows[0]))
    summary = {
        "images": len(images),
        "tiles": sum(int(row["tile_count"]) for row in image_rows),
        "raw_detections": sum(int(row["raw_detection_count"]) for row in image_rows),
        "merged_detections": len(prediction_rows),
        "tile_size": tile_size,
        "stride": stride,
        "overlap": tile_size - stride,
        "confidence": confidence,
        "tile_iou": tile_iou,
        "merge_iou": merge_iou,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_ultralytics_model(model_path: Path) -> _PredictModel:
    """Load an Ultralytics model lazily so dataset building has no trainer dependency."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is required only for the Stage 37 infer command") from error
    return YOLO(str(model_path.resolve()))
