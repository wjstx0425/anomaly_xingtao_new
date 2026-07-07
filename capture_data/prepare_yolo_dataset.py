# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export C789 part-crop manifests and bbox annotations to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


CLASS_NAMES = ("defect",)
YOLO_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class BBoxAnnotation:
    """One pixel-space bounding box annotation for a part crop."""

    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def _clean_text(value: Any) -> str | None:
    """Return a stripped string or ``None`` for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_value(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    """Return the first non-empty value from candidate CSV columns."""
    for name in names:
        value = _clean_text(row.get(name))
        if value is not None:
            return value
    return None


def _path_aliases(path_text: str) -> set[str]:
    """Return stable aliases used to match annotation rows to manifest rows."""
    path = Path(path_text)
    aliases = {path_text}
    try:
        aliases.add(str(path.resolve(strict=False)))
    except OSError:
        pass
    return {alias for alias in aliases if alias}


def _manifest_aliases(row: Mapping[str, Any]) -> set[str]:
    """Return image aliases from one part-crop manifest row."""
    aliases: set[str] = set()
    for name in ("processed_path", "image_path", "path", "file_path"):
        value = _clean_text(row.get(name))
        if value is not None:
            aliases.update(_path_aliases(value))
    value = _clean_text(row.get("sample_id"))
    if value is not None:
        aliases.add(value)
    return aliases


def _annotation_aliases(row: Mapping[str, Any]) -> set[str]:
    """Return aliases from one annotation row."""
    aliases: set[str] = set()
    for name in ("processed_path", "image_path", "path", "file_path", "filename", "image"):
        value = _clean_text(row.get(name))
        if value is not None:
            aliases.update(_path_aliases(value))
    value = _clean_text(row.get("sample_id"))
    if value is not None:
        aliases.add(value)
    return aliases


def _class_id(row: Mapping[str, Any], class_names: Sequence[str]) -> int:
    """Resolve a zero-based YOLO class id from annotation columns."""
    value = _first_value(row, ("class_id", "label_id"))
    if value is not None:
        class_id = int(float(value))
        if class_id < 0 or class_id >= len(class_names):
            msg = f"class_id {class_id} is outside configured classes {list(class_names)}"
            raise ValueError(msg)
        return class_id
    class_name = _first_value(row, ("class_name", "class", "label", "rectanglelabels", "defect_type"))
    if class_name is None:
        return 0
    normalized = class_name.strip()
    if normalized not in class_names:
        msg = f"Unsupported YOLO class {normalized!r}; expected one of {list(class_names)}"
        raise ValueError(msg)
    return list(class_names).index(normalized)


def _bbox_from_row(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Read bbox coordinates from common pixel CSV schemas."""
    xyxy_names = ("x_min", "y_min", "x_max", "y_max")
    if all(_clean_text(row.get(name)) is not None for name in xyxy_names):
        return tuple(float(str(row[name])) for name in xyxy_names)  # type: ignore[return-value]

    xyxy_aliases = ("xmin", "ymin", "xmax", "ymax")
    if all(_clean_text(row.get(name)) is not None for name in xyxy_aliases):
        return tuple(float(str(row[name])) for name in xyxy_aliases)  # type: ignore[return-value]

    xywh_names = ("x", "y", "width", "height")
    if all(_clean_text(row.get(name)) is not None for name in xywh_names):
        x = float(str(row["x"]))
        y = float(str(row["y"]))
        width = float(str(row["width"]))
        height = float(str(row["height"]))
        return x, y, x + width, y + height

    msg = "Annotation row must contain x_min/y_min/x_max/y_max or x/y/width/height columns"
    raise ValueError(msg)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_bbox_annotations(path: Path, class_names: Sequence[str] = CLASS_NAMES) -> dict[str, list[BBoxAnnotation]]:
    """Load bbox annotations keyed by any image alias found in the annotation CSV."""
    annotations_by_alias: dict[str, list[BBoxAnnotation]] = defaultdict(list)
    for row in _read_csv_rows(path):
        x_min, y_min, x_max, y_max = _bbox_from_row(row)
        if x_max <= x_min or y_max <= y_min:
            msg = f"Invalid bbox with non-positive size in {path}: {row}"
            raise ValueError(msg)
        aliases = _annotation_aliases(row)
        if not aliases:
            msg = f"Annotation row has no image identifier columns: {row}"
            raise ValueError(msg)
        annotation = BBoxAnnotation(
            class_id=_class_id(row, class_names),
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
        for alias in aliases:
            annotations_by_alias[alias].append(annotation)
    return dict(annotations_by_alias)


def _annotations_for_manifest_row(
    row: Mapping[str, Any],
    annotations_by_alias: Mapping[str, list[BBoxAnnotation]],
) -> list[BBoxAnnotation]:
    """Return annotations matching a manifest row."""
    annotations: list[BBoxAnnotation] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for alias in _manifest_aliases(row):
        for annotation in annotations_by_alias.get(alias, []):
            key = (annotation.class_id, annotation.x_min, annotation.y_min, annotation.x_max, annotation.y_max)
            if key in seen:
                continue
            annotations.append(annotation)
            seen.add(key)
    return annotations


def _positive_split(row: Mapping[str, Any], val_ratio: float, test_ratio: float, seed: int) -> str:
    """Assign an annotated defect row to train/val/test deterministically by source sample."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        msg = "positive val/test ratios must be non-negative and sum to less than 1"
        raise ValueError(msg)
    key = _first_value(row, ("source_path", "sample_id", "processed_path", "frame_id")) or "unknown"
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def _split_for_row(
    row: Mapping[str, Any],
    annotations: Sequence[BBoxAnnotation],
    *,
    positive_val_ratio: float,
    positive_test_ratio: float,
    seed: int,
    normal_test_split: str,
) -> str:
    """Resolve the YOLO split for a manifest row."""
    explicit = _clean_text(row.get("yolo_split"))
    if explicit is not None:
        if explicit not in YOLO_SPLITS:
            msg = f"Unsupported yolo_split {explicit!r}; expected one of {YOLO_SPLITS}"
            raise ValueError(msg)
        return explicit
    if annotations:
        return _positive_split(row, positive_val_ratio, positive_test_ratio, seed)
    label = (_clean_text(row.get("label")) or "").lower()
    if label == "normal_test":
        return normal_test_split
    return "train"


def _image_path_for_row(row: Mapping[str, Any]) -> Path:
    """Return the processed crop path for a manifest row."""
    value = _first_value(row, ("processed_path", "image_path", "path", "file_path"))
    if value is None:
        msg = f"Manifest row is missing processed_path: {row}"
        raise ValueError(msg)
    return Path(value)


def _image_size(row: Mapping[str, Any], image_path: Path) -> tuple[int, int]:
    """Return crop width/height from manifest columns or by reading the image."""
    width = _clean_text(row.get("crop_width"))
    height = _clean_text(row.get("crop_height"))
    if width is not None and height is not None:
        return int(float(width)), int(float(height))
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image to determine size: {image_path}"
        raise ValueError(msg)
    return image.shape[1], image.shape[0]


def _normalize_bbox(annotation: BBoxAnnotation, width: int, height: int) -> str:
    """Return one YOLO normalized xywh label line."""
    if width <= 0 or height <= 0:
        msg = f"Image dimensions must be positive, got {width}x{height}"
        raise ValueError(msg)
    x_min = max(0.0, min(annotation.x_min, float(width)))
    x_max = max(0.0, min(annotation.x_max, float(width)))
    y_min = max(0.0, min(annotation.y_min, float(height)))
    y_max = max(0.0, min(annotation.y_max, float(height)))
    if x_max <= x_min or y_max <= y_min:
        msg = f"Clipped bbox has non-positive size for image {width}x{height}: {annotation}"
        raise ValueError(msg)
    x_center = ((x_min + x_max) / 2.0) / width
    y_center = ((y_min + y_max) / 2.0) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return (
        f"{annotation.class_id} {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def _safe_image_name(row: Mapping[str, Any], image_path: Path, used: set[str]) -> str:
    """Return a collision-safe image filename for YOLO output folders."""
    candidate = image_path.name
    if candidate not in used:
        used.add(candidate)
        return candidate
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    candidate = f"{Path(candidate).stem}_{digest}{image_path.suffix.lower()}"
    used.add(candidate)
    return candidate


def _write_data_yaml(output_root: Path, class_names: Sequence[str], include_test: bool) -> None:
    """Write an Ultralytics dataset YAML file."""
    lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend(["names:", *[f"  {index}: {name}" for index, name in enumerate(class_names)]])
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preview(image_path: Path, annotations: Sequence[BBoxAnnotation], output_path: Path) -> None:
    """Write a bbox overlay preview for annotation review."""
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Could not read image for preview: {image_path}"
        raise ValueError(msg)
    for annotation in annotations:
        top_left = (int(round(annotation.x_min)), int(round(annotation.y_min)))
        bottom_right = (int(round(annotation.x_max)), int(round(annotation.y_max)))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        msg = f"Could not write preview: {output_path}"
        raise RuntimeError(msg)


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    """Create or clean the YOLO output root before exporting."""
    if output_root.exists() and not output_root.is_dir():
        msg = f"Output root exists but is not a directory: {output_root}"
        raise ValueError(msg)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            msg = f"Output root is non-empty: {output_root}. Pass --overwrite to replace it."
            raise ValueError(msg)
        resolved = output_root.resolve(strict=False)
        forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if resolved in forbidden:
            msg = f"Refusing to overwrite unsafe output root: {output_root}"
            raise ValueError(msg)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def export_yolo_dataset(
    manifest_path: Path,
    annotations_path: Path,
    output_root: Path,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    positive_val_ratio: float = 0.2,
    positive_test_ratio: float = 0.0,
    seed: int = 0,
    normal_test_split: str = "val",
    on_unlabeled_defect: str = "error",
    preview_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Export a C789 part-crop manifest to Ultralytics YOLO detection format."""
    if normal_test_split not in YOLO_SPLITS:
        msg = f"normal_test_split must be one of {YOLO_SPLITS}"
        raise ValueError(msg)
    if on_unlabeled_defect not in {"error", "skip", "empty"}:
        msg = "on_unlabeled_defect must be one of: error, skip, empty"
        raise ValueError(msg)

    manifest_rows = _read_csv_rows(manifest_path)
    annotations_by_alias = load_bbox_annotations(annotations_path, class_names)
    _prepare_output_root(output_root, overwrite=overwrite)
    for split in YOLO_SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {split: 0 for split in YOLO_SPLITS}
    export_rows: list[dict[str, str]] = []
    used_names: dict[str, set[str]] = {split: set() for split in YOLO_SPLITS}

    for row in manifest_rows:
        image_path = _image_path_for_row(row)
        annotations = _annotations_for_manifest_row(row, annotations_by_alias)
        label = (_clean_text(row.get("label")) or "").lower()
        if label == "defect" and not annotations:
            if on_unlabeled_defect == "error":
                msg = f"Defect crop is missing bbox annotations: {image_path}"
                raise ValueError(msg)
            if on_unlabeled_defect == "skip":
                continue
        split = _split_for_row(
            row,
            annotations,
            positive_val_ratio=positive_val_ratio,
            positive_test_ratio=positive_test_ratio,
            seed=seed,
            normal_test_split=normal_test_split,
        )
        output_name = _safe_image_name(row, image_path, used_names[split])
        output_image = output_root / "images" / split / output_name
        output_label = output_root / "labels" / split / f"{Path(output_name).stem}.txt"
        shutil.copy2(image_path, output_image)
        width, height = _image_size(row, image_path)
        label_lines = [_normalize_bbox(annotation, width, height) for annotation in annotations]
        output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        if preview_dir is not None and annotations:
            _write_preview(image_path, annotations, preview_dir / split / output_name)
        export_rows.append(
            {
                "source_path": _clean_text(row.get("source_path")) or "",
                "processed_path": str(image_path),
                "yolo_image_path": str(output_image),
                "yolo_label_path": str(output_label),
                "yolo_split": split,
                "label": _clean_text(row.get("label")) or "",
                "slot": _clean_text(row.get("slot")) or "",
                "slot_row": _clean_text(row.get("slot_row")) or "",
                "slot_col": _clean_text(row.get("slot_col")) or "",
                "defect_type": _clean_text(row.get("defect_type")) or "",
                "sample_id": _clean_text(row.get("sample_id")) or "",
                "frame_id": _clean_text(row.get("frame_id")) or "",
                "annotation_count": str(len(annotations)),
            },
        )
        summary[split] += 1

    if not export_rows:
        msg = "No YOLO images were exported. Check manifest, annotations, and filtering options."
        raise RuntimeError(msg)

    with (output_root / "export_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(export_rows[0]))
        writer.writeheader()
        writer.writerows(export_rows)
    _write_data_yaml(output_root, class_names, include_test=summary["test"] > 0)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the YOLO dataset export CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="C789 part_crop_manifest.csv.")
    parser.add_argument("--annotations", type=Path, required=True, help="BBox annotation CSV in pixel coordinates.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output Ultralytics dataset root.")
    parser.add_argument(
        "--positive-val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for defect crops.",
    )
    parser.add_argument("--positive-test-ratio", type=float, default=0.0, help="Test split ratio for defect crops.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed.")
    parser.add_argument(
        "--normal-test-split",
        choices=YOLO_SPLITS,
        default="val",
        help="YOLO split used for manifest rows labelled normal_test.",
    )
    parser.add_argument(
        "--on-unlabeled-defect",
        choices=("error", "skip", "empty"),
        default="error",
        help="How to handle defect crops that do not have bbox annotations.",
    )
    parser.add_argument("--preview-dir", type=Path, help="Optional directory for bbox overlay previews.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output root.")
    return parser


def main() -> None:
    """Run YOLO dataset export."""
    args = build_parser().parse_args()
    summary = export_yolo_dataset(
        manifest_path=args.manifest,
        annotations_path=args.annotations,
        output_root=args.output_root,
        positive_val_ratio=args.positive_val_ratio,
        positive_test_ratio=args.positive_test_ratio,
        seed=args.seed,
        normal_test_split=args.normal_test_split,
        on_unlabeled_defect=args.on_unlabeled_defect,
        preview_dir=args.preview_dir,
        overwrite=args.overwrite,
    )
    print(f"YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
