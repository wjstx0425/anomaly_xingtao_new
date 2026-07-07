# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build a split-aware same-distribution YOLO dataset for C789."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from capture_data.augment_yolo_dataset import (
    IMAGE_EXTENSIONS,
    _draw_preview,
    _read_labels,
    _write_labels,
    build_augmentations,
)


CLASS_NAMES = ("defect",)
DEFAULT_VAL_GROUPS = ("g002", "g008")


@dataclass(frozen=True)
class SourceSample:
    """One source sample before split-aware export."""

    image_path: Path
    label_path: Path
    source_kind: str
    split: str
    group_id: str
    slot_id: str
    source_stem: str


def _parse_identifier(stem: str, prefix: str) -> str:
    """Extract an identifier like ``g001`` or ``slot01`` from a filename stem."""
    for chunk in stem.split("_"):
        if chunk.startswith(prefix):
            suffix = chunk[len(prefix) :]
            if suffix and suffix.isdigit():
                return chunk
    msg = f"Filename stem must contain {prefix}NNN-style token: {stem}"
    raise ValueError(msg)


def _iter_flat_images(input_root: Path) -> list[Path]:
    """Return sorted flat YOLO images from ``input_root/images``."""
    images_dir = input_root / "images"
    if not images_dir.is_dir():
        msg = f"Input root must contain images/: {input_root}"
        raise ValueError(msg)
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _load_defect_samples(input_root: Path, val_groups: set[str]) -> list[SourceSample]:
    """Load flat defect YOLO samples and assign train/val by group id."""
    labels_dir = input_root / "labels"
    samples: list[SourceSample] = []
    for image_path in _iter_flat_images(input_root):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            msg = f"Missing YOLO label for image: {image_path}"
            raise ValueError(msg)
        group_id = _parse_identifier(image_path.stem, "g")
        slot_id = _parse_identifier(image_path.stem, "slot")
        split = "val" if group_id in val_groups else "train"
        samples.append(
            SourceSample(
                image_path=image_path,
                label_path=label_path,
                source_kind="defect_source",
                split=split,
                group_id=group_id,
                slot_id=slot_id,
                source_stem=image_path.stem,
            ),
        )
    if not samples:
        msg = f"No images found under {input_root / 'images'}"
        raise ValueError(msg)
    return samples


def _iter_split_images(root: Path, split: str) -> list[Path]:
    """Return YOLO image files from an existing split-aware dataset."""
    images_dir = root / "images" / split
    if not images_dir.is_dir():
        return []
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _is_empty_label(path: Path) -> bool:
    """Return whether a YOLO label file has no boxes."""
    return not path.read_text(encoding="utf-8").strip()


def _load_normal_samples(
    normal_source_root: Path | None,
    *,
    seed: int,
    train_limit: int,
    val_limit: int,
) -> list[SourceSample]:
    """Load optional normal samples from an existing YOLO dataset split."""
    if normal_source_root is None:
        return []
    rng = random.Random(seed)
    samples_by_split: dict[str, list[SourceSample]] = {"train": [], "val": []}
    for split in ("train", "val"):
        for image_path in _iter_split_images(normal_source_root, split):
            label_path = normal_source_root / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                msg = f"Missing normal-source label for image: {image_path}"
                raise ValueError(msg)
            if not _is_empty_label(label_path):
                continue
            samples_by_split[split].append(
                SourceSample(
                    image_path=image_path,
                    label_path=label_path,
                    source_kind="normal_source",
                    split=split,
                    group_id=_parse_identifier(image_path.stem, "g"),
                    slot_id=_parse_identifier(image_path.stem, "slot"),
                    source_stem=image_path.stem,
                ),
            )
    limited_samples: list[SourceSample] = []
    for split, limit in (("train", train_limit), ("val", val_limit)):
        chosen = list(samples_by_split[split])
        rng.shuffle(chosen)
        if limit >= 0:
            chosen = chosen[:limit]
        limited_samples.extend(sorted(chosen, key=lambda sample: sample.image_path.name))
    return limited_samples


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    """Create a clean output directory tree."""
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            msg = f"Output root is non-empty: {output_root}. Pass --overwrite to replace it."
            raise ValueError(msg)
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "previews").mkdir(parents=True, exist_ok=True)


def _copy_sample(sample: SourceSample, output_root: Path) -> tuple[Path, Path]:
    """Copy one real sample into the requested split directory."""
    output_image = output_root / "images" / sample.split / sample.image_path.name
    output_label = output_root / "labels" / sample.split / f"{sample.image_path.stem}.txt"
    shutil.copy2(sample.image_path, output_image)
    shutil.copy2(sample.label_path, output_label)
    return output_image, output_label


def _write_data_yaml(output_root: Path) -> None:
    """Write Ultralytics dataset metadata."""
    content = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: defect",
            "",
        ],
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a CSV file from dictionaries."""
    if not rows:
        msg = f"Cannot write CSV without rows: {path}"
        raise ValueError(msg)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_split_counts(split_manifest_rows: Sequence[dict[str, Any]], augmented_train_count: int) -> list[dict[str, str]]:
    """Build dataset summary rows for train and val."""
    summary_rows: list[dict[str, str]] = []
    for split in ("train", "val"):
        real_rows = [row for row in split_manifest_rows if row["split"] == split]
        copied_train_rows = [
            row for row in real_rows if row["split"] == "train" and row["source_kind"] != "defect_source"
        ]
        image_count = augmented_train_count + len(copied_train_rows) if split == "train" else len(real_rows)
        summary_rows.append(
            {
                "split": split,
                "real_image_count": str(len(real_rows)),
                "image_count": str(image_count),
                "defect_source_count": str(sum(row["source_kind"] == "defect_source" for row in real_rows)),
                "normal_source_count": str(sum(row["source_kind"] == "normal_source" for row in real_rows)),
            },
        )
    return summary_rows


def _leakage_rows(split_manifest_rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Report whether defect group or source stems appear in both train and val."""
    defect_rows = [row for row in split_manifest_rows if row["source_kind"] == "defect_source"]
    rows: list[dict[str, str]] = []
    for leakage_kind in ("group_id", "source_stem"):
        train_values = {str(row[leakage_kind]) for row in defect_rows if row["split"] == "train"}
        val_values = {str(row[leakage_kind]) for row in defect_rows if row["split"] == "val"}
        shared_values = sorted(train_values & val_values)
        rows.append(
            {
                "leakage_kind": leakage_kind,
                "train_count": str(len(train_values)),
                "val_count": str(len(val_values)),
                "shared_count": str(len(shared_values)),
                "shared_values": "|".join(shared_values),
                "is_leakage": str(bool(shared_values)).lower(),
            },
        )
    return rows


def build_same_dist_yolo_dataset(
    input_root: Path,
    output_root: Path,
    *,
    normal_source_root: Path | None = None,
    val_groups: Sequence[str] = DEFAULT_VAL_GROUPS,
    seed: int = 0,
    preview_limit: int = 80,
    overwrite: bool = False,
    train_normal_limit: int = 600,
    val_normal_limit: int = 150,
) -> dict[str, Any]:
    """Create a same-distribution YOLO dataset with train-only augmentation."""
    val_group_set = set(val_groups)
    _prepare_output_root(output_root, overwrite)
    source_samples = _load_defect_samples(input_root, val_group_set)
    source_samples.extend(
        _load_normal_samples(
            normal_source_root,
            seed=seed,
            train_limit=train_normal_limit,
            val_limit=val_normal_limit,
        ),
    )

    split_manifest_rows: list[dict[str, Any]] = []
    augmentation_rows: list[dict[str, Any]] = []
    preview_count = 0
    augmentations = build_augmentations(seed)

    for sample in source_samples:
        if sample.split == "train" and sample.source_kind == "defect_source":
            output_image = None
            output_label = None
        else:
            output_image, output_label = _copy_sample(sample, output_root)
        split_manifest_rows.append(
            {
                "split": sample.split,
                "source_kind": sample.source_kind,
                "group_id": sample.group_id,
                "slot_id": sample.slot_id,
                "source_stem": sample.source_stem,
                "source_image": str(sample.image_path),
                "source_label": str(sample.label_path),
                "output_image": str(output_image) if output_image is not None else "",
                "output_label": str(output_label) if output_label is not None else "",
            },
        )
        if sample.split == "val":
            if preview_count < preview_limit:
                image = cv2.imread(str(output_image), cv2.IMREAD_COLOR)
                if image is None:
                    msg = f"Could not read image: {output_image}"
                    raise ValueError(msg)
                _draw_preview(image, _read_labels(output_label), output_root / "previews" / output_image.name)
                preview_count += 1
            continue

        if sample.source_kind != "defect_source":
            if preview_count < preview_limit:
                image = cv2.imread(str(output_image), cv2.IMREAD_COLOR)
                if image is None:
                    msg = f"Could not read image: {output_image}"
                    raise ValueError(msg)
                _draw_preview(image, _read_labels(output_label), output_root / "previews" / output_image.name)
                preview_count += 1
            continue

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Could not read image: {sample.image_path}"
            raise ValueError(msg)
        boxes = _read_labels(sample.label_path)
        for augmentation in augmentations:
            augmented_image, augmented_boxes = augmentation.apply(image, boxes)
            variant_name = f"{sample.image_path.stem}__{augmentation.name}{sample.image_path.suffix.lower()}"
            variant_image = output_root / "images" / "train" / variant_name
            variant_label = output_root / "labels" / "train" / f"{Path(variant_name).stem}.txt"
            if not cv2.imwrite(str(variant_image), augmented_image):
                msg = f"Could not write image: {variant_image}"
                raise RuntimeError(msg)
            _write_labels(variant_label, augmented_boxes)
            if preview_count < preview_limit:
                _draw_preview(augmented_image, augmented_boxes, output_root / "previews" / variant_name)
                preview_count += 1
            augmentation_rows.append(
                {
                    "split": "train",
                    "source_kind": sample.source_kind,
                    "group_id": sample.group_id,
                    "slot_id": sample.slot_id,
                    "source_stem": sample.source_stem,
                    "variant": augmentation.name,
                    "output_image": str(variant_image),
                    "output_label": str(variant_label),
                    "input_box_count": str(len(boxes)),
                    "output_box_count": str(len(augmented_boxes)),
                },
            )

    if not split_manifest_rows:
        msg = "No samples were collected for same-distribution YOLO dataset build"
        raise ValueError(msg)

    _write_data_yaml(output_root)
    _write_csv(output_root / "split_manifest.csv", split_manifest_rows)
    _write_csv(output_root / "augmentation_manifest.csv", augmentation_rows)
    summary_rows = _summarize_split_counts(split_manifest_rows, len(augmentation_rows))
    _write_csv(output_root / "dataset_summary.csv", summary_rows)
    leakage_rows = _leakage_rows(split_manifest_rows)
    _write_csv(output_root / "leakage_report.csv", leakage_rows)

    return {
        "status": "ok",
        "train_real_images": sum(row["split"] == "train" for row in split_manifest_rows),
        "val_real_images": sum(row["split"] == "val" for row in split_manifest_rows),
        "train_images": len(list((output_root / "images" / "train").glob("*"))),
        "val_images": sum(row["split"] == "val" for row in split_manifest_rows),
        "leakage_rows": len(leakage_rows),
        "leakage_flagged_rows": sum(row["is_leakage"] == "true" for row in leakage_rows),
        "preview_images": preview_count,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="Flat YOLO root with images/ and labels/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output split-aware YOLO root.")
    parser.add_argument(
        "--normal-source-root",
        type=Path,
        default=None,
        help="Optional existing YOLO dataset root with images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--val-groups",
        nargs="+",
        default=list(DEFAULT_VAL_GROUPS),
        help="Group ids reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for sampling and augmentation noise.")
    parser.add_argument("--preview-limit", type=int, default=80, help="Maximum preview images to write.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output root.")
    parser.add_argument(
        "--train-normal-limit",
        type=int,
        default=600,
        help="Maximum normal-source train samples to include before augmentation.",
    )
    parser.add_argument(
        "--val-normal-limit",
        type=int,
        default=150,
        help="Maximum normal-source val samples to include as real images.",
    )
    return parser


def main() -> None:
    """Run the same-distribution YOLO dataset builder CLI."""
    args = build_parser().parse_args()
    summary = build_same_dist_yolo_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        normal_source_root=args.normal_source_root,
        val_groups=args.val_groups,
        seed=args.seed,
        preview_limit=args.preview_limit,
        overwrite=args.overwrite,
        train_normal_limit=args.train_normal_limit,
        val_normal_limit=args.val_normal_limit,
    )
    print(f"same-distribution YOLO dataset: {args.output_root}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
