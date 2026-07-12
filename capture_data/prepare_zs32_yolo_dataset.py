# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM101, EM102, S311, TRY003

"""Build a trainable ZS32 six-view YOLO dataset from Label Studio labels."""

from __future__ import annotations

import csv
import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
MIRRORED_VIEW = {
    "front": "front",
    "front_left": "front_right",
    "front_right": "front_left",
    "back": "back",
    "back_left": "back_right",
    "back_right": "back_left",
}
GROUP_PATTERN = re.compile(r"_(group\d{3})_")
DEFAULT_EXCLUDED_SAMPLE_IDS = ("left/20260711_181850_552955/less/group027",)


@dataclass(frozen=True)
class NormalImage:
    """One real right-hand normal image."""

    source_path: Path
    view: str
    session_id: str
    group_id: str

    @property
    def sample_id(self) -> str:
        """Physical normal group identifier."""
        return f"normal/right/{self.session_id}/{self.group_id}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _discover_normals(dataset_root: Path) -> list[NormalImage]:
    """Discover complete six-view right-hand normal groups."""
    images: list[NormalImage] = []
    for path in sorted(dataset_root.glob("right/*/normal/*/images/*.png")):
        _, view, _, session_id, _, _ = path.relative_to(dataset_root).parts
        if view not in VIEWS:
            continue
        match = GROUP_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Could not parse normal group from: {path}")
        images.append(NormalImage(path, view, session_id, match.group(1)))
    grouped: dict[str, set[str]] = defaultdict(set)
    for image in images:
        grouped[image.sample_id].add(image.view)
    incomplete = {key: sorted(views) for key, views in grouped.items() if views != set(VIEWS)}
    if not images or incomplete:
        raise ValueError(f"Normal groups must contain six views: {incomplete}")
    return images


def _assign_splits(
    group_strata: dict[str, str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign whole physical groups to deterministic stratified splits."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    strata: dict[str, list[str]] = defaultdict(list)
    for sample_id, stratum in group_strata.items():
        strata[stratum].append(sample_id)
    assignments: dict[str, str] = {}
    for stratum in sorted(strata):
        sample_ids = sorted(strata[stratum])
        random.Random(f"{seed}:{stratum}").shuffle(sample_ids)
        size = len(sample_ids)
        val_count = max(1, round(size * val_ratio)) if val_ratio > 0 and size >= 3 else 0
        test_count = max(1, round(size * test_ratio)) if test_ratio > 0 and size >= 3 else 0
        while val_count + test_count >= size:
            if val_count >= test_count and val_count > 0:
                val_count -= 1
            elif test_count > 0:
                test_count -= 1
        for sample_id in sample_ids[:test_count]:
            assignments[sample_id] = "test"
        for sample_id in sample_ids[test_count : test_count + val_count]:
            assignments[sample_id] = "val"
        for sample_id in sample_ids[test_count + val_count :]:
            assignments[sample_id] = "train"
    return assignments


def _prepare_output(output_root: Path, *, overwrite: bool, forbidden_roots: tuple[Path, ...]) -> None:
    """Create a clean output directory without allowing source deletion."""
    resolved = output_root.resolve(strict=False)
    forbidden = [path.resolve(strict=False) for path in forbidden_roots]
    if any(resolved == path or path.is_relative_to(resolved) for path in forbidden):
        raise ValueError(f"Unsafe output root: {output_root}")
    if any(resolved.is_relative_to(path) for path in forbidden[1:]):
        raise ValueError(f"Unsafe output root: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise ValueError(f"Output root is non-empty: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _validate_label(path: Path) -> tuple[str, int]:
    """Validate a Label Studio YOLO label and return its text and box count."""
    text = path.read_text(encoding="utf-8")
    count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if class_id != 0 or any(value < 0 or value > 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Invalid YOLO box at {path}:{line_number}")
        count += 1
    return text, count


def _link_image(source: Path, target: Path) -> None:
    """Hard-link a real image into the YOLO dataset."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def _write_mirror(source: Path, target: Path) -> None:
    """Write a horizontal mirror for a synthetic left-hand normal."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read normal image: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), cv2.flip(image, 1), [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"Could not write mirrored image: {target}")


def build_zs32_yolo_dataset(
    *,
    repo_root: Path,
    dataset_root: Path,
    labeling_root: Path,
    label_export_root: Path,
    output_root: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    excluded_sample_ids: tuple[str, ...] = DEFAULT_EXCLUDED_SAMPLE_IDS,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build the complete split-aware ZS32 YOLO detection dataset."""
    manifest_rows = _read_csv(labeling_root / "labeling_manifest.csv")
    excluded = set(excluded_sample_ids)
    defect_rows = [row for row in manifest_rows if row["sample_id"] not in excluded]
    labels_by_stem = {path.stem: path for path in (label_export_root / "labels").glob("*.txt")}
    known_stems = {Path(row["source_path"]).stem for row in manifest_rows}
    unknown_labels = sorted(set(labels_by_stem) - known_stems)
    if unknown_labels:
        raise ValueError(f"Label export contains unknown images: {unknown_labels[:5]}")
    normal_images = _discover_normals(dataset_root)

    group_strata: dict[str, str] = {}
    for row in defect_rows:
        group_strata[row["sample_id"]] = f"defect:{row['hand']}:{row['defect_type']}"
    for image in normal_images:
        group_strata[image.sample_id] = "normal:right"
    split_for_group = _assign_splits(group_strata, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    _prepare_output(
        output_root,
        overwrite=overwrite,
        forbidden_roots=(repo_root, dataset_root / "left", dataset_root / "right", labeling_root),
    )

    export_rows: list[dict[str, str]] = []
    annotated_defects = 0
    for row in defect_rows:
        source = repo_root / row["source_path"]
        split = split_for_group[row["sample_id"]]
        output_image = output_root / "images" / split / source.name
        output_label = output_root / "labels" / split / f"{source.stem}.txt"
        _link_image(source, output_image)
        label_path = labels_by_stem.get(source.stem)
        label_text, box_count = _validate_label(label_path) if label_path is not None else ("", 0)
        output_label.write_text(label_text, encoding="utf-8")
        annotated_defects += box_count > 0
        export_rows.append(
            {
                "source_path": str(source.relative_to(repo_root)),
                "output_image": str(output_image.relative_to(repo_root)),
                "output_label": str(output_label.relative_to(repo_root)),
                "kind": "defect",
                "split": split,
                "sample_id": row["sample_id"],
                "hand": row["hand"],
                "view": row["view"],
                "source_view": row["view"],
                "defect_type": row["defect_type"],
                "annotation_count": str(box_count),
            },
        )

    for image in normal_images:
        split = split_for_group[image.sample_id]
        real_output = output_root / "images" / split / image.source_path.name
        real_label = output_root / "labels" / split / f"{image.source_path.stem}.txt"
        _link_image(image.source_path, real_output)
        real_label.write_text("", encoding="utf-8")
        export_rows.append(
            {
                "source_path": str(image.source_path.relative_to(repo_root)),
                "output_image": str(real_output.relative_to(repo_root)),
                "output_label": str(real_label.relative_to(repo_root)),
                "kind": "normal_real",
                "split": split,
                "sample_id": image.sample_id,
                "hand": "right",
                "view": image.view,
                "source_view": image.view,
                "defect_type": "",
                "annotation_count": "0",
            },
        )
        mirrored_view = MIRRORED_VIEW[image.view]
        mirror_name = f"mirror_left_{mirrored_view}__{image.source_path.name}"
        mirror_output = output_root / "images" / split / mirror_name
        mirror_label = output_root / "labels" / split / f"{Path(mirror_name).stem}.txt"
        _write_mirror(image.source_path, mirror_output)
        mirror_label.write_text("", encoding="utf-8")
        export_rows.append(
            {
                "source_path": str(image.source_path.relative_to(repo_root)),
                "output_image": str(mirror_output.relative_to(repo_root)),
                "output_label": str(mirror_label.relative_to(repo_root)),
                "kind": "normal_mirror",
                "split": split,
                "sample_id": image.sample_id,
                "hand": "left",
                "view": mirrored_view,
                "source_view": image.view,
                "defect_type": "",
                "annotation_count": "0",
            },
        )

    with (output_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(export_rows[0]))
        writer.writeheader()
        writer.writerows(export_rows)
    (output_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    (output_root / "excluded_samples.txt").write_text("\n".join(sorted(excluded)) + "\n", encoding="utf-8")
    defect_count = len(defect_rows)
    normal_count = len(normal_images)
    return {
        "defect_images": defect_count,
        "annotated_defect_images": annotated_defects,
        "empty_defect_images": defect_count - annotated_defects,
        "normal_real_images": normal_count,
        "normal_mirror_images": normal_count,
        "total_images": defect_count + 2 * normal_count,
        "groups": len(group_strata),
    }
