# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build a symlink-based BMW multisource laboratory training release."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER, _atomic_publish_noreplace
from bmw_inspection.lab.eight_view_training_data import _validate_yolo_text

_TRAINING_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_YOLO_SPLITS = ("train", "val", "test")
_SPLIT_TO_YOLO = {"train": "train", "calibration": "val", "final_test": "test"}
_TEMPLATE_FIELDS = ("sample_id", "part_id", "view_id", "image_path", "split", "label")
_BRIGHT_FIELDS = (
    "sample_id",
    "physical_part_id",
    "session_id",
    "group_id",
    "view_id",
    "camera_serial",
    "source_path",
    "source_sha256",
    "source_class",
    "business_label",
    "split",
    "expected_status",
    "review_reason",
)
_QUEUE_FIELDS = {
    "sample_id",
    "physical_part_id",
    "view_id",
    "source_class",
    "split",
    "crop_path",
    "expected_label_filename",
}


def _read_csv(path: Path, required: set[str]) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if not required.issubset(fields):
            missing = ", ".join(sorted(required - set(fields)))
            raise ValueError(f"CSV is missing required fields ({missing}): {path}")
        rows = list(reader)
    return fields, rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_release(path: Path, *, require_yolo: bool = False) -> tuple[Path, dict[str, Any]]:
    release = Path(path).expanduser().resolve()
    report_path = release / "report.json"
    if not report_path.is_file():
        raise ValueError(f"training release report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("release_status") != "published":
        raise ValueError(f"training release is not published: {release}")
    if require_yolo and (report.get("yolo_training_ready") is not True or not (release / "yolo/data.yaml").is_file()):
        raise ValueError(f"YOLO training release is incomplete: {release}")
    return release, report


def _resolved_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path.resolve()


def _add_link(links: dict[Path, Path], relative: Path, source: Path) -> None:
    resolved = _resolved_file(source, "composite source file")
    previous = links.setdefault(relative, resolved)
    if previous != resolved:
        raise ValueError(f"composite destination collision: {relative}")


def _collect_efficientad(
    release: Path,
    links: dict[Path, Path],
) -> int:
    count = 0
    for view in VIEW_ORDER:
        view_root = release / "efficientad" / view
        if not view_root.is_dir():
            raise ValueError(f"EfficientAD view is missing: {view_root}")
        view_count = 0
        for image in sorted(view_root.rglob("*.png")):
            relative = image.relative_to(view_root)
            if not relative.parts or relative.parts[0] not in {"normal", "normal_test", "defect"}:
                raise ValueError(f"unsupported EfficientAD layout: {image}")
            destination = Path("efficientad") / view / relative.parts[0] / release.name / Path(*relative.parts[1:])
            _add_link(links, destination, image)
            view_count += 1
        if view_count == 0:
            raise ValueError(f"EfficientAD view has no images: {view_root}")
        count += view_count
    return count


def _collect_template(release: Path) -> list[dict[str, str]]:
    manifest = release / "template/trainer_manifest.csv"
    _fields, rows = _read_csv(manifest, set(_TEMPLATE_FIELDS))
    output: list[dict[str, str]] = []
    for row in rows:
        image = Path(row["image_path"])
        image = image if image.is_absolute() else manifest.parent / image
        output.append({
            "sample_id": f"{release.name}::{row['sample_id']}",
            "part_id": f"{release.name}::{row['part_id']}",
            "view_id": row["view_id"],
            "image_path": str(_resolved_file(image, "Template image")),
            "split": row["split"],
            "label": row["label"],
        })
    if not output:
        raise ValueError(f"Template manifest is empty: {manifest}")
    return output


def _collect_bright(release: Path, report: Mapping[str, Any]) -> list[dict[str, str]]:
    prepared_manifest = report.get("prepared_manifest")
    if not isinstance(prepared_manifest, str):
        raise ValueError(f"training release has no prepared_manifest: {release}")
    manifest = Path(prepared_manifest).expanduser().resolve().parent / "bright_streak.csv"
    fields, rows = _read_csv(manifest, set(_BRIGHT_FIELDS))
    if fields != _BRIGHT_FIELDS:
        raise ValueError(f"bright-streak manifest header differs from the frozen schema: {manifest}")
    output: list[dict[str, str]] = []
    for row in rows:
        source = _resolved_file(Path(row["source_path"]).expanduser(), "bright-streak source image")
        output.append({
            **row,
            "sample_id": f"{release.name}::{row['sample_id']}",
            "physical_part_id": f"{release.name}::{row['physical_part_id']}",
            "source_path": str(source),
        })
    return output


def _collect_ready_yolo(release: Path, links: dict[Path, Path]) -> tuple[int, int]:
    image_count = 0
    positive_count = 0
    for split in _YOLO_SPLITS:
        image_root = release / "yolo/images" / split
        label_root = release / "yolo/labels" / split
        for image in sorted(image_root.glob("*.png")):
            label = label_root / f"{image.stem}.txt"
            _add_link(links, Path("yolo/images") / split / image.name, image)
            _add_link(links, Path("yolo/labels") / split / label.name, label)
            positive_count += int(bool(label.read_text(encoding="utf-8").strip()))
            image_count += 1
    if image_count == 0:
        raise ValueError(f"YOLO release contains no images: {release}")
    return image_count, positive_count


def _collect_reviewed_right_yolo(
    release: Path,
    label_root: Path,
    links: dict[Path, Path],
    text_files: dict[Path, str],
    template_rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    image_count = 0
    positive_count = 0
    efficientad_count = 0
    for split in _YOLO_SPLITS:
        image_root = release / "yolo/images" / split
        source_label_root = release / "yolo/labels" / split
        for image in sorted(image_root.glob("*.png")):
            label = source_label_root / f"{image.stem}.txt"
            _add_link(links, Path("yolo/images") / split / image.name, image)
            _add_link(links, Path("yolo/labels") / split / label.name, label)
            positive_count += int(bool(label.read_text(encoding="utf-8").strip()))
            image_count += 1

    queue = release / "yolo/annotation_queue.csv"
    _fields, rows = _read_csv(queue, _QUEUE_FIELDS)
    labels = Path(label_root).expanduser().resolve()
    expected_names = {row["expected_label_filename"] for row in rows}
    actual_names = {path.name for path in labels.glob("*.txt") if path.is_file() and not path.is_symlink()}
    if actual_names != expected_names:
        raise ValueError(
            "reviewed right YOLO labels do not match annotation queue: "
            f"expected={len(expected_names)}, actual={len(actual_names)}"
        )
    for row in rows:
        split = _SPLIT_TO_YOLO.get(row["split"])
        if split is None or row["view_id"] not in VIEW_ORDER:
            raise ValueError(f"right YOLO queue contains an unknown split/view: {row}")
        crop = _resolved_file(release / row["crop_path"], "reviewed right ROI crop")
        label_path = labels / row["expected_label_filename"]
        text = _validate_yolo_text(label_path.read_text(encoding="utf-8"), path=label_path)
        _add_link(links, Path("yolo/images") / split / crop.name, crop)
        destination_label = Path("yolo/labels") / split / f"{crop.stem}.txt"
        if destination_label in links or destination_label in text_files:
            raise ValueError(f"composite destination collision: {destination_label}")
        text_files[destination_label] = text
        image_count += 1
        positive_count += int(bool(text))
        if not text or row["split"] not in {"calibration", "final_test"}:
            continue
        namespace = release.name
        if row["split"] == "calibration":
            destination = (
                Path("efficientad")
                / row["view_id"]
                / "defect"
                / namespace
                / row["source_class"]
                / row["physical_part_id"]
                / "images"
                / crop.name
            )
        else:
            destination = (
                Path("efficientad/held_out")
                / row["view_id"]
                / "defect"
                / namespace
                / row["source_class"]
                / row["physical_part_id"]
                / "images"
                / crop.name
            )
        _add_link(links, destination, crop)
        efficientad_count += 1
        template_rows.append({
            "sample_id": f"{namespace}::{row['sample_id']}",
            "part_id": f"{namespace}::{row['physical_part_id']}",
            "view_id": row["view_id"],
            "image_path": str(crop),
            "split": row["split"],
            "label": "defect",
        })
    return image_count, positive_count, efficientad_count


def _validate_yolo_pairing(links: Mapping[Path, Path], text_files: Mapping[Path, str]) -> int:
    images = {
        (path.parts[2], path.stem)
        for path in links
        if len(path.parts) == 4 and path.parts[:2] == ("yolo", "images") and path.suffix == ".png"
    }
    labels = {
        (path.parts[2], path.stem)
        for path in (*links, *text_files)
        if len(path.parts) == 4 and path.parts[:2] == ("yolo", "labels") and path.suffix == ".txt"
    }
    if images != labels:
        raise ValueError(f"composite YOLO image/label identities differ: images={len(images)}, labels={len(labels)}")
    return len(images)


def _publish(
    staging: Path,
    links: Mapping[Path, Path],
    text_files: Mapping[Path, str],
    template_rows: Sequence[Mapping[str, str]],
    bright_rows: Sequence[Mapping[str, str]],
    report: Mapping[str, Any],
) -> None:
    for relative, source in sorted(links.items(), key=lambda item: item[0].as_posix()):
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.path.relpath(source, destination.parent))
    for relative, text in sorted(text_files.items(), key=lambda item: item[0].as_posix()):
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    _write_csv(staging / "template/trainer_manifest.csv", _TEMPLATE_FIELDS, template_rows)
    _write_csv(staging / "manifests/bright_streak.csv", _BRIGHT_FIELDS, bright_rows)
    (staging / "yolo/data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    (staging / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_multisource_training_data(
    *,
    branch_releases: Sequence[Path],
    left_yolo_release: Path,
    right_yolo_release: Path,
    right_yolo_label_root: Path,
    output_root: Path,
    training_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build one immutable composite release for the BMW laboratory trainers.

    Args:
        branch_releases (Sequence[Path]): Right-hand releases used by EfficientAD, Template, and bright-streak.
        left_yolo_release (Path): Published left-hand YOLO-ready release.
        right_yolo_release (Path): Right-hand ROI release paired with reviewed labels.
        right_yolo_label_root (Path): Reviewed right-hand YOLO text directory.
        output_root (Path): Parent directory for the composite release.
        training_id (str): Immutable composite release identifier.
        dry_run (bool): Validate and count inputs without publishing output.

    Returns:
        dict[str, Any]: Composite release report and branch counts.

    Raises:
        FileExistsError: If the destination already exists with a different source contract.
        OSError: If source assets cannot be read or output cannot be published.
        ValueError: If a release, manifest, label, split, or image/label pairing is invalid.
    """
    if not isinstance(training_id, str) or not _TRAINING_ID.fullmatch(training_id):
        raise ValueError("training_id contains unsupported characters")
    if len(branch_releases) < 2:
        raise ValueError("at least two right branch releases are required")
    loaded_branches = [_load_release(path) for path in branch_releases]
    resolved_branches = [item[0] for item in loaded_branches]
    if len(set(resolved_branches)) != len(resolved_branches):
        raise ValueError("right branch releases must be unique")
    left, _left_report = _load_release(left_yolo_release, require_yolo=True)
    right, _right_report = _load_release(right_yolo_release)
    if right not in resolved_branches:
        raise ValueError("right YOLO release must also be one of the right branch releases")

    links: dict[Path, Path] = {}
    text_files: dict[Path, str] = {}
    template_rows: list[dict[str, str]] = []
    bright_rows: list[dict[str, str]] = []
    efficientad_count = 0
    view_counts = {view: 0 for view in VIEW_ORDER}
    for release, release_report in loaded_branches:
        counts = release_report.get("view_crop_counts")
        if not isinstance(counts, dict) or set(counts) != set(VIEW_ORDER):
            raise ValueError(f"right branch release has invalid view counts: {release}")
        for view in VIEW_ORDER:
            value = counts[view]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"right branch release has invalid view count: {release}/{view}")
            view_counts[view] += value
        efficientad_count += _collect_efficientad(release, links)
        template_rows.extend(_collect_template(release))
        bright_rows.extend(_collect_bright(release, release_report))

    left_yolo_count, left_positive_count = _collect_ready_yolo(left, links)
    right_yolo_count, right_positive_count, reviewed_ea_count = _collect_reviewed_right_yolo(
        right,
        right_yolo_label_root,
        links,
        text_files,
        template_rows,
    )
    efficientad_count += reviewed_ea_count
    yolo_count = _validate_yolo_pairing(links, text_files)
    if yolo_count != left_yolo_count + right_yolo_count:
        raise ValueError("composite YOLO count differs from source counts")

    source_contract = {
        "right_branch_releases": [str(path) for path in resolved_branches],
        "left_yolo_release": str(left),
        "right_yolo_release": str(right),
        "right_yolo_label_root": str(Path(right_yolo_label_root).expanduser().resolve()),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "training_id": training_id,
        "release_status": "dry_run" if dry_run else "published",
        "materialization": "multisource_relative_symlinks",
        "experimental_only": True,
        "source_contract": source_contract,
        "view_crop_counts": view_counts,
        "efficientad_image_count": efficientad_count,
        "template_row_count": len(template_rows),
        "bright_streak_row_count": len(bright_rows),
        "yolo_image_count": yolo_count,
        "yolo_label_count": yolo_count,
        "yolo_positive_count": left_positive_count + right_positive_count,
        "yolo_pending_count": 0,
        "yolo_training_ready": True,
    }
    destination = Path(output_root).expanduser().resolve() / training_id
    if destination.exists() or destination.is_symlink():
        existing_path = destination / "report.json"
        if not existing_path.is_file():
            raise FileExistsError(f"incomplete composite release exists: {destination}")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("source_contract") != source_contract or existing.get("yolo_training_ready") is not True:
            raise FileExistsError(f"composite release uses a different source contract: {destination}")
        return {**existing, "release_status": "reused"}
    if dry_run:
        return report

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{training_id}.", dir=destination.parent))
    try:
        _publish(staging, links, text_files, template_rows, bright_rows, report)
        _atomic_publish_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
