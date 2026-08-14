"""Materialize ROI crops and branch datasets for BMW eight-view experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER, _atomic_publish_noreplace
from bmw_inspection.lab.eight_view_roi import EightViewRoiConfig, load_roi_config

_DATASET_FIELDS = (
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
)
_CROP_FIELDS = _DATASET_FIELDS + (
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
)
_TEMPLATE_FIELDS = ("sample_id", "part_id", "view_id", "image_path", "split", "label")
_QUEUE_FIELDS = _DATASET_FIELDS + ("crop_path", "expected_label_filename")
_TRAINING_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SPLIT_TO_YOLO = {"train": "train", "calibration": "val", "final_test": "test"}
_YOLO_EDGE_TOLERANCE = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_dataset_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _DATASET_FIELDS:
            raise ValueError("prepared dataset manifest header differs from the frozen schema")
        rows = list(reader)
    if not rows:
        raise ValueError("prepared dataset manifest must not be empty")
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        identity = (row["session_id"], row["sample_id"], row["view_id"])
        if identity in identities:
            raise ValueError(f"duplicate prepared image identity: {identity}")
        identities.add(identity)
        if row["view_id"] not in VIEW_ORDER or row["split"] not in _SPLIT_TO_YOLO:
            raise ValueError("prepared row has an unknown view or split")
    return rows


def _export_stem(row: Mapping[str, str]) -> str:
    return f"{row['session_id']}__{row['sample_id']}__{row['view_id']}"


def _validate_yolo_text(text: str, *, path: Path) -> str:
    normalized_lines: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 5 or tokens[0] != "0":
            raise ValueError(f"YOLO label must use class 0 and five tokens: {path}:{line_number}")
        try:
            center_x, center_y, width, height = (float(value) for value in tokens[1:])
        except ValueError as error:
            raise ValueError(f"YOLO label contains a non-numeric coordinate: {path}:{line_number}") from error
        if not all(math.isfinite(value) for value in (center_x, center_y, width, height)):
            raise ValueError(f"YOLO label contains a non-finite coordinate: {path}:{line_number}")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"YOLO label width and height must be positive: {path}:{line_number}")
        if not (
            -_YOLO_EDGE_TOLERANCE <= center_x - width / 2.0
            and center_x + width / 2.0 <= 1.0 + _YOLO_EDGE_TOLERANCE
            and -_YOLO_EDGE_TOLERANCE <= center_y - height / 2.0
            and center_y + height / 2.0 <= 1.0 + _YOLO_EDGE_TOLERANCE
        ):
            raise ValueError(f"YOLO label box lies outside normalized crop coordinates: {path}:{line_number}")
        normalized_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    return "" if not normalized_lines else "\n".join(normalized_lines) + "\n"


def _reviewed_yolo_labels(
    rows: Sequence[Mapping[str, str]],
    label_root: Path | None,
) -> tuple[dict[tuple[str, str, str], str], list[dict[str, str]]]:
    pending_rows = [row for row in rows if row["source_class"] not in {"normal", "no_streak"}]
    if label_root is None:
        return {}, pending_rows
    root = Path(label_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"reviewed YOLO label root does not exist: {root}")
    reviewed: dict[tuple[str, str, str], str] = {}
    expected_names: set[str] = set()
    for row in pending_rows:
        name = f"{_export_stem(row)}.txt"
        expected_names.add(name)
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing reviewed YOLO label: {path}")
        reviewed[(row["session_id"], row["sample_id"], row["view_id"])] = _validate_yolo_text(
            path.read_text(encoding="utf-8"),
            path=path,
        )
    actual_names = {path.name for path in root.glob("*.txt")}
    extras = sorted(actual_names - expected_names)
    if extras:
        raise ValueError(f"reviewed YOLO label root contains unexpected files: {extras[:3]}")
    return reviewed, []


def _relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise ValueError(f"branch export destination collision: {link}")
    link.symlink_to(os.path.relpath(target, link.parent))


def _crop_one(
    row: Mapping[str, str],
    roi_config: EightViewRoiConfig,
    target: Path,
) -> tuple[np.ndarray, str]:
    source = Path(row["source_path"]).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source image is missing or a symlink: {source}")
    if row["source_sha256"] and _sha256(source) != row["source_sha256"]:
        raise ValueError(f"source image SHA-256 mismatch: {source}")
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0 or image.dtype != np.uint8 or image.ndim not in {2, 3}:
        raise ValueError(f"source image is not a supported uint8 image: {source}")
    if image.shape[:2] != (roi_config.image_height, roi_config.image_width):
        raise ValueError(f"source image dimensions differ from ROI config: {source}")
    x1, y1, x2, y2 = roi_config.part_rois[row["view_id"]]
    crop = image[y1:y2, x1:x2].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"cannot write canonical ROI crop: {target}")
    decoded = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
    if decoded is None or not np.array_equal(decoded, crop):
        raise OSError(f"canonical ROI crop failed lossless round-trip validation: {target}")
    return crop, _sha256(target)


def _template_label(row: Mapping[str, str], yolo_text: str | None) -> str | None:
    if row["split"] == "train":
        return "normal" if row["source_class"] == "normal" else None
    if row["source_class"] in {"normal", "no_streak"}:
        return "normal"
    if yolo_text:
        return "defect"
    return None


def _materialize(
    *,
    staging: Path,
    rows: Sequence[dict[str, str]],
    roi_config: EightViewRoiConfig,
    reviewed_labels: Mapping[tuple[str, str, str], str],
    pending_rows: Sequence[dict[str, str]],
) -> dict[str, object]:
    crop_rows: list[dict[str, str]] = []
    template_rows: list[dict[str, str]] = []
    queue_rows: list[dict[str, str]] = []
    yolo_label_count = 0
    yolo_positive_count = 0
    for row in rows:
        stem = _export_stem(row)
        crop_relative = Path("crops") / row["view_id"] / f"{stem}.png"
        crop_target = staging / crop_relative
        crop, crop_sha256 = _crop_one(row, roi_config, crop_target)
        x1, y1, x2, y2 = roi_config.part_rois[row["view_id"]]
        crop_rows.append(
            {
                **row,
                "roi_x1": str(x1),
                "roi_y1": str(y1),
                "roi_x2": str(x2),
                "roi_y2": str(y2),
                "crop_path": crop_relative.as_posix(),
                "crop_sha256": crop_sha256,
                "crop_width": str(crop.shape[1]),
                "crop_height": str(crop.shape[0]),
            }
        )
        identity = (row["session_id"], row["sample_id"], row["view_id"])
        yolo_text = reviewed_labels.get(identity)

        if row["source_class"] in {"normal", "no_streak"}:
            if row["split"] == "train":
                ea_relative = (
                    Path("efficientad")
                    / row["view_id"]
                    / "normal"
                    / row["physical_part_id"]
                    / "images"
                    / f"{stem}.png"
                )
            elif row["split"] == "calibration":
                ea_relative = (
                    Path("efficientad")
                    / row["view_id"]
                    / "normal_test"
                    / row["physical_part_id"]
                    / "images"
                    / f"{stem}.png"
                )
            else:
                ea_relative = (
                    Path("efficientad/held_out")
                    / row["view_id"]
                    / "good"
                    / row["physical_part_id"]
                    / "images"
                    / f"{stem}.png"
                )
            _relative_symlink(crop_target, staging / ea_relative)
        elif yolo_text and row["split"] in {"calibration", "final_test"}:
            if row["split"] == "calibration":
                ea_relative = (
                    Path("efficientad")
                    / row["view_id"]
                    / "defect"
                    / row["source_class"]
                    / row["physical_part_id"]
                    / "images"
                    / f"{stem}.png"
                )
            else:
                ea_relative = (
                    Path("efficientad/held_out")
                    / row["view_id"]
                    / "defect"
                    / row["source_class"]
                    / row["physical_part_id"]
                    / "images"
                    / f"{stem}.png"
                )
            _relative_symlink(crop_target, staging / ea_relative)

        template_label = _template_label(row, yolo_text)
        if template_label is not None:
            template_relative = (
                Path("template")
                / row["view_id"]
                / row["split"]
                / template_label
                / f"{stem}.png"
            )
            _relative_symlink(crop_target, staging / template_relative)
            template_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "part_id": row["physical_part_id"],
                    "view_id": row["view_id"],
                    "image_path": os.path.relpath(crop_target, staging / "template"),
                    "split": row["split"],
                    "label": template_label,
                }
            )

        if row["source_class"] in {"normal", "no_streak"} or identity in reviewed_labels:
            yolo_split = _SPLIT_TO_YOLO[row["split"]]
            image_link = staging / "yolo/images" / yolo_split / f"{stem}.png"
            label_target = staging / "yolo/labels" / yolo_split / f"{stem}.txt"
            _relative_symlink(crop_target, image_link)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_text = yolo_text or ""
            label_target.write_text(label_text, encoding="utf-8")
            yolo_label_count += 1
            yolo_positive_count += int(bool(label_text))
        else:
            queue_rows.append(
                {
                    **row,
                    "crop_path": crop_relative.as_posix(),
                    "expected_label_filename": f"{stem}.txt",
                }
            )

    _write_csv(staging / "manifests/crop_manifest.csv", _CROP_FIELDS, crop_rows)
    _write_csv(staging / "template/trainer_manifest.csv", _TEMPLATE_FIELDS, template_rows)
    _write_csv(staging / "yolo/annotation_queue.csv", _QUEUE_FIELDS, queue_rows)
    if not pending_rows:
        (staging / "yolo/data.yaml").write_text(
            "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n",
            encoding="utf-8",
        )
    return {
        "crop_count": len(crop_rows),
        "template_row_count": len(template_rows),
        "yolo_label_count": yolo_label_count,
        "yolo_positive_count": yolo_positive_count,
        "yolo_pending_count": len(queue_rows),
        "yolo_training_ready": not pending_rows,
        "view_crop_counts": dict(sorted(Counter(row["view_id"] for row in rows).items())),
    }


def materialize_training_data(
    *,
    prepared_root: Path,
    roi_config_path: Path,
    output_root: Path,
    training_id: str,
    yolo_label_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Build one immutable ROI-crop release and three branch adapters."""
    if not isinstance(training_id, str) or not _TRAINING_ID.fullmatch(training_id):
        raise ValueError("training_id contains unsupported characters")
    prepared = Path(prepared_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    destination = output / training_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"training dataset release already exists: {destination}")
    roi_config = load_roi_config(roi_config_path)
    manifest_path = prepared / "manifests/dataset_manifest.csv"
    rows = _read_dataset_manifest(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    if roi_config.binding_mode == "prepared_manifest" and (
        prepared.name != roi_config.dataset_id
        or manifest_path.resolve() != roi_config.source_manifest
        or manifest_sha256 != roi_config.source_manifest_sha256
    ):
        raise ValueError("ROI config is bound to a different prepared dataset")
    if roi_config.binding_mode == "fixed_setup":
        report_path = prepared / "report.json"
        try:
            prepared_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read prepared report for capture_scope validation: {report_path}") from error
        prepared_capture_scope = prepared_report.get("capture_scope")
        if prepared_capture_scope != roi_config.capture_scope:
            raise ValueError(
                "fixed_setup ROI capture_scope differs from the prepared report capture_scope"
            )
    reviewed_labels, pending_rows = _reviewed_yolo_labels(rows, yolo_label_root)
    base_report: dict[str, object] = {
        "schema_version": 1,
        "training_id": training_id,
        "prepared_dataset_id": prepared.name,
        "prepared_manifest": str(manifest_path),
        "prepared_manifest_sha256": manifest_sha256,
        "roi_config": str(Path(roi_config_path).expanduser().resolve()),
        "roi_config_sha256": _sha256(Path(roi_config_path).expanduser().resolve()),
        "roi_binding_mode": roi_config.binding_mode,
        "roi_capture_scope": roi_config.capture_scope,
        "source_row_count": len(rows),
        "materialization": "canonical_png_plus_relative_branch_symlinks",
        "experimental_only": True,
    }
    if dry_run:
        for row in rows:
            source = Path(row["source_path"]).expanduser().resolve()
            if not source.is_file() or (row["source_sha256"] and _sha256(source) != row["source_sha256"]):
                raise ValueError(f"source image is missing or has a different SHA-256: {source}")
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[:2] != (roi_config.image_height, roi_config.image_width):
                raise ValueError(f"source image dimensions differ from ROI config: {source}")
        return {
            **base_report,
            "release_status": "dry_run",
            "yolo_pending_count": len(pending_rows),
            "yolo_training_ready": not pending_rows,
        }

    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{training_id}.", dir=output))
    try:
        shutil.copy2(Path(roi_config_path).expanduser().resolve(), staging / "roi_config.json")
        branch_report = _materialize(
            staging=staging,
            rows=rows,
            roi_config=roi_config,
            reviewed_labels=reviewed_labels,
            pending_rows=pending_rows,
        )
        report = {**base_report, **branch_report, "release_status": "published"}
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _atomic_publish_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
