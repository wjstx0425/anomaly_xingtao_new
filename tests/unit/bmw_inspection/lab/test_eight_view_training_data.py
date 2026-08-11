"""Tests for BMW eight-view ROI training-data materialization."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_roi import EightViewRoiConfig, save_roi_config
from bmw_inspection.lab.eight_view_training_data import _validate_yolo_text, materialize_training_data

FIELDS = (
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


def _release(root: Path) -> tuple[Path, Path, dict[tuple[str, str], Path]]:
    release = root / "bmw-test-v1"
    specs = (
        ("normal-sample", "normal", "train"),
        ("no-streak-sample", "no_streak", "calibration"),
        ("edge-sample", "edge", "train"),
    )
    paths: dict[tuple[str, str], Path] = {}
    rows: list[dict[str, str]] = []
    for sample_index, (sample_id, source_class, split) in enumerate(specs):
        for view_index, view in enumerate(VIEW_ORDER):
            image_path = root / "source" / f"{sample_id}__{view}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            grid = np.arange(120, dtype=np.uint8).reshape(10, 12)
            image = np.stack((grid, grid + sample_index * 20, grid + view_index), axis=2)
            assert cv2.imwrite(str(image_path), image)
            paths[(sample_id, view)] = image_path
            rows.append(
                {
                    "sample_id": sample_id,
                    "physical_part_id": sample_id,
                    "session_id": f"session-{sample_index}",
                    "group_id": f"group{sample_index}",
                    "view_id": view,
                    "camera_serial": "serial",
                    "source_path": str(image_path),
                    "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "source_class": source_class,
                    "business_label": "OK" if source_class == "normal" else "NG",
                    "split": split,
                }
            )
    manifest = release / "manifests/dataset_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (release / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_status": "published",
                "dataset_id": release.name,
                "image_width": 12,
                "image_height": 10,
                "manifest_sha256": {"dataset_manifest.csv": manifest_hash},
            }
        ),
        encoding="utf-8",
    )
    roi_path = root / "rois.json"
    save_roi_config(
        roi_path,
        EightViewRoiConfig(
            dataset_id=release.name,
            source_manifest=manifest,
            source_manifest_sha256=manifest_hash,
            representative_sample_id="normal-sample",
            image_width=12,
            image_height=10,
            part_rois={view: (1, 2, 9, 8) for view in VIEW_ORDER},
        ),
    )
    return release, roi_path, paths


def _stem(session: str, sample: str, view: str) -> str:
    return f"{session}__{sample}__{view}"


def test_materializes_exact_crops_and_truthful_branch_layouts(tmp_path: Path) -> None:
    release, roi_path, source_paths = _release(tmp_path)

    report = materialize_training_data(
        prepared_root=release,
        roi_config_path=roi_path,
        output_root=tmp_path / "training",
        training_id="bmw-train-v1",
    )

    output = tmp_path / "training/bmw-train-v1"
    assert report["crop_count"] == 24
    assert report["yolo_pending_count"] == 8
    assert report["yolo_training_ready"] is False
    assert not (output / "yolo/data.yaml").exists()

    crop = output / "crops/front/session-0__normal-sample__front.png"
    expected = cv2.imread(str(source_paths[("normal-sample", "front")]), cv2.IMREAD_UNCHANGED)[2:8, 1:9]
    assert np.array_equal(cv2.imread(str(crop), cv2.IMREAD_UNCHANGED), expected)

    efficientad_train = output / "efficientad/front/normal/normal-sample/images/session-0__normal-sample__front.png"
    efficientad_calibration = (
        output
        / "efficientad/front/normal_test/no-streak-sample/images/session-1__no-streak-sample__front.png"
    )
    assert efficientad_train.is_symlink()
    assert efficientad_train.resolve() == crop.resolve()
    assert efficientad_calibration.is_symlink()

    template_rows = list(csv.DictReader((output / "template/trainer_manifest.csv").open()))
    assert len(template_rows) == 16
    assert {row["label"] for row in template_rows} == {"normal"}
    assert {row["sample_id"] for row in template_rows} == {"normal-sample", "no-streak-sample"}

    negative_labels = sorted((output / "yolo/labels").glob("*/*.txt"))
    assert len(negative_labels) == 16
    assert all(path.read_text(encoding="utf-8") == "" for path in negative_labels)
    queue = list(csv.DictReader((output / "yolo/annotation_queue.csv").open()))
    assert len(queue) == 8
    assert {row["source_class"] for row in queue} == {"edge"}


def test_complete_yolo_review_publishes_training_yaml(tmp_path: Path) -> None:
    release, roi_path, _paths = _release(tmp_path)
    labels = tmp_path / "reviewed_labels"
    labels.mkdir()
    for view_index, view in enumerate(VIEW_ORDER):
        path = labels / f"{_stem('session-2', 'edge-sample', view)}.txt"
        path.write_text("0 0.500000 0.500000 0.250000 0.250000\n" if view_index == 0 else "", encoding="utf-8")

    report = materialize_training_data(
        prepared_root=release,
        roi_config_path=roi_path,
        output_root=tmp_path / "training",
        training_id="bmw-train-reviewed-v1",
        yolo_label_root=labels,
    )

    output = tmp_path / "training/bmw-train-reviewed-v1"
    assert report["yolo_pending_count"] == 0
    assert report["yolo_training_ready"] is True
    assert (output / "yolo/data.yaml").read_text(encoding="utf-8").endswith("names:\n  0: defect\n")
    yolo_labels = sorted((output / "yolo/labels").glob("*/*.txt"))
    assert len(yolo_labels) == 24
    positive = output / "yolo/labels/train/session-2__edge-sample__front.txt"
    assert positive.read_text(encoding="utf-8").startswith("0 0.500000")


def test_fixed_setup_roi_materializes_a_different_prepared_release_with_same_dimensions(tmp_path: Path) -> None:
    release, _bound_roi_path, _paths = _release(tmp_path)
    manifest = release / "manifests/dataset_manifest.csv"
    reusable_roi_path = tmp_path / "right-reusable-rois.json"
    save_roi_config(
        reusable_roi_path,
        EightViewRoiConfig(
            dataset_id="bmw-right-fixed-setup-v1",
            source_manifest=manifest,
            source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            representative_sample_id="normal-sample",
            image_width=12,
            image_height=10,
            part_rois={view: (1, 2, 9, 8) for view in VIEW_ORDER},
            binding_mode="fixed_setup",
            capture_scope="right",
        ),
    )

    report = materialize_training_data(
        prepared_root=release,
        roi_config_path=reusable_roi_path,
        output_root=tmp_path / "training",
        training_id="bmw-right-reused-v1",
    )

    assert report["roi_binding_mode"] == "fixed_setup"
    assert report["roi_capture_scope"] == "right"
    assert report["prepared_dataset_id"] == "bmw-test-v1"
    assert report["prepared_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["crop_count"] == 24


def test_missing_or_invalid_review_labels_fail_without_release(tmp_path: Path) -> None:
    release, roi_path, _paths = _release(tmp_path)
    labels = tmp_path / "reviewed_labels"
    labels.mkdir()
    for view in VIEW_ORDER[:-1]:
        (labels / f"{_stem('session-2', 'edge-sample', view)}.txt").write_text("", encoding="utf-8")
    output_root = tmp_path / "training"

    with pytest.raises(ValueError, match="missing reviewed YOLO label"):
        materialize_training_data(
            prepared_root=release,
            roi_config_path=roi_path,
            output_root=output_root,
            training_id="bmw-train-invalid-v1",
            yolo_label_root=labels,
        )

    assert not (output_root / "bmw-train-invalid-v1").exists()


def test_yolo_validation_tolerates_only_six_decimal_serialization_error(tmp_path: Path) -> None:
    path = tmp_path / "boundary.txt"

    normalized = _validate_yolo_text(
        "0 0.454308 0.959184 0.146040 0.081633\n",
        path=path,
    )

    assert normalized == "0 0.454308 0.959184 0.146040 0.081633\n"
    with pytest.raises(ValueError, match="outside normalized"):
        _validate_yolo_text(
            "0 0.454308 0.959186 0.146040 0.081633\n",
            path=path,
        )
