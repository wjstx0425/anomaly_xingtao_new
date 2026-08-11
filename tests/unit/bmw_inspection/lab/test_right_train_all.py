# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""BMW right-hand one-click training contracts."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import zipfile
from pathlib import Path

from bmw_inspection.lab import eight_view_train_all as train_all
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_right_module():
    spec = importlib.util.find_spec("bmw_inspection.lab.right_train_all")
    assert spec is not None, "right_train_all module must exist"
    return importlib.import_module("bmw_inspection.lab.right_train_all")


def _write_label_studio_export(root: Path) -> tuple[Path, Path]:
    names = ("session-a__right-defect-001__front.txt", "session-a__right-defect-001__back.txt")
    json_path = root / "project.json"
    json_path.write_text(
        json.dumps([{"data": {"expected_label_filename": name}, "annotations": [{"result": []}]} for name in names]),
        encoding="utf-8",
    )
    zip_path = root / "project.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("classes.txt", "defect\n无可见缺陷\n有可见缺陷\n")
        archive.writestr(f"labels/{names[0]}", "0 0.500000 0.500000 0.100000 0.100000\n")
        archive.writestr(f"labels/{names[1]}", "")
    return zip_path, json_path


def test_required_yolo_labels_are_derived_from_the_selected_prepared_dataset(tmp_path: Path) -> None:
    assert hasattr(train_all, "required_yolo_label_names")
    manifest = tmp_path / "manifests/dataset_manifest.csv"
    manifest.parent.mkdir(parents=True)
    fields = (
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
    rows = [
        {
            "sample_id": "right-defect-001",
            "physical_part_id": "right-defect",
            "session_id": "session-a",
            "group_id": "group001",
            "view_id": "front",
            "camera_serial": "camera-a",
            "source_path": "/tmp/defect.png",
            "source_sha256": "0" * 64,
            "source_class": "deform",
            "business_label": "NG",
            "split": "train",
        },
        {
            "sample_id": "right-normal-001",
            "physical_part_id": "right-normal",
            "session_id": "session-b",
            "group_id": "group001",
            "view_id": "front",
            "camera_serial": "camera-a",
            "source_path": "/tmp/normal.png",
            "source_sha256": "1" * 64,
            "source_class": "normal",
            "business_label": "OK",
            "split": "train",
        },
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    names = train_all.required_yolo_label_names(tmp_path)

    assert names == ("session-a__right-defect-001__front.txt",)


def test_complete_release_report_accepts_dataset_specific_balanced_view_count() -> None:
    assert hasattr(train_all, "training_release_report_is_complete")
    report = {
        "yolo_training_ready": True,
        "view_crop_counts": {view: 129 for view in VIEW_ORDER},
    }

    assert train_all.training_release_report_is_complete(report, VIEW_ORDER)
    assert not train_all.training_release_report_is_complete(
        {**report, "view_crop_counts": {**report["view_crop_counts"], "front": 128}},
        VIEW_ORDER,
    )


def test_prepares_and_reuses_reviewed_yolo_label_cache(tmp_path: Path) -> None:
    module = _load_right_module()
    zip_path, json_path = _write_label_studio_export(tmp_path)
    output = tmp_path / "reviewed_yolo_labels"

    created = module.prepare_reviewed_yolo_labels(zip_path, json_path, output)
    reused = module.prepare_reviewed_yolo_labels(zip_path, json_path, output)

    assert created == {"status": "created", "label_count": 2, "positive_label_count": 1, "label_root": str(output)}
    assert reused == {**created, "status": "reused"}
    assert sorted(path.name for path in output.glob("*.txt")) == [
        "session-a__right-defect-001__back.txt",
        "session-a__right-defect-001__front.txt",
    ]


def test_right_cli_defaults_to_supplied_dataset_and_yolo_batch_32() -> None:
    script = REPO_ROOT / "pipeline/bmw_lab_train_right.py"
    assert script.is_file(), "right-hand training CLI must exist"
    spec = importlib.util.spec_from_file_location("bmw_lab_train_right_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.prepared_root == REPO_ROOT / "dataset/bmw_lab_prepared/bmw_right_complete_20260810_v1"
    assert args.roi_config == REPO_ROOT / "configs/bmw/rois/bmw_right_hdr_eight_view_v1.json"
    assert args.training_id == "bmw_right_complete_roi_reviewed_v1"
    assert args.run_id == "bmw_right_eight_view_v1"
    assert args.efficientad_epochs == 30
    assert args.yolo_epochs == 100
    assert args.yolo_batch == 32
    assert args.yolo_imgsz == 640


def test_multisource_cli_defaults_add_right_batch_and_left_yolo() -> None:
    assert hasattr(train_all, "preflight_model_assets")
    script = REPO_ROOT / "pipeline/bmw_lab_train_right_multisource.py"
    assert script.is_file(), "multisource training CLI must exist"
    spec = importlib.util.spec_from_file_location("bmw_lab_train_right_multisource_cli", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.right_release == REPO_ROOT / "dataset/bmw_lab_training/bmw_right_complete_roi_v1"
    assert args.additional_right_release == (
        REPO_ROOT / "dataset/bmw_lab_training/bmw_right_batch_20260810_21_roi_v1"
    )
    assert args.left_yolo_release == (
        REPO_ROOT / "dataset/bmw_lab_training/bmw_hdr_roi_training_reviewed_v1"
    )
    assert args.training_id == "bmw_right_multisource_left_yolo_v1"
    assert args.run_id == "bmw_right_multisource_left_yolo_v1"
    assert args.yolo_batch == 32
