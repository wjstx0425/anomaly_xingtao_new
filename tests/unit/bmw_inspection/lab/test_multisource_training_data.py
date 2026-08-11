# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""BMW multisource composite training-data tests."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
from pathlib import Path

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


def _load_module():
    spec = importlib.util.find_spec("bmw_inspection.lab.multisource_training_data")
    assert spec is not None, "multisource training-data module must exist"
    return importlib.import_module("bmw_inspection.lab.multisource_training_data")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _branch_release(root: Path, name: str, session: str) -> Path:
    release = root / name
    prepared = root / f"{name}_prepared"
    bright_fields = (
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
    source = prepared / f"{session}_bright.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"bright")
    _write_csv(
        prepared / "manifests/bright_streak.csv",
        bright_fields,
        [
            {
                "sample_id": "normal-001",
                "physical_part_id": "normal-part",
                "session_id": session,
                "group_id": "group001",
                "view_id": "front_left",
                "camera_serial": "camera-left",
                "source_path": str(source),
                "source_sha256": "0" * 64,
                "source_class": "normal",
                "business_label": "OK",
                "split": "calibration",
                "expected_status": "OK",
                "review_reason": "known_normal",
            }
        ],
    )
    template_rows: list[dict[str, str]] = []
    for view in VIEW_ORDER:
        crop = release / "crops" / view / f"{session}__normal-001__{view}.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(view.encode())
        ea = release / "efficientad" / view / "normal" / "normal-part" / "images" / crop.name
        ea.parent.mkdir(parents=True, exist_ok=True)
        ea.symlink_to(crop)
        template_rows.append({
            "sample_id": "normal-001",
            "part_id": "normal-part",
            "view_id": view,
            "image_path": str(crop),
            "split": "train",
            "label": "normal",
        })
    _write_csv(
        release / "template/trainer_manifest.csv",
        ("sample_id", "part_id", "view_id", "image_path", "split", "label"),
        template_rows,
    )
    report = {
        "release_status": "published",
        "prepared_manifest": str(prepared / "manifests/dataset_manifest.csv"),
        "view_crop_counts": {view: 1 for view in VIEW_ORDER},
    }
    (release / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return release


def _right_yolo_source(release: Path, label_root: Path) -> None:
    normal_name = "right_session__right-normal__front.png"
    image = release / "yolo/images/train" / normal_name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"right-normal")
    label = release / "yolo/labels/train" / normal_name.replace(".png", ".txt")
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("", encoding="utf-8")

    defect_name = "right_session__right-defect__front"
    crop = release / "crops/front" / f"{defect_name}.png"
    crop.write_bytes(b"right-defect")
    _write_csv(
        release / "yolo/annotation_queue.csv",
        (
            "sample_id",
            "physical_part_id",
            "view_id",
            "source_class",
            "split",
            "crop_path",
            "expected_label_filename",
        ),
        [
            {
                "sample_id": "right-defect",
                "physical_part_id": "right-defect-part",
                "view_id": "front",
                "source_class": "deform",
                "split": "calibration",
                "crop_path": f"crops/front/{defect_name}.png",
                "expected_label_filename": f"{defect_name}.txt",
            }
        ],
    )
    label_root.mkdir(parents=True)
    (label_root / f"{defect_name}.txt").write_text(
        "0 0.500000 0.500000 0.100000 0.100000\n",
        encoding="utf-8",
    )


def _left_yolo_release(root: Path) -> Path:
    release = root / "left_reviewed"
    image = release / "yolo/images/test/left_session__left-defect__front.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"left-defect")
    label = release / "yolo/labels/test/left_session__left-defect__front.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.400000 0.400000 0.100000 0.100000\n", encoding="utf-8")
    (release / "yolo/data.yaml").write_text("names:\n  0: defect\n", encoding="utf-8")
    (release / "report.json").write_text(
        json.dumps({"release_status": "published", "yolo_training_ready": True}),
        encoding="utf-8",
    )
    return release


def test_builds_namespaced_right_branches_and_paired_left_right_yolo(tmp_path: Path) -> None:
    module = _load_module()
    base = _branch_release(tmp_path, "right_base", "session_a")
    additional = _branch_release(tmp_path, "right_additional", "session_b")
    labels = tmp_path / "reviewed_labels"
    _right_yolo_source(base, labels)
    left = _left_yolo_release(tmp_path)
    output = tmp_path / "training"

    dry = module.build_multisource_training_data(
        branch_releases=(base, additional),
        left_yolo_release=left,
        right_yolo_release=base,
        right_yolo_label_root=labels,
        output_root=output,
        training_id="combined-v1",
        dry_run=True,
    )
    assert dry["release_status"] == "dry_run"
    assert dry["template_row_count"] == 17
    assert dry["bright_streak_row_count"] == 2
    assert dry["efficientad_image_count"] == 17
    assert dry["yolo_image_count"] == 3
    assert not (output / "combined-v1").exists()

    report = module.build_multisource_training_data(
        branch_releases=(base, additional),
        left_yolo_release=left,
        right_yolo_release=base,
        right_yolo_label_root=labels,
        output_root=output,
        training_id="combined-v1",
    )
    release = output / "combined-v1"
    assert report["release_status"] == "published"
    assert report["yolo_training_ready"] is True
    assert len(list((release / "yolo/images").rglob("*.png"))) == 3
    assert len(list((release / "yolo/labels").rglob("*.txt"))) == 3
    assert (release / "yolo/data.yaml").is_file()
    template_rows = list(csv.DictReader((release / "template/trainer_manifest.csv").open()))
    assert len({row["sample_id"] for row in template_rows}) == 3
    assert sum("::normal-001" in row["sample_id"] for row in template_rows) == 16
    assert len(list((release / "efficientad/front/normal").rglob("*.png"))) == 2
    assert len(list((release / "efficientad/front/defect").rglob("*.png"))) == 1
