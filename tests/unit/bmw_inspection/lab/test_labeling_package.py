"""Tests for the portable BMW YOLO labeling handoff package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.labeling_package import prepare_labeling_package


QUEUE_FIELDS = (
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
    "crop_path",
    "expected_label_filename",
)
CROP_FIELDS = QUEUE_FIELDS[:-2] + (
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
)


def _training_release(root: Path, *, missing_second_crop: bool = False) -> Path:
    release = root / "training-release"
    rows = []
    for index, (view, defect) in enumerate((("front", "deform"), ("back_left", "edge"))):
        filename = f"session-{index}__sample-{index}__{view}.png"
        crop = release / "crops" / view / filename
        if not (missing_second_crop and index == 1):
            crop.parent.mkdir(parents=True, exist_ok=True)
            crop.write_bytes(f"portable-image-{index}".encode())
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "physical_part_id": f"part-{index}",
                "session_id": f"session-{index}",
                "group_id": f"group{index:03d}",
                "view_id": view,
                "camera_serial": f"camera-{index}",
                "source_path": f"/source/{filename}",
                "source_sha256": str(index) * 64,
                "source_class": defect,
                "business_label": "NG",
                "split": "train",
                "crop_path": f"crops/{view}/{filename}",
                "expected_label_filename": f"{Path(filename).stem}.txt",
            }
        )
    queue = release / "yolo/annotation_queue.csv"
    queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return release


def _add_normal_reference_crops(release: Path, *, part_count: int = 2) -> None:
    rows = []
    for part_index in range(part_count):
        for view in VIEW_ORDER:
            filename = f"normal-session__normal-part-{part_index}__{view}.png"
            crop = release / "crops" / view / filename
            crop.parent.mkdir(parents=True, exist_ok=True)
            crop.write_bytes(f"normal-{part_index}-{view}".encode())
            rows.append(
                {
                    "sample_id": f"normal-sample-{part_index}",
                    "physical_part_id": f"normal-part-{part_index}",
                    "session_id": "normal-session",
                    "group_id": f"group{part_index:03d}",
                    "view_id": view,
                    "camera_serial": f"camera-{view}",
                    "source_path": f"/source/{filename}",
                    "source_sha256": "a" * 64,
                    "source_class": "normal",
                    "business_label": "OK",
                    "split": "train",
                    "roi_x1": "0",
                    "roi_y1": "0",
                    "roi_x2": "10",
                    "roi_y2": "10",
                    "crop_path": f"crops/{view}/{filename}",
                    "crop_sha256": "",
                    "crop_width": "10",
                    "crop_height": "10",
                }
            )
    manifest = release / "manifests/crop_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CROP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_prepares_portable_images_manifest_and_label_studio_tasks(tmp_path: Path) -> None:
    release = _training_release(tmp_path)

    report = prepare_labeling_package(
        training_root=release,
        output_root=tmp_path / "handoff",
        package_id="bmw-yolo-labeling-v1",
    )

    package = tmp_path / "handoff/bmw-yolo-labeling-v1"
    images = sorted((package / "images").glob("*.png"))
    assert report["task_count"] == 2
    assert report["image_bytes"] == sum(path.stat().st_size for path in images)
    assert len(images) == 2
    assert all(path.is_file() and not path.is_symlink() for path in images)

    tasks = json.loads((package / "label_studio/tasks.json").read_text(encoding="utf-8"))
    assert [task["data"]["image"] for task in tasks] == [
        "/data/local-files/?d=images/session-0__sample-0__front.png",
        "/data/local-files/?d=images/session-1__sample-1__back_left.png",
    ]
    assert tasks[0]["data"]["source_class"] == "deform"
    assert tasks[1]["data"]["expected_label_filename"].endswith("back_left.txt")

    manifest = list(
        csv.DictReader((package / "annotation_tasks.csv").open(newline="", encoding="utf-8"))
    )
    assert [row["image_filename"] for row in manifest] == [path.name for path in images]
    assert "<Label value=\"defect\"" in (package / "label_studio/label_config.xml").read_text(
        encoding="utf-8"
    )
    assert "没有可见缺陷" in (package / "标注说明.md").read_text(encoding="utf-8")
    assert len((package / "SHA256SUMS").read_text(encoding="utf-8").splitlines()) == 2


def test_missing_crop_does_not_publish_partial_package(tmp_path: Path) -> None:
    release = _training_release(tmp_path, missing_second_crop=True)
    output = tmp_path / "handoff"

    with pytest.raises(ValueError, match="missing crop image"):
        prepare_labeling_package(
            training_root=release,
            output_root=output,
            package_id="bmw-yolo-labeling-v1",
        )

    assert not (output / "bmw-yolo-labeling-v1").exists()


def test_adds_complete_normal_parts_as_per_view_reference_images(tmp_path: Path) -> None:
    release = _training_release(tmp_path)
    _add_normal_reference_crops(release, part_count=2)

    report = prepare_labeling_package(
        training_root=release,
        output_root=tmp_path / "handoff",
        package_id="bmw-yolo-labeling-with-references-v2",
        normal_references_per_view=2,
    )

    package = tmp_path / "handoff/bmw-yolo-labeling-with-references-v2"
    assert report["normal_reference_count"] == 16
    assert report["normal_references_per_view"] == 2
    assert report["normal_reference_part_ids"] == ["normal-part-0", "normal-part-1"]
    for view in VIEW_ORDER:
        references = sorted((package / "normal_references" / view).glob("*.png"))
        assert len(references) == 2
        assert all(path.is_file() and not path.is_symlink() for path in references)
    manifest = list(
        csv.DictReader((package / "normal_reference_manifest.csv").open(newline="", encoding="utf-8-sig"))
    )
    assert len(manifest) == 16
    assert {row["source_class"] for row in manifest} == {"normal"}
    assert "每个视角：2张" in (package / "标注说明.md").read_text(encoding="utf-8")
